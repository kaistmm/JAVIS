"""Clip manifest {id, wav, ...} -> CosyVoice3 speech tokens (25 Hz), one row per clip:
{id, wav, token = 10 x <|audio_bos|> + [AU####]..., n_codes}. Extra manifest keys listed in CARRY are copied through.

  python data_prep/extract_speech_tokens.py --manifest clips.jsonl --output_dir data/<set>/tokens [--speech_cpu_threads 8]
  # sharded: --rank r --world_size n, then cat speech_tokens.jsonl.rank* > speech_tokens.jsonl
The ONNX CUDA provider is ~10x slower than CPU on some machines; --speech_cpu_threads N uses the CPU provider.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.constant import AUDIO_BOS_TOKEN  # noqa: E402
from utils.cosyvoice_tokenizer import extract_speech_token, get_audio_tokenizer  # noqa: E402

CARRY = ("mp4", "text", "interaction_key", "idx", "episode", "speaker", "start", "end",
         "gap_prev", "dialog_uid", "turn_index", "gender", "emotion")


def collect_processed_ids(out_path):
    """Ids already in the rank shard; truncates a partial trailing line (same as
    va_speech_token_extract.collect_processed_ids)."""
    if not os.path.exists(out_path):
        return set()
    processed = set()
    valid_end = 0
    with open(out_path, "rb") as f:
        for raw in f:
            try:
                processed.add(json.loads(raw.decode("utf-8"))["id"])
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
                break
            valid_end += len(raw)
    if valid_end < os.path.getsize(out_path):
        with open(out_path, "r+b") as f:
            f.truncate(valid_end)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="speech_tokens.jsonl")
    parser.add_argument("--speech_onnx", default=str(PROJECT_ROOT / "pretrained_models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx"))
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--speech_workers", type=int, default=4)
    parser.add_argument("--speech_cpu_threads", type=int, default=0, help=">0: CPU EP with N intra-op threads")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--skip_ids_glob", default="", help="also skip ids found in these shards — makes a world_size change lossless")
    parser.add_argument("--flush_every", type=int, default=50)
    args = parser.parse_args()
    rank, world = args.rank, args.world_size

    rows = []
    with open(args.manifest) as f:
        for line in f:
            rows.append(json.loads(line))
            if args.num_samples is not None and len(rows) >= args.num_samples * world:
                break
    my_rows = rows[rank::world]
    if args.num_samples is not None:
        my_rows = my_rows[: args.num_samples]
    print(f"[rank{rank}/{world}] {len(rows)} rows in manifest, {len(my_rows)} mine", flush=True)

    if args.speech_cpu_threads > 0:
        session = get_audio_tokenizer(args.speech_onnx, use_cuda=False, intra_op_num_threads=args.speech_cpu_threads)
    else:
        session = get_audio_tokenizer(args.speech_onnx, device_id=0)  # pinned via CUDA_VISIBLE_DEVICES
    print(f"[rank{rank}] ONNX providers: {session.get_providers()}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.rank{rank}" if world > 1 else args.output_name)
    processed = collect_processed_ids(out_path) if args.resume else set()
    if args.skip_ids_glob:
        import glob as _glob
        n0 = len(processed)
        for p in _glob.glob(args.skip_ids_glob):
            with open(p) as f:
                for raw in f:
                    try:
                        processed.add(json.loads(raw)["id"])
                    except (json.JSONDecodeError, KeyError):
                        break  # partial tail line
        print(f"[rank{rank}] skip_ids_glob: +{len(processed) - n0} ids from {args.skip_ids_glob}", flush=True)
    if processed:
        before = len(my_rows)
        my_rows = [r for r in my_rows if r["id"] not in processed]
        print(f"[rank{rank}] resume: {len(processed)} done in {out_path}, {before - len(my_rows)} skipped, {len(my_rows)} remaining", flush=True)

    n_ok = n_missing = n_err = 0
    pool = ThreadPoolExecutor(max_workers=args.speech_workers)
    try:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(my_rows, desc=f"rank{rank}")
        except ImportError:
            iterator = my_rows
        with open(out_path, "a" if processed else "w") as fout:
            futures = {}
            def emit(meta, codes):
                nonlocal n_ok
                row = {
                    "id": meta["id"],
                    "wav": meta["wav"],
                    "token": AUDIO_BOS_TOKEN * args.delay + "".join(f"[AU{c:04d}]" for c in codes),
                    "n_codes": len(codes),
                }
                row.update({k: meta[k] for k in CARRY if k in meta})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1
                if n_ok % args.flush_every == 0:
                    fout.flush()
            for meta in iterator:
                wav = meta["wav"]
                if not os.path.exists(wav):
                    n_missing += 1
                    print(f"[rank{rank}] missing wav id={meta['id']}", flush=True)
                    continue
                futures[pool.submit(extract_speech_token, session, wav)] = meta
                if len(futures) >= args.speech_workers * 2:  # bounded in-flight, ordered-ish drain
                    for fut in list(futures):
                        m = futures.pop(fut)
                        try:
                            emit(m, fut.result())
                        except Exception as e:
                            n_err += 1
                            print(f"[rank{rank}] extract failed id={m['id']}: {e}", flush=True)
            for fut, m in futures.items():
                try:
                    emit(m, fut.result())
                except Exception as e:
                    n_err += 1
                    print(f"[rank{rank}] extract failed id={m['id']}: {e}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"[rank{rank}] done: {n_ok} ok, {n_missing} missing_wav, {n_err} errors → {out_path}", flush=True)


if __name__ == "__main__":
    main()
