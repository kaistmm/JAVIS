"""Stage-1 rows (monologue "repeat after me"): the user turn is a clip's audio, the assistant re-emits the same
utterance as text + speech tokens + that clip's mcoeff.

  user : {"path": wav, "token": <|audio_pad|> * M}
  asst : {"ref_path": wav, "text": ..., "token": 10 x <|audio_bos|> + [AU####] * M, "mcoeff_bin", "mcoeff_offset", "mcoeff_frames": M}

M = min(#speech tokens, #mcoeff frames); a clip whose two lengths differ by more than --max_len_diff is a bad
speech/motion pairing and is dropped. clips.jsonl rows: `id`, `text`, optional `split` (train|val, default train).

  python data_prep/build_stage1_rows.py --clips clips.jsonl --tokens tokens/speech_tokens.jsonl --store store --out_dir rows
"""
import argparse
import json
import os
import re
from collections import Counter

SYSTEM = "You are asked to generate text, speech tokens, and motion tokens at the same time."
USER_PROMPT = "Repeat the following exactly: <|audio_bos|><|AUDIO|><|audio_eos|>"
AUDIO = "<|audio_bos|><|AUDIO|><|audio_eos|>"
AU = re.compile(r"\[AU\d{4}\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--tokens", required=True, help="extract_speech_tokens.py output")
    ap.add_argument("--store", required=True, help="pack_mcoeff.py output dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude", default=None, help="text file of clip ids to drop (e.g. the Talker-T2AV filters)")
    ap.add_argument("--max_len_diff", type=int, default=5)
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=2048, help="same cap as build_stage2_rows.py; longer clips do not fit the sequence cutoff or a 24 GB GPU")
    a = ap.parse_args()

    tok = {}
    for line in open(a.tokens):
        d = json.loads(line)
        tok[d["id"]] = d["token"]
    store = {}
    for line in open(os.path.join(a.store, "index.jsonl")):
        d = json.loads(line)
        store[d["id"]] = (d["offset"], d["frames"])
    binp = os.path.abspath(os.path.join(a.store, "feats.bin"))
    ex = set(open(a.exclude).read().split()) if a.exclude else set()

    os.makedirs(a.out_dir, exist_ok=True)
    fh = {s: open(os.path.join(a.out_dir, f"{s}.jsonl"), "w") for s in ("train", "val")}
    stats = Counter()
    for line in open(a.clips):
        c = json.loads(line)
        if c["id"] in ex:
            stats["excluded"] += 1
            continue
        if c["id"] not in tok or c["id"] not in store:
            stats["no_tokens_or_mcoeff"] += 1
            continue
        t = tok[c["id"]]
        au = AU.findall(t)
        off, n_mc = store[c["id"]]
        if abs(len(au) - n_mc) > a.max_len_diff:
            stats["len_diff"] += 1
            continue
        M = min(len(au), n_mc)
        if M < a.min_frames:
            stats["too_short"] += 1
            continue
        if M > a.max_frames:
            stats["too_long"] += 1
            continue
        bos = t[:t.index("[AU")]
        row = {"system": SYSTEM,
               "messages": [{"role": "user", "content": USER_PROMPT}, {"role": "assistant", "content": AUDIO}],
               "audio": [json.dumps({"path": c["wav"], "ref_path": "", "ref_text": "", "text": "", "token": "<|audio_pad|>" * M}, ensure_ascii=False),
                         json.dumps({"path": "", "ref_path": c["wav"], "ref_text": "", "text": c.get("text", ""), "token": bos + "".join(au[:M]),
                                     "mcoeff_bin": binp, "mcoeff_offset": off, "mcoeff_frames": M}, ensure_ascii=False)]}
        split = c.get("split", "train")
        fh[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] += 1
    for f in fh.values():
        f.close()
    print(dict(stats))


if __name__ == "__main__":
    main()
