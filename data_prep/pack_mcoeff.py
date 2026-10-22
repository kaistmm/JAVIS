"""Per-clip FLOAT mcoeff .pt files -> one packed store: feats.bin (float32 frames x 20) + index.jsonl + meta.json.

The training collator reads a memmap slice (mcoeff_bin / mcoeff_offset / mcoeff_frames) instead of opening one file per
utterance, and refuses a store whose meta.json feat_dim is not 20.

clips.jsonl rows need `id`, `wav`, `video`; the mcoeff of a clip is <mcoeff_dir>/<video path relative to video_root, no
extension>.pt, i.e. what extract_mcoeff.py --dir_path <video_root> --output_dir <out> wrote under <out>/mcoeffs.
`group` / `turn` (same person + same video, utterance order) are copied into the index for build_gcond_tables.py.

  python data_prep/pack_mcoeff.py --clips clips.jsonl --video_root videos/ --mcoeff_dir mc/mcoeffs --out data/<set>/store
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--mcoeff_dir", required=True)
    ap.add_argument("--out", required=True, help="store directory; its name is the key of its gcond table")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.clips)]
    os.makedirs(a.out, exist_ok=True)

    def load(r):
        p = os.path.join(a.mcoeff_dir, os.path.splitext(os.path.relpath(r["video"], a.video_root))[0] + ".pt")
        try:
            t = torch.load(p, weights_only=True)
        except Exception:
            return None
        return t.float().numpy() if t.ndim == 2 and t.shape[1] == 20 and t.shape[0] > 0 else None

    off = missing = 0
    with open(os.path.join(a.out, "feats.bin"), "wb") as fb, open(os.path.join(a.out, "index.jsonl"), "w") as fi, \
            ThreadPoolExecutor(a.workers) as pool:
        for r, mc in zip(rows, pool.map(load, rows, chunksize=32)):
            if mc is None:
                missing += 1
                continue
            fb.write(np.ascontiguousarray(mc, dtype="<f4").tobytes())
            rec = {"id": r["id"], "wav": r["wav"], "offset": off, "frames": len(mc)}
            rec.update({k: r[k] for k in ("group", "turn") if k in r})
            fi.write(json.dumps(rec, ensure_ascii=False) + "\n")
            off += len(mc)
    json.dump(dict(feat_dim=20, frame_rate=25, dtype_feat="<f4", feat_space="float_mcoeff", n_clips=len(rows) - missing,
                   missing=missing, total_frames=off), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"{len(rows) - missing} clips ({missing} without mcoeff), {off} frames -> {a.out}")


if __name__ == "__main__":
    main()
