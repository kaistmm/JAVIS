"""Per-dimension mcoeff mean / std over packed stores -> stats.pt, the head's normalization (`mc_mean`, `mc_std`).
JAVIS computed it once on all kept Talker-T2AV clips (train + val) and kept it for stage 2 (the stats travel inside the checkpoint).

  python data_prep/compute_mcoeff_stats.py --stores data/talker/store --out data/talker/stats.pt
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--exclude_ids", default=None, help="text file, one clip id per line (e.g. validation clips)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    skip = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()

    N, s1, s2 = 0, np.zeros(20), np.zeros(20)
    for store in a.stores:
        mm = np.memmap(os.path.join(store, "feats.bin"), dtype="<f4", mode="r")
        for line in open(os.path.join(store, "index.jsonl")):
            r = json.loads(line)
            if r["id"] in skip:
                continue
            x = np.asarray(mm[r["offset"] * 20:(r["offset"] + r["frames"]) * 20], dtype=np.float64).reshape(-1, 20)
            N += len(x)
            s1 += x.sum(0)
            s2 += (x ** 2).sum(0)
    mean = s1 / N
    std = np.sqrt(np.maximum(s2 / N - mean ** 2, 0.0))
    torch.save({"mean": torch.tensor(mean, dtype=torch.float32), "std": torch.tensor(std, dtype=torch.float32)}, a.out)
    print(f"{N} frames; std min {std.min():.4g} max {std.max():.4g} -> {a.out}")


if __name__ == "__main__":
    main()
