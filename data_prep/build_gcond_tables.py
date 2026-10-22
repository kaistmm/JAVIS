"""Sibling tables for the gcond policy: during training, a target clip's global condition is the middle frame of
another clip of the same person in the same video, at least 2 utterances away (falling back to any other clip of that
group). This makes the training condition mean what it means at inference (a reference portrait of the speaker) and
removes the target turn's own motion state from it.

Reads a packed store whose index.jsonl carries `group` (same person + same video) and `turn` (utterance order) from
the clip manifest and writes <store>/gcond.pt, which training reads with FUNAUDIOCHAT_GCOND_SIBLING=1. A clip alone in its
group falls back to its own middle frame.

  python data_prep/build_gcond_tables.py --stores data/talker/store data/seamless/store data/multidialog/store
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    a = ap.parse_args()
    for store in a.stores:
        rows = sorted((json.loads(l) for l in open(os.path.join(store, "index.jsonl"))), key=lambda r: int(r["offset"]))
        if rows and ("group" not in rows[0] or "turn" not in rows[0]):
            raise ValueError(f"{store}: index.jsonl needs `group` and `turn` (put them in the clip manifest before packing)")
        gids, members = {}, []
        group = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            if r["group"] not in gids:
                gids[r["group"]] = len(gids)
                members.append([])
            group[i] = gids[r["group"]]
            members[group[i]].append(i)
        torch.save({"offsets": torch.tensor([int(r["offset"]) for r in rows]), "frames": torch.tensor([int(r["frames"]) for r in rows]),
                    "tidx": torch.tensor([int(r["turn"]) for r in rows]), "group": torch.from_numpy(group), "members": members},
                   os.path.join(store, "gcond.pt"))
        print(f"[{store}] clips {len(rows)} groups {len(gids)} -> {os.path.join(store, 'gcond.pt')}")


if __name__ == "__main__":
    main()
