"""After `llamafactory-cli export`: the exported motion_cfm_head must equal the one stored in the adapter checkpoint.

A head that failed to load is silently re-initialized by the export, so this compares the tensors instead of trusting
the log. Usage: python training/check_merged_head.py <adapter dir> <export dir>
"""
import json
import os
import sys

import torch
from safetensors import safe_open

PREFIX = "audio_invert_tower.motion_cfm_head."


def adapter_head(d):
    p = os.path.join(d, "adapter_model.safetensors")
    if os.path.exists(p):
        with safe_open(p, "pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
    else:
        sd = torch.load(os.path.join(d, "adapter_model.bin"), map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        k = k.replace("base_model.model.", "", 1).replace("modules_to_save.default.", "")
        if k.startswith(PREFIX):
            out[k] = v
    return out


def export_head(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for shard in sorted({s for k, s in idx.items() if k.startswith(PREFIX)}):
        with safe_open(os.path.join(d, shard), "pt") as f:
            out.update({k: f.get_tensor(k) for k in f.keys() if k.startswith(PREFIX)})
    return out


def main():
    a, e = adapter_head(sys.argv[1]), export_head(sys.argv[2])
    if not a:
        sys.exit(f"[err] no motion_cfm_head tensors in the adapter at {sys.argv[1]}")
    missing = sorted(set(a) - set(e))
    if missing:
        sys.exit(f"[err] export lacks {len(missing)} motion_cfm_head tensors, e.g. {missing[0]}")
    bad = [k for k in a if not torch.equal(a[k].to(e[k].dtype), e[k])]
    if bad:
        sys.exit(f"[err] {len(bad)}/{len(a)} motion_cfm_head tensors differ from the adapter (head not loaded?), e.g. {bad[0]}")
    print(f"[ok] motion_cfm_head: {len(a)} tensors match the adapter")


if __name__ == "__main__":
    main()
