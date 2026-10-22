"""Talker-T2AV-Data -> stage-1 clip manifest + exclusion list.

  # 1. python data_prep/extract_mcoeff.py --dir_path <root>/video --output_dir data/talker/mc --no_crop --stats_jsonl
  # 2. exclusion list (needs the extraction stats)
  python data_prep/datasets/talker_t2av.py filter --root <root> --mcoeff_out data/talker/mc --out data/talker/exclude.txt
  # 3. clips.jsonl (id, video, wav, text, split, group, turn) without the excluded clips, with the JAVIS validation split
  python data_prep/datasets/talker_t2av.py manifest --root <root> --exclude data/talker/exclude.txt --out data/talker/clips.jsonl

Exclusion criteria (one threshold each; the union removed 23,699 of 968,894 clips, 2.45%):
  A  |video s - audio s| > 0.12   broken lip sync     (video s = pt_length/fps, audio s = wav data bytes / 2 / 24000)
  B  vel_max / vel_p99   > 8      scene / camera cut  (frame-to-frame mcoeff L2 velocity)
  C  vel_mean            < 0.2    no face / still image
  D  voiced-frame ratio  < 0.05   listening-only clip (25 Hz frames above max(p95 - 25, -55) dBFS)
  X  no extraction stats          not extracted or failed, so B and C cannot be checked (such a clip has no mcoeff anyway)
"""
import argparse
import csv
import glob
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SEAMLESS_I = re.compile(r"_I(\d+)_P")
MEAD_WAV = re.compile(r"^audio/mead/([^/]+)/audio/([^/]+)/(level_\d)/(\d+)\.wav$")
TALKVID_WAV = re.compile(r"^audio/talkvid/[^/]+/[^/]+/(.+?)_separated/")


def read_csv(root):
    csv.field_size_limit(10 ** 7)
    with open(os.path.join(root, "metadata", "train.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def pick_val(rows, seed=0):
    """Content-novel validation: 4 Seamless interactions, 2 MEAD sentences (all speakers), ~590 clips of whole TalkVid
    source videos, 900 dh_facevid and 440 hallo3 clips at random; no RAVDESS."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["dataset_source"]].append(r)
    val = set()
    inter = defaultdict(list)
    for r in by["seamless"]:
        inter[SEAMLESS_I.search(r["wav_path"]).group(1)].append(r["sample_id"])
    for k in rng.sample(sorted(inter), 4):
        val.update(inter[k])
    combo = defaultdict(list)
    for r in by["mead"]:
        m = MEAD_WAV.match(r["wav_path"])
        combo[(m.group(2), m.group(4))].append(r["sample_id"])
    for e in rng.sample(sorted({e for e, _ in combo}), 2):
        val.update(combo[rng.choice(sorted(k for k in combo if k[0] == e))])
    vids = defaultdict(list)
    for r in by["talkvid"]:
        vids[TALKVID_WAV.match(r["wav_path"]).group(1)].append(r["sample_id"])
    n = 0
    for k in rng.sample(sorted(vids), len(vids)):
        if n >= 590:
            break
        val.update(vids[k])
        n += len(vids[k])
    for src, k in (("dh_facevid", 900), ("hallo3", 440)):
        val.update(rng.sample(sorted(r["sample_id"] for r in by[src]), k))
    return val


def voiced_ratio(path, sr=24000, fps=25):
    with open(path, "rb") as fh:
        buf = fh.read()
    off, data = 12, None
    while off + 8 <= len(buf):                       # walk RIFF chunks: some sources put LIST before data
        cid, sz = buf[off:off + 4], int.from_bytes(buf[off + 4:off + 8], "little")
        if cid == b"data":
            data = buf[off + 8:off + 8 + sz]
            break
        off += 8 + sz + (sz & 1)
    if data is None:
        return None
    a = np.frombuffer(data[:len(data) // 2 * 2], np.int16)
    hop = sr // fps
    m = len(a) // hop
    if m < 4:
        return None
    db = 20 * np.log10(np.sqrt((a[:m * hop].astype(np.float32).reshape(m, hop) ** 2).mean(1) + 1e-6) / 32768.0 + 1e-12)
    return float((db > max(np.percentile(db, 95) - 25.0, -55.0)).mean())


def person_group(r):
    """Clips of one person in one recording: Seamless participant-in-interaction, TalkVid source video, MEAD / RAVDESS
    actor. dh_facevid and hallo3 clips stand alone (their gcond falls back to the clip's own middle frame)."""
    src, w = r["dataset_source"], r["wav_path"]
    if src == "seamless":
        return "seamless/" + os.path.basename(os.path.dirname(w))
    if src == "talkvid":
        return "talkvid/" + TALKVID_WAV.match(w).group(1)
    if src == "mead":
        return "mead/" + MEAD_WAV.match(w).group(1)
    if src == "ravdess":
        return "ravdess/" + w.split("/")[2]
    return r["sample_id"]


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def cmd_manifest(a):
    ex = set(open(a.exclude).read().split()) if a.exclude else set()
    rows = [r for r in read_csv(a.root) if r["sample_id"] not in ex]
    val = pick_val(rows, a.seed)
    groups = defaultdict(list)
    for r in rows:
        groups[person_group(r)].append(r)
    turn = {}
    for g in groups.values():
        for k, r in enumerate(sorted(g, key=lambda r: natural_key(r["wav_path"]))):
            turn[r["sample_id"]] = k
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["sample_id"], "video": os.path.join(a.root, r["video_path"]), "wav": os.path.join(a.root, r["wav_path"]),
                                "text": r["text"], "split": "val" if r["sample_id"] in val else "train",
                                "group": person_group(r), "turn": turn[r["sample_id"]]}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips ({len(val)} val) -> {a.out}")


def cmd_filter(a):
    rows = read_csv(a.root)
    stats = {}
    for p in glob.glob(os.path.join(a.mcoeff_out, "stats_rank*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            stats[r["stem"]] = r
    if not stats:
        raise SystemExit(f"no stats_rank*.jsonl in {a.mcoeff_out}: run extract_mcoeff.py --no_crop --stats_jsonl over all videos first")
    with ThreadPoolExecutor(a.workers) as pool:
        vrs = list(pool.map(lambda r: voiced_ratio(os.path.join(a.root, r["wav_path"])), rows, chunksize=64))
    out = []
    for r, vr in zip(rows, vrs):
        st = stats.get(os.path.splitext(r["video_path"][len("video/"):])[0])
        if st is None:                                     # not extracted or failed: cannot be checked, so not kept
            out.append((r["sample_id"], "X"))
            continue
        adur = (os.path.getsize(os.path.join(a.root, r["wav_path"])) - 44) / 2 / 24000
        d = int(r["pt_length"]) / float(r["fps"]) - adur
        crit = ("A" if abs(d) > 0.12 else "") + ("B" if st["vel_max"] / max(st["vel_p99"], 1e-6) > 8 else "") + \
               ("C" if st["vel_mean"] < 0.2 else "") + ("D" if vr is not None and vr < 0.05 else "")
        if crit:
            out.append((r["sample_id"], crit))
    with open(a.out, "w") as f:
        f.writelines(f"{i}\n" for i, _ in out)
    n_x = sum(c == "X" for _, c in out)
    print(f"{len(out)} clips excluded -> {a.out}" + (f" ({n_x} of them have no extraction stats)" if n_x else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "filter"])
    ap.add_argument("--root", required=True, help="Talker-T2AV-Data root (metadata/train.csv, video/, audio/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcoeff_out", help="filter: extract_mcoeff.py --output_dir (holds stats_rank*.jsonl)")
    ap.add_argument("--exclude", help="manifest: output of the filter step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    (cmd_manifest if a.cmd == "manifest" else cmd_filter)(a)


if __name__ == "__main__":
    main()
