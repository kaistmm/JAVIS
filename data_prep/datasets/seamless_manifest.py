"""seamless_segment.py output -> clips.jsonl + dialogues.jsonl for the common pipeline.

  python data_prep/datasets/seamless_manifest.py --out_root data/seamless/train --split train --out_dir data/seamless/train

clip id = <interaction>/<idx>_<side>; group (same person, same video) = <interaction>_<side>; turn = idx.
Dialogues use both sides as the assistant; build the rows with --episode_break --min_target_frames 50.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="seamless_segment.py --out_root")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    a.out_root = os.path.abspath(a.out_root)
    root = Path(a.out_root)
    turn_dir = root / "turns" / a.split
    os.makedirs(a.out_dir, exist_ok=True)
    n_c = n_d = 0
    with open(os.path.join(a.out_dir, "clips.jsonl"), "w") as fc, open(os.path.join(a.out_dir, "dialogues.jsonl"), "w") as fd:
        for p in sorted(turn_dir.glob("*.json")):
            key, turns = p.stem, json.load(open(p))
            dlg = []
            for t in sorted(turns, key=lambda t: t["idx"]):
                stem = f"{t['idx']:03d}_{t['speaker']}"
                cid = f"{key}/{stem}"
                fc.write(json.dumps({"id": cid, "wav": str(root / "turn_wavs" / a.split / key / f"{stem}.wav"),
                                     "video": str(root / "turn_mp4s" / a.split / key / f"{stem}.mp4"), "text": t.get("text", ""),
                                     "group": f"{key}_{t['speaker']}", "turn": t["idx"]}, ensure_ascii=False) + "\n")
                dlg.append({"clip": cid, "speaker": t["speaker"], "episode": t.get("episode") or 0})
                n_c += 1
            fd.write(json.dumps({"id": key, "turns": dlg, "assistant": ["a", "b"]}) + "\n")
            n_d += 1
    print(f"{n_c} clips, {n_d} dialogues -> {a.out_dir}")


if __name__ == "__main__":
    main()
