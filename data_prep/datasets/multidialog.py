"""MultiDialog -> clips.jsonl + dialogues.jsonl per split for the common pipeline.

  python data_prep/datasets/multidialog.py --root /data/MultiDialog --video_dir /data/MultiDialog/video_25fps --out_dir data/multidialog

<root>/metadata.jsonl rows carry conv_id, utterance_id, from (gpt|human), value (text) and audpath (<split>/<conv>/<file>.wav);
videos are expected at <video_dir>/<audpath without .wav>.mp4 at 25 fps (extract mcoeff WITH face cropping).
JAVIS trained on `train` with both speakers as the assistant and validated on `valid_freq` with the gpt side only.
"""
import argparse
import json
import os
from collections import defaultdict

SPLITS = ("train", "valid_freq", "test_freq", "test_rare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    a.root, a.video_dir = os.path.abspath(a.root), os.path.abspath(a.video_dir)
    convs = defaultdict(list)
    split_of = {}
    for line in open(os.path.join(a.root, "metadata.jsonl")):
        d = json.loads(line)
        split = d["audpath"].split("/", 1)[0]
        if split in SPLITS:
            convs[d["conv_id"]].append(d)
            split_of[d["conv_id"]] = split
    os.makedirs(a.out_dir, exist_ok=True)
    fc = {s: open(os.path.join(a.out_dir, f"clips_{s}.jsonl"), "w") for s in SPLITS}
    fd = {s: open(os.path.join(a.out_dir, f"dialogues_{s}.jsonl"), "w") for s in SPLITS}
    for conv, uts in convs.items():
        s = split_of[conv]
        dlg = []
        for u in sorted(uts, key=lambda x: x["utterance_id"]):
            cid = u["audpath"][:-4]
            tail = cid.rsplit("_", 1)[-1]                      # <turn index><side letter>, e.g. 12e
            fc[s].write(json.dumps({"id": cid, "wav": os.path.join(a.root, "audio", u["audpath"]),
                                    "video": os.path.join(a.video_dir, cid + ".mp4"), "text": u["value"],
                                    "group": f"{conv}_{tail[-1]}", "turn": int(tail[:-1] or 0)}, ensure_ascii=False) + "\n")
            dlg.append({"clip": cid, "speaker": u["from"]})
        fd[s].write(json.dumps({"id": conv, "turns": dlg, "assistant": ["gpt", "human"] if s == "train" else ["gpt"]}) + "\n")
    for f in list(fc.values()) + list(fd.values()):
        f.close()
    print(f"{len(convs)} conversations -> {a.out_dir}")


if __name__ == "__main__":
    main()
