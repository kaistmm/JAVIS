"""Stage-2 rows (dialogue): one row per assistant turn that has speech tokens and mcoeff, with up to 3 preceding turns
as context. Context user turns are audio, context assistant turns are text only; the target assistant turn carries
speech tokens + mcoeff. The window starts with a user turn and never crosses an `episode` boundary.

dialogues.jsonl, one dialogue per line:
  {"id": ..., "turns": [{"clip": <clip id>, "speaker": "a", "episode": 0}, ...], "assistant": ["a", "b"]}
Every speaker listed in "assistant" yields one pass over the dialogue with that speaker as the assistant (JAVIS used
both directions for Seamless / MultiDialog-train dialogues and one for MultiDialog-val).
clips.jsonl gives each clip's `wav` and `text`; context user turns need only the wav.

  python data_prep/build_stage2_rows.py --dialogues dialogues.jsonl --clips clips.jsonl --tokens speech_tokens.jsonl \
      --store store --out rows/train.jsonl [--min_target_frames 50]
(JAVIS used --min_target_frames 50, i.e. 2 s, for Seamless, whose short backchannels stay usable as context only.)
"""
import argparse
import json
import os
import re
from collections import Counter

import soundfile as sf

SYSTEM = "You are asked to generate text, speech tokens, and motion tokens at the same time."
AUDIO = "<|audio_bos|><|AUDIO|><|audio_eos|>"
AU = re.compile(r"\[AU\d{4}\]")
MAX_TARGET_FRAMES = 2048
MAX_LEN_DIFF = 5          # |n_speech_tok - n_mcoeff| beyond this means a bad pairing
WINDOW_TURNS = 4
LEN_GUARD = int(2048 * 0.95)
GROUP_SIZE = 5
stats = Counter()


def turn_M(t):
    if t["tok"] is not None:
        return len(AU.findall(t["tok"]))
    if t["mc"] is not None:
        return t["mc"][2]
    info = sf.info(t["wav"])
    return round(info.frames / info.samplerate * 25)


def est_positions(window):
    """Cheap LM-position estimate for the cutoff guard."""
    pos = 24
    for t in window:
        if t["role"] == "user" or t is window[-1]:
            pos += (turn_M(t) + GROUP_SIZE - 1) // GROUP_SIZE + 2
        else:
            pos += len(t["text"].split()) * 2 + 4
    return pos


def build_window(turns, k, min_target, episode_break):
    tgt = turns[k]
    if tgt["tok"] is None or tgt["mc"] is None:
        stats["skip_target_no_tokens_or_mcoeff"] += 1
        return None
    n_tok, n_mc = len(AU.findall(tgt["tok"])), tgt["mc"][2]
    if abs(n_tok - n_mc) > MAX_LEN_DIFF:
        stats["skip_len_diff"] += 1
        return None
    M = min(n_tok, n_mc)
    if not min_target <= M <= MAX_TARGET_FRAMES:
        stats["skip_target_length"] += 1
        return None
    context = turns[max(0, k - (WINDOW_TURNS - 1)):k]
    if episode_break:
        for j in range(len(context) - 1, -1, -1):
            if context[j]["episode"] != tgt["episode"]:
                context = context[j + 1:]
                break
    while context and context[0]["role"] == "assistant":
        context = context[1:]
    if not context:
        stats["skip_no_user_context"] += 1
        return None
    window = context + [tgt]
    try:
        if est_positions(window) > LEN_GUARD:
            stats["skip_len_guard"] += 1
            return None
        messages, audio = [], []
        for t in context:
            if t["role"] == "user":
                messages.append({"role": "user", "content": AUDIO})
                audio.append(json.dumps({"path": t["wav"], "ref_path": "", "ref_text": "", "text": "",
                                         "token": "<|audio_pad|>" * turn_M(t)}, ensure_ascii=False))
            else:
                messages.append({"role": "assistant", "content": t["text"]})
        binp, off, _ = tgt["mc"]
        au = AU.findall(tgt["tok"])
        messages.append({"role": "assistant", "content": AUDIO})
        audio.append(json.dumps({"path": "", "ref_path": tgt["wav"], "ref_text": "", "text": tgt["text"],
                                 "token": tgt["tok"][:tgt["tok"].index("[AU")] + "".join(au[:M]),
                                 "mcoeff_bin": binp, "mcoeff_offset": off, "mcoeff_frames": M}, ensure_ascii=False))
    except Exception:
        stats["skip_build_error"] += 1
        return None
    stats["rows"] += 1
    return {"system": SYSTEM, "messages": messages, "audio": audio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_target_frames", type=int, default=10)
    ap.add_argument("--episode_break", action="store_true", help="window never crosses an `episode` change (Seamless)")
    a = ap.parse_args()

    clips = {}
    for line in open(a.clips):
        c = json.loads(line)
        clips[c["id"]] = c
    tok = {}
    for line in open(a.tokens):
        d = json.loads(line)
        tok[d["id"]] = d["token"]
    binp = os.path.abspath(os.path.join(a.store, "feats.bin"))
    mc = {}
    for line in open(os.path.join(a.store, "index.jsonl")):
        d = json.loads(line)
        mc[d["id"]] = (binp, d["offset"], d["frames"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fo:
        for line in open(a.dialogues):
            d = json.loads(line)
            for spk in d.get("assistant") or sorted({t["speaker"] for t in d["turns"]}):
                turns = [{"wav": clips[t["clip"]]["wav"], "text": clips[t["clip"]].get("text", ""),
                          "role": "assistant" if t["speaker"] == spk else "user", "episode": t.get("episode", 0),
                          "tok": tok.get(t["clip"]), "mc": mc.get(t["clip"])} for t in d["turns"]]
                for k in range(1, len(turns)):
                    if turns[k]["role"] == "assistant":
                        row = build_window(turns, k, a.min_target_frames, a.episode_break)
                        if row is not None:
                            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main()
