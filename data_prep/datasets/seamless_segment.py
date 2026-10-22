#!/usr/bin/env python3
"""Seamless Interaction (full-duplex) -> half-duplex turns for JAVIS stage 2.

Invariants measured on the dev/test ground truth and enforced here:
  - a/b assignment: `a` is the participant with the smaller P-number (240/240).
  - turns alternate a,b,a,b,... within an episode (240/240 conversations).
  - turn span == [first word start, last word end] of its words, exactly (7158/7158).
  - min turn span is 0.5s; shorter runs are backchannels and are dropped (97.1% of words kept).
  - turns may OVERLAP in time and are NOT sorted by start (161/240 have inversions).

Three rules on top of that exist for the training objective, not for fidelity, each measured
against the reference before being kept (see `eval`, which prints both):
  - backchannel filter (`--bc_max_dur`): a short run of nothing but "yeah/okay/mm-hmm" is not a
    turn. Brings turn count to 7179 vs the reference's 7158 -- the original almost certainly had
    this list too.
  - finish-the-sentence cut (`--sent_ext`): cutting at the takeover leaves 19.8% of turns ending
    mid-sentence (ref 7.9%), which trains the target stream to stop mid-utterance; letting the
    speaker close the sentence takes that to 13.9% at no cost in boundary F1. Cutting at the first
    *pause* instead was also tried and did nothing (19.8 -> 18.9%), so it is not in here.
  - episode break (`--absorb_gap_max`): two halves of one turn separated by more than a few seconds
    of dead air (seen up to 166s) are not one clip. Turns whose span is majority silence drop from
    12.2% to 7.5%. `episode` increments at the break and alternation restarts.

What is deliberately NOT filtered here: turns overlapped by the next speaker (~15%) and turns
following a long mutual silence are real properties of the data, not segmentation errors. They are
flagged per turn (`end_overlap`, `gap_prev`, `ends_punct`, `speech_ratio`) and kept: a turn that is a bad assistant
*target* is usually still fine as user audio or as context, and role-flip means dropping it outright costs twice.

Pipeline (phases are independent and resumable):
  manifest  filelist.csv + interactions.csv -> manifest.jsonl   (pairs + interaction-type filter)
  segment   per-participant transcripts     -> turns/{split}/{conv}.json
  slice     manifest + turns                -> turn_wavs/, turn_mp4s/  (ffmpeg)
  eval      compare a turns/ tree against a reference turns/ tree

Usage:
  python data_prep/datasets/seamless_segment.py manifest --split train --src <download root> --assets <seamless_interaction repo>/assets --out_root data/seamless/train
  python data_prep/datasets/seamless_segment.py segment  --split train --src ... --assets ... --out_root data/seamless/train
  python data_prep/datasets/seamless_segment.py slice    --split train --src ... --assets ... --out_root data/seamless/train --num_workers 32
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# prompt_hash 129 is labelled `charades` in interactions.csv but its prompt text is a plain
# conversation ("Ask your partner what they truly want in life?"); keep it.
DEFAULT_TYPES = ("ipc_conversation",)
DEFAULT_KEEP_PROMPTS = (129,)

FILE_RE = re.compile(r"^(?P<vendor>V\d+)_S(?P<session>\d+)_I(?P<interaction>\d+)_P(?P<participant>\d+)(?P<suffix>[A-Z]*)$")

SENT_END = (".", "!", "?")
BACKCHANNEL = frozenset("""
yeah yes yep yup no nope okay ok right sure exactly really wow oh ooh ah aha uh-huh mm-hmm mhm
hmm hm huh gotcha totally absolutely definitely true cool nice great awesome interesting alright
""".split())
_WORD_RE = re.compile(r"[^a-z'\-]")


def _norm(word):
    return _WORD_RE.sub("", word.lower())


def log(msg):
    print(f"[{time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}] {msg}", flush=True)


# ---------------------------------------------------------------- phase: manifest

def phase_manifest(args):
    import pandas as pd

    fl = pd.read_csv(args.filelist or os.path.join(args.src, "filelist.csv"))
    it = pd.read_csv(Path(args.assets) / "interactions.csv")
    type_by_prompt = it.set_index("prompt_hash")["interaction_type"].to_dict()

    sel = fl[(fl.label == args.label) & (fl.split == args.split)].copy()
    if args.vendors:
        sel = sel[sel.file_id.str.startswith(tuple(args.vendors))]
    sel["interaction_id"] = sel.file_id.str.extract(r"I(\d+)").astype(int)
    sel["interaction_type"] = sel.interaction_id.map(type_by_prompt)

    keep = sel.interaction_type.isin(args.types) | sel.interaction_id.isin(args.keep_prompts)
    dropped_types = sel.loc[~keep, "interaction_type"].value_counts().to_dict()
    sel = sel[keep]

    groups = defaultdict(list)
    for row in sel.itertuples():
        m = FILE_RE.match(row.file_id)
        groups[(m["vendor"], m["session"], m["interaction"])].append((row, m))

    n_written = n_incomplete = n_missing = 0
    out = Path(args.out_root) / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for (vendor, session, interaction), rows in sorted(groups.items()):
            if len(rows) != 2:
                n_incomplete += 1
                continue
            rows.sort(key=lambda rm: rm[1]["participant"])  # a = smaller P-number
            rec = {
                "interaction_key": f"{vendor}_S{session}_I{interaction}",
                "split": args.split,
                "session": session,
                "interaction": interaction,
                "interaction_type": rows[0][0].interaction_type,
            }
            ok = True
            for side, (row, m) in zip("ab", rows):
                d = Path(args.src) / args.label / args.split / f"{int(row.batch_idx):04d}" / f"{int(row.archive_idx):04d}"
                paths = {ext: str(d / f"{row.file_id}.{ext}") for ext in ("wav", "mp4", "json")}
                if not all(os.path.exists(p) for p in paths.values()):
                    ok = False
                    break
                rec[side] = {"stem": row.file_id, "participant": m["participant"], "suffix": m["suffix"], **paths}
            if not ok:
                n_missing += 1
                continue
            f.write(json.dumps(rec) + "\n")
            n_written += 1

    log(f"[manifest] {n_written} pairs -> {out}")
    log(f"[manifest] dropped: {n_incomplete} single-sided, {n_missing} with files not on disk")
    log(f"[manifest] excluded interaction types: {dropped_types}")


# ---------------------------------------------------------------- phase: segment

def load_words(json_path):
    """Flat, time-sorted word list. Words with null timestamps are unusable and dropped."""
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for seg in d["metadata:transcript"]:
        for w in seg["words"]:
            if w.get("start") is None or w.get("end") is None:
                continue
            out.append((float(w["start"]), float(w["end"]), w["word"]))
    out.sort()
    return out


def _runs(words, run_gap):
    runs, cur = [], []
    for w in words:
        if cur and w[0] - cur[-1][1] > run_gap:
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def _is_backchannel(run, max_dur):
    """A short run made of nothing but acknowledgement tokens. Promoting these to turns both
    creates junk user/assistant rows and splits the other speaker's utterance in two."""
    if run[-1][1] - run[0][0] > max_dur:
        return False
    return all(_norm(w[2]) in BACKCHANNEL or not _norm(w[2]) for w in run)


def _split_at(run, onsets, sent_ext):
    """Cut a run wherever the other speaker takes the floor.

    The cut lands after the speaker finishes the sentence already in progress: the first word
    boundary at/after the onset that follows sentence-final punctuation, searched at most sent_ext
    seconds past the onset. Cutting at the onset itself leaves 19.8% of turns ending mid-sentence
    (ref 7.9%), which teaches the target stream to stop mid-utterance; finishing the sentence takes
    that to 14.1% at no cost in dev/test boundary F1. Falls back to the onset if the speaker keeps
    running without closing a sentence.
    """
    cuts = set()
    for t in onsets:
        if not run[0][0] < t <= run[-1][0]:  # onset outside this run cannot split it
            continue
        fallback, done = None, False
        for i in range(1, len(run)):
            if run[i][0] < t:
                continue
            if fallback is None:
                fallback = i
            if run[i - 1][2].rstrip().endswith(SENT_END):
                cuts.add(i)
                done = True
                break
            if run[i][0] - t > sent_ext:
                break
        if not done and fallback is not None:
            cuts.add(fallback)
    pieces, cur = [], []
    for i, w in enumerate(run):
        if i in cuts and cur:
            pieces.append(cur)
            cur = []
        cur.append(w)
    if cur:
        pieces.append(cur)
    return pieces


def segment_conversation(words_a, words_b, run_gap, min_turn_dur,
                         bc_max_dur=1.5, sent_ext=2.0, absorb_gap_max=3.0):
    """-> [{idx, episode, speaker, start, end, text, gap_prev, end_overlap, ends_punct, speech_ratio}].

    Speakers alternate strictly within an episode; `episode` increments where a same-speaker merge
    was refused because the two halves were more than absorb_gap_max apart.
    """
    kept = {}
    for side, words in (("a", words_a), ("b", words_b)):
        kept[side] = [r for r in _runs(words, run_gap)
                      if r[-1][1] - r[0][0] >= min_turn_dur
                      and not (bc_max_dur > 0 and _is_backchannel(r, bc_max_dur))]

    pieces = []
    for side in ("a", "b"):
        onsets = [r[0][0] for r in kept["b" if side == "a" else "a"]]
        for run in kept[side]:
            # re-apply the floor after splitting: a fragment left over by a mid-run takeover is a
            # backchannel too (ref's shortest turn is exactly min_turn_dur).
            pieces.extend((p, side) for p in _split_at(run, onsets, sent_ext)
                          if p[-1][1] - p[0][0] >= min_turn_dur)
    pieces.sort(key=lambda ps: ps[0][0][0])

    turns, episode = [], 0
    for words, side in pieces:
        if turns and turns[-1]["speaker"] == side:
            # absorb into the previous turn, unless the dead air between the halves would dominate
            # the clip (seen up to 166s) -- then start a fresh episode instead.
            if absorb_gap_max <= 0 or words[0][0] - turns[-1]["_w"][-1][1] <= absorb_gap_max:
                turns[-1]["_w"].extend(words)
                continue
            episode += 1
        turns.append({"speaker": side, "_w": list(words), "episode": episode})

    turns = [t for t in turns if " ".join(x[2] for x in t["_w"]).strip()]
    out = []
    for idx, t in enumerate(turns):
        w = t["_w"]
        start, end = w[0][0], w[-1][1]
        prev, nxt = (turns[idx - 1]["_w"] if idx else None), (turns[idx + 1]["_w"] if idx + 1 < len(turns) else None)
        out.append({
            "idx": idx,
            "episode": t["episode"],
            "speaker": t["speaker"],
            "start": start,
            "end": end,
            "text": " ".join(x[2] for x in w).strip(),
            # builder-facing eligibility signals; see module docstring.
            "gap_prev": None if prev is None else round(start - prev[-1][1], 3),
            "end_overlap": 0.0 if nxt is None else round(max(0.0, end - nxt[0][0]), 3),
            "ends_punct": w[-1][2].rstrip().endswith(SENT_END),
            "speech_ratio": round(sum(x[1] - x[0] for x in w) / (end - start), 3) if end > start else 0.0,
        })
    return out


def phase_segment(args):
    pairs = [json.loads(l) for l in open(Path(args.out_root) / "manifest.jsonl")]
    log(f"Loaded {len(pairs)} pairs from {Path(args.out_root) / 'manifest.jsonl'}")
    turns_root = Path(args.out_root) / "turns"
    n_turns = n_conv = 0
    for r in pairs:
        d = turns_root / r["split"]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{r['interaction_key']}.json"
        if dst.exists() and not args.overwrite:
            continue
        turns = segment_conversation(load_words(r["a"]["json"]), load_words(r["b"]["json"]),
                                     args.run_gap, args.min_turn_dur,
                                     args.bc_max_dur, args.sent_ext, args.absorb_gap_max)
        if not turns:
            continue
        with dst.open("w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False)
        n_turns += len(turns)
        n_conv += 1
    log(f"[segment] {n_conv} pairs, {n_turns} total turns -> {turns_root}")
    log("[segment] " + " ".join(f"{k}={v}" for k, v in turn_health(turns_root).items()))


# ---------------------------------------------------------------- phase: slice

def _slice_side(job):
    """Cut every turn of one participant: one ffmpeg per turn, input-seeking the source in place.

    Two attempts to beat this were measured and both lost, so don't reintroduce them:
      - one ffmpeg per source with many outputs: 450 vs 247 core-s for the same 41 turns. Every
        output instantiates its own encoder, and the single decode has to cover the whole source
        rather than just the turn spans (turns are ~half the recording).
      - copying the source to local disk first, then seeking the copy: same CPU per second of video
        (1.15 vs 1.21) but 1.9x worse wall clock (3.43 vs 6.66 turns/s at 40 workers), because the
        per-source copy is serial and the worker idles through it.
    Input seeking (-ss before -i) already reads only the neighbourhood of the turn, so the repeated
    opens cost latency that parallelism hides rather than bandwidth.

    Video is written at 25 fps because data_prep/extract_mcoeff.py refuses anything else
    (`--expect_fps 25`); emitting 30 fps here would force a second full re-encode pass.
    """
    src_wav, src_mp4, out_wav_dir, out_mp4_dir, turns, fps, ffmpeg = job
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_mp4_dir, exist_ok=True)
    errs = []
    for src, out_dir, ext, codec in (
        (src_wav, out_wav_dir, "wav", ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
        (src_mp4, out_mp4_dir, "mp4", ["-an", "-r", str(fps), "-c:v", "libx264",
                                       "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]),
    ):
        for t in turns:
            dst = os.path.join(out_dir, f"{t['idx']:03d}_{t['speaker']}.{ext}")
            if os.path.exists(dst):
                continue
            # Write aside and rename: ffmpeg creates dst as soon as it starts, so a run killed
            # mid-encode would otherwise leave a header-only file that the resume check counts as
            # done and skips forever (verified: 52-byte mp4, "moov atom not found"). -f is required
            # because the .part suffix hides the container from ffmpeg's extension sniffing.
            tmp = f"{dst}.part"
            cmd = [ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{t['start']:.3f}", "-i", src,
                   "-t", f"{t['end'] - t['start']:.3f}", *codec, "-threads", "1", "-f", ext, tmp]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                errs.append(f"{ext} {dst}: {p.stderr.strip()[:200]}")
                if os.path.exists(tmp):
                    os.remove(tmp)
            else:
                os.replace(tmp, dst)
    return len(turns), errs


def phase_slice(args):
    root = Path(args.out_root)
    pairs = [json.loads(l) for l in open(root / "manifest.jsonl")]
    log(f"Loaded {len(pairs)} pairs from {root / 'manifest.jsonl'}")
    if args.num_shards > 1:  # one unit == one source file, so shards never touch the same input
        ids = {int(x) for x in str(args.shard_id).split(",") if x.strip() != ""}
        pairs = [p for i, p in enumerate(pairs) if i % args.num_shards in ids]
        log(f"[slice] shard {sorted(ids)}/{args.num_shards}: {len(pairs)} pairs")

    jobs, n_turns, secs = [], 0, 0.0
    for r in pairs:
        tp = root / "turns" / r["split"] / f"{r['interaction_key']}.json"
        if not tp.exists():
            continue
        turns = json.load(open(tp))
        conv = f"{r['split']}/{r['interaction_key']}"
        for side in ("a", "b"):
            side_turns = [t for t in turns if t["speaker"] == side]
            if not side_turns:
                continue
            jobs.append((r[side]["wav"], r[side]["mp4"],
                         str(root / "turn_wavs" / conv), str(root / "turn_mp4s" / conv),
                         side_turns, args.fps, args.ffmpeg))
            n_turns += len(side_turns)
            secs += sum(t["end"] - t["start"] for t in side_turns)
    log(f"[slice] {len(jobs)} source files, {n_turns} turns, {secs/3600:.1f} h, est {secs*0.286/1e3:.0f} GB "
        f"(measured dev/test rate: 0.032 MB/s wav + 0.254 MB/s mp4)")
    if args.dry_run:
        return

    log(f"=== phase slice (num_workers={args.num_workers}, {args.fps} fps, ffmpeg -threads 1) ===")
    n_done = n_src = n_err = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        # map+chunksize rather than submit-all: materialising a Future per job costs memory before
        # any work starts.
        for done, errs in ex.map(_slice_side, jobs, chunksize=4):
            for e in errs:
                n_err += 1
                print(f"[slice error] {e}", file=sys.stderr, flush=True)
            n_done += done
            n_src += 1
            if n_src % 200 == 0:
                log(f"[slice] {n_src}/{len(jobs)} sources, {n_done}/{n_turns} turns ({n_err} errors)")
    log(f"[slice] done {n_src}/{len(jobs)} sources, {n_done}/{n_turns} turns, {n_err} errors")


# ---------------------------------------------------------------- phase: eval

def phase_eval(args):
    ref_root, hyp_root = Path(args.ref), Path(args.out_root) / "turns"
    tot_r = tot_h = tp = 0
    n_conv = n_alt = 0
    for rf in sorted(ref_root.glob("*/*.json")):
        hf = hyp_root / rf.parent.name / rf.name
        if not hf.exists():
            continue
        ref, hyp = json.load(open(rf)), json.load(open(hf))
        n_conv += 1
        # alternation is an invariant WITHIN an episode; an episode break is where it may repeat.
        n_alt += all(hyp[i]["speaker"] != hyp[i + 1]["speaker"]
                     or hyp[i].get("episode") != hyp[i + 1].get("episode")
                     for i in range(len(hyp) - 1))
        R = {(t["speaker"], round(t["start"], 3), round(t["end"], 3)) for t in ref}
        H = {(t["speaker"], round(t["start"], 3), round(t["end"], 3)) for t in hyp}
        tot_r += len(R)
        tot_h += len(H)
        tp += len(R & H)
    if not n_conv:
        log("[eval] no overlapping conversations between --ref and --out_root/turns")
        return
    prec = tp / tot_h if tot_h else 0.0
    rec = tp / tot_r if tot_r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    log(f"[eval] {n_conv} conversations | ref turns {tot_r}, hyp turns {tot_h}")
    log(f"[eval] exact-boundary  P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    log(f"[eval] in-episode a/b alternation: {n_alt}/{n_conv}")
    log("[eval] " + " ".join(f"{k}={v}" for k, v in turn_health(hyp_root).items()))


def turn_health(turns_root):
    """Training-facing quality of a turns/ tree: the share of turns that are unusable as an
    assistant target (cut mid-sentence, overlapped by the next speaker, mostly silence)."""
    n = midcut = deep = lowsr = long20 = 0
    for f in Path(turns_root).glob("*/*.json"):
        for t in json.load(open(f)):
            n += 1
            ov = t.get("end_overlap", 0.0)
            if ov > 0 and not t.get("ends_punct", True):
                midcut += 1
            if ov > 2.0:
                deep += 1
            if t.get("speech_ratio", 1.0) < 0.5:
                lowsr += 1
            if t["end"] - t["start"] > 20:
                long20 += 1
    if not n:
        return {}
    pc = lambda x: f"{100 * x / n:.1f}%"
    return {"turns": n, "midcut": pc(midcut), "overlap>2s": pc(deep),
            "speech_ratio<0.5": pc(lowsr), "dur>20s": pc(long20)}


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["manifest", "segment", "slice", "eval"])
    p.add_argument("--out_root", required=True, help="Cache root to write (manifest.jsonl, turns/, turn_wavs/, turn_mp4s/).")
    p.add_argument("--src", required=True, help="Seamless download root (seamless_download.py --root).")
    p.add_argument("--assets", required=True, help="Dir holding interactions.csv (assets/ of github.com/facebookresearch/seamless_interaction).")
    p.add_argument("--filelist", default=None, help="default <src>/filelist.csv")
    p.add_argument("--label", default="improvised", choices=["improvised", "naturalistic"])
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--vendors", nargs="*", default=["V00"], help="Vendor prefixes to keep; empty = all.")
    p.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES), help="interaction_type values to keep.")
    p.add_argument("--keep_prompts", nargs="*", type=int, default=list(DEFAULT_KEEP_PROMPTS),
                   help="Interaction ids kept regardless of type (mislabelled prompts).")
    p.add_argument("--run_gap", type=float, default=1.0, help="Within-speaker silence that starts a new run.")
    p.add_argument("--min_turn_dur", type=float, default=0.5, help="Runs shorter than this are backchannels.")
    p.add_argument("--bc_max_dur", type=float, default=1.5,
                   help="Runs up to this long made only of acknowledgement tokens are dropped; 0 disables.")
    p.add_argument("--sent_ext", type=float, default=2.0,
                   help="Seconds past a takeover to look for a sentence end before cutting; 0 cuts at the takeover.")
    p.add_argument("--absorb_gap_max", type=float, default=3.0,
                   help="Dead air tolerated when merging two halves of one turn; beyond it a new episode starts. 0 = unlimited.")
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--fps", type=int, default=25,
                   help="Output video fps. Must stay 25: data_prep/extract_mcoeff.py has an --expect_fps 25 guard.")
    p.add_argument("--ffmpeg", default=str(Path(sys.executable).with_name("ffmpeg")) if Path(sys.executable).with_name("ffmpeg").exists() else "ffmpeg",
                   help="ffmpeg binary. Defaults to the one beside this interpreter -- remote hosts run non-login shells where conda's PATH is absent.")
    p.add_argument("--num_shards", type=int, default=1, help="slice: split source files across machines.")
    p.add_argument("--shard_id", default="0",
                   help="Shard index, or a comma-separated list to give one machine several shards "
                        "(e.g. --num_shards 24 --shard_id 0,1,2,3 for a big host).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="slice: print the plan and size estimate, then exit.")
    p.add_argument("--ref", default=None,
                   help="eval: reference turns/ tree.")
    args = p.parse_args()

    log(f"=== phase {args.phase} ===")
    {"manifest": phase_manifest, "segment": phase_segment,
     "slice": phase_slice, "eval": phase_eval}[args.phase](args)


if __name__ == "__main__":
    main()
