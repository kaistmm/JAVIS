#!/usr/bin/env python3
"""Download the Seamless Interaction subset needed for FLOAT mcoeff extraction, from S3.

Only 4 modalities per file (video, audio, transcript, vad) -> ~94MB/file vs ~146MB for the full
S3 set and ~1.05GB/archive for the HF tars, which bundle every modality and cannot be filtered.


The unit of work is an INTERACTION, not a file: turn-level segmentation needs both participants,
so an interaction is downloaded whole or not at all. Both sides share one timeline (verified: 120/120
sampled pairs match to 0.0ms), and pairs always sit in the same batch but almost never in the same
archive (1.6%), so pairing has to come from filelist.csv rather than the directory layout.

Output layout mirrors the HF WebDataset extraction:
  <root>/<label>/<split>/<XXXX batch>/<YYYY archive>/<file_id>.{mp4,wav,json}
with the json carrying webdataset-style keys: {"id", "metadata:transcript", "metadata:vad"}.

Usage (inside tmux):
  python data_prep/datasets/seamless_download.py --root data/seamless_interaction --label improvised --split train
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

S3_BASE = "https://dl.fbaipublicfiles.com/seamless_interaction"
FILELIST_URL = "https://raw.githubusercontent.com/facebookresearch/seamless_interaction/main/assets/filelist.csv"
SKIPLIST_NAME = "_skipped_interactions.txt"
FAILLIST_NAME = "_failed_interactions.txt"
MEAN_MB = 94.5  # average mp4+wav size per file, for the size/time estimate only
UPLINK_MBPS = 9.0  # assumed download bandwidth, for the time estimate only


def make_session(pool_size: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    ad = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", ad)
    return s


def load_filelist(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[info] filelist not found, fetching {FILELIST_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(FILELIST_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pd.read_csv(path)


def fetch(sess: requests.Session, url: str, timeout: int) -> bytes | None:
    """Return body, or None if the object does not exist (S3 answers 403 for missing keys)."""
    r = sess.get(url, timeout=timeout)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    n = r.headers.get("Content-Length")
    if n is not None and len(r.content) != int(n):
        raise IOError(f"short read {len(r.content)}/{n} for {url}")
    return r.content


def write_atomic(dst: Path, data: bytes) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def parse_jsonl(raw: bytes) -> list:
    return [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = self.skipped = self.failed = self.existed = 0
        self.bytes = 0
        self.t0 = time.time()

    def bump(self, *, done=0, skipped=0, failed=0, existed=0, nbytes=0):
        with self.lock:
            self.done += done; self.skipped += skipped; self.failed += failed; self.existed += existed; self.bytes += nbytes

    def line(self, total: int) -> str:
        el = max(1e-9, time.time() - self.t0)
        seen = self.done + self.skipped + self.failed + self.existed
        rate = self.bytes / 1048576 / el
        eta = (total - seen) / (self.done / el) / 3600 if self.done else float("inf")
        return (f"[{seen}/{total} ix] ok={self.done} cached={self.existed} skip={self.skipped} fail={self.failed} "
                f"| {self.bytes/2**30:.1f} GiB @ {rate:.1f} MB/s | ETA {eta:.1f}h")


def paths_for(root: Path, label: str, split: str, row) -> tuple[Path, Path, Path]:
    d = root / label / split / f"{int(row.batch_idx):04d}" / f"{int(row.archive_idx):04d}"
    return d / f"{row.file_id}.mp4", d / f"{row.file_id}.wav", d / f"{row.file_id}.json"


def process(ik: str, rows: list, root: Path, label: str, split: str, sess: requests.Session, timeout: int, stats: Stats, skip_f, fail_f, flock: threading.Lock):
    if all(p.exists() and p.stat().st_size > 0 for r in rows for p in paths_for(root, label, split, r)):
        stats.bump(existed=1)
        return
    base = f"{S3_BASE}/{label}/{split}"
    try:
        # transcripts are tiny and mandatory downstream; probe every participant before spending
        # ~190MB on the interaction's media, and drop the whole interaction if any side lacks one
        # (a half-downloaded interaction is useless for turn-level segmentation)
        meta = {}
        for r in rows:
            tr_raw = fetch(sess, f"{base}/metadata/transcript/{r.file_id}.jsonl", timeout)
            transcript = parse_jsonl(tr_raw) if tr_raw else []
            if not transcript:
                with flock:
                    skip_f.write(f"{ik}\tno transcript for {r.file_id}\n"); skip_f.flush()
                stats.bump(skipped=1)
                return
            vad_raw = fetch(sess, f"{base}/metadata/vad/{r.file_id}.jsonl", timeout)
            meta[r.file_id] = (transcript, parse_jsonl(vad_raw) if vad_raw else [], len(tr_raw))

        nbytes = 0
        for r in rows:
            mp4, wav, js = paths_for(root, label, split, r)
            if all(p.exists() and p.stat().st_size > 0 for p in (mp4, wav, js)):
                continue
            mp4_raw = fetch(sess, f"{base}/video/{r.file_id}.mp4", timeout)
            wav_raw = fetch(sess, f"{base}/audio/{r.file_id}.wav", timeout)
            if mp4_raw is None or wav_raw is None:
                raise IOError(f"missing media object for {r.file_id}")
            transcript, vad, tr_len = meta[r.file_id]
            mp4.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(mp4, mp4_raw)
            write_atomic(wav, wav_raw)
            write_atomic(js, json.dumps({"id": r.file_id, "metadata:transcript": transcript, "metadata:vad": vad}).encode())
            nbytes += len(mp4_raw) + len(wav_raw) + tr_len
        stats.bump(done=1, nbytes=nbytes)
    except Exception as e:
        with flock:
            fail_f.write(f"{ik}\t{type(e).__name__}: {e}\n"); fail_f.flush()
        stats.bump(failed=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Download root.")
    ap.add_argument("--filelist", default=None, help="filelist.csv path (default <root>/filelist.csv, auto-fetched).")
    ap.add_argument("--label", default="improvised", choices=["improvised", "naturalistic"])
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--vendors", nargs="*", default=["V00"], help="Vendor prefixes to keep; empty = all.")
    ap.add_argument("--min-participants", type=int, default=2, help="Drop interactions with fewer sides (turn-level needs both).")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent interactions. Uplink caps ~9 MB/s at P>=8; higher only steals CPU from FLOAT extraction.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--num-shards", type=int, default=1, help="Split the work across machines.")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N interactions (smoke test).")
    ap.add_argument("--retry-failed", action="store_true", help="Also retry interactions in the fail list.")
    ap.add_argument("--only-failed", action="store_true", help="Retry ONLY the fail list. Safe to run alongside a main run, which --retry-failed is not (it would race on not-yet-downloaded interactions).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = ap.parse_args()

    root = Path(args.root)
    fl_path = Path(args.filelist) if args.filelist else root / "filelist.csv"
    fl = load_filelist(fl_path)

    sel = fl[(fl.label == args.label) & (fl.split == args.split)].copy()
    if args.vendors:
        sel = sel[sel.file_id.str.startswith(tuple(args.vendors))]
    sel["ik"] = sel.file_id.str.extract(r"(V\d+_S\d+_I\d+)")[0]

    groups = {ik: list(d.itertuples(index=False)) for ik, d in sel.sort_values("file_id").groupby("ik")}
    dropped = {ik for ik, rows in groups.items() if len(rows) < args.min_participants}
    keys = sorted(k for k in groups if k not in dropped)
    if args.num_shards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_id]

    skip_path, fail_path = root / SKIPLIST_NAME, root / FAILLIST_NAME
    failed = {ln.split("\t")[0] for ln in fail_path.read_text().splitlines() if ln.strip()} if fail_path.exists() else set()
    known = set()
    if args.only_failed:
        keys = [k for k in keys if k in failed]
    else:
        known = {ln.split("\t")[0] for ln in skip_path.read_text().splitlines() if ln.strip()} if skip_path.exists() else set()
        if not args.retry_failed:
            known |= failed
        keys = [k for k in keys if k not in known]
    if args.limit:
        keys = keys[: args.limit]

    total = len(keys)
    nfiles = sum(len(groups[k]) for k in keys)
    est = nfiles * MEAN_MB / 1024
    print(f"[info] {args.label}/{args.split} vendors={args.vendors or 'ALL'} shard={args.shard_id}/{args.num_shards}")
    print(f"[info] {total} interactions / {nfiles} files (~{est/1024:.2f} TB, ~{est*1024/UPLINK_MBPS/3600:.0f}h at {UPLINK_MBPS} MB/s) -> {root}")
    if dropped:
        print(f"[info] dropped {len(dropped)} interactions with <{args.min_participants} participants")
    if known:
        print(f"[info] {len(known)} interactions excluded from prior skip/fail lists")
    if args.dry_run or total == 0:
        return

    root.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    flock = threading.Lock()
    sess = make_session(args.workers)
    stop = threading.Event()

    def ticker():
        while not stop.wait(30):
            print("[prog] " + stats.line(total), flush=True)
    threading.Thread(target=ticker, daemon=True).start()

    with open(skip_path, "a") as skip_f, open(fail_path, "a") as fail_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process, k, groups[k], root, args.label, args.split, sess, args.timeout, stats, skip_f, fail_f, flock) for k in keys]
            for f in as_completed(futs):
                f.result()
    stop.set()

    print("[done] " + stats.line(total))
    if stats.failed:
        print(f"[warn] {stats.failed} failures logged to {fail_path}; rerun with --retry-failed", file=sys.stderr)


if __name__ == "__main__":
    main()
