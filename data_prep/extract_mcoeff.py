"""
Extract motion coefficient (20-dim) and motion vector (512-dim) from video frames using FLOAT's LIA autoencoder.
Only float.pth's `motion_autoencoder.enc` goes to GPU.

Two-pass ffmpeg pipeline:
  pass 1 — decode + frame-select -> cv2 h=360 downscale -> batched SFD face detection -> tube bbox
  pass 2 — decode + integer crop -> raw rgb24 pipe -> prefetch thread does the cv2 512 resize -> encoder
ffmpeg does only lossless work (color convert, integer crop); every resample stays on cv2, so the crops match a plain
cv2 decode-and-crop of the same video.

Tube bbox strategy (inference crops portraits the same way):
  - Per probe frame, detect face as (cx, cy, bsx, bsy) where bsx/bsy are half-widths.
  - Average all per-frame detections -> avg center (avg_cx, avg_cy), avg half-sizes (avg_bsx, avg_bsy).
  - Reconstruct a representative bbox: [avg_cx - avg_bsx, avg_cy - avg_bsy, avg_cx + avg_bsx, avg_cy + avg_bsy].
  - Expand with compute_bbox(bbox, frame_shape, increase_area, increase_area_top).

Usage:
    # Single video
    CUDA_VISIBLE_DEVICES=0 python data_prep/extract_mcoeff.py --video_path /path/to/video.mp4

    # Directory (single process)
    CUDA_VISIBLE_DEVICES=0 python data_prep/extract_mcoeff.py --dir_path /path/to/videos --output_dir /path/to/output

    # Directory (multi-process, run each command in a separate terminal)
    CUDA_VISIBLE_DEVICES=0 python data_prep/extract_mcoeff.py --dir_path /path/to/videos --output_dir /path/to/output --rank 0 --num_processes 8
    CUDA_VISIBLE_DEVICES=1 python data_prep/extract_mcoeff.py --dir_path /path/to/videos --output_dir /path/to/output --rank 1 --num_processes 8
    ...

    # Skip face cropping (videos that are already the 512x512 face crop, e.g. Talker-T2AV)
    python data_prep/extract_mcoeff.py --video_path /path/to/video.mp4 --no_crop
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import torch
import cv2
import face_alignment
import numpy as np

from glob import glob
from tqdm import tqdm

FLOAT_DIR = os.environ.get("FLOAT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "float"))   # github.com/deepbrainai-research/float
sys.path.insert(0, os.path.join(FLOAT_DIR, "models", "float"))
from encoder import Encoder  # noqa: E402  (imported directly: models/__init__.py drags in transformers)

PROBE_HEIGHT = 360

# cv2's FFmpeg backend ignores bt709 tags and always converts YUV->RGB with bt601/limited, and it
# converts the whole frame before any cropping. Forcing both here makes the pipe bit-identical to
# cv2.VideoCapture (verified 0/255 diff on tagged and untagged sources alike); without it, bt709
# clips drift ~1.8/255 and a yuv420 crop at an odd offset misaligns chroma by ~1.3/255.
RGB_VF = "scale=in_color_matrix=bt601:in_range=tv:out_range=pc,format=rgb24"


def compute_bbox(bbox, frame_shape, increase_area=0.1, increase_area_top=0.):
    """Face box -> aspect-preserving crop box, grown by `increase_area` (and `increase_area_top` above the head)."""
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height * (1 + increase_area_top))
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    return w, h, left, top


def crop_rect(bbox, frame_shape, increase_area, increase_area_top):
    """compute_bbox() + square re-centering, reduced to an ffmpeg crop rect.

    crop() always yields exactly size x size: size = min(h, w) <= min(frame_h, frame_w), and the
    out-of-bounds shift can never push the origin negative, so the trailing clamps are no-ops.
    """
    w, h, left, top = compute_bbox(bbox, frame_shape, increase_area, increase_area_top)
    size = min(h, w)
    center_y, center_x = top + h // 2, left + w // 2
    new_top, new_left = max(0, center_y - size // 2), max(0, center_x - size // 2)
    if new_top + size > frame_shape[0]:
        new_top -= (new_top + size - frame_shape[0])
    if new_left + size > frame_shape[1]:
        new_left -= (new_left + size - frame_shape[1])
    return size, max(0, new_left), max(0, new_top)


def video_meta(video_path):
    """(n_frames, width, height, fps) from container metadata — no decoding."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, w, h, fps


def _ffmpeg_frames(cmd, frame_bytes):
    """Run ffmpeg and yield exactly-frame_bytes chunks. Raw pipes short-read, so refill until full."""
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 24)
    buf = bytearray(frame_bytes)
    mv = memoryview(buf)
    try:
        while True:
            got = 0
            while got < frame_bytes:
                k = pr.stdout.readinto(mv[got:])
                if not k:
                    break
                got += k
            if got < frame_bytes:
                break
            yield buf
    finally:
        pr.stdout.close()
        err = pr.stderr.read().decode(errors="replace")
        pr.stderr.close()
        pr.wait()
        if pr.returncode not in (0, None) and err.strip():
            raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")


def probe_bbox(video_path, fa, w, h, n_frames, increase_area, increase_area_top,
               detect_every_n, probe_samples, threads):
    """Pass 1: subsampled h=360 decode -> batched SFD -> averaged tube bbox -> ffmpeg crop rect."""
    # ffmpeg only selects frames; the h=360 downscale stays on cv2 (ffmpeg's `area` scaler is not cv2's INTER_AREA)
    mult = PROBE_HEIGHT / h
    interp = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
    stride = max(detect_every_n, math.ceil(max(n_frames, 1) / probe_samples))

    cmd = ["ffmpeg", "-v", "error", "-threads", str(threads), "-i", video_path,
           "-vf", f"select='not(mod(n\\,{stride}))',{RGB_VF}",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    probes = [cv2.resize(np.frombuffer(b, np.uint8).reshape(h, w, 3), (0, 0), fx=mult, fy=mult, interpolation=interp)
              for b in _ffmpeg_frames(cmd, w * h * 3)]
    if not probes:
        raise ValueError(f"No probe frames decoded from {video_path}")

    # detect_from_batch takes (B,3,H,W) float in the channel order detect_from_image gets (it flips RGB->BGR internally)
    batch = torch.from_numpy(np.stack(probes)).permute(0, 3, 1, 2).float().to(fa.device)
    dets = fa.face_detector.detect_from_batch(batch)

    cx_list, cy_list, bsx_list, bsy_list = [], [], [], []
    for bboxes in dets:
        if len(bboxes) == 0:
            continue
        x1, y1, x2, y2 = np.asarray(bboxes[0])[:4] / mult
        cx_list.append((x1 + x2) / 2)
        cy_list.append((y1 + y2) / 2)
        bsx_list.append((x2 - x1) / 2)
        bsy_list.append((y2 - y1) / 2)

    if not cx_list:
        raise ValueError(f"Face not detected in any probed frame of {video_path}")

    avg_cx = sum(cx_list) / len(cx_list)
    avg_cy = sum(cy_list) / len(cy_list)
    avg_bsx = sum(bsx_list) / len(bsx_list)
    avg_bsy = sum(bsy_list) / len(bsy_list)
    avg_bbox = [avg_cx - avg_bsx, avg_cy - avg_bsy,
                avg_cx + avg_bsx, avg_cy + avg_bsy]
    return crop_rect(avg_bbox, (h, w), increase_area, increase_area_top)


def _prefetch_batches(cmd, in_hw, out_size, batch_size, out_q, depth=3):
    """Reader thread: decode -> cv2 resize -> recycled batch buffers, so ffmpeg overlaps the GPU.

    ffmpeg only crops (integer, lossless); the resize stays on cv2 (ffmpeg's `area` scaler shifts the mcoeff by ~3% of std).
    """
    ih, iw = in_hw
    in_bytes = ih * iw * 3
    interp = cv2.INTER_AREA if ih > out_size else cv2.INTER_CUBIC

    free_q = queue.Queue()
    for _ in range(depth):
        free_q.put(np.empty((batch_size, out_size, out_size, 3), np.uint8))
    it = _ffmpeg_frames(cmd, in_bytes)
    try:
        done = False
        while not done:
            batch = free_q.get()
            n = 0
            for frame in it:
                src = np.frombuffer(frame, np.uint8).reshape(ih, iw, 3)
                cv2.resize(src, (out_size, out_size), dst=batch[n], interpolation=interp)
                n += 1
                if n == batch_size:
                    break
            else:
                done = True
            if n:
                out_q.put((batch, n, free_q))
            else:
                free_q.put(batch)
    except Exception as e:                     # surface ffmpeg/decode errors on the consumer side
        out_q.put(e)
    finally:
        it.close()                             # runs _ffmpeg_frames' finally -> reaps the ffmpeg process
    out_q.put(None)


class MotionLatentExtractor:
    def __init__(self, ckpt_path, motion_dim=20, size=512, style_dim=512, fp16=True, compile_encoder=False):
        self.size = size
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        # float.pth holds the whole FLOAT model; only motion_autoencoder.enc is needed for extraction
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        prefix = 'motion_autoencoder.'
        enc_sd = {k[len(prefix + 'enc.'):]: v for k, v in state_dict.items() if k.startswith(prefix + 'enc.')}
        if not enc_sd:
            raise ValueError(f"No '{prefix}enc.*' weights in {ckpt_path}")

        self.enc = Encoder(size, style_dim, motion_dim)
        self.enc.load_state_dict(enc_sd, strict=True)
        self.enc = self.enc.to('cuda:0').eval().to(self.dtype)

        # Cache Q matrix from Direction (weight is fixed, QR result is constant)
        with torch.no_grad():
            weight = state_dict[prefix + 'dec.direction.weight'].to('cuda:0') + 1e-8
            Q, _ = torch.linalg.qr(weight)
            self.Q_T = Q.T.contiguous()  # (motion_dim, style_dim)

        # Free unused parameters from CPU memory
        del state_dict, enc_sd

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device='cuda:0',
        )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 1.53x (332 -> 509 fps @bs32 fp16). The win needs inductor fusion AND cudagraphs together:
        # 'max-autotune-no-cudagraphs' gave nothing, and a hand-captured graph alone gave nothing.
        # Safe here because the batch shape is fixed (last batch is padded) and every result is
        # copied off the static output buffer immediately. Drift vs eager: 0.05% of alpha std.
        self._encode = torch.compile(self.enc.enc_motion, mode='reduce-overhead') if compile_encoder \
            else self.enc.enc_motion

    @torch.inference_mode()
    def extract(self, video_path, no_crop=False, batch_size=32,
                increase_area=0.25, increase_area_top=0.5, detect_every_n=10,
                probe_samples=64, threads=4):
        """
        Extract motion coefficient and motion vector from video.

        Returns:
            motion_coeffs:  (T, 20)   — alpha from encoder
            motion_vectors: (T, 512)  — direction(alpha) from decoder
        """
        n_frames, w, h, fps = video_meta(video_path)

        if no_crop:
            vf, in_hw = RGB_VF, (h, w)
        else:
            s, left, top = probe_bbox(video_path, self.fa, w, h, n_frames, increase_area,
                                      increase_area_top, detect_every_n, probe_samples, threads)
            vf, in_hw = f"{RGB_VF},crop={s}:{s}:{left}:{top}", (s, s)

        cmd = ["ffmpeg", "-v", "error", "-threads", str(threads), "-i", video_path, "-vf", vf]
        # -vsync 0: emit every decoded frame verbatim. Default vsync drops/duplicates to hit a CFR
        # target, which silently desynced mcoeff from audio on VFR-ish sources (TalkVid2, HDTF).
        cmd += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        out_q = queue.Queue(maxsize=2)
        reader = threading.Thread(target=_prefetch_batches,
                                  args=(cmd, in_hw, self.size, batch_size, out_q), daemon=True)
        reader.start()

        motion_coeffs, motion_vectors = [], []
        # Fixed-shape batches keep cudnn.benchmark from re-tuning on every trailing partial batch.
        staging = torch.empty(batch_size, self.size, self.size, 3, dtype=torch.uint8, device='cuda:0')
        while True:
            item = out_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            batch, n, free_q = item
            staging[:n].copy_(torch.from_numpy(batch[:n]))
            if n < batch_size:
                staging[n:] = staging[n - 1]
            free_q.put(batch)

            x = staging.to(self.dtype).permute(0, 3, 1, 2).contiguous().div_(127.5).sub_(1.0)

            # Encoder: image -> style latent -> motion coefficient (alpha, 20-dim)
            alpha = self._encode(x)[:n].float()  # (n, 20)
            motion_coeffs.append(alpha.cpu())

            # Direction: alpha @ Q.T (cached, no QR decomposition)
            mvec = alpha @ self.Q_T  # (n, 512)
            motion_vectors.append(mvec.cpu())

        reader.join()
        if not motion_coeffs:
            raise ValueError("No valid frames found in video")

        motion_coeffs = torch.cat(motion_coeffs, dim=0)     # (T, 20)
        motion_vectors = torch.cat(motion_vectors, dim=0)   # (T, 512)
        assert motion_coeffs.shape[0] == motion_vectors.shape[0]
        return motion_coeffs, motion_vectors, fps


def get_video_list(dir_path, video_list=None):
    """Get all video files in directory recursively, or read a prebuilt list.

    The glob costs ~30s on TalkVid2 and minutes on seamless_improvised (377k files); with N ranks it
    is paid N times, so the launcher builds the list once and passes it via --video_list.
    """
    if video_list is not None:
        with open(video_list) as f:
            return [line.strip() for line in f if line.strip()]
    patterns = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']
    videos = []
    for pattern in patterns:
        videos.extend(glob(os.path.join(dir_path, '**', pattern), recursive=True))
    return sorted(set(videos))


def clip_stats(mc, fps):
    """Per-clip mcoeff summary for downstream outlier screening. Costs <1ms; the alpha is already here.

    Screening rationale: real 25 Hz face motion is smooth (lag-1 autocorr ~0.93), so low `lag1` or a
    `vel_max` spike flags noise / scene cuts, and `tstd`~0 flags a frozen frame or no face. `norm_*`
    is domain-dependent (framing), so threshold it per dataset, never globally.
    """
    a = mc.float()
    n = a.norm(dim=1)
    d = (a[1:] - a[:-1]).norm(dim=1) if len(a) > 1 else torch.zeros(1)
    c = a - a.mean(0, keepdim=True)
    lag1 = ((c[1:] * c[:-1]).sum(0) / (c.pow(2).sum(0).clamp_min(1e-8))).mean() if len(a) > 2 else torch.tensor(0.)
    return dict(T=int(len(a)), fps=round(float(fps), 3),
                norm_mean=round(float(n.mean()), 4), norm_max=round(float(n.max()), 4),
                vel_mean=round(float(d.mean()), 4), vel_p99=round(float(d.quantile(0.99)), 4),
                vel_max=round(float(d.max()), 4), tstd=round(float(a.std(0).mean()), 4),
                lag1=round(float(lag1), 4), nan=bool(torch.isnan(a).any() or torch.isinf(a).any()))


def process_single_video(extractor, video_path, mcoeffs_path, mvecs_path, args, stats_fh=None, rel_stem=None):
    """Process a single video and save results. Returns True on success."""
    try:
        motion_coeffs, motion_vectors, fps = extractor.extract(
            video_path=video_path,
            no_crop=args.no_crop,
            batch_size=args.batch_size,
            increase_area=args.increase_area,
            increase_area_top=args.increase_area_top,
            detect_every_n=args.detect_every_n,
            probe_samples=args.probe_samples,
            threads=args.ffmpeg_threads,
        )
        if args.expect_fps > 0 and abs(fps - args.expect_fps) > 0.1:
            raise ValueError(f"fps {fps:.3f} != expected {args.expect_fps} (re-encode before extracting)")
        torch.save(motion_coeffs, mcoeffs_path)
        if args.save_mvecs:
            torch.save(motion_vectors, mvecs_path)
        if stats_fh is not None:
            stats_fh.write(json.dumps(dict(stem=rel_stem, **clip_stats(motion_coeffs, fps))) + "\n")
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False


if __name__ == '__main__':
    # Limit CPU threads per process to avoid oversubscription when running multiple processes.

    parser = argparse.ArgumentParser(description="Extract FLOAT/LIA motion latents from videos (avg-center bbox, ffmpeg two-pass)")
    parser.add_argument("--video_path", default=None, type=str, help="Input video path (single video)")
    parser.add_argument("--dir_path", default=None, type=str, help="Input directory containing videos")
    parser.add_argument("--output_path", default=None, type=str, help="Output .pt prefix (single video)")
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory (batch)")
    parser.add_argument("--ckpt_path", default=os.environ.get("FLOAT_CKPT", os.path.join(FLOAT_DIR, "checkpoints", "float.pth")), type=str, help="FLOAT checkpoint")
    parser.add_argument("--motion_dim", default=20, type=int, help="Motion dimension")
    parser.add_argument("--style_dim", default=512, type=int, help="Style (w) dimension")
    parser.add_argument("--size", default=512, type=int, help="Input image size")
    parser.add_argument("--no_crop", action='store_true', help="Skip face detection and cropping")
    parser.add_argument("--increase_area", default=0.25, type=float, help="Face crop increase area")
    parser.add_argument("--increase_area_top", default=0.5, type=float, help="Face crop increase top area")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for processing")
    parser.add_argument("--rank", default=0, type=int, help="Index of this process (0-based)")
    parser.add_argument("--num_processes", default=1, type=int, help="Total number of processes")
    parser.add_argument("--detect_every_n", default=10, type=int, help="Minimum stride between probed frames")
    parser.add_argument("--probe_samples", default=64, type=int, help="Cap on probed frames per video; clips up to probe_samples*detect_every_n frames are probed every detect_every_n frames")
    parser.add_argument("--num_threads", default=2, type=int, help="Number of CPU threads per process (torch/cv2)")
    parser.add_argument("--ffmpeg_threads", default=4, type=int, help="Decoder threads per ffmpeg pass")
    parser.add_argument("--fp32", action='store_true', help="Run the encoder in fp32 (default fp16; fp16 shifts the mcoeff by ~0.3%% of std)")
    parser.add_argument("--save_mvecs", action='store_true', help="Also write the 512-d mvecs (~26x the mcoeff bytes; recoverable as mcoeff @ Q.T)")
    parser.add_argument("--expect_fps", default=25.0, type=float, help="Fail videos whose fps differs by >0.1 (0 disables)")
    parser.add_argument("--compile", action='store_true', help="torch.compile the encoder (reduce-overhead/cudagraphs): 1.53x, ~12s warmup, 0.05%% of alpha std drift; its graph pool holds ~21 GB, so with face cropping it needs a 48 GB GPU (24 GB cards OOM in face detection)")
    parser.add_argument("--stats_jsonl", action='store_true', help="Append a per-clip mcoeff summary to <output_dir>/stats_rank<N>.jsonl (free; used for outlier screening)")
    parser.add_argument("--video_list", default=None, type=str, help="Prebuilt newline-separated video paths (skips the recursive glob; build once, share across ranks)")
    parser.add_argument("--done_list", default=None, type=str, help="Path to manifest of already-processed rel_stems (one per line, no extension). Skips these without fs check.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    cv2.setNumThreads(args.num_threads)

    if args.video_path is None and args.dir_path is None:
        raise ValueError("Either --video_path or --dir_path must be provided")

    extractor = MotionLatentExtractor(
        ckpt_path=args.ckpt_path,
        motion_dim=args.motion_dim,
        size=args.size,
        style_dim=args.style_dim,
        fp16=not args.fp32,
        compile_encoder=args.compile,
    )

    # Single video mode
    if args.video_path is not None:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        if args.output_path is not None:
            base = os.path.splitext(args.output_path)[0]
            mcoeffs_path = f"{base}_mcoeffs.pt"
            mvecs_path = f"{base}_mvecs.pt"
        else:
            mcoeffs_path = f"{video_name}_mcoeffs.pt"
            mvecs_path = f"{video_name}_mvecs.pt"

        success = process_single_video(extractor, args.video_path, mcoeffs_path, mvecs_path, args)
        if success:
            print(f"Saved: {mcoeffs_path}" + (f", {mvecs_path}" if args.save_mvecs else ""))
        else:
            print("Failed to process video")

    # Directory mode (batch)
    else:
        if args.output_dir is None:
            args.output_dir = args.dir_path

        os.makedirs(args.output_dir, exist_ok=True)

        video_list = get_video_list(args.dir_path, args.video_list)
        print(f"Found {len(video_list)} videos in {args.video_list or args.dir_path}")

        if len(video_list) == 0:
            print("No videos found")
            exit()

        # Assign videos to this process: every num_processes-th video starting at rank
        assigned = [v for i, v in enumerate(video_list) if i % args.num_processes == args.rank]
        print(f"[rank {args.rank}/{args.num_processes}] Processing {len(assigned)} / {len(video_list)} videos")

        mcoeffs_base = os.path.join(args.output_dir, "mcoeffs")
        mvecs_base = os.path.join(args.output_dir, "mvecs")

        # Q is all that is needed to rebuild mvecs from mcoeffs later; write it once per output dir
        q_path = os.path.join(args.output_dir, "direction_Q.pt")
        if args.rank == 0 and not os.path.exists(q_path):
            torch.save(extractor.Q_T.cpu(), q_path)

        # Load pre-computed done set (avoids per-video NFS stat)
        done_stems = set()
        if args.done_list is not None and os.path.exists(args.done_list):
            with open(args.done_list) as f:
                done_stems = {line.strip() for line in f if line.strip()}
            print(f"[rank {args.rank}] Loaded {len(done_stems)} done stems from manifest")

        failed_list = []
        stats_fh = open(os.path.join(args.output_dir, f"stats_rank{args.rank}.jsonl"), "a", buffering=1) if args.stats_jsonl else None

        for video_path in tqdm(assigned, desc=f"[rank {args.rank}] Extracting", smoothing=0.7):
            rel_stem = os.path.splitext(os.path.relpath(video_path, args.dir_path))[0]
            mcoeffs_path = os.path.join(mcoeffs_base, f"{rel_stem}.pt")
            mvecs_path = os.path.join(mvecs_base, f"{rel_stem}.pt")

            # Fast in-memory skip (NAS already done)
            if rel_stem in done_stems:
                continue

            if os.path.exists(mcoeffs_path) and (not args.save_mvecs or os.path.exists(mvecs_path)):
                continue

            os.makedirs(os.path.dirname(mcoeffs_path), exist_ok=True)
            if args.save_mvecs:
                os.makedirs(os.path.dirname(mvecs_path), exist_ok=True)

            success = process_single_video(extractor, video_path, mcoeffs_path, mvecs_path, args, stats_fh, rel_stem)
            if not success:
                failed_list.append(video_path)

        if stats_fh is not None:
            stats_fh.close()

        if failed_list:
            print(f"\nFailed: {len(failed_list)} video(s):")
            for v in failed_list:
                print(f"  - {v}")
            fail_path = os.path.join(args.output_dir, f"failed_rank{args.rank}.txt")
            with open(fail_path, "w") as f:
                f.write("\n".join(failed_list) + "\n")
            print(f"Failed list written to {fail_path}")
        else:
            print("\nAll assigned videos processed successfully!")
        print(f"Results saved to {args.output_dir}")
