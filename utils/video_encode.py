#!/usr/bin/env python
"""One encode path for every rendered clip in this workspace.

Why this exists: the pattern it replaces wrote frames through `cv2.VideoWriter(..., "mp4v")` into a
temp file and re-encoded THAT with x264. mp4v is a lossy codec at OpenCV's own default rate, so
every clip was compressed twice. Measured on a 351-frame 512x512 render, against the decoder's own
frames: mp4v -> x264 crf20 = 38.11 dB, raw frames -> x264 crf12 = 42.7 dB. The loss was invisible
in the code and cost 4.6 dB in the file.

Keeping one function also keeps the ENCODER constant across conditions. FVD/FID read pixel
statistics, so a comparison whose rows were written by different encoders confounds the model
difference with the codec difference.

  from utils.video_encode import write_frames
  write_frames(frames, "out.mp4", fps=25, audio="turn.wav")      # frames: (T,H,W,3) uint8 RGB
"""
import os
import subprocess

CRF = int(os.environ.get("FVC_VIDEO_CRF", "12"))
PRESET = os.environ.get("FVC_VIDEO_PRESET", "slow")
FFMPEG = os.environ.get("FVC_FFMPEG", "ffmpeg")


def write_frames(frames, output, fps=25, audio=None, crf=None, preset=None, audio_bitrate="192k"):
    """(T,H,W,3) uint8 RGB -> mp4, piped straight into x264. `audio` is muxed when it exists."""
    import numpy as np
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"expected (T,H,W,3) uint8 RGB, got {frames.shape} {frames.dtype}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio and os.path.exists(audio):
        cmd += ["-i", audio, "-c:a", "aac", "-b:a", audio_bitrate, "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", preset or PRESET, "-crf", str(CRF if crf is None else crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode:
        raise RuntimeError(f"ffmpeg exited {p.returncode} writing {output}")
    if not (os.path.exists(output) and os.path.getsize(output) > 0):
        raise RuntimeError(f"no output at {output}: is ffmpeg on PATH?")
    return output


def write_bgr_frames(frames, output, **kw):
    """Same, for call sites that already hold BGR (anything built with cv2 drawing calls)."""
    return write_frames(frames[..., ::-1], output, **kw)
