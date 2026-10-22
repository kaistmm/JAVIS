"""Packet-arrival accounting for the streaming pipeline: each stream (audio, video) is logged separately, and a
packet's playout deadline is playback start + its media START.

t0 is taken after the user wav is read and before feature extraction, so latencies answer "how long after the user
stopped talking".
"""
import json
import os
import time

TOKEN_FPS = 25.0


class PacketLog:
    """Per-packet arrival times for one run, plus the metrics the paper table needs."""

    def __init__(self, t0, hop_tokens, arm, clip_id):
        self.t0, self.hop_tokens, self.arm, self.clip_id = t0, int(hop_tokens), arm, clip_id
        self.pkts = []                      # (stream, media_start_s, media_end_s, wall_s)
        self.stage = {}                     # name -> accumulated seconds
        self.load = {}

    # ---------------- recording ----------------
    def add(self, stream, media_start_s, media_end_s):
        """Call at the instant the packet is ready for playout, i.e. after any device->host copy."""
        if stream not in ("audio", "video"):
            raise ValueError(f"stream must be audio|video, got {stream!r}")
        if media_end_s <= media_start_s:
            raise ValueError(f"empty packet [{media_start_s}, {media_end_s})")
        self.pkts.append((stream, float(media_start_s), float(media_end_s),
                          time.perf_counter() - self.t0))

    def acc(self, name, dt):
        self.stage[name] = self.stage.get(name, 0.0) + float(dt)

    # ---------------- metrics ----------------
    def _of(self, stream):
        return [p for p in self.pkts if p[0] == stream]

    def first(self, stream):
        p = self._of(stream)
        return min(x[3] for x in p) if p else float("nan")

    def first_av(self):
        """When media time 0 is playable on BOTH streams. nan if either stream never produced it."""
        out = []
        for s in ("audio", "video"):
            head = [p[3] for p in self._of(s) if p[1] <= 1e-9]
            if not head:
                return float("nan")
            out.append(min(head))
        return max(out)

    def media_s(self, stream):
        p = self._of(stream)
        return max(x[2] for x in p) if p else 0.0

    def stalls(self):
        """Gapless-playout check. Playback starts at first_av; packet k must be ready by
        first_av + its media START. Returns (n_late, n_deadlines, worst_slack_s)."""
        t_play = self.first_av()
        if t_play != t_play:                               # nan
            return 0, 0, float("nan")
        late, n, worst = 0, 0, float("inf")
        for _, ms, _, w in self.pkts:
            n += 1
            slack = (t_play + ms) - w
            worst = min(worst, slack)
            if slack < 0:
                late += 1
        return late, n, (worst if worst != float("inf") else float("nan"))

    # ---------------- output ----------------
    def result(self, wall_s, checkpoint, prompt_kind, hardware, extra=None, cap_terminated=None):
        # the denominator is the EMITTED media, not the untruncated token count: a cap-terminated run
        # makes those differ by one group and the two harnesses then disagree by ~5% on the same run
        a_s, v_s = self.media_s("audio"), self.media_s("video")
        media = min(a_s, v_s) if (a_s and v_s) else max(a_s, v_s)
        late, n_dl, worst = self.stalls()
        res = {
            "arm": self.arm, "clip_id": self.clip_id, "hop_tokens": self.hop_tokens,
            "hardware": hardware, "checkpoint": checkpoint, "prompt_kind": prompt_kind,
            "cap_terminated": cap_terminated,
            "media_s": round(media, 4), "media_audio_s": round(a_s, 4), "media_video_s": round(v_s, 4),
            "wall_s": round(wall_s, 4),
            "rtf": round(wall_s / media, 4) if media else None,
            "first_audio_s": round(self.first("audio"), 4),
            "first_video_s": round(self.first("video"), 4),
            "first_av_s": round(self.first_av(), 4),
            "stalls": late, "n_deadlines": n_dl, "stall_free": (n_dl > 0 and late == 0),
            "worst_slack_s": round(worst, 4) if worst == worst else None,
            "packets": [{"stream": s, "media_start_s": round(ms, 4),
                         "media_end_s": round(me, 4), "wall_s": round(w, 4)}
                        for s, ms, me, w in self.pkts],
            "stage_s": {k: round(v, 4) for k, v in self.stage.items()},
            "load_s": {k: round(v, 3) for k, v in self.load.items()},
        }
        if extra:
            res.update(extra)
        return res

    def save(self, path, **kw):
        res = self.result(**kw)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        return res


def summarize(res):
    """One human-readable line per run, for the launcher log."""
    return (f"[{res['arm']}] {res['clip_id']}  media {res['media_s']:.2f}s  wall {res['wall_s']:.2f}s  "
            f"RTF {res['rtf']}  first-audio {res['first_audio_s']:.2f}s  "
            f"first-video {res['first_video_s']:.2f}s  first-AV {res['first_av_s']:.2f}s  "
            f"stalls {res['stalls']}/{res['n_deadlines']}  worst-slack {res['worst_slack_s']}")
