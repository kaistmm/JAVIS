"""Renderer process for streaming: one per stream, normally on a second GPU.

  --role audio : speech tokens -> CosyVoice3 chunked token2wav
  --role video : mcoeff        -> FLOAT decoder frames

A packet covers `hop` frames (hop/25 s). An audio packet needs hop + PRE_LOOKAHEAD_LEN content tokens, a video packet hop
mcoeff frames; each goes out as soon as its own input is there and is stamped with time.perf_counter() once it is on the
host (CLOCK_MONOTONIC, so the parent subtracts its own t0). A packet's media span is what was actually written: CosyVoice3
holds the last 0.16 s of every non-final chunk back until the next one, so audio spans lag the token count.

Protocol: pickled dicts. stdin <- {"op": new|tok|mot|end|quit}; reply fd -> {"op": ready|packet|done}. Library prints go
to stderr so they cannot corrupt the reply stream. Media is appended to <out_dir>/<cid>.{pcm,rgb} as packets are made.
"""
import argparse
import os
import pickle
import queue
import sys
import threading
import time
import uuid

_reply = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MOTION_FPS = 25
PRE_LOOKAHEAD_LEN = 3      # CosyVoice3 flow pre-lookahead: extra tokens a non-final chunk needs
SPEECH_CONTENT_MAX = 6561  # >= this is speech bos/eos, not a content frame


def send(msg):
    pickle.dump(msg, _reply, protocol=pickle.HIGHEST_PROTOCOL)
    _reply.flush()


def reader(q):
    inp = sys.stdin.buffer
    while True:
        try:
            m = pickle.load(inp)
        except EOFError:
            q.put({"op": "quit"})
            return
        q.put(m)
        if m["op"] == "quit":
            return


class SpeechStreamer:
    """Chunked CosyVoice3 token2wav with the hift/mel caches held across calls."""

    def __init__(self, cv, embedding, prompt_token, prompt_feat, hop):
        self.cv, self.emb, self.pt, self.pf = cv, embedding, prompt_token, prompt_feat
        self.uid = str(uuid.uuid4())
        cv.model.hift_cache_dict[self.uid] = None
        # chunk-causal attention span must match the hop, or the seam between chunks is audible
        cv.model.flow.decoder.estimator.static_chunk_size = 2 * hop

    def __call__(self, tokens, offset, finalize):
        t = torch.tensor(tokens, dtype=torch.int32).view(1, -1)
        return self.cv.model.token2wav(t, self.pt, self.pf, self.emb, offset, self.uid, stream=True, finalize=finalize, speed=1.0)

    def close(self):
        self.cv.model.hift_cache_dict.pop(self.uid, None)


class FloatRenderer:
    """mcoeff -> frames with the checkpoint load and the portrait encoding done once."""

    def __init__(self, source, float_ckpt, direction_q, size=512, device="cuda:0"):
        from inference.javis import _load_float, _read_source_frame, load_direction_q
        self.enc, self.dec, deps = _load_float(size, float_ckpt, device)
        self.Q = load_direction_q(direction_q or None, float_ckpt).to(device)
        self.size, self.device = size, device
        src = _read_source_frame(source, size, *deps)          # source is the already-cropped 512 portrait
        with torch.inference_mode():
            x0 = torch.from_numpy(src[None]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.
            self.wa0, self.feats0 = self.enc.net_app(x0)

    @torch.inference_mode()
    def decode(self, mc, batch_size=25):
        """(n,20) raw mcoeff -> (n,size,size,3) uint8 RGB."""
        out = []
        for i in range(0, mc.shape[0], batch_size):
            a = mc[i:i + batch_size].to(self.device).float()
            f = [t.expand(len(a), -1, -1, -1) for t in self.feats0]
            img, _ = self.dec(self.wa0 + a @ self.Q, alpha=None, feats=f)
            out.append(((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.size, self.size, 3), np.uint8)


class AudioStream:
    def __init__(self, a):
        from utils.cosyvoice_detokenizer import get_audio_detokenizer, get_prompt_from_wav
        self.cv = get_audio_detokenizer()
        self.emb, self.pt, self.pf = get_prompt_from_wav(self.cv, a.voice_ref)
        self.cv.model.fp16 = a.cosy_fp16
        self.cv.model.llm = None              # text->token LLM; token2wav never uses it
        torch.cuda.empty_cache()
        self.hop, self.sr, self.ext = a.hop, self.cv.sample_rate, "pcm"
        self.unit_bytes, self.rate = 4, self.sr        # f32 samples

    def new(self):
        self.content, self.offset, self.written = [], 0, 0
        self.speech = SpeechStreamer(self.cv, self.emb, self.pt, self.pf, self.hop)

    def feed(self, m):
        self.content += [t for t in m["ids"] if 0 <= t < SPEECH_CONTENT_MAX]

    def ready(self):
        return len(self.content) >= self.offset + self.hop + PRE_LOOKAHEAD_LEN

    def render(self, end, finalize):
        need = end + (0 if finalize else PRE_LOOKAHEAD_LEN)
        wav = self.speech(self.content[:need], self.offset, finalize).cpu().numpy().reshape(-1)
        return np.ascontiguousarray(wav, dtype=np.float32).tobytes()   # f32le mono @ self.sr

    def n_avail(self):
        return len(self.content)

    def close(self):
        self.speech.close()


class VideoStream:
    def __init__(self, a):
        self.r = FloatRenderer(a.source, a.float_ckpt, a.direction_q, a.size, "cuda:0")
        self.hop, self.batch, self.ext = a.hop, a.float_batch, "rgb"
        self.unit_bytes, self.rate = a.size * a.size * 3, MOTION_FPS     # rgb24 frames

    def new(self):
        self.mot, self.n, self.offset, self.written = [], 0, 0, 0

    def feed(self, m):
        self.mot.append(m["m"])
        self.n += m["m"].shape[0]

    def ready(self):
        return self.n >= self.offset + self.hop

    def render(self, end, finalize):
        mc = torch.from_numpy(np.concatenate(self.mot, 0)[self.offset:end])
        return self.r.decode(mc, self.batch).tobytes()                 # rgb24 frames, size x size

    def n_avail(self):
        return self.n

    def close(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["audio", "video"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hop", type=int, default=25)
    p.add_argument("--voice_ref", default="")
    p.add_argument("--source", default="")
    p.add_argument("--float_ckpt", default="")
    p.add_argument("--direction_q", default="")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--float_batch", type=int, default=25)
    p.add_argument("--cosy_fp16", action="store_true")
    a = p.parse_args()

    t = time.perf_counter()
    st = AudioStream(a) if a.role == "audio" else VideoStream(a)
    torch.cuda.synchronize()
    send({"op": "ready", "role": a.role, "load_s": time.perf_counter() - t, "device": torch.cuda.get_device_name(0),
          "sr": getattr(st, "sr", None)})

    q = queue.Queue()
    threading.Thread(target=reader, args=(q,), daemon=True).start()
    fh, cid, pkts = None, None, []

    def emit(end, finalize):
        data = st.render(end, finalize)
        now = time.perf_counter()
        fh.write(data)
        fh.flush()
        start = st.written / st.rate
        st.written += len(data) // st.unit_bytes
        pkt = (a.role, start, st.written / st.rate, now)
        pkts.append(pkt)
        send({"op": "packet", "cid": cid, "pkt": pkt})
        st.offset = end

    while True:
        msgs = [q.get()]
        while True:                                   # drain: render against the newest state
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                break
        for m in msgs:
            op = m["op"]
            if op == "quit":
                return
            if op == "new":
                cid, pkts = m["cid"], []
                st.new()
                fh = open(os.path.join(a.out_dir, f"{cid}.{st.ext}"), "wb")
            elif op in ("tok", "mot"):
                st.feed(m)
                while st.ready():
                    emit(st.offset + st.hop, finalize=False)
            elif op == "end":
                while st.ready():
                    emit(st.offset + st.hop, finalize=False)
                end = min(m["n"], st.n_avail())
                if end > st.offset:
                    emit(end, finalize=True)
                elif a.role == "audio" and st.offset > 0:   # nothing new, but flush the held-back tail
                    emit(st.offset, finalize=True)
                st.close()
                fh.close()
                send({"op": "done", "cid": cid, "pkts": pkts})


if __name__ == "__main__":
    main()
