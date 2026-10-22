#!/usr/bin/env python
"""Real-time streaming reply: speech and face come out in 1 s packets while the reply is still being generated.

One process generates (LM + SRH + motion head) on --gen_gpu; two renderer processes (CosyVoice3 for speech, FLOAT for
the face, inference/stream_worker.py) run on --render_gpu and turn every new speech-token group / motion group into
playable media as soon as a packet's worth exists. Each packet's arrival time is printed live and logged; at the end the
streamed PCM and frames are muxed into <out_dir>/<tag>.mp4.

  # 2 GPUs (RTX 4090 x2: first audio+video packet ~1.0 s, RTF ~0.45)
  python inference/stream.py --audio user.wav --portrait face.png --voice_ref voice.wav --gen_gpu 0 --render_gpu 1 --cosy_fp16 --fe_gpu
  # one GPU with ~33 GB
  python inference/stream.py ... --gen_gpu 0 --render_gpu 0

--history json: earlier turns [{"wav": user.wav, "text": assistant reply}, ...]; prefilled off the clock.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True, help="user turn (wav)")
ap.add_argument("--portrait", required=True, help="reference face image or video")
ap.add_argument("--voice_ref", required=True, help="reference wav for the reply voice")
ap.add_argument("--history", default=None)
ap.add_argument("--model_path", default=None, help="default: $JAVIS_MODEL or pretrained_models/JAVIS")
ap.add_argument("--out_dir", default="outputs/stream")
ap.add_argument("--tag", default=None)
ap.add_argument("--gen_gpu", default="0")
ap.add_argument("--render_gpu", default="1")
ap.add_argument("--hop", type=int, default=25, help="packet length in 25 Hz frames")
ap.add_argument("--no_fast", action="store_true", help="plain HF SRH + motion sampler instead of the CUDA-graph versions (not real-time: RTF ~1.1)")
ap.add_argument("--cosy_fp16", action="store_true", help="CosyVoice3 flow in fp16")
ap.add_argument("--fe_gpu", action="store_true", help="Whisper features on the generation GPU")
ap.add_argument("--no_crop", action="store_true", help="portrait is already the 512x512 face crop")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_new_tokens", type=int, default=2048)
ap.add_argument("--srh_max_len", type=int, default=8192)
ap.add_argument("--no_warmup", action="store_true", help="skip the discarded first pass (graph capture, autotuning, CosyVoice init)")
a = ap.parse_args()

# pin this process to the generation GPU before torch initialises CUDA; the workers get the render GPU
os.environ["CUDA_VISIBLE_DEVICES"] = a.gen_gpu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "inference"))

import cv2  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402

import javis  # noqa: E402
from packet_log import PacketLog, summarize  # noqa: E402
from utils.constant import DEFAULT_S2M_GEN_KWARGS, DEFAULT_SP_GEN_KWARGS  # noqa: E402


class Worker:
    def __init__(self, role, args, on_packet):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.render_gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.role = role
        self.p = subprocess.Popen([sys.executable, os.path.join(ROOT, "inference", "stream_worker.py"), "--role", role] + args,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=ROOT)
        self.ready = pickle.load(self.p.stdout)
        self.done = threading.Event()
        self.result = None
        self.on_packet = on_packet
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                m = pickle.load(self.p.stdout)
            except EOFError:                          # renderer exited (crash, OOM): release the waiter
                self.result = None
                self.done.set()
                return
            if m["op"] == "packet":
                self.on_packet(m["pkt"])
            elif m["op"] == "done":
                self.result = m
                self.done.set()

    def send(self, m):
        try:
            pickle.dump(m, self.p.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"{self.role} renderer exited (exit code {self.p.poll()}); see its stderr above") from None

    def close(self):
        try:
            self.send({"op": "quit"})
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=30)


def prepare_portrait(path, out_png):
    """Face-crop the portrait exactly as the training data were cropped; the 512 PNG is both FLOAT's source frame and
    the gcond image, so the two live in one appearance latent."""
    Encoder, Synthesis, RGB_VF, video_meta, _ffmpeg_frames = javis._load_float_deps(512)
    crop = None if a.no_crop else javis._face_crop_rect(path, "cuda:0")
    img = javis._read_source_frame(path, 512, RGB_VF, video_meta, _ffmpeg_frames, crop=crop)
    cv2.imwrite(out_png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return javis.image_gcond(out_png, crop_face=False)


def main():
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or os.path.splitext(os.path.basename(a.audio))[0]
    model_path = a.model_path or javis.DEFAULT_MODEL
    portrait = os.path.join(a.out_dir, f"{tag}_portrait512.png")
    gcond = prepare_portrait(a.portrait, portrait)
    torch.cuda.empty_cache()

    st = {"t0": None}

    def on_packet(pkt):
        if st["t0"] is not None:
            role, ms, me, t = pkt
            print(f"[packet] {role:5s} {ms:5.2f}-{me:5.2f} s ready at +{t - st['t0']:.2f} s", flush=True)

    aw = Worker("audio", ["--out_dir", a.out_dir, "--hop", str(a.hop), "--voice_ref", a.voice_ref]
                + (["--cosy_fp16"] if a.cosy_fp16 else []), on_packet)
    vw = Worker("video", ["--out_dir", a.out_dir, "--hop", str(a.hop), "--source", portrait, "--float_ckpt", javis.DEFAULT_FLOAT_CKPT,
                          "--direction_q", javis.direction_q_path(model_path) or "", "--float_batch", str(a.hop)], on_packet)
    model, processor = javis.load_model_and_processor(model_path, "cuda:0")
    fast = fast_reset = None
    if not a.no_fast:
        from fast_gen import install_fast_srh, install_graphed_cfm
        fast, fast_reset = install_fast_srh(model, a.srh_max_len, a.max_new_tokens)
        install_graphed_cfm(model)

    model.sp_gen_kwargs.update({**DEFAULT_SP_GEN_KWARGS, "text_greedy": True})
    model.gen_mcoeff_ref = gcond.float()
    ait = model.audio_invert_tower
    eos = int(model.config.audio_config.eos_token_id)
    orig_crq, orig_cfm = ait.crq_generate_forward, ait.motion_cfm_sample
    n = {"tok": 0, "mot": 0}

    def crq_hooked(*args, **kw):
        out = orig_crq(*args, **kw)
        if bool(model.generate_speech.any()):
            t = ait.crq_generate_tokens[0]
            ids = [eos] * t.shape[0] if bool((t == eos).any()) else t.tolist()
            ids = [i for i in ids if 0 <= i < javis.SPEECH_CONTENT_MAX]
            if ids:
                aw.send({"op": "tok", "ids": ids})
                n["tok"] += len(ids)
        return out

    def cfm_hooked(*args, **kw):
        m = orig_cfm(*args, **kw)
        mc = ait.denormalize_mcoeff(m.float())[0].cpu().numpy()
        vw.send({"op": "mot", "m": mc})
        n["mot"] += mc.shape[0]
        return m

    history = json.load(open(a.history)) if a.history else []
    hist_wavs = [librosa.load(h["wav"], sr=16000)[0] for h in history]
    wav_in = librosa.load(a.audio, sr=16000)[0]
    proc_kw = {"audio_kwargs": {"device": "cuda"}} if a.fe_gpu else {}
    gk = {**DEFAULT_S2M_GEN_KWARGS, "max_new_tokens": a.max_new_tokens}

    def respond(cid, log):
        pkv = None
        n.update(tok=0, mot=0)
        if fast_reset is not None:
            fast_reset()
        if history and fast is not None:
            from prefix_cache import build_prefix, split_texts
            pre_text, cur_text = split_texts(processor, javis.SYSTEM_PROMPT, history)
            prefix_ids, pkv = build_prefix(model, processor, fast, pre_text, hist_wavs, proc_kw)   # off the clock
        for w in (aw, vw):
            w.done.clear()
            w.send({"op": "new", "cid": cid})
        ait.crq_generate_forward, ait.motion_cfm_sample = crq_hooked, cfm_hooked
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        javis._sync()
        t0 = time.perf_counter()
        st["t0"] = t0 if log is not None else None
        if pkv is not None:
            from prefix_cache import generate_with_prefix, timed_inputs
            inputs = timed_inputs(processor, cur_text, wav_in, prefix_ids, model.device, proc_kw)
            gen_ids, _, _ = generate_with_prefix(model, inputs, pkv, **gk)
        else:
            from prefix_cache import conversation
            text = processor.apply_chat_template(conversation(javis.SYSTEM_PROMPT, history), add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audio=hist_wavs + [wav_in], return_tensors="pt", return_token_type_ids=False, **proc_kw).to(model.device)
            gen_ids, _, _ = model.generate(**inputs, **gk)
        javis._sync()
        t_gen = time.perf_counter() - t0
        n_fin = min(n["tok"], n["mot"])
        for w in (aw, vw):
            w.send({"op": "end", "n": n_fin})
        for w in (aw, vw):
            w.done.wait()
            if w.result is None:
                raise RuntimeError(f"{w.role} renderer exited before finishing (exit code {w.p.poll()}); see its stderr above")
        ait.crq_generate_forward, ait.motion_cfm_sample = orig_crq, orig_cfm
        return t0, t_gen, gen_ids, inputs

    if not a.no_warmup:
        respond("warmup", None)
        for ext in ("rgb", "pcm"):
            os.remove(os.path.join(a.out_dir, f"warmup.{ext}"))
        print("[warmup] done", flush=True)
    log = PacketLog(0.0, a.hop, "joint", tag)
    t0, t_gen, gen_ids, inputs = respond(tag, log)
    log.t0 = t0
    pk = aw.result["pkts"] + vw.result["pkts"]
    for s, ms, me, tabs in pk:
        log.pkts.append((s, ms, me, tabs - t0))
    wall = max(p[3] for p in pk) - t0 if pk else t_gen
    reply = processor.decode(gen_ids[0, inputs.input_ids.size(1):], skip_special_tokens=True)
    res = log.result(wall_s=wall, checkpoint=model_path, prompt_kind="persona_en", hardware=f"gen={torch.cuda.get_device_name(0)} render={vw.ready['device']}",
                     extra={"text": reply, "n_speech_tokens": n["tok"], "n_motion_frames": n["mot"], "generate_s": round(t_gen, 3)})
    json.dump(res, open(os.path.join(a.out_dir, f"{tag}.json"), "w"), ensure_ascii=False, indent=1)
    aw.close()
    vw.close()

    mp4 = os.path.join(a.out_dir, f"{tag}.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512", "-r", "25", "-i", os.path.join(a.out_dir, f"{tag}.rgb"),
                    "-f", "f32le", "-ar", str(aw.ready["sr"]), "-ac", "1", "-i", os.path.join(a.out_dir, f"{tag}.pcm"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", mp4], check=True)
    for ext in ("rgb", "pcm"):
        os.remove(os.path.join(a.out_dir, f"{tag}.{ext}"))
    print(f"[reply] {reply}")
    print(summarize(res))
    print(f"[done] {mp4}")


if __name__ == "__main__":
    main()
