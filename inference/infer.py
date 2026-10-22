#!/usr/bin/env python
"""User speech -> JAVIS reply as text, speech (wav) and a talking-face video (mp4).

  python inference/infer.py --audio user.wav --portrait face.png --voice_ref voice.wav --out_dir outputs/offline

--history is an optional json list of earlier rounds, [{"wav": user.wav, "text": assistant reply}, ...] (the same format
as inference/stream.py; assistant history is text only, as in training). The portrait decides identity: its own
mcoeff is the gcond, and the video is rendered relative to it.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import javis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="user turn to answer (wav)")
    ap.add_argument("--portrait", required=True, help="reference face image or video (first frame)")
    ap.add_argument("--voice_ref", required=True, help="reference wav for the reply voice (zero-shot clone)")
    ap.add_argument("--history", default=None, help="json file: earlier turns")
    ap.add_argument("--model_path", default=javis.DEFAULT_MODEL)
    ap.add_argument("--out_dir", default="outputs/offline")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_crop", action="store_true", help="portrait is already the 512x512 face crop")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or os.path.splitext(os.path.basename(a.audio))[0]
    turns = [t for h in (json.load(open(a.history)) if a.history else []) for t in (("user", h["wav"]), ("assistant", h["text"]))]
    turns.append(("user", a.audio))

    gcond = javis.image_gcond(a.portrait, crop_face=not a.no_crop)
    model, processor = javis.load_model_and_processor(a.model_path)
    out = javis.generate_reply(model, processor, turns, gcond=gcond, seed=a.seed)
    print(f"[javis] reply: {out['text']}")
    del model
    torch.cuda.empty_cache()

    from utils.cosyvoice_detokenizer import get_audio_detokenizer
    wav = javis.vocode(get_audio_detokenizer(), out["speech_tokens"], a.voice_ref)
    wav_path = os.path.join(a.out_dir, f"{tag}.wav")
    sf.write(wav_path, wav.detach().cpu().numpy().reshape(-1), 24000)

    video_path, n = javis.mcoeff2video_float(out["mcoeff"], a.portrait, wav_path, os.path.join(a.out_dir, f"{tag}.mp4"),
                                             direction_q=javis.direction_q_path(a.model_path), crop_face=not a.no_crop)
    torch.save({**out, "gcond": gcond, "portrait": a.portrait, "voice_ref": a.voice_ref}, os.path.join(a.out_dir, f"{tag}.pt"))
    print(f"[javis] {n} frames ({n / javis.MOTION_FPS:.2f} s) -> {video_path}")


if __name__ == "__main__":
    main()
