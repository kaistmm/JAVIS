"""JAVIS inference library: one AR pass emits text + speech tokens + FLOAT mcoeff; CosyVoice3 renders the
speech, the FLOAT decoder renders the face.

    frame_t = dec(wa0 + mcoeff_t @ Q, feats=feats0)   wa0/feats0: appearance latent of the source portrait

mcoeff is relative to the source frame, so the portrait decides identity; gcond (the head's global
condition) is that portrait's own mcoeff (`image_gcond`).
"""
import gc
import os
import sys

import cv2
import librosa
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import funaudiochat.register as _fac_register  # noqa: E402

_fac_register.register_funaudiochat()
_fac_register.register_funaudiochat_motion_cfm()

from utils.constant import (AUDIO_TEMPLATE, DEFAULT_S2M_GEN_KWARGS, DEFAULT_S2M_MOTION_PROMPT,  # noqa: E402
                            DEFAULT_SP_GEN_KWARGS, SPOKEN_PERSONA_EN)
from utils.video_encode import write_frames  # noqa: E402

DEFAULT_MODEL = os.environ.get("JAVIS_MODEL", f"{ROOT}/pretrained_models/JAVIS")
FLOAT_DIR = os.environ.get("FLOAT_DIR", f"{ROOT}/third_party/float")
DEFAULT_FLOAT_CKPT = os.environ.get("FLOAT_CKPT", f"{FLOAT_DIR}/checkpoints/float.pth")
MOTION_FPS = 25
SPEECH_CONTENT_MAX = 6561   # >= this is speech bos/eos, not a content frame
# training rows' system prompt + the English persona (the paper's VStyle setting)
SYSTEM_PROMPT = DEFAULT_S2M_MOTION_PROMPT + " " + SPOKEN_PERSONA_EN

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------- model
def load_model_and_processor(model_path=DEFAULT_MODEL, device=device):
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    if not hasattr(model.audio_invert_tower, "motion_cfm_head"):
        raise RuntimeError(f"{model_path}: no motion_cfm_head (FUNAUDIOCHAT variant not registered?)")
    return model, processor


def generate_reply(model, processor, turns, gcond=None, seed=42, do_sample=True, temperature=0.8, top_p=0.9,
                   max_new_tokens=2048, system_prompt=SYSTEM_PROMPT):
    """turns: [("user", wav_path) | ("assistant", text), ...] ending with the user turn to answer.
    gcond: raw (20,) mcoeff of the reference portrait. Returns text, speech tokens and (n,20) raw mcoeff,
    trimmed to the same n (both are 25 Hz content-frame indexed)."""
    if gcond is None:
        raise ValueError("gcond is required: a zero gcond is outside the head's training distribution")
    model.sp_gen_kwargs.update(dict(DEFAULT_SP_GEN_KWARGS, text_greedy=True))
    model.gen_mcoeff_ref = torch.as_tensor(gcond).float().view(-1)
    gen_kwargs = dict(DEFAULT_S2M_GEN_KWARGS, do_sample=do_sample, temperature=temperature, top_p=top_p,
                      max_new_tokens=max_new_tokens)
    gen_kwargs.pop("top_k", None)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    conversation = [{"role": "system", "content": system_prompt}]
    audio = []
    for role, x in turns:
        if role == "user":
            conversation.append({"role": "user", "content": AUDIO_TEMPLATE})
            audio.append(librosa.load(x, sr=16000)[0])
        else:
            conversation.append({"role": "assistant", "content": x})
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
    generate_ids, audio_ids, motion_latents = model.generate(**inputs, **gen_kwargs)
    reply = processor.decode(generate_ids[0, inputs.input_ids.size(1):], skip_special_tokens=True)

    speech_tokens = [t for t in audio_ids[0].tolist() if 0 <= t < SPEECH_CONTENT_MAX]
    mc = motion_latents[0].float().cpu()
    n = min(len(speech_tokens), mc.shape[0])
    return {"text": reply, "speech_tokens": speech_tokens[:n], "mcoeff": mc[:n], "n_speech_tok": len(speech_tokens),
            "prompt_tokens": int(inputs.input_ids.size(1))}


# ---------------------------------------------------------------------------------------- speech
def vocode(cosyvoice, speech_tokens, voice_ref):
    """Speech tokens -> 24 kHz waveform (1, T) in the voice of `voice_ref` (zero-shot prompt)."""
    from utils.cosyvoice_detokenizer import get_prompt_from_wav, token2wav
    emb, ptok, pfeat = get_prompt_from_wav(cosyvoice, voice_ref)
    return token2wav(cosyvoice, speech_tokens, embedding=emb, prompt_token=ptok, prompt_feat=pfeat,
                     token_hop_len=25 * 30, pre_lookahead_len=3)


# ------------------------------------------------------------------------------------------ face
def _load_float_deps(size):
    """FLOAT's autoencoder + the extractor's frame decode. `encoder`/`styledecoder` are bare top-level
    module names that collide with CosyVoice, so they are swapped in only for the import."""
    saved_path, saved_mods = list(sys.path), {}
    for name in ("encoder", "styledecoder"):
        if name in sys.modules:
            saved_mods[name] = sys.modules.pop(name)
    sys.path.insert(0, os.path.join(FLOAT_DIR, "models", "float"))
    try:
        from encoder import Encoder            # noqa: E402
        from styledecoder import Synthesis     # noqa: E402
        from data_prep.extract_mcoeff import RGB_VF, video_meta, _ffmpeg_frames  # noqa: E402
        return Encoder, Synthesis, RGB_VF, video_meta, _ffmpeg_frames
    finally:
        sys.path[:] = saved_path
        for name, mod in saved_mods.items():
            sys.modules[name] = mod


_FACE_ALIGNER = None


def _face_crop_rect(path, device="cuda:0"):
    """The extractor's face crop (size, left, top), same parameters as the training-data extraction
    (increase_area 0.25 / top 0.5); stills get one SFD detection, videos the probed tube bbox."""
    global _FACE_ALIGNER
    saved_path = list(sys.path)
    sys.path.insert(0, os.path.join(FLOAT_DIR, "models", "float"))
    try:
        from data_prep.extract_mcoeff import probe_bbox, crop_rect, video_meta  # noqa: E402
        import face_alignment  # noqa: E402
        if _FACE_ALIGNER is None:
            _FACE_ALIGNER = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot read image {path}")
            h, w = bgr.shape[:2]
            batch = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[None]).permute(0, 3, 1, 2).float().to(_FACE_ALIGNER.device)
            dets = _FACE_ALIGNER.face_detector.detect_from_batch(batch)[0]
            if len(dets) == 0:
                raise RuntimeError(f"no face detected in {path}")
            x1, y1, x2, y2 = np.asarray(dets[0])[:4]
            return crop_rect([x1, y1, x2, y2], (h, w), 0.25, 0.5)
        n, w, h, _ = video_meta(path)
        return probe_bbox(path, _FACE_ALIGNER, w, h, n, 0.25, 0.5, 10, 64, 4)
    finally:
        sys.path[:] = saved_path


def _read_source_frame(path, size, RGB_VF, video_meta, _ffmpeg_frames, crop=None):
    """First frame of a video (decoded exactly as the extractor did) or a still image, as uint8 RGB (size, size, 3)."""
    if os.path.splitext(path)[1].lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        _, w, h, _ = video_meta(path)
        vf, (dh, dw) = RGB_VF, (h, w)
        if crop is not None:   # crop after the color conversion, as in the extractor's vf string
            s, left, top = crop
            vf, (dh, dw) = f"{RGB_VF},crop={s}:{s}:{left}:{top}", (s, s)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-i", path, "-vf", vf,
               "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        img = None
        for b in _ffmpeg_frames(cmd, dh * dw * 3):
            img = np.frombuffer(b, np.uint8).reshape(dh, dw, 3).copy()
        if img is None:
            raise RuntimeError(f"no frames decoded from {path}")
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read source image {path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if crop is not None:
            s, left, top = crop
            img = img[top:top + s, left:left + s].copy()
    if img.shape[:2] != (size, size):
        interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
        img = cv2.resize(img, (size, size), interpolation=interp)
    return img


def _load_float(size, float_ckpt, device):
    Encoder, Synthesis, RGB_VF, video_meta, _ffmpeg_frames = _load_float_deps(size)
    sd = torch.load(float_ckpt, map_location="cpu", weights_only=True)
    enc, dec = Encoder(size, 512, 20), Synthesis(size, 512, 20)
    enc.load_state_dict({k[len("motion_autoencoder.enc."):]: v for k, v in sd.items() if k.startswith("motion_autoencoder.enc.")})
    dec.load_state_dict({k[len("motion_autoencoder.dec."):]: v for k, v in sd.items() if k.startswith("motion_autoencoder.dec.")})
    del sd
    return enc.to(device).eval(), dec.to(device).eval(), (RGB_VF, video_meta, _ffmpeg_frames)


@torch.inference_mode()
def image_gcond(path, size=512, float_ckpt=DEFAULT_FLOAT_CKPT, device=device, crop_face=True):
    """Raw (20,) FLOAT mcoeff of a portrait, cropped exactly as the render's source frame."""
    enc, _, deps = _load_float(size, float_ckpt, device)
    crop = _face_crop_rect(path, device) if crop_face else None
    img = _read_source_frame(path, size, *deps, crop=crop)
    x = torch.from_numpy(img[None]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.
    return enc.enc_motion(x)[0].float().cpu()


def direction_q_path(model_path=DEFAULT_MODEL):
    """The (20, 512) motion basis the mcoeff live in: $JAVIS_DIRECTION_Q, else <model>/direction_Q.pt, else None."""
    p = os.environ.get("JAVIS_DIRECTION_Q") or os.path.join(model_path, "direction_Q.pt")
    return p if os.path.exists(p) else None


def load_direction_q(path, float_ckpt=DEFAULT_FLOAT_CKPT):
    """direction_Q.pt if given, else the same basis recomputed from the FLOAT checkpoint (QR of the decoder direction)."""
    if path:
        Q = torch.load(path, map_location="cpu", weights_only=True).float()
    else:
        w = torch.load(float_ckpt, map_location="cpu", weights_only=True)["motion_autoencoder.dec.direction.weight"].float()
        Q = torch.linalg.qr(w + 1e-8)[0].T.contiguous()
    if tuple(Q.shape) != (20, 512):
        raise ValueError(f"direction_Q must be (20,512), got {tuple(Q.shape)}")
    return Q


def mcoeff2video_float(mcoeff, source, audio, output, float_ckpt=DEFAULT_FLOAT_CKPT, direction_q=None,
                       size=512, batch_size=8, fps=MOTION_FPS, device=device, crop_face=False, crf=None):
    """(T,20) raw mcoeff + source portrait -> mp4 with `audio` muxed in. crop_face=True for uncropped
    portraits; leave False for frames that are already the 512 face crop (cropping twice zooms in)."""
    feats = torch.as_tensor(mcoeff).float().cpu()
    if feats.ndim != 2 or feats.shape[1] != 20:
        raise ValueError(f"expected (T, 20) FLOAT mcoeff, got {tuple(feats.shape)}")
    enc, dec, deps = _load_float(size, float_ckpt, device)
    Q = load_direction_q(direction_q, float_ckpt).to(device)
    crop = _face_crop_rect(source, device) if crop_face else None
    src = _read_source_frame(source, size, *deps, crop=crop)
    with torch.inference_mode():
        x0 = torch.from_numpy(src[None]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.
        wa0, feats0 = enc.net_app(x0)
        frames = []
        for i in range(0, feats.shape[0], batch_size):
            a = feats[i:i + batch_size].to(device)
            f = [t.expand(len(a), -1, -1, -1) for t in feats0]
            img, _ = dec(wa0 + a @ Q, alpha=None, feats=f)
            frames.append(((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
    frames = np.concatenate(frames)
    write_frames(frames, output, fps=fps, audio=audio, crf=crf)
    del enc, dec, Q
    gc.collect()
    torch.cuda.empty_cache()
    return output, frames.shape[0]
