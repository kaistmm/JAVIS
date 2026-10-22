"""Dialogue-history prefix caching for the motion_cfm S2SM loop (fast SRH only).

History uses the training layout (user audio, assistant TEXT only). The prefix
[system, user_1, assistant_1, ...] is run once off the clock: LM KV cache + FastSRH KV cache. The timed
step then prefills only the current user turn on top of both caches, so it is token-for-token the
same input as a full re-prefill of the whole conversation.
"""
import torch

from funaudiochat.modeling_motion_cfm import FunAudioChatMotionCFMDecoderOutput
from utils.constant import AUDIO_TEMPLATE


def conversation(conv_sys, history, with_current=True):
    msgs = [{"role": "system", "content": conv_sys}]
    for h in history:
        msgs += [{"role": "user", "content": AUDIO_TEMPLATE}, {"role": "assistant", "content": h["text"]}]
    if with_current:
        msgs.append({"role": "user", "content": AUDIO_TEMPLATE})
    return msgs


def split_texts(processor, conv_sys, history):
    """(prefix text, current-turn text); prefix + current == the full re-prefill template."""
    # cut the full template (templating the history alone adds Qwen3's empty <think> to a trailing assistant turn)
    full = processor.apply_chat_template(conversation(conv_sys, history), add_generation_prompt=True, tokenize=False)
    k = full.rindex("<|im_start|>user\n")
    if full.count(AUDIO_TEMPLATE, k) != 1 or not full[:k].endswith("<|im_end|>\n"):
        raise ValueError("chat template: unexpected layout of the current user turn")
    return full[:k], full[k:]


@torch.inference_mode()
def build_prefix(model, processor, fast, prefix_text, prefix_wavs, proc_kwargs=None):
    """Runs the prefix through the real forward (same audio merge as a full prefill), capturing the
    SRH input instead of running the SRH, then writes it into FastSRH's cache. Call fast reset first."""
    base = model.base_model.model if hasattr(model, "peft_config") else model
    ait = base.audio_invert_tower
    inp = processor(text=prefix_text, audio=prefix_wavs, return_tensors="pt", return_token_type_ids=False,
                    **(proc_kwargs or {})).to(model.device)
    cap = {}

    def capture(inputs_embeds=None, **kw):
        cap["h"] = inputs_embeds
        return FunAudioChatMotionCFMDecoderOutput(loss=None, logits=None)

    saved = ait.forward
    ait.forward = capture
    try:
        out = base(**inp, use_cache=True, return_dict=True)
    finally:
        ait.forward = saved
    if fast.pos.item() != 0:
        raise RuntimeError("FastSRH not reset before build_prefix")
    fast.prefix(cap["h"])
    return inp.input_ids, out.past_key_values


def timed_inputs(processor, cur_text, wav, prefix_ids, device, proc_kwargs=None):
    """Current-turn processor call (on the clock) + full input_ids for generate(past_key_values=...)."""
    inp = processor(text=cur_text, audio=[wav], return_tensors="pt", return_token_type_ids=False,
                    **(proc_kwargs or {})).to(device)
    ids = torch.cat([prefix_ids, inp.input_ids], dim=1)
    inp["input_ids"] = ids
    inp["attention_mask"] = torch.ones_like(ids)
    return inp


def generate_with_prefix(model, inputs, pkv, **gk):
    """generate() rejects a Cache kwarg because the class declares _supports_cache_class=False, although its
    LM builds and threads a DynamicCache itself. Skip only that start-up check; the flag stays False."""
    orig = model._validate_model_kwargs

    def _validate(kw):
        return orig({k: v for k, v in kw.items() if k != "past_key_values"})

    model._validate_model_kwargs = _validate
    try:
        return model.generate(**inputs, **gk, past_key_values=pkv)
    finally:
        del model._validate_model_kwargs
