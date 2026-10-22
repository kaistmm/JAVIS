# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""JAVIS: Fun-Audio-Chat + a flow-matching motion head that predicts FLOAT mcoeff (20-d @ 25 Hz)
off the SRH's pre-output_matching hidden states, in the same autoregressive pass as text and speech.

Timing (group_size G=5 frames = 0.2 s; E[g] := the SRH block that EMITS speech group g, located at
LM audio slot p0+g-1 because the SRH output at flat j predicts frame j+G):

    target M(g)  <-  cond [ E[g-1] | E[g] ]   (COND_GROUPS=2, LOOKAHEAD_GROUPS=1)
                     prev  M(g-1)
                     gcond reference-portrait mcoeff

Assistant streams start with a 10-frame bos onset, so speech group g holds mcoeff group g-2 (g>=2)
and the onset groups supply the past context for the first real target.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import contextlib
import logging
import math
import os

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor
from transformers.modeling_outputs import CausalLMOutput

from .configuration_funaudiochat import FunAudioChatConfig
from .modeling_funaudiochat import (
    FunAudioChatCausalLMOutputWithPast,
    FunAudioChatDecoder,
    FunAudioChatForConditionalGeneration,
)

logger = logging.getLogger(__name__)

MCOEFF_DIM = 20
# first motion group the free-run loop emits; with it len(motion) == len(content speech)
FIRST_MOTION_GROUP = 3


def _envf(key, default):
    v = os.environ.get(key, "").strip()
    return float(v) if v else default


def _envi(key, default):
    v = os.environ.get(key, "").strip()
    return int(v) if v else default


def _envb(key, default=False):
    v = os.environ.get(key, "").strip().lower()
    return v in ("1", "true", "yes") if v else default


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x, scale=1000):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=x.device) * -emb)
        emb = scale * x.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner = int(8 * dim / 3)
        self.fc1 = nn.Linear(dim, inner, bias=False)
        self.fc2 = nn.Linear(dim, inner, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        return self.proj(nn.functional.silu(self.fc1(x)) * self.fc2(x))


class _Attn(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0
        self.n_head, self.head_dim = n_head, dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        y = nn.functional.scaled_dot_product_attention(q, k, v)  # bidirectional, no mask
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class _Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = _Attn(dim, n_head)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = _SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MotionCFMHead(nn.Module):
    """Bidirectional DiT over [tau, gcond, SRH hidden (n_cond*G), prev (G), noisy target (G)].

    Media positions align prev/target with the hidden block that sits on the target's group
    (anchor c = n_cond-1-la); position 0 is shared by tau and gcond. Output is the velocity field
    at the target positions."""

    def __init__(self, crq_hidden=1024, motion_dim=MCOEFF_DIM, dim=1024, n_layers=8, n_head=8,
                 group_size=5, n_cond_groups=2, n_prev_groups=1, n_lookahead_groups=1):
        super().__init__()
        if not 0 <= n_lookahead_groups <= n_cond_groups:
            raise ValueError(f"n_lookahead_groups={n_lookahead_groups} must be in [0, n_cond_groups]")
        self.dim, self.group_size = dim, group_size
        self.n_cond_groups, self.n_prev_groups = n_cond_groups, n_prev_groups
        self.n_lookahead_groups = n_lookahead_groups
        self.n_hidden_tok = n_cond_groups * group_size
        self.n_prev_tok = n_prev_groups * group_size

        self.hidden_proj = nn.Linear(crq_hidden, dim, bias=True)
        self.prev_proj = nn.Linear(motion_dim, dim, bias=True)
        self.x_proj = nn.Linear(motion_dim, dim, bias=True)
        self.global_proj = nn.Linear(motion_dim, dim, bias=True)

        self.time_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        c = n_cond_groups - 1 - n_lookahead_groups
        h_rel = torch.arange(self.n_hidden_tok) - c * group_size
        prev_rel = torch.arange(self.n_prev_tok) - self.n_prev_tok
        tgt_rel = torch.arange(group_size)
        lo = int(min(h_rel.min(), prev_rel.min(), tgt_rel.min()))
        h_pos, prev_pos, tgt_pos = (x - lo + 1 for x in (h_rel, prev_rel, tgt_rel))
        n_pos = int(max(h_pos.max(), prev_pos.max(), tgt_pos.max())) + 1
        self.pos_emb = nn.Embedding(n_pos, dim)
        zero = torch.zeros(2, dtype=torch.long)
        self.register_buffer("_pos_ids", torch.cat([zero, h_pos, prev_pos, tgt_pos]), persistent=False)

        self.blocks = nn.ModuleList([_Block(dim, n_head) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        # not zero-init: out_proj is the head's only output path
        self.out_proj = nn.Linear(dim, motion_dim, bias=True)
        self.grad_checkpointing = False

        # normalization stats travel with the head so a saved checkpoint is self-describing
        self.register_buffer("mc_mean", torch.zeros(motion_dim), persistent=True)
        self.register_buffer("mc_std", torch.ones(motion_dim), persistent=True)

    def forward(self, x, t, h_cond, prev, gcond):
        """x (N,G,Dm) noisy target; t (N,); h_cond (N,n_cond*G,C); prev (N,G,Dm); gcond (N,Dm)."""
        tok_t = self.time_mlp(self.time_emb(t).to(x.dtype)).unsqueeze(1)
        tok_g = self.global_proj(gcond).unsqueeze(1)
        tok_h = self.hidden_proj(h_cond)
        seq = torch.cat([tok_t, tok_g, tok_h, self.prev_proj(prev), self.x_proj(x)], dim=1)
        seq = seq + self.pos_emb(self._pos_ids).unsqueeze(0).to(seq.dtype)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                seq = torch.utils.checkpoint.checkpoint(blk, seq, use_reentrant=False)
            else:
                seq = blk(seq)
        return self.out_proj(self.norm(seq[:, -x.shape[1]:]))


def cfm_item_errors(head, x1, h_cond, prev, gcond, t, z):
    """Per-frame squared velocity error, fp32 (N, G). t fp32 (N,), z has x1's dtype."""
    tt = t.to(x1.dtype)[:, None, None]
    y = (1 - tt) * x1 + tt * z
    v = z - x1
    pred = head(y, t, h_cond, prev, gcond)
    return (pred.float() - v.float()).pow(2).mean(-1)


@torch.no_grad()
def cfm_sample(head, h_cond, prev, gcond, n_steps=10, temperature=1.0, cfg_scale=2.0,
               sway=1.0, cfg_zero_star=True, initial_noise=None):
    """Euler ODE from tau=1 (noise) to tau=0 (data); CFG drops the SRH hidden only.
    FUNAUDIOCHAT_CFM_SAMPLE_SEED makes the initial noise depend only on the seed and item shape."""
    N = h_cond.shape[0]
    P = head.group_size
    dev, dt_ = h_cond.device, h_cond.dtype
    if initial_noise is not None:
        x = initial_noise.to(device=dev, dtype=dt_)
    else:
        _sd = os.environ.get("FUNAUDIOCHAT_CFM_SAMPLE_SEED", "")
        if _sd:
            gen = torch.Generator(device=dev).manual_seed(int(_sd) + N * 1000003 + P)
            x = torch.randn(N, P, MCOEFF_DIM, device=dev, dtype=dt_, generator=gen) * temperature
        else:
            x = torch.randn(N, P, MCOEFF_DIM, device=dev, dtype=dt_) * temperature
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=dev, dtype=torch.float32)
    if sway:
        ts = ts + sway * (torch.cos(math.pi / 2 * ts) - 1 + ts)
    zero_init = max(1, int(n_steps * 0.04)) if cfg_zero_star else 0
    use_cfg = cfg_scale is not None and cfg_scale != 1.0

    for step in range(n_steps):
        t_cur, dt = ts[step], ts[step] - ts[step + 1]
        if step < zero_init:
            continue  # cfg-zero*: hold x for the first steps
        tb = t_cur.expand(N)
        if use_cfg:
            out = head(torch.cat([x, x], 0), torch.cat([tb, tb], 0), torch.cat([h_cond, torch.zeros_like(h_cond)], 0),
                       torch.cat([prev, prev], 0), torch.cat([gcond, gcond], 0))
            cond, uncond = out.split(N, dim=0)
            if cfg_zero_star:
                pf, nf = cond.reshape(N, -1).float(), uncond.reshape(N, -1).float()
                st = ((pf * nf).sum(1, keepdim=True) / (nf.pow(2).sum(1, keepdim=True) + 1e-8))
                st = st.to(cond.dtype).view(N, 1, 1)
            else:
                st = 1.0
            v = uncond * st + cfg_scale * (cond - uncond * st)
        else:
            v = head(x, tb, h_cond, prev, gcond)
        x = x - dt.to(x.dtype) * v
    return x


@dataclass
class FunAudioChatMotionCFMDecoderOutput(CausalLMOutput):
    speech_loss: Optional[torch.FloatTensor] = None
    speech_logits: torch.FloatTensor = None
    speech_shifted_labels: Optional[torch.LongTensor] = None
    crq_hidden: Optional[torch.FloatTensor] = None


@dataclass
class FunAudioChatMotionCFMCausalLMOutputWithPast(FunAudioChatCausalLMOutputWithPast):
    motion_cfm_loss: Optional[torch.FloatTensor] = None
    speech_shifted_labels: Optional[torch.LongTensor] = None


class FunAudioChatMotionCFMDecoder(FunAudioChatDecoder):
    """SRH that also exposes its pre-output_matching hidden states (crq_hidden) to the motion head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # geometry: COND/LOOKAHEAD env overrides exist for the paper's "w/o lookahead" ablation (cond 1, la 0)
        self.motion_cfm_head = MotionCFMHead(
            crq_hidden=self.output_matching.in_features,
            group_size=self.group_size,
            n_cond_groups=_envi("FUNAUDIOCHAT_CFM_COND_GROUPS", 2),
            n_lookahead_groups=_envi("FUNAUDIOCHAT_CFM_LOOKAHEAD_GROUPS", 1),
        )

        # train-only knobs (both stages)
        self.cfm_k_repeat = _envi("FUNAUDIOCHAT_CFM_K_REPEAT", 2)
        self.cfm_cond_drop = _envf("FUNAUDIOCHAT_CFM_COND_DROP", 0.1)
        self.cfm_prev_drop = _envf("FUNAUDIOCHAT_CFM_PREV_DROP", 0.25)
        self.cfm_gcond_drop = _envf("FUNAUDIOCHAT_CFM_GCOND_DROP", 0.5)
        self.cfm_t_mu = _envf("FUNAUDIOCHAT_CFM_T_MU", -0.4)
        self.cfm_t_sigma = _envf("FUNAUDIOCHAT_CFM_T_SIGMA", 1.0)
        # cap on CFM items per micro-batch (uniform random subsample -> unbiased)
        self.cfm_max_items = _envi("FUNAUDIOCHAT_CFM_MAX_ITEMS", 1024)
        self.motion_cfm_head.grad_checkpointing = _envb("FUNAUDIOCHAT_CFM_GRAD_CKPT", True)
        self._cfm_logged = False

        self.post_init()

    @property
    def speech_head(self):
        return self.lm_head

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.RMSNorm) and module.weight is not None:
            module.weight.data.fill_(1.0)

    def normalize_mcoeff(self, x):
        h = self.motion_cfm_head
        return (x - h.mc_mean.to(x.dtype)) / h.mc_std.to(x.dtype)

    def denormalize_mcoeff(self, x):
        h = self.motion_cfm_head
        return x * h.mc_std.to(x.dtype) + h.mc_mean.to(x.dtype)

    def motion_cfm_loss(self, x1, h_cond, prev, gcond, tgt_mask):
        """x1 (N,G,Dm) normalized GT; tgt_mask (N,G). OT-CFM with logit-normal t and independent
        cond/prev/gcond drops. Always returns a scalar."""
        K = max(1, self.cfm_k_repeat)
        if K > 1:
            x1, h_cond, prev, gcond, tgt_mask = (
                t.repeat_interleave(K, dim=0) for t in (x1, h_cond, prev, gcond, tgt_mask)
            )
        N = x1.shape[0]
        if self.training:
            if self.cfm_cond_drop > 0:
                keep = (torch.rand(N, device=x1.device) >= self.cfm_cond_drop).to(x1.dtype)
                h_cond = h_cond * keep[:, None, None]
            if self.cfm_prev_drop > 0:
                keep = (torch.rand(N, device=x1.device) >= self.cfm_prev_drop).to(x1.dtype)
                prev = prev * keep[:, None, None]
            if self.cfm_gcond_drop > 0:
                keep = (torch.rand(N, device=x1.device) >= self.cfm_gcond_drop).to(x1.dtype)
                gcond = gcond * keep[:, None]
        t = torch.sigmoid(torch.randn(N, device=x1.device, dtype=torch.float32) * self.cfm_t_sigma + self.cfm_t_mu)
        z = torch.randn_like(x1)
        per_frame = cfm_item_errors(self.motion_cfm_head, x1, h_cond, prev, gcond, t, z)
        denom = tgt_mask.float().sum().clamp(min=1.0)
        return (per_frame * tgt_mask.float()).sum() / denom

    @torch.no_grad()
    def motion_cfm_sample(self, h_cond, prev, gcond, n_steps=10, temperature=1.0, cfg_scale=2.0,
                          sway=1.0, cfg_zero_star=True, initial_noise=None):
        return cfm_sample(self.motion_cfm_head, h_cond, prev, gcond, n_steps=n_steps,
                          temperature=temperature, cfg_scale=cfg_scale, sway=sway,
                          cfg_zero_star=cfg_zero_star, initial_noise=initial_noise)

    def crq_generate_forward(
        self,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Base speech generation + the group's 5 pre-output_matching hidden states in
        `self.crq_hidden_block` (B,G,C), the motion head's condition."""
        inputs_embeds = self.pre_matching(inputs_embeds)
        bs, slen, hs = inputs_embeds.shape
        hidden_states = inputs_embeds.reshape(bs, slen * self.group_size, -1)
        self.crq_audio_embeds = (
            self.get_embeddings(self.config.bos_token_id)[None, None, :]
            .repeat(bs, 1, 1)
            .to(dtype=hidden_states.dtype, device=hidden_states.device)
            if self.crq_audio_embeds is None
            else self.crq_audio_embeds.unsqueeze(1)
        )

        all_logits = []
        blk = []
        self.crq_generate_tokens = []
        for i in range(self.group_size):
            if i == 0:
                input_embeds = (
                    hidden_states[:, : slen * self.group_size - (self.group_size - i - 1)] + self.crq_audio_embeds
                )
            else:
                input_embeds = (
                    hidden_states[:, slen * self.group_size - (self.group_size - i)] + self.crq_audio_embeds.squeeze(1)
                ).unsqueeze(1)
            input_embeds = self.input_matching(input_embeds)
            outputs = self.crq_transformer(
                inputs_embeds=input_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                past_key_values=self.crq_past_key_values,
                use_cache=True,
                return_dict=True,
            )
            self.crq_past_key_values = outputs.past_key_values
            lhidden_states = outputs.last_hidden_state
            blk.append(lhidden_states[:, -1, :])
            lhidden_states = self.output_matching(lhidden_states)

            logits = self.lm_head(lhidden_states)
            crq_audio_tokens, logits = self.sampling_step(logits)
            self.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
            if i == 0:
                all_logits.append(logits)
            else:
                all_logits.append(logits[:, -1, :].unsqueeze(1))

            self.crq_audio_embeds = self.get_embeddings(crq_audio_tokens)

        self.crq_hidden_block = torch.stack(blk, dim=1)  # (B,G,C)
        self.crq_generate_tokens = torch.cat(self.crq_generate_tokens, dim=1)
        logits = torch.cat(all_logits, dim=1)
        return FunAudioChatMotionCFMDecoderOutput(
            loss=None, logits=logits, speech_loss=None, speech_logits=logits, crq_hidden=self.crq_hidden_block
        )

    def forward(
        self,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inputs_embeds = self.pre_matching(inputs_embeds)
        bs, slen, _ = inputs_embeds.shape

        hidden_states = inputs_embeds.reshape(bs, slen * self.group_size, -1)

        if audio_embeds is not None:
            audio_embeds = audio_embeds.view(bs, slen * self.group_size, -1)
            audio_embeds = torch.roll(audio_embeds, shifts=-(self.group_size - 1), dims=1)
            my_inputs_embeds = hidden_states + audio_embeds
        else:
            my_inputs_embeds = hidden_states

        my_inputs_embeds = self.input_matching(my_inputs_embeds)

        crq_attention_mask = None
        crq_position_ids = None
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask should be 2D"
            crq_attention_mask = attention_mask.repeat_interleave(self.group_size, dim=1)

        if position_ids is not None:
            # [bs, slen] -> [bs, slen * group_size] with sub-position offsets p*G + (0..G-1)
            base_positions = position_ids * self.group_size
            crq_position_ids = base_positions.unsqueeze(-1).repeat(1, 1, self.group_size)
            offsets = torch.arange(self.group_size, device=position_ids.device, dtype=position_ids.dtype)
            crq_position_ids = crq_position_ids + offsets.view(1, 1, -1)
            crq_position_ids = crq_position_ids.view(bs, -1)

            # padding from position_ids (rightmost non-zero), not attention_mask (all 1s without neat_packing)
            nonzero_mask = position_ids != 0
            reversed_nonzero = torch.flip(nonzero_mask, dims=[-1])
            first_nonzero_from_end = reversed_nonzero.to(torch.long).argmax(dim=-1)
            has_nonzero = nonzero_mask.any(dim=-1)
            orig_slen = position_ids.shape[1]
            ending_pos = torch.where(has_nonzero, orig_slen - first_nonzero_from_end, torch.zeros_like(first_nonzero_from_end))
            seq_indices = torch.arange(orig_slen, device=position_ids.device).unsqueeze(0)
            padding_mask = seq_indices >= ending_pos.unsqueeze(-1)
            crq_padding_mask = padding_mask.repeat_interleave(self.group_size, dim=1)
            crq_position_ids = crq_position_ids.masked_fill(crq_padding_mask, 0)

        outputs = self.crq_transformer(
            inputs_embeds=my_inputs_embeds,
            attention_mask=crq_attention_mask,
            position_ids=crq_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        crq_hidden = outputs.last_hidden_state  # (B, slen*G, C), the CFM condition
        hidden_states = self.output_matching(crq_hidden)

        logits = self.lm_head(hidden_states)

        loss = None
        shift_labels = None
        if labels is not None:
            labels = nn.functional.pad(labels, (0, self.group_size), value=-100)
            shift_labels = labels[..., self.group_size :].contiguous()
            loss = self.loss_function(logits, labels, vocab_size=self.config.codebook_size, shift_labels=shift_labels)

        encoder_hidden_states = (hidden_states,) if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [loss, logits, encoder_hidden_states] if v is not None)
        return FunAudioChatMotionCFMDecoderOutput(
            loss=loss,
            logits=logits,
            speech_loss=loss,
            speech_logits=logits,
            speech_shifted_labels=shift_labels,
            hidden_states=encoder_hidden_states,
            crq_hidden=crq_hidden,
        )


class FunAudioChatMotionCFMForConditionalGeneration(FunAudioChatForConditionalGeneration):
    def __init__(self, config: FunAudioChatConfig):
        _enable = getattr(config.audio_config, "enable_audio_invert_tower", True)
        config.audio_config.enable_audio_invert_tower = False
        try:
            super().__init__(config)
        finally:
            config.audio_config.enable_audio_invert_tower = _enable
        self.audio_invert_tower = FunAudioChatMotionCFMDecoder(config.audio_config)

        # S2S mode: skip motion sampling
        self.sp_gen_kwargs['disable_motion'] = False
        # repeat the last real speech frame into the trailing partial group (as at generation)
        self._speech_pad_repeat = _envb("FUNAUDIOCHAT_SPEECH_PAD_REPEAT", True)
        self._cfm_assert_align = _envb("FUNAUDIOCHAT_CFM_ASSERT_ALIGN", False)
        # sampler (standing setting: 10 steps / temperature 0.7 / CFG 2.0)
        self.cfm_steps = _envi("FUNAUDIOCHAT_CFM_STEPS", 10)
        self.cfm_temperature = _envf("FUNAUDIOCHAT_CFM_TEMPERATURE", 0.7)
        self.cfm_cfg = _envf("FUNAUDIOCHAT_CFM_CFG", 2.0)

        self._load_mcoeff_stats()
        self.tie_weights()

    def _load_mcoeff_stats(self):
        # training from the base model needs FUNAUDIOCHAT_MCOEFF_STATS; a JAVIS checkpoint carries the stats in its buffers
        path = os.environ.get("FUNAUDIOCHAT_MCOEFF_STATS", "").strip()
        head = self.audio_invert_tower.motion_cfm_head
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"FUNAUDIOCHAT_MCOEFF_STATS={path} does not exist")
        st = torch.load(path, map_location="cpu")
        mean, std = st["mean"].float(), st["std"].float().clamp_min(1e-3)
        head.mc_mean.data.copy_(mean)
        head.mc_std.data.copy_(std)
        logger.info(f"[motion_cfm] loaded mcoeff stats from {path} (std min={std.min():.4g})")

    def _speech_feedback_ids(self, speech_labels):
        ig, pad = self.config.ignore_index, self.audio_pad_token_id
        if not self._speech_pad_repeat:
            return speech_labels.masked_fill(speech_labels == ig, pad)
        G = self.audio_tower.group_size
        eos = self.config.audio_config.eos_token_id
        B = speech_labels.shape[0]
        g = speech_labels.view(B, -1, G)
        is_real = (g != ig) & (g != eos)
        has_real = is_real.any(dim=-1, keepdim=True)
        idx = torch.arange(G, device=g.device)
        last_idx = torch.where(is_real, idx, torch.zeros_like(idx)).amax(dim=-1, keepdim=True)
        last_val = g.gather(-1, last_idx)
        fill = has_real & (g == ig)
        g = torch.where(fill, last_val.expand_as(g), g)
        ids = g.reshape(B, -1)
        return ids.masked_fill(ids == ig, pad)

    def _build_mcoeff_entry_grids(self, mcoeff, mcoeff_lengths, speech_ids, speech_attention_mask, n_frames):
        """Place each utterance's (T,Dm) mcoeff on its own speech frame grid: the k-th content
        speech token holds mcoeff[k]. Returns normalized (mgrid, mvalid), (num_audios, n_frames[, Dm])."""
        NA = mcoeff.shape[0]
        dev = mcoeff.device
        bos_id = self.config.audio_config.bos_token_id

        content = (speech_attention_mask == 1) & (speech_ids < bos_id)  # (NA, Ts)
        if content.shape[1] < n_frames:
            content = nn.functional.pad(content, (0, n_frames - content.shape[1]), value=False)
        elif content.shape[1] > n_frames:
            content = content[:, :n_frames]

        mgrid = torch.zeros(NA, n_frames, MCOEFF_DIM, device=dev, dtype=mcoeff.dtype)
        mvalid = torch.zeros(NA, n_frames, device=dev, dtype=mcoeff.dtype)

        lens = mcoeff_lengths.detach().cpu().tolist()
        n_slot = content.sum(dim=1).detach().cpu().tolist()
        for a in range(NA):
            n = int(min(lens[a], n_slot[a]))
            if n <= 0:
                continue
            pos = content[a].nonzero(as_tuple=True)[0][:n]
            mgrid[a, pos] = mcoeff[a, :n].to(mgrid.dtype)
            mvalid[a, pos] = 1.0

        mgrid = self.audio_invert_tower.normalize_mcoeff(mgrid) * mvalid.unsqueeze(-1)
        return mgrid, mvalid

    @staticmethod
    def _scatter_groups(src_e, audio_features_mask, special_audio_mask, B, slen, G, trail_shape):
        """(num_audios, n_groups*G, *trail) -> (B, slen, G, *trail) over the LM audio slots."""
        src = src_e.reshape(src_e.shape[0], -1, G, *trail_shape)
        sel = audio_features_mask
        for _ in range(2 + len(trail_shape) - 1):
            sel = sel.unsqueeze(-1)
        src = src[sel.expand_as(src)]
        dest = torch.zeros(B, slen, G, *trail_shape, device=src_e.device, dtype=src_e.dtype)
        mask = special_audio_mask
        while mask.dim() < dest.dim():
            mask = mask.unsqueeze(-1)
        return dest.masked_scatter(mask.expand_as(dest), src)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        speech_ids: Optional[torch.LongTensor] = None,
        text_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        feature_exist_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mcoeff: Optional[torch.FloatTensor] = None,
        mcoeff_lengths: Optional[torch.LongTensor] = None,
        mcoeff_ref: Optional[torch.FloatTensor] = None,
        mcoeff_exist_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FunAudioChatMotionCFMCausalLMOutputWithPast]:
        """Upstream forward plus, per assistant audio slot, a motion CFM loss on the mcoeff grid.
        mcoeff (num_audios, T, Dm) raw / mcoeff_lengths (num_audios,) / mcoeff_ref (num_audios, Dm)
        gcond frame / mcoeff_exist_mask (num_audios,) come from the training collator."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        speech_labels = None
        text_features = None
        special_audio_mask = None
        audio_features_mask = None
        mgrid_b = mvalid_b = ref_b = None

        if text_ids is not None and text_ids.numel() == 0:
            text_ids = None

        target_device = self.audio_tower.device
        G = self.audio_tower.group_size

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # merge text and audio
            if speech_ids is not None and input_ids.shape[1] != 1:
                speech_ids = speech_ids.to(device=target_device, dtype=input_ids.dtype)
                speech_attention_mask = speech_attention_mask.to(target_device)
                mid_speech_labels = torch.where(speech_attention_mask == 1, speech_ids, self.config.ignore_index)
                # pad to a multiple of group_size
                speech_padding_target_length = (
                    (speech_ids.shape[-1] + self.audio_tower.group_size - 1) // self.audio_tower.group_size
                ) * self.audio_tower.group_size
                if speech_padding_target_length > speech_ids.shape[-1]:
                    padding_length = speech_padding_target_length - speech_ids.shape[-1]
                    speech_ids = torch.nn.functional.pad(
                        speech_ids, (0, padding_length), value=self.audio_pad_token_id
                    )
                    mid_speech_labels = torch.nn.functional.pad(
                        mid_speech_labels, (0, padding_length), value=self.config.ignore_index
                    )
                    # same width as the ids: the mcoeff grid builder ANDs the two
                    speech_attention_mask = torch.nn.functional.pad(
                        speech_attention_mask, (0, padding_length), value=0
                    )

                if self._speech_pad_repeat:
                    real_len = speech_attention_mask.sum(-1)
                    last_val = speech_ids.gather(1, (real_len - 1).clamp(min=0)[:, None]).squeeze(1)
                    grp_end = ((real_len + G - 1) // G) * G
                    cols = torch.arange(speech_ids.shape[1], device=speech_ids.device)[None, :]
                    fill = (cols >= real_len[:, None]) & (cols < grp_end[:, None])
                    speech_ids = torch.where(fill, last_val[:, None], speech_ids)

                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    speech_attention_mask.sum(-1)
                )

                continuous_audio_features = None
                continuous_audio_output_lengths = None
                if input_features is not None and hasattr(self, "continuous_audio_tower"):
                    continuous_audio_features, continuous_audio_output_lengths = self.get_audio_features(
                        input_features.to(target_device),
                        feature_attention_mask=feature_attention_mask.to(target_device),
                        speech_maxlen=speech_ids.shape[-1],
                    )
                    continuous_audio_features = continuous_audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

                audio_features, *_ = self.audio_tower(
                    speech_ids,
                    continuous_audio_features=continuous_audio_features,
                    continuous_audio_output_lengths=continuous_audio_output_lengths,
                    feature_exist_mask=feature_exist_mask,
                )

                mid_text_labels = None
                if text_ids is not None:
                    # align text length with audio
                    max_audio_output_length = int(speech_padding_target_length // self.audio_tower.group_size)
                    if text_ids.shape[1] > max_audio_output_length:
                        text_ids = text_ids[:, :max_audio_output_length]
                    elif text_ids.shape[1] < max_audio_output_length:
                        text_ids = torch.nn.functional.pad(
                            text_ids,
                            (0, max_audio_output_length - text_ids.shape[1]),
                            value=self.config.text_config.pad_token_id or self.config.text_config.eos_token_id,
                        )
                    text_ids[text_ids == self.config.text_config.eos_token_id] = self.config.text_config.sil_index
                    if self.config.text_config.pad_token_id is not None:
                        text_ids[text_ids == self.config.text_config.pad_token_id] = self.config.text_config.sil_index
                    text_ids = text_ids.to(device=target_device, dtype=input_ids.dtype)
                    mid_text_labels = torch.where(
                        speech_attention_mask[:, :: self.config.audio_config.group_size] == 1,
                        text_ids,
                        self.config.ignore_index,
                    )

                    text_features = self.get_input_embeddings()(text_ids)
                    text_features = text_features.masked_fill(~text_attention_mask[:, None, None], 0.0).to(
                        dtype=text_features.dtype
                    )
                    mid_text_labels.masked_fill_(~text_attention_mask[:, None], self.config.ignore_index)
                    mid_text_labels = mid_text_labels.to(dtype=input_ids.dtype)

                    audio_features = (text_features + audio_features) / 2

                num_audios, max_audio_tokens, embed_dim = audio_features.shape
                audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                audio_features = audio_features[audio_features_mask]

                n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
                n_audio_features = audio_features.shape[0]
                if n_audio_tokens != n_audio_features:
                    logger.warning(
                        f"Audio feature/token mismatch (tokens={n_audio_tokens}, features={n_audio_features}); "
                        f"truncating/padding to align."
                    )
                    if n_audio_features > n_audio_tokens:
                        audio_features = audio_features[:n_audio_tokens]
                    else:
                        pad = torch.zeros(
                            n_audio_tokens - n_audio_features, audio_features.shape[-1],
                            dtype=audio_features.dtype, device=audio_features.device,
                        )
                        audio_features = torch.cat([audio_features, pad], dim=0)
                special_audio_mask = (input_ids == self.config.audio_token_index).to(inputs_embeds.device)
                special_audio_mask = special_audio_mask.unsqueeze(-1)
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_audio_mask.expand_as(inputs_embeds), audio_features
                )

                # motion grids need exactly one (entry, group) per LM audio slot; the mismatch fixup
                # above only patches audio_features, so skip motion for such a batch
                grids_ok = int(audio_features_mask.sum()) == int(special_audio_mask.sum())
                if not grids_ok:
                    logger.warning("[motion_cfm] audio slot/group count mismatch; skipping motion grids for this batch")
                if grids_ok and mcoeff is not None and mcoeff.numel() > 0:
                    B, slen = input_ids.shape
                    mcoeff = mcoeff.to(device=target_device, dtype=torch.float32)
                    mgrid_e, mvalid_e = self._build_mcoeff_entry_grids(
                        mcoeff, mcoeff_lengths.to(target_device), speech_ids.to(target_device),
                        speech_attention_mask.to(target_device), speech_padding_target_length,
                    )
                    mgrid_b = self._scatter_groups(mgrid_e, audio_features_mask, special_audio_mask, B, slen, G, (MCOEFF_DIM,))
                    mvalid_b = self._scatter_groups(
                        mvalid_e.unsqueeze(-1), audio_features_mask, special_audio_mask, B, slen, G, (1,)
                    ).squeeze(-1)
                    if mcoeff_ref is not None:
                        ref_e = mcoeff_ref.to(device=target_device, dtype=torch.float32)
                        ref_e = self.audio_invert_tower.normalize_mcoeff(ref_e)
                        if mcoeff_exist_mask is not None:
                            ref_e = ref_e * mcoeff_exist_mask.to(target_device, ref_e.dtype)[:, None]
                        ref_e = ref_e[:, None, None, :].expand(-1, mgrid_e.shape[1] // G, G, -1)
                        ref_b = self._scatter_groups(ref_e, audio_features_mask, special_audio_mask, B, slen, G, (MCOEFF_DIM,))[:, :, 0]

                if labels is not None:
                    # speech labels
                    speech_labels = torch.full(
                        (*labels.shape, self.audio_tower.group_size),
                        self.config.ignore_index,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    mid_speech_labels = mid_speech_labels.view(
                        mid_speech_labels.shape[0], -1, self.audio_tower.group_size
                    )
                    mid_speech_labels = mid_speech_labels[
                        audio_features_mask.unsqueeze(-1).expand_as(mid_speech_labels)
                    ]
                    speech_labels = speech_labels.masked_scatter(
                        special_audio_mask.expand_as(speech_labels), mid_speech_labels
                    )
                    non_labels_mask = labels == self.config.ignore_index
                    speech_labels = speech_labels.masked_fill(non_labels_mask.unsqueeze(-1), self.config.ignore_index)

                    audio_eos_mask = labels == self.config.text_config.audio_eos_index
                    speech_labels = speech_labels.masked_fill(
                        audio_eos_mask.unsqueeze(-1), self.config.audio_config.eos_token_id
                    )
                    speech_labels = speech_labels.reshape(speech_labels.shape[0], -1)

                    # assistant-only motion targets
                    if mgrid_b is not None:
                        mgrid_b = mgrid_b.masked_fill(non_labels_mask[:, :, None, None], 0.0)
                        mvalid_b = mvalid_b.masked_fill(non_labels_mask[:, :, None], 0.0)
                        if ref_b is not None:
                            ref_b = ref_b.masked_fill(non_labels_mask[:, :, None], 0.0)

                    # text labels
                    labels[labels == self.config.audio_token_index] = self.config.ignore_index
                    if mid_text_labels is not None:
                        mid_text_labels = mid_text_labels[audio_features_mask]
                        labels = labels.masked_scatter(special_audio_mask.squeeze(-1), mid_text_labels)
                        labels.masked_fill_(non_labels_mask, self.config.ignore_index)

        # the LM is frozen in both training stages; no graph is kept for it then
        llm_frozen = not any(p.requires_grad for p in self.language_model.parameters())
        with torch.no_grad() if llm_frozen else contextlib.nullcontext():
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
        aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else None
        speech_loss = None
        speech_logits = None
        speech_shifted_labels = None
        motion_cfm_loss = None
        if self.audio_invert_tower is not None:
            last_hidden_state = outputs.hidden_states[-1]

            if speech_labels is not None:
                audio_embeds = self.audio_tower(
                    self._speech_feedback_ids(speech_labels),
                    return_dict=True,
                ).hidden_states[0]
            else:
                audio_embeds = None

            if not self.sp_gen_kwargs['disable_speech']:
                speech_inputs_embeds = last_hidden_state
                if text_embeds is None:
                    text_embeds = self.get_input_embeddings()(input_ids)
                    if text_features is not None:
                        text_embeds = text_embeds.masked_scatter(special_audio_mask.expand_as(text_embeds), text_features[audio_features_mask])
                speech_inputs_embeds = speech_inputs_embeds + text_embeds.detach()
                if self._cfm_assert_align and mgrid_b is not None:
                    self._assert_alignment(mgrid_b, mvalid_b, speech_labels)
                speech_output = self.audio_invert_tower(
                    audio_embeds=audio_embeds,
                    inputs_embeds=speech_inputs_embeds,
                    labels=speech_labels if labels is not None else None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
                speech_loss = speech_output.loss
                speech_logits = speech_output.logits
                speech_shifted_labels = speech_output.speech_shifted_labels

                # training/eval only (generation runs its own per-group sampler)
                if labels is not None:
                    motion_cfm_loss = self._motion_cfm_from_hidden(
                        speech_output.crq_hidden, mgrid_b, mvalid_b, ref_b, special_audio_mask
                    )

        loss = None
        text_loss = None
        if labels is not None:
            text_loss = outputs.loss
            if not llm_frozen:
                loss = text_loss
            if speech_loss is not None:
                loss = speech_loss if loss is None else loss + speech_loss
            if motion_cfm_loss is not None:
                loss = motion_cfm_loss if loss is None else loss + motion_cfm_loss

        if not return_dict:
            output = (outputs.logits, speech_logits) + outputs[1:]
            return (loss, text_loss, speech_loss, motion_cfm_loss) + output if loss is not None else output

        return FunAudioChatMotionCFMCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            text_loss=text_loss,
            speech_loss=speech_loss,
            motion_cfm_loss=motion_cfm_loss,
            text_logits=outputs.logits,
            speech_logits=speech_logits,
            speech_shifted_labels=speech_shifted_labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def _motion_cfm_from_hidden(self, crq_hidden, mgrid_b, mvalid_b, ref_b, special_audio_mask):
        """One CFM item per LM audio slot q that holds real mcoeff and whose condition window
        (q+la-n_cond+1 .. q+la) and prev slot q-1 are audio slots of the same span."""
        ait = self.audio_invert_tower
        if mgrid_b is None or crq_hidden is None:
            return self._motion_cfm_dummy(crq_hidden)

        B, slen = mvalid_b.shape[0], mvalid_b.shape[1]
        C = crq_hidden.shape[-1]
        G = self.audio_tower.group_size
        H = crq_hidden.view(B, slen, G, C)
        n_cond = ait.motion_cfm_head.n_cond_groups
        la = ait.motion_cfm_head.n_lookahead_groups
        blk_off = [la - n_cond + i for i in range(n_cond)]

        is_aud = special_audio_mask.squeeze(-1)
        ok = (mvalid_b.sum(-1) > 0) & is_aud
        for d in sorted(set(blk_off) | {-1}):  # condition blocks + prev group
            if d == 0:
                continue
            sh = torch.zeros_like(is_aud)
            if d < 0:
                sh[:, -d:] = is_aud[:, :d]
            else:
                sh[:, :-d] = is_aud[:, d:]
            ok = ok & sh
        if not bool(ok.any()):
            return self._motion_cfm_dummy(crq_hidden)

        b_idx, q_idx = ok.nonzero(as_tuple=True)
        n_avail = b_idx.numel()
        cap = ait.cfm_max_items
        if cap and n_avail > cap:
            keep = torch.randperm(n_avail, device=b_idx.device)[:cap]
            b_idx, q_idx = b_idx[keep], q_idx[keep]
        if not ait._cfm_logged:
            logger.warning(
                f"[motion_cfm] items/micro-batch: {b_idx.numel()} used of {n_avail} available "
                f"(cap={cap}, k_repeat={ait.cfm_k_repeat}, grad_ckpt={ait.motion_cfm_head.grad_checkpointing}) "
                f"drops: cond={ait.cfm_cond_drop} prev={ait.cfm_prev_drop} gcond={ait.cfm_gcond_drop}"
            )
            ait._cfm_logged = True
        h_cond = torch.cat([H[b_idx, q_idx + o] for o in blk_off], dim=1)  # (N, n_cond*G, C)
        x1 = mgrid_b[b_idx, q_idx].to(h_cond.dtype)                        # (N, G, Dm)
        prev = mgrid_b[b_idx, q_idx - 1].to(h_cond.dtype)                  # zeros at the first group
        tgt_mask = mvalid_b[b_idx, q_idx]
        gcond = (ref_b[b_idx, q_idx] if ref_b is not None else torch.zeros_like(x1[:, 0])).to(h_cond.dtype)
        return ait.motion_cfm_loss(x1, h_cond, prev, gcond, tgt_mask)

    def _assert_alignment(self, mgrid_b, mvalid_b, speech_labels):
        """Every slot carrying mcoeff also carries real speech labels (FUNAUDIOCHAT_CFM_ASSERT_ALIGN=1)."""
        G = self.audio_tower.group_size
        has_mc = mvalid_b.sum(-1) > 0
        if has_mc.any():
            b, q = has_mc.nonzero(as_tuple=True)
            sl = speech_labels.view(speech_labels.shape[0], -1, G)[b, q]
            n_bad = int((sl == self.config.ignore_index).all(-1).sum())
            assert n_bad == 0, f"[motion_cfm] {n_bad} slots carry mcoeff but no speech labels"

    def _motion_cfm_dummy(self, crq_hidden):
        """Keep the head in the DDP graph on motion-less batches (x0 contribution)."""
        ait = self.audio_invert_tower
        head = ait.motion_cfm_head
        dev = crq_hidden.device if crq_hidden is not None else self.audio_tower.device
        dt = crq_hidden.dtype if crq_hidden is not None else torch.float32
        G = head.group_size
        x1 = torch.zeros(1, G, MCOEFF_DIM, device=dev, dtype=dt)
        h = torch.zeros(1, head.n_hidden_tok, head.hidden_proj.in_features, device=dev, dtype=dt)
        if crq_hidden is not None:
            h = h + crq_hidden.reshape(-1)[0] * 0.0
        prev = torch.zeros(1, head.n_prev_tok, MCOEFF_DIM, device=dev, dtype=dt)
        g = torch.zeros(1, MCOEFF_DIM, device=dev, dtype=dt)
        mask = torch.ones(1, G, device=dev, dtype=dt)
        return ait.motion_cfm_loss(x1, h, prev, g, mask) * 0.0

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: "LogitsProcessorList",
        stopping_criteria: "StoppingCriteriaList",
        generation_config: "GenerationConfig",
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union["GenerateNonBeamOutput", torch.LongTensor]:
        """Upstream _sample + one motion group per SRH block. Returns (input_ids, speech_ids,
        motion_latents); motion_latents (B, T, Dm) are DE-normalized mcoeff, 1:1 with content speech frames.

        sp_gen_kwargs['forced_text_ids']: reply text ids replace the text stream while speech is on,
        then <|sil|> (training layout); speech and motion stay sampled.
        self.gen_mcoeff_ref: raw reference-portrait mcoeff (Dm,) used as gcond."""
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        text_greedy = self.sp_gen_kwargs.get('text_greedy', False)
        only_crq_sampling = self.sp_gen_kwargs.get('only_crq_sampling', False)
        force_text_abos = self.sp_gen_kwargs.get('force_text_abos', False)
        disable_motion = self.sp_gen_kwargs.get('disable_motion', False)
        forced_text = self.sp_gen_kwargs.get('forced_text_ids')
        _ft_i = 0

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        speech_ids = torch.empty(input_ids.size(0), 0, dtype=input_ids.dtype, device=input_ids.device)
        self.generate_speech = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        self.audio_invert_tower.forward = self.audio_invert_tower.crq_generate_forward
        self.audio_invert_tower.crq_past_key_values = None
        self.audio_invert_tower.crq_audio_embeds = None
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_speech_ids = speech_ids
        # NoBadWordsLogitsProcessor is for text only
        crq_logits_processor = LogitsProcessorList([p for p in logits_processor if not isinstance(p, NoBadWordsLogitsProcessor)])
        self.audio_invert_tower.crq_logits_processor = crq_logits_processor
        self.audio_invert_tower.crq_grobal_step = 0

        G = self.config.audio_config.group_size
        ait = self.audio_invert_tower
        ref = getattr(self, "gen_mcoeff_ref", None)
        if ref is None:
            gcond = torch.zeros(input_ids.size(0), MCOEFF_DIM, device=input_ids.device)
        else:
            gcond = ait.normalize_mcoeff(ref.to(input_ids.device, torch.float32).view(1, -1)).expand(input_ids.size(0), -1)
        h_blocks = []   # E[0..j]
        m_groups = []   # emitted motion groups, normalized (B,G,Dm)
        n_cond = ait.motion_cfm_head.n_cond_groups
        la = ait.motion_cfm_head.n_lookahead_groups
        j = 0           # index of the next SRH block E[j]

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            self.is_prefill = is_prefill
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            if not text_greedy:
                next_token_scores = logits_processor(input_ids, next_token_logits)
            else:
                next_token_scores = next_token_logits

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            if do_sample and not text_greedy:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if force_text_abos:
                next_tokens = torch.zeros_like(next_tokens) + self.config.text_config.audio_bos_index
                force_text_abos = False
            elif forced_text is not None and bool(self.generate_speech.any()):
                next_tokens = torch.full_like(next_tokens, forced_text[_ft_i] if _ft_i < len(forced_text) else self.config.text_config.sil_index)
                _ft_i += 1

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            if any(self.generate_speech) and not self.sp_gen_kwargs["disable_speech"]:
                if only_crq_sampling:
                    next_speech_tokens = self.audio_invert_tower.crq_generate_tokens
                else:
                    next_speech_token_logits = outputs.speech_logits.clone()[
                        :, -self.config.audio_config.group_size :, :
                    ].float()
                    next_speech_token_logits = next_speech_token_logits.to(input_ids.device)

                    next_speech_token_scores = logits_processor(
                        speech_ids.view(speech_ids.size(0) * self.config.audio_config.group_size, -1),
                        next_speech_token_logits.view(-1, next_speech_token_logits.size(-1)),
                    )
                    if do_sample:
                        speech_probs = nn.functional.softmax(next_speech_token_scores, dim=-1)
                        next_speech_tokens = torch.multinomial(speech_probs, num_samples=1)
                    else:
                        next_speech_tokens = torch.argmax(next_speech_token_scores, dim=-1)
                    next_speech_tokens = next_speech_tokens.view(-1, self.config.audio_config.group_size)

                # E[j] emits speech group j, and training's window for M(g) ends at E[g+la], so this
                # step emits M(j-la); q_emit = target group + 1
                if not disable_motion:
                    h_blocks.append(ait.crq_hidden_block)
                    q_emit = j - la + 1
                    if q_emit >= FIRST_MOTION_GROUP and len(h_blocks) >= n_cond:
                        h_cond = torch.cat(h_blocks[-n_cond:], dim=1)
                        prev = m_groups[-1] if m_groups else torch.zeros(input_ids.size(0), G, MCOEFF_DIM, device=input_ids.device)
                        m = ait.motion_cfm_sample(
                            h_cond, prev.to(h_cond.dtype), gcond.to(h_cond.dtype),
                            n_steps=self.cfm_steps, temperature=self.cfm_temperature, cfg_scale=self.cfm_cfg,
                        )
                        m_groups.append(m.float())
                j += 1

                finish_speech = (next_speech_tokens == self.config.audio_config.eos_token_id).any(dim=-1)

                combined_next_speech_tokens = torch.where(
                    finish_speech[:, None], self.config.audio_config.eos_token_id, next_speech_tokens
                )

                if streamer is not None:
                    streamer.put(combined_next_speech_tokens.cpu())

                speech_ids = torch.cat([speech_ids, combined_next_speech_tokens], dim=-1)
                self.audio_invert_tower.crq_speech_ids = speech_ids
                model_kwargs["speech_ids"] = speech_ids

                input_ids[:, -1].masked_fill_(finish_speech, self.config.text_config.audio_eos_index)

                self.generate_speech = torch.logical_and(self.generate_speech, ~finish_speech)
            else:
                model_kwargs["speech_ids"] = None

            # drop the reference so the first-iteration logits can be freed
            del outputs

        if streamer is not None:
            streamer.end()

        if m_groups:
            motion_latents = ait.denormalize_mcoeff(torch.cat(m_groups, dim=1).float())
        else:
            motion_latents = torch.zeros(input_ids.size(0), 0, MCOEFF_DIM, device=input_ids.device)
        return input_ids, speech_ids, motion_latents


__all__ = [
    "cfm_item_errors",
    "cfm_sample",
    "FunAudioChatMotionCFMForConditionalGeneration",
    "FunAudioChatMotionCFMDecoder",
    "FunAudioChatMotionCFMDecoderOutput",
    "FunAudioChatMotionCFMCausalLMOutputWithPast",
    "MotionCFMHead",
]
