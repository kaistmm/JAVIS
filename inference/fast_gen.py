"""Generation-side speedups for the JAVIS streaming loop, installed onto a loaded HF model.

1. FastSRHMotion: the speech-refinement head with a static KV cache and one CUDA graph per sub-step. It
   - exports the pre-output_matching hidden per sub-step -> ait.crq_hidden_block (the motion decoder's condition)
   - keeps HF numerics: unfused q/k/v and gate/up aliasing HF's own parameters, Qwen3RMSNorm's op order
   - samples through HF's own crq_logits_processor with HF's token history, so repetition penalty matches
2. GraphedCFM: the whole Euler/CFG loop of cfm_sample captured as one CUDA graph. The initial noise is drawn outside
   the graph with the same call cfm_sample makes, so RNG consumption is unchanged.

Both fall back to the eager path on any configuration they do not reproduce.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srh_fast import FastSRH  # noqa: E402


def _rms_hf(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rope(x, cos, sin):
    d = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., d:], x[..., :d]), dim=-1) * sin


class FastSRHMotion(FastSRH):
    def __init__(self, ait, audio_config, max_srh_len):
        dev = next(ait.parameters()).device
        super().__init__(audio_config, max_srh_len, device=str(dev), dtype=torch.bfloat16)
        self.ait = ait
        sd = ait.state_dict()
        need = ["pre_matching.weight", "pre_matching.bias", "input_matching.weight",
                "output_matching.weight", "lm_head.weight", "crq_transformer.norm.weight"]
        miss = [k for k in need if k not in sd]
        if miss:
            raise KeyError(f"SRH weights missing from the live module: {miss}")
        g = lambda k: sd[k].to(device=dev, dtype=torch.bfloat16)
        W = self.W
        W.update(pre_w=g("pre_matching.weight"), pre_b=g("pre_matching.bias"),
                 in_w=g("input_matching.weight"), out_w=g("output_matching.weight"),
                 lm_w=g("lm_head.weight"), norm=g("crq_transformer.norm.weight"), layers=[])
        for i in range(self.n_layers):
            p = f"crq_transformer.layers.{i}."
            W["layers"].append(dict(
                ln1=g(p + "input_layernorm.weight"), ln2=g(p + "post_attention_layernorm.weight"),
                q=g(p + "self_attn.q_proj.weight"), k=g(p + "self_attn.k_proj.weight"),
                v=g(p + "self_attn.v_proj.weight"), o=g(p + "self_attn.o_proj.weight"),
                qn=g(p + "self_attn.q_norm.weight"), kn=g(p + "self_attn.k_norm.weight"),
                gt=g(p + "mlp.gate_proj.weight"), up=g(p + "mlp.up_proj.weight"),
                down=g(p + "mlp.down_proj.weight")))
        self._qs = self.n_heads * self.hd
        self._g_hid = None

    def _layers(self, x, cos, sin, write_idx, kv_len, causal_prefill=False, decode=False):
        B, S, _ = x.shape
        nrep, eps = self.n_heads // self.n_kv, self.eps
        for li, L in enumerate(self.W["layers"]):
            r = _rms_hf(x, L["ln1"], eps)
            q = _rms_hf(F.linear(r, L["q"]).view(B, S, self.n_heads, self.hd), L["qn"], eps).transpose(1, 2)
            k = _rms_hf(F.linear(r, L["k"]).view(B, S, self.n_kv, self.hd), L["kn"], eps).transpose(1, 2)
            v = F.linear(r, L["v"]).view(B, S, self.n_kv, self.hd).transpose(1, 2)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            self.kc[li].index_copy_(2, write_idx, k)
            self.vc[li].index_copy_(2, write_idx, v)
            kk, vv = self.kc[li][:, :, :kv_len], self.vc[li][:, :, :kv_len]
            if decode:
                qg = q.reshape(B, self.n_kv, nrep, self.hd)
                sc = torch.matmul(qg.float(), kk.float().transpose(-1, -2)) * self.scaling + self._mask
                a = torch.matmul(sc.softmax(-1).to(vv.dtype), vv).reshape(B, S, self._qs)
            else:
                if causal_prefill:
                    a = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=self.scaling, enable_gqa=True)
                else:
                    a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=self._mask.to(x.dtype),
                                                       scale=self.scaling, enable_gqa=True)
                a = a.transpose(1, 2).reshape(B, S, self._qs)
            x = x + F.linear(a, L["o"])
            r = _rms_hf(x, L["ln2"], eps)
            x = x + F.linear(F.silu(F.linear(r, L["gt"])) * F.linear(r, L["up"]), L["down"])
        return _rms_hf(x, self.W["norm"], eps)

    def _decode_body(self, kv_len: int):
        x = F.linear(self._g_h + self._g_ae, self.W["in_w"])
        self._mask = torch.where(self._ar[:kv_len] <= self.pos, 0.0, self._neg).view(1, 1, 1, -1)
        cos = self.cos_t.index_select(0, self.pos).view(1, 1, 1, self.hd)
        sin = self.sin_t.index_select(0, self.pos).view(1, 1, 1, self.hd)
        h = self._layers(x, cos, sin, self.pos, kv_len, decode=True)
        self._g_hid.copy_(h[:, -1])                                   # pre-output_matching -> CFM cond
        self._g_logits.copy_(F.linear(F.linear(h, self.W["out_w"]), self.W["lm_w"])[:, -1, :].float())
        self.pos += 1

    @torch.inference_mode()
    def capture(self, max_decode_steps: int = 0):
        """FastSRH.capture with ONE memory pool shared by every bucket's graph. Safe because graphs
        are never replayed concurrently and every output leaves through a static buffer; separate
        pools cost ~70 MB per bucket at 8k slots, too much next to the 8B on 24 GB."""
        if self._g_h is None:
            self._g_h = torch.zeros(1, 1, self.hs, device=self.device, dtype=self.dtype)
            self._g_ae = torch.zeros(1, 1, self.hs, device=self.device, dtype=self.dtype)
            self._g_logits = torch.zeros(1, self.codebook_size, device=self.device, dtype=torch.float32)
            self._g_hid = torch.zeros(1, self.d, device=self.device, dtype=self.dtype)
            self._pool = torch.cuda.graph_pool_handle()
        need_hi = self._hpos + max(max_decode_steps, 1) * self.gs
        lens = sorted({self._kv_len_for(n) for n in range(self._hpos + 1, min(need_hi, self.max_len) + 1, max(1, self.bucket // 2))}
                      | {self._kv_len_for(self._hpos + 1), self._kv_len_for(min(need_hi, self.max_len))})
        saved = int(self.pos.item())
        for kvl in lens:
            if kvl in self._graphs:
                continue
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._decode_body(kvl)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.pos.fill_(saved)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._decode_body(kvl)
            self._graphs[kvl] = g
            self.pos.fill_(saved)

    def _causal_mask(self, p0, S, kvl):
        # bottom-right aligned: query p0+i sees keys <= p0+i (SDPA is_causal is top-left when q_len != kv_len)
        q = torch.arange(p0, p0 + S, device=self.device)[:, None]
        return torch.where(self._ar[:kvl][None, :] <= q, 0.0, self._neg).view(1, 1, S, kvl)

    @torch.inference_mode()
    def prefix(self, h_lm):
        """Cache-only pass over history LM positions: every sub-slot gets the speech-BOS embed, exactly
        what sub-step 0 of a full prefill gives them. No sampling, no outputs."""
        hid = F.linear(h_lm, self.W["pre_w"], self.W["pre_b"]).reshape(1, h_lm.shape[1] * self.gs, self.hs)
        ae = self.W["lm_w"][self.bos_token_id].view(1, 1, self.hs)
        S, p0 = hid.shape[1], int(self.pos.item())
        if p0 + S > self.max_len:
            raise RuntimeError(f"SRH cache exhausted by prefix ({p0 + S}/{self.max_len}); raise --srh_max_len")
        x = F.linear(hid + ae, self.W["in_w"])
        cos = self.cos_t[p0:p0 + S].view(1, 1, S, self.hd)
        sin = self.sin_t[p0:p0 + S].view(1, 1, S, self.hd)
        widx = torch.arange(p0, p0 + S, device=self.device)
        if p0 == 0:
            self._layers(x, cos, sin, widx, S, causal_prefill=True)
        else:
            self._mask = self._causal_mask(p0, S, p0 + S)
            self._layers(x, cos, sin, widx, p0 + S)
        self.pos += S
        self._hpos += S

    # sampler(logits [1,V] fp32, toks_so_far list of [1] long) -> [1] long
    @torch.inference_mode()
    def prefill(self, h_lm, sampler):
        gs = self.gs
        hid = F.linear(h_lm, self.W["pre_w"], self.W["pre_b"]).reshape(1, h_lm.shape[1] * gs, self.hs)
        slen5 = hid.shape[1]
        ae = self.W["lm_w"][self.bos_token_id].view(1, 1, self.hs)
        toks, blk = [], []
        for i in range(gs):
            x_in = hid[:, : slen5 - (gs - 1)] + ae if i == 0 else (hid[:, slen5 - (gs - i)] + ae.squeeze(1)).unsqueeze(1)
            S = x_in.shape[1]
            p0 = int(self.pos.item())
            widx = torch.arange(p0, p0 + S, device=self.device)
            x = F.linear(x_in, self.W["in_w"])
            cos = self.cos_t[p0:p0 + S].view(1, 1, S, self.hd)
            sin = self.sin_t[p0:p0 + S].view(1, 1, S, self.hd)
            if i == 0 and p0 == 0:
                out = self._layers(x, cos, sin, widx, p0 + S, causal_prefill=True)
            elif i == 0:                                   # after a cached prefix
                self._mask = self._causal_mask(p0, S, p0 + S)
                out = self._layers(x, cos, sin, widx, p0 + S)
            else:
                kvl = self._kv_len_for(p0 + 1)
                self._mask = torch.where(self._ar[:kvl] <= self.pos, 0.0, self._neg).view(1, 1, 1, -1)
                out = self._layers(x, cos, sin, widx, kvl, decode=True)
            self.pos += S
            self._hpos += S
            blk.append(out[:, -1])
            logits = F.linear(F.linear(out, self.W["out_w"]), self.W["lm_w"])[:, -1, :].float()
            tok = sampler(logits, toks)
            toks.append(tok)
            ae = self.W["lm_w"][tok].view(1, 1, self.hs)
        self._last_ae = ae
        return torch.stack(toks, dim=1), torch.stack(blk, dim=1)

    @torch.inference_mode()
    def step(self, h_lm1, sampler):
        gs = self.gs
        if self._hpos + gs > self.max_len:
            raise RuntimeError(f"SRH cache exhausted at {self._hpos}/{self.max_len}; raise --srh_max_len")
        hid = F.linear(h_lm1, self.W["pre_w"], self.W["pre_b"]).reshape(1, gs, self.hs)
        ae = self._last_ae
        toks, blk = [], []
        for i in range(gs):
            self._g_h.copy_(hid[:, i:i + 1])
            self._g_ae.copy_(ae)
            kvl = self._kv_len_for(self._hpos + 1)
            g = self._graphs.get(kvl)
            if g is None:
                self.capture()
                g = self._graphs[kvl]
            g.replay()
            self._hpos += 1
            blk.append(self._g_hid.clone())        # clone: the static buffer is overwritten next sub-step
            tok = sampler(self._g_logits, toks)
            toks.append(tok)
            ae = self.W["lm_w"][tok].view(1, 1, self.hs)
        self._last_ae = ae
        return torch.stack(toks, dim=1), torch.stack(blk, dim=1)


def hf_sampler(ait):
    """FunAudioChatDecoder.sampling_step minus the logits copy: same processor, same history."""
    def f(logits, toks):
        hist = torch.cat([ait.crq_speech_ids, *[t.view(1, 1) for t in toks]], dim=-1)
        sc = ait.crq_logits_processor(hist, logits)
        if ait.crq_do_sample:
            return torch.multinomial(torch.softmax(sc, dim=-1), num_samples=1).squeeze(1)
        return torch.argmax(sc, dim=-1)
    return f


def install_fast_srh(model, max_srh_len, max_decode_steps):
    """Replace ait.crq_generate_forward. generate() rebinds ait.forward to it at each call
    (FunAudioChatMotionCFMForConditionalGeneration._sample), so the swap must be on this attribute. Returns reset()."""
    base = model.base_model.model if hasattr(model, "peft_config") else model
    ait = base.audio_invert_tower
    if getattr(ait, "motion_cfm_head", None) is not None:
        from funaudiochat.modeling_motion_cfm import FunAudioChatMotionCFMDecoderOutput as Out
    else:                                               # stock 8B: forward reads .loss/.logits only
        from transformers.modeling_outputs import CausalLMOutput
        Out = lambda crq_hidden=None, speech_logits=None, **kw: CausalLMOutput(**kw)
    fast = FastSRHMotion(ait, base.config.audio_config, max_srh_len)
    samp = hf_sampler(ait)
    st = {"prefilled": False}
    dummy = torch.zeros(1, fast.gs, fast.codebook_size, device=fast.device, dtype=torch.float32)

    def fast_forward(inputs_embeds=None, **kw):
        if inputs_embeds.shape[0] != 1:
            raise RuntimeError("FastSRH is batch 1")
        if not st["prefilled"]:
            toks, blk = fast.prefill(inputs_embeds, samp)
            fast.capture(max_decode_steps=max_decode_steps)
            st["prefilled"] = True
        else:
            toks, blk = fast.step(inputs_embeds, samp)
        ait.crq_generate_tokens = toks
        ait.crq_hidden_block = blk
        ait.crq_audio_embeds = fast.W["lm_w"][toks[:, -1]]
        # only_crq_sampling=True: the outer loop reads crq_generate_tokens, never these logits
        return Out(loss=None, logits=dummy, speech_logits=dummy, crq_hidden=blk)

    def reset():
        fast.reset()
        st["prefilled"] = False

    ait.crq_generate_forward = fast_forward
    return fast, reset


def _head_fwd(head, pos, x, t, h_cond, prev, gcond):
    """MotionCFMHead.forward without grad checkpointing; pos = head._pos_ids moved to device beforehand."""
    tok_t = head.time_mlp(head.time_emb(t).to(x.dtype)).unsqueeze(1)
    tok_g = head.global_proj(gcond).unsqueeze(1)
    seq = torch.cat([tok_t, tok_g, head.hidden_proj(h_cond), head.prev_proj(prev), head.x_proj(x)], dim=1)
    seq = seq + head.pos_emb(pos).unsqueeze(0).to(seq.dtype)
    for blk in head.blocks:
        seq = blk(seq)
    return head.out_proj(head.norm(seq[:, -x.shape[1]:]))


class GraphedCFM:
    """Drop-in for ait.motion_cfm_sample at a fixed (n_steps, temperature, cfg, sway, zero*)."""

    def __init__(self, ait, n_steps, temperature, cfg_scale, sway=1.0, cfg_zero_star=True):
        from funaudiochat.modeling_motion_cfm import MCOEFF_DIM, cfm_sample
        self.ait, self.head, self.eager = ait, ait.motion_cfm_head, cfm_sample
        self.cfg = (int(n_steps), float(temperature), float(cfg_scale), float(sway), bool(cfg_zero_star))
        self.Dm = MCOEFF_DIM
        self.graph = None
        self.n_graph = self.n_eager = 0

    def _body(self):
        n, _, cfg, sway, zstar = self.cfg
        head, N = self.head, 1
        x, h, prev, g = self.s_x, self.s_h, self.s_prev, self.s_g
        ts = torch.linspace(1.0, 0.0, n + 1, device=x.device, dtype=torch.float32)
        if sway:
            ts = ts + sway * (torch.cos(math.pi / 2 * ts) - 1 + ts)
        zero_init = max(1, int(n * 0.04)) if zstar else 0
        h2 = torch.cat([h, torch.zeros_like(h)], 0)
        p2, g2 = torch.cat([prev] * 2, 0), torch.cat([g] * 2, 0)
        for step in range(n):
            if step < zero_init:
                continue
            t_cur, dt = ts[step], ts[step] - ts[step + 1]
            tb = t_cur.expand(N)
            out = _head_fwd(head, self.pos, torch.cat([x] * 2, 0), torch.cat([tb] * 2, 0), h2, p2, g2)
            cond, uncond = out.split(N, dim=0)
            if zstar:
                pf, nf = cond.reshape(N, -1).float(), uncond.reshape(N, -1).float()
                s = ((pf * nf).sum(1, keepdim=True) / (nf.pow(2).sum(1, keepdim=True) + 1e-8)).to(cond.dtype).view(N, 1, 1)
            else:
                s = 1.0
            v = uncond * s + cfg * (cond - uncond * s)
            x = x - dt.to(x.dtype) * v
        self.s_out.copy_(x)

    @torch.inference_mode()
    def _capture(self, h_cond, prev, gcond):
        dev, dt = h_cond.device, h_cond.dtype
        self.pos = self.head._pos_ids.to(dev)
        self.s_x = torch.zeros(1, self.head.group_size, self.Dm, device=dev, dtype=dt)
        self.s_h, self.s_prev, self.s_g = torch.zeros_like(h_cond), torch.zeros_like(prev), torch.zeros_like(gcond)
        self.s_out = torch.zeros_like(self.s_x)
        self.shape = (tuple(h_cond.shape), tuple(prev.shape), tuple(gcond.shape), dt)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    @torch.inference_mode()
    def __call__(self, h_cond, prev, gcond, n_steps=10, temperature=1.0, cfg_scale=2.0, sway=1.0,
                 cfg_zero_star=True, initial_noise=None):
        ok = (h_cond.shape[0] == 1 and initial_noise is None
              and (int(n_steps), float(temperature), float(cfg_scale), float(sway), bool(cfg_zero_star)) == self.cfg
              and self.cfg[2] != 1.0 and not os.environ.get("FUNAUDIOCHAT_CFM_SAMPLE_SEED"))
        if ok and self.graph is not None and self.shape != (tuple(h_cond.shape), tuple(prev.shape), tuple(gcond.shape), h_cond.dtype):
            ok = False
        if not ok:
            self.n_eager += 1
            return self.eager(self.head, h_cond, prev, gcond, n_steps=n_steps, temperature=temperature,
                              cfg_scale=cfg_scale, sway=sway, cfg_zero_star=cfg_zero_star, initial_noise=initial_noise)
        if self.graph is None:
            self._capture(h_cond, prev, gcond)
        # same draw as cfm_sample's unseeded branch -> same RNG consumption
        x0 = torch.randn(1, self.head.group_size, self.Dm, device=h_cond.device, dtype=h_cond.dtype) * temperature
        self.s_x.copy_(x0); self.s_h.copy_(h_cond); self.s_prev.copy_(prev); self.s_g.copy_(gcond)
        self.graph.replay()
        self.n_graph += 1
        return self.s_out.clone()

    @torch.inference_mode()
    def replay_with_noise(self, h_cond, prev, gcond, x0):
        """Verification only: graph output for a given initial noise."""
        if self.graph is None:
            self._capture(h_cond, prev, gcond)
        self.s_x.copy_(x0); self.s_h.copy_(h_cond); self.s_prev.copy_(prev); self.s_g.copy_(gcond)
        self.graph.replay()
        return self.s_out.clone()


def install_graphed_cfm(model):
    base = model.base_model.model if hasattr(model, "peft_config") else model
    ait = base.audio_invert_tower
    gc = GraphedCFM(ait, base.cfm_steps, base.cfm_temperature, base.cfm_cfg)
    ait.motion_cfm_sample = gc
    return gc
