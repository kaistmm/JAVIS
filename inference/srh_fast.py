"""Static-KV-cache state for the CUDA-graph speech-refinement head (FastSRHMotion in fast_gen.py does the compute).

The HF SRH costs ~121 ms per 5 Hz LM step on a 4090 and is ~97% CPU launch overhead (28 Qwen3 layers x 5 sequential
sub-steps, batch 1, seq 1); a static cache plus one captured graph per sub-step removes it.
"""
import torch


class FastSRH:
    """Static-cache, CUDA-graphed SRH. Batch size 1 (single dialogue stream)."""

    def __init__(self, audio_config, max_srh_len: int, device="cuda", dtype=torch.bfloat16):
        ac = audio_config
        _c = ac.crq_transformer_config
        crq = dict(_c) if isinstance(_c, dict) else _c.to_dict()
        self.device, self.dtype = torch.device(device), dtype
        self.gs = int(ac.group_size)
        self.hs = int(ac.output_dim)
        self.codebook_size = int(ac.codebook_size)
        self.bos_token_id = int(ac.bos_token_id)
        self.eos_token_id = int(ac.eos_token_id)

        self.n_layers = int(crq["num_hidden_layers"])
        self.d = int(crq["hidden_size"])
        self.n_heads = int(crq["num_attention_heads"])
        self.n_kv = int(crq["num_key_value_heads"])
        self.hd = int(crq.get("head_dim", self.d // self.n_heads))
        self.inter = int(crq["intermediate_size"])
        self.eps = float(crq["rms_norm_eps"])
        self.theta = float(crq["rope_theta"])
        self.scaling = self.hd ** -0.5
        self.max_len = int(max_srh_len)

        # rope tables
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32, device=device) / self.hd))
        pos = torch.arange(self.max_len, dtype=torch.float32, device=device)
        fr = torch.outer(pos, inv)
        emb = torch.cat((fr, fr), dim=-1)
        self.cos_t = emb.cos().to(dtype)          # [max_len, hd]
        self.sin_t = emb.sin().to(dtype)

        # kv cache [L, B=1, n_kv, max_len, hd]
        self.kc = torch.zeros(self.n_layers, 1, self.n_kv, self.max_len, self.hd, device=device, dtype=dtype)
        self.vc = torch.zeros_like(self.kc)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)   # next write slot
        self._ar = torch.arange(self.max_len, device=device)
        self._neg = -1e30

        self.W = {}          # weight dict, filled by the subclass
        self.bucket = 256    # attention is O(kv_len) per sub-step, so graph per length bucket
        self._graphs = {}    # kv_len -> CUDAGraph
        self._hpos = 0       # host mirror of self.pos
        self._g_h = None     # static [1,1,hs] LM-side input
        self._g_ae = None    # static [1,1,hs] previous speech-token embed
        self._g_logits = None

    def reset(self):
        self.pos.zero_()
        self._hpos = 0

    def _kv_len_for(self, need: int) -> int:
        return min(self.max_len, ((need + self.bucket - 1) // self.bucket) * self.bucket)
