# -----------------------------------------------------------
# core/decoder.py
# transformer decoder that generates LaTeX tokens step-by-step.
# includes rmsnorm (prenorm), rotary positional encoding (rope),
# swiglu feed-forward blocks, droppath regularization, and mha.
# each layer: self-attn → cross-attn → ff. outputs logits for vocab.
# -----------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return xr.flatten(-2)

def build_rotary_cache(seq_len, head_dim, device, base=10000.0):
    pos = torch.arange(seq_len, device=device).float()
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.einsum("i,j->ij", pos, inv)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        n = x.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return self.weight * x * n

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)
    def _merge(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, rope_cache=None, need_weights=False):
        q = self._split(self.q_proj(q))
        k = self._split(self.k_proj(k))
        v = self._split(self.v_proj(v))
        if rope_cache is not None:
            cos, sin = rope_cache
            Tq, Tk = q.size(2), k.size(2)
            q = apply_rotary(q, cos[:, :, :Tq, :], sin[:, :, :Tq, :])
            k = apply_rotary(k, cos[:, :, :Tk, :], sin[:, :, :Tk, :])
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :].to(scores.dtype)
            scores = scores.masked_fill(pad.bool(), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        out = self.o_proj(self._merge(out))
        if need_weights:
            return out, attn.mean(dim=1)
        return out, None

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        return x + (torch.rand(shape, device=x.device) < self.p).float() * (-x / keep)

class StrongDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop=0.1, droppath=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MHA(d_model, n_heads, dropout=drop)
        self.drop1 = nn.Dropout(drop)
        self.dp1 = DropPath(droppath)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MHA(d_model, n_heads, dropout=drop)
        self.drop2 = nn.Dropout(drop)
        self.dp2 = DropPath(droppath)
        self.norm3 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.drop3 = nn.Dropout(drop)
        self.dp3 = DropPath(droppath)

    def forward(self, x, mem, self_mask, mem_pad, rope_cache, need_xattn=False):
        y, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_mask, rope_cache=rope_cache)
        x = x + self.dp1(self.drop1(y))
        y, xattn = self.cross_attn(self.norm2(x), mem, mem, key_padding_mask=mem_pad, need_weights=need_xattn)
        x = x + self.dp2(self.drop2(y))
        y = self.ff(self.norm3(x))
        x = x + self.dp3(self.drop3(y))
        return x, xattn

class StrongTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=320, n_heads=4, d_ff=768, n_layers=6, drop=0.1, droppath=0.1, max_len=4096):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            StrongDecoderLayer(d_model, n_heads, d_ff, drop=drop, droppath=droppath * (i+1) / n_layers)
            for i in range(n_layers)
        ])
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def _causal_mask(self, T, device):
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, memory, tgt_ids, mem_pad=None, need_xattn=False):
        B, T = tgt_ids.size()
        x = self.emb(tgt_ids)
        mask = self._causal_mask(T, x.device)
        cos, sin = build_rotary_cache(T, x.size(-1), x.device)
        last_xattn = None
        for i, layer in enumerate(self.layers):
            need = (need_xattn and i == len(self.layers) - 1)
            x, xattn = layer(x, memory, mask, mem_pad, (cos, sin), need_xattn=need)
            if xattn is not None:
                last_xattn = xattn
        logits = self.proj(x)
        return logits, last_xattn
