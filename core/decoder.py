# -----------------------------------------------------------
# core/decoder.py
# unidirectional lstm decoder with bahdanau attention,
# using input-feeding (emb_t + prev context) and coverage
# (cumulative attention over encoder steps) to reduce skips
# and repeats when generating LaTeX tokens.
# -----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int, memory_size: int, attn_size: int, use_coverage: bool = True):
        super().__init__()
        self.use_coverage = use_coverage
        self.W_h = nn.Linear(hidden_size, attn_size, bias=False)
        self.W_m = nn.Linear(memory_size, attn_size, bias=False)
        if use_coverage:
            self.W_c = nn.Linear(1, attn_size, bias=False)
        else:
            self.W_c = None
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,                   # (B, H)
        memory: torch.Tensor,                   # (B, S, D)
        mem_pad: Optional[torch.Tensor] = None, # (B, S) True for PAD
        coverage: Optional[torch.Tensor] = None # (B, S) cumulative attn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          context: (B, D)
          attn_weights: (B, S)
        """
        B, S, _ = memory.shape

        h = self.W_h(hidden).unsqueeze(1)  # (B,1,A)
        m = self.W_m(memory)               # (B,S,A)

        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros(B, S, device=memory.device, dtype=memory.dtype)
            c_feat = self.W_c(coverage.unsqueeze(-1))   # (B,S,1)->(B,S,A)
            score = self.v(torch.tanh(h + m + c_feat)).squeeze(-1)  # (B,S)
        else:
            score = self.v(torch.tanh(h + m)).squeeze(-1)          # (B,S)

        if mem_pad is not None:
            score = score.masked_fill(mem_pad, float("-inf"))

        attn_weights = F.softmax(score, dim=-1)                    # (B,S)
        context = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)  # (B,D)
        return context, attn_weights


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 320,
        hidden_size: int = 320,
        attn_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, d_model)

        # input-feeding: [emb_t, ctx_{t-1}] -> LSTMCell
        self.lstm_cell = nn.LSTMCell(d_model + d_model, hidden_size)

        self.attn = BahdanauAttention(
            hidden_size=hidden_size,
            memory_size=d_model,
            attn_size=attn_size,
            use_coverage=True,
        )

        self.fc_ctx = nn.Linear(hidden_size + d_model, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.drop = nn.Dropout(dropout)

        self.init_h = nn.Linear(d_model, hidden_size)
        self.init_c = nn.Linear(d_model, hidden_size)

    def init_state(
        self,
        memory: torch.Tensor,
        mem_pad: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        memory: (B, S, D)
        mem_pad: (B, S) or None
        returns: h0, c0 of shape (B, H)
        """
        if mem_pad is not None:
            mask = ~mem_pad
            mask_f = mask.float()
            denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            m = (memory * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        else:
            m = memory.mean(dim=1)

        h0 = torch.tanh(self.init_h(m))
        c0 = torch.tanh(self.init_c(m))
        return h0, c0

    def forward(
        self,
        memory: torch.Tensor,                   # (B, S, D)
        tgt_ids: torch.Tensor,                  # (B, T)
        mem_pad: Optional[torch.Tensor] = None, # (B, S) True for PAD
        need_xattn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Teacher forcing: processes all target tokens in one pass.

        returns:
          logits: (B, T, V)
          attn_seq: (B, T, S) if need_xattn else None
        """
        B, S, D = memory.shape
        _, T = tgt_ids.shape
        device = memory.device

        emb = self.emb(tgt_ids)  # (B,T,E)

        h_t, c_t = self.init_state(memory, mem_pad)  # (B,H), (B,H)

        ctx_prev = torch.zeros(B, D, device=device, dtype=memory.dtype)  # (B,D)
        coverage = torch.zeros(B, S, device=device, dtype=memory.dtype)  # (B,S)

        logits = []
        attn_seq = [] if need_xattn else None

        for t in range(T):
            x_t = emb[:, t, :]                          # (B,E)
            lstm_in = torch.cat([x_t, ctx_prev], dim=-1)  # (B,E+D)
            h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))

            ctx_t, alpha_t = self.attn(h_t, memory, mem_pad=mem_pad, coverage=coverage)  # (B,D),(B,S)

            coverage = coverage + alpha_t

            dec_feat = torch.cat([h_t, ctx_t], dim=-1)  # (B,H+D)
            dec_feat = self.drop(torch.tanh(self.fc_ctx(dec_feat)))  # (B,D)

            logit_t = self.fc_out(dec_feat)  # (B,V)
            logits.append(logit_t.unsqueeze(1))

            if need_xattn:
                attn_seq.append(alpha_t.unsqueeze(1))

            ctx_prev = ctx_t

        logits = torch.cat(logits, dim=1)  # (B,T,V)

        if need_xattn and attn_seq:
            attn_seq = torch.cat(attn_seq, dim=1)  # (B,T,S)
        else:
            attn_seq = None

        return logits, attn_seq
