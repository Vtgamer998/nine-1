"""
NINE-1: Transformer decoder-only minimal, implementado do zero usando PyTorch.
Arquitetura:
- Embeddings de tokens + positional encodings (learned)
- N blocos: pre-norm Transformer
  - Multi-head causal self-attention com KV Cache opcional
  - MLP (Linear -> GELU -> Linear) com expansao 4x
- Final: RMSNorm + Linear (tied weights com embedding)

Features:
- KV Cache para geracao O(n) em vez de O(n^2)
- Suporte a FlashAttention (PyTorch >=2.0)
- Inicializacao estavel
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NINEConfig:
    vocab_size: int = 512
    block_size: int = 256          # maximo de contexto (tokens)
    n_layer: int = 6               # blocos transformer
    n_head: int = 6                # cabecas de atencao
    n_embd: int = 384              # dimensao do embedding
    dropout: float = 0.0
    bias: bool = False             # sem bias em Linear (nanoGPT style)
    mlp_ratio: float = 4.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (mais estavel que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    """Atencao causal multi-cabeca com projecao QKV e KV Cache."""

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()

        # Com KV Cache: so calculamos Q,K,V para o NOVO token
        if kv_cache is not None:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            k_prev, v_prev = kv_cache
            # Concatena com cache anterior
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

            T_full = k.size(2)
            if hasattr(F, "scaled_dot_product_attention"):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                att = att.masked_fill(~self.mask[:, :, :T, :T_full], float("-inf"))
                att = F.softmax(att, dim=-1)
                y = att @ v

            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            return y, (k, v)

        # Sem KV Cache (training / batch)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: NINEConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: NINEConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class NINE1(nn.Module):
    """Modelo NINE-1: decoder Transformer pequeno com KV Cache."""

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=RMSNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"contexto {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.transformer.h):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv_cache=cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss, new_kv_caches
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None, new_kv_caches

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 use_cache: bool = True) -> torch.Tensor:
        """Gera tokens um a um com KV Cache opcional."""
        kv_caches = None

        for _ in range(max_new_tokens):
            if use_cache and kv_caches is not None:
                # So passa o ultimo token
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cum > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tiny_config(vocab_size: int = 512, block_size: int = 256) -> NINEConfig:
    """Configuracao 'tiny' para um modelo beem leve (~10M params)."""
    return NINEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
    )


if __name__ == "__main__":
    cfg = tiny_config()
    m = NINE1(cfg)
    print(f"Parametros: {m.num_params()/1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = m(x, y)
    print(f"logits: {logits.shape}, loss: {loss.item():.3f}")

    # Teste KV Cache
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 16))
    out = m.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10, use_cache=True)
    print(f"generate (cached): {out.shape}")
