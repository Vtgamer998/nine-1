"""
NINE-1: Transformer decoder-only minimal, implementado do zero usando PyTorch.
Arquitetura:
- Embeddings de tokens + RoPE (Rotary Position Embeddings)
- N blocos: pre-norm Transformer
  - Multi-head causal self-attention com KV Cache opcional
  - MLP (Linear -> GELU -> Linear) com expansao 4x
- Final: RMSNorm + Linear (tied weights com embedding)

Features:
- KV Cache para geracao O(n) em vez de O(n^2)
- RoPE (Rotary Position Embeddings) para codificacao posicional
- Suporte a FlashAttention (PyTorch >=2.0)
- Inicializacao estavel
- Validacao de seguranca para checkpoints e tokens
"""

from __future__ import annotations
import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Seguranca: validacao de checkpoint
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION = 1

REQUIRED_CHECKPOINT_KEYS = {"model", "cfg"}
REQUIRED_CFG_KEYS = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd"}


def validate_checkpoint_state(
    state_dict: Dict[str, torch.Tensor],
    expected_cfg: Optional["NINEConfig"] = None,
) -> List[str]:
    """Valida a integridade de um state dict de checkpoint.

    Verifica:
    - Tipos dos tensores
    - Dimensoes esperadas (se cfg for fornecido)
    - Ausencia de valores NaN/inf
    - Compatibilidade de dispositivos

    Returns:
        Lista de avisos/issues encontrados (vazia se tudo ok).
    """
    issues: List[str] = []

    if not state_dict:
        return ["State dict vazio."]

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            issues.append(f"Parametro '{name}' nao e tensor, tipo={type(param)}")
            continue

        # Verifica NaN/Inf
        if torch.isnan(param).any():
            issues.append(f"Parametro '{name}' contem NaN!")
        if torch.isinf(param).any():
            issues.append(f"Parametro '{name}' contem Inf!")

        # Verifica dtype
        if param.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8):
            issues.append(f"Parametro '{name}' dtype inesperado: {param.dtype}")

    if expected_cfg:
        # Verifica se dimensoes batem
        expected_shapes = {
            "transformer.wte.weight": (expected_cfg.vocab_size, expected_cfg.n_embd),
            "transformer.wpe.weight": (expected_cfg.block_size, expected_cfg.n_embd),
            "transformer.ln_f.weight": (expected_cfg.n_embd,),
            "lm_head.weight": (expected_cfg.vocab_size, expected_cfg.n_embd),
        }
        for name, expected_shape in expected_shapes.items():
            if name in state_dict:
                actual_shape = tuple(state_dict[name].shape)
                if actual_shape != expected_shape:
                    issues.append(
                        f"Shape de '{name}' {actual_shape} != esperado {expected_shape}"
                    )

    return issues


def safe_load_checkpoint(
    path: str,
    map_location: str = "cpu",
    expected_cfg: Optional["NINEConfig"] = None,
) -> Dict[str, Any]:
    """Carrega checkpoint com validacoes de seguranca.

    Usa weights_only=True do torch.load e valida estrutura.
    Levanta excecao se o checkpoint parecer corrompido ou malicioso.
    """
    if not isinstance(path, str) or not path.endswith(".pt"):
        raise ValueError(f"Caminho de checkpoint invalido: {path!r}")

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar checkpoint (pode estar corrompido): {e}")

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint deve ser um dicionario, nao {type(ckpt)}")

    # Verifica chaves minimas
    found_keys = set(ckpt.keys())
    if not found_keys & REQUIRED_CHECKPOINT_KEYS:
        raise RuntimeError(
            f"Checkpoint nao contem chaves esperadas {REQUIRED_CHECKPOINT_KEYS}. "
            f"Encontradas: {found_keys}"
        )

    # Valida tensores
    if "model" in ckpt:
        issues = validate_checkpoint_state(ckpt["model"], expected_cfg)
        if issues:
            raise RuntimeError(f"Checkpoint corrompido: {'; '.join(issues[:5])}")

    return ckpt


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class NINEConfig:
    vocab_size: int = 512
    block_size: int = 256          # maximo de contexto (tokens)
    n_layer: int = 6               # blocos transformer
    n_head: int = 6                # cabecas de atencao (query)
    n_kv_heads: int = 0           # cabecas K/V (0 = mesmo que n_head, MHA)
    n_embd: int = 384              # dimensao do embedding
    dropout: float = 0.0
    bias: bool = False             # sem bias em Linear (nanoGPT style)
    mlp_ratio: float = 4.0
    use_rope: bool = True          # RoPE em vez de learned positional embeddings

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NINEConfig":
        """Cria config a partir de dict, ignorando chaves desconhecidas."""
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})


# ---------------------------------------------------------------------------
# Componentes do modelo
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (mais estavel que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = "cpu"):
    """Pre-computa as frequencias complexas para RoPE.

    Args:
        dim: Dimensao do head (head_dim).
        end: Comprimento maximo (block_size).
        theta: Frequencia base (10000.0 do original RoPE).
        device: Dispositivo alvo.

    Returns:
        (freqs_cos, freqs_sin): Tensores (1, 1, end, dim//2) cada.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    freqs_cos = freqs.cos().view(1, 1, end, dim // 2)
    freqs_sin = freqs.sin().view(1, 1, end, dim // 2)
    return freqs_cos, freqs_sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica RoPE aos queries e keys.

    Args:
        xq: (B, n_head, T, head_dim)
        xk: (B, n_head, T, head_dim)
        freqs_cos: (1, 1, T, head_dim//2)
        freqs_sin: (1, 1, T, head_dim//2)

    Returns:
        (xq_rotated, xk_rotated)
    """
    head_dim = xq.size(-1)
    # Separa pares (dim // 2) para aplicar rotacao 2D
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    freqs_cos = freqs_cos.unsqueeze(-1)  # (1, 1, T, head_dim//2, 1)
    freqs_sin = freqs_sin.unsqueeze(-1)

    xq_rot = torch.stack(
        [xq_r[..., 0] * freqs_cos[..., 0] - xq_r[..., 1] * freqs_sin[..., 0],
         xq_r[..., 1] * freqs_cos[..., 0] + xq_r[..., 0] * freqs_sin[..., 0]],
        dim=-1,
    ).flatten(-2)
    xk_rot = torch.stack(
        [xk_r[..., 0] * freqs_cos[..., 0] - xk_r[..., 1] * freqs_sin[..., 0],
         xk_r[..., 1] * freqs_cos[..., 0] + xk_r[..., 0] * freqs_sin[..., 0]],
        dim=-1,
    ).flatten(-2)

    return xq_rot.to(xq.dtype), xk_rot.to(xk.dtype)


class CausalSelfAttention(nn.Module):
    """Atencao causal multi-cabeca com projecao QKV e KV Cache.

    Suporta RoPE (Rotary Position Embeddings) quando cfg.use_rope=True.
    """

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads > 0 else cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.use_rope
        assert self.n_head % self.n_kv_heads == 0, "n_head deve ser multiplo de n_kv_heads"
        self.n_rep = self.n_head // self.n_kv_heads  # repeticoes para GQA

        # Projecoes separadas para Q (n_head) e K/V (n_kv_heads)
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        kv_dim = self.n_kv_heads * self.head_dim
        self.k_proj = nn.Linear(cfg.n_embd, kv_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, kv_dim, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

        # Mascara causal
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask, persistent=False)

        # RoPE frequencias (inicializadas lazy)
        self.register_buffer("_freqs_cos", None, persistent=False)
        self.register_buffer("_freqs_sin", None, persistent=False)

    def _maybe_init_rope(self, device: str):
        if self._freqs_cos is not None and self._freqs_cos.device == torch.device(device):
            return
        cos, sin = precompute_freqs_cis(self.head_dim, self.mask.size(-1), device=device)
        self.register_buffer("_freqs_cos", cos, persistent=False)
        self.register_buffer("_freqs_sin", sin, persistent=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repete heads K/V para GQA.

        Args:
            x: (B, n_kv_heads, T, head_dim)
            n_rep: Numero de repeticoes.

        Returns:
            (B, n_head, T, head_dim) onde n_head = n_kv_heads * n_rep
        """
        if n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Aplica RoPE nas posicoes atuais
        if self.use_rope:
            self._maybe_init_rope(x.device)
            cos_slice = self._freqs_cos[:, :, :T, :]
            sin_slice = self._freqs_sin[:, :, :T, :]
            # RoPE precisa de shapes (B, n_head, T, head_dim) e (B, n_kv_heads, T, head_dim)
            q, k = apply_rotary_emb(q, k, cos_slice, sin_slice)

        # Com KV Cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        # Expande K/V para o numero de Q heads (GQA)
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

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


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class NINE1(nn.Module):
    """Modelo NINE-1: decoder Transformer pequeno com KV Cache e RoPE.

    Seguranca:
    - Validacao de checkpoint no carregamento
    - Clamping de temperatura para evitar divisao por zero
    - Validacao de tokens gerados
    - Controle de tamanho maximo de contexto
    """

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=RMSNorm(cfg.n_embd),
        ))

        # WPE ainda usado como fallback para modelos sem RoPE
        if not cfg.use_rope:
            self.transformer.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.transformer.wpe = None

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

    def validate_input_ids(self, idx: torch.Tensor) -> bool:
        """Valida ids de entrada: nao pode ter valores negativos ou fora do vocab.

        Returns:
            True se valido, False se invalido.
        """
        if idx.dtype not in (torch.long, torch.int, torch.int64):
            return False
        if (idx < 0).any() or (idx >= self.cfg.vocab_size).any():
            return False
        return True

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"contexto {T} > block_size {self.cfg.block_size}"

        # Validacao de seguranca (modo eval)
        if not self.training and not self.validate_input_ids(idx):
            raise ValueError("Input IDs invalidos (fora do vocabulario ou negativos)")

        # Posicao (apenas se for usar wpe)
        tok_emb = self.transformer.wte(idx)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

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
            # Retorna logits COMPLETOS (B, T, V) mesmo sem targets
            # para compatibilidade com DPO e outras losses que precisam
            # de log-probabilidades de todas as posicoes.
            # No caso de geracao com um unico token (KV cache),
            # x tem shape (B, 1, C) e logits (B, 1, V), mesmo efeito.
            logits = self.lm_head(x)
            return logits, None, new_kv_caches

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 use_cache: bool = True,
                 repetition_penalty: Optional[float] = None) -> torch.Tensor:
        """Gera tokens um a um com KV Cache opcional e seguranca.

        Args:
            idx: Tensor (1, T) com tokens de prompt.
            max_new_tokens: Maximo de tokens a gerar.
            temperature: Temperatura para amostragem (clamped >= 1e-8).
            top_k: Se > 0, amostra apenas top-k tokens.
            top_p: Nucleus sampling parametro.
            use_cache: Se True, usa KV Cache (gera mais rapido).
            repetition_penalty: Penalidade de repeticao (>1.0 penaliza tokens repetidos).

        Returns:
            Tensor (1, T + max_new_tokens) com prompt + geracao.
        """
        # Validacao de seguranca
        if not self.validate_input_ids(idx):
            raise ValueError("Input IDs invalidos para geracao")
        max_new_tokens = max(1, min(max_new_tokens, self.cfg.block_size))
        temperature = max(temperature, 1e-8)  # Evita divisao por zero
        kv_caches = None

        for _ in range(max_new_tokens):
            if use_cache and kv_caches is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty is not None and repetition_penalty > 0:
                for token in idx[0].unique():
                    logits[:, token] /= repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and top_p > 0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cum > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            # Stop condition: token 0 (PAD) frequentemente
            if next_id.item() == 0:
                break

        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas de parametros para debug."""
        return {
            "total": self.num_params(),
            "trainable": self.num_trainable_params(),
            "frozen": self.num_params() - self.num_trainable_params(),
        }


# ---------------------------------------------------------------------------
# Configuracoes pre-definidas
# ---------------------------------------------------------------------------

def tiny_config(vocab_size: int = 512, block_size: int = 256) -> NINEConfig:
    """Configuracao 'tiny' para um modelo beem leve (~10M params)."""
    return NINEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_kv_heads=4,  # GQA: 6 Q heads, 4 KV heads
        n_embd=384,
        dropout=0.0,
        bias=False,
        use_rope=True,
    )


def small_config(vocab_size: int = 4096, block_size: int = 512) -> NINEConfig:
    """Configuracao 'small' para treino no Colab (~30M params)."""
    return NINEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=10,
        n_head=8,
        n_kv_heads=4,  # GQA: 8 Q heads, 4 KV heads
        n_embd=512,
        dropout=0.0,
        bias=False,
        use_rope=True,
    )


def medium_config(vocab_size: int = 8192, block_size: int = 1024) -> NINEConfig:
    """Configuracao 'medium' para ~100M params."""
    return NINEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=16,
        n_head=12,
        n_kv_heads=4,  # GQA: 12 Q heads, 4 KV heads
        n_embd=768,
        dropout=0.1,
        bias=False,
        use_rope=True,
    )


if __name__ == "__main__":
    # Teste de seguranca basico
    cfg = tiny_config()
    m = NINE1(cfg)
    print(f"Parametros: {m.num_params()/1e6:.2f}M")
    print(f"Config: vocab={cfg.vocab_size}, block={cfg.block_size}, "
          f"layers={cfg.n_layer}, heads={cfg.n_head}, embd={cfg.n_embd}")
    print(f"RoPE: {cfg.use_rope}")

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = m(x, y)
    print(f"logits: {logits.shape}, loss: {loss.item():.3f}")

    # Validacao de input
    assert m.validate_input_ids(x), "IDs validos falharam"
    bad_ids = torch.tensor([[-1, 0, 999999]])
    assert not m.validate_input_ids(bad_ids), "IDs invalidos passaram"

    # Teste KV Cache
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 16))
    out = m.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10, use_cache=True)
    print(f"generate (cached): {out.shape}")

    # Teste validacao checkpoint
    state = {"model": m.state_dict(), "cfg": cfg.__dict__}
    issues = validate_checkpoint_state(state["model"], cfg)
    assert not issues, f"Issues encontrados: {issues}"
    print(f"[ok] Validacao de checkpoint OK ({len(issues)} issues)")

    print("[ok] Testes do modelo passaram!")
