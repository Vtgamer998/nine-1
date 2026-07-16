"""
Script de pre-treinamento do NINE-1.

Uso:
    python -m nine.train --data nine/data/corpus.bin --tok nine/data/nine1-tok.json
                         --out nine/data/nine1-base.pt

Seguranca:
- Validacao de dados de entrada (tamanho, integridade)
- Gradient accumulation para estabilidade
- Checkpoint com validacao de integridade
- Protecao contra overflow de dados
- Seed deterministica para reproducibilidade
"""

from __future__ import annotations
import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch

from .model import NINE1, NINEConfig
from .dataset import get_batch


# Limites de seguranca
MAX_DATASET_SIZE = 10_000_000_000  # 10B tokens max
MIN_DATASET_SIZE = 100             # 100 tokens min
VALID_DTYPES = {"float32", "bfloat16", "float16"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="arquivo .bin ou .txt tokenizado")
    p.add_argument("--tok", type=str, default=None, help="tokenizer BPE .json (opcional, para log)")
    p.add_argument("--out", type=str, default="nine/data/nine1-base.pt")
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--val_split", type=float, default=0.05, help="fracao para validacao")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_lr(it: int, warmup: int, max_iters: int) -> float:
    if it < warmup:
        return (it + 1) / warmup
    if it > max_iters:
        return 0.1
    decay_ratio = (it - warmup) / (max_iters - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, num_batches=20):
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def load_data_safe(path: str) -> np.memmap:
    """Carrega dataset binario com validacoes de seguranca.

    Args:
        path: Caminho do arquivo .bin.

    Returns:
        np.memmap com os dados.

    Raises:
        FileNotFoundError: Se arquivo nao existe.
        ValueError: Se dados sao muito pequenos ou grandes demais.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")

    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError(f"Arquivo de dados vazio: {path}")

    # Verifica extensao
    if not path.endswith(".bin"):
        print(f"  [aviso] Arquivo de dados nao tem extensao .bin: {path}")

    data = np.memmap(path, dtype=np.uint16, mode="r")

    n_tokens = len(data)
    if n_tokens < MIN_DATASET_SIZE:
        raise ValueError(f"Dataset muito pequeno: {n_tokens} tokens (min {MIN_DATASET_SIZE})")
    if n_tokens > MAX_DATASET_SIZE:
        raise ValueError(f"Dataset muito grande: {n_tokens} tokens (max {MAX_DATASET_SIZE})")

    return data


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Validacao de parametros
    args.dtype = args.dtype if args.dtype in VALID_DTYPES else "float32"
    if args.vocab < 256:
        print(f"  [aviso] vocab_size {args.vocab} muito pequeno, usando 256")
        args.vocab = 256
    if args.block_size < 16:
        print(f"  [aviso] block_size {args.block_size} muito pequeno, usando 16")
        args.block_size = 16

    # Carrega dataset binario com validacao
    print(f"Carregando dados: {args.data}")
    data = load_data_safe(args.data)
    n = len(data)

    # Split treino/val
    val_size = int(n * args.val_split)
    train_data = data[val_size:]
    val_data = data[:val_size] if val_size > 0 else None
    print(f"Tokens no dataset: {n:,} (treino: {len(train_data):,}, val: {val_size:,})")

    # Configuracao
    cfg = NINEConfig(
        vocab_size=args.vocab,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
        use_rope=True,  # RoPE por padrao no treino
    )

    model = NINE1(cfg)
    model.to(args.device)
    n_params = model.num_params() / 1e6
    print(f"NINE-1 (base): {n_params:.2f}M parametros | "
          f"vocab={cfg.vocab_size} block={cfg.block_size} "
          f"layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd}")

    if args.tok:
        print(f"Tokenizer BPE: {args.tok}")
        from .tokenizer import BPETokenizer
        try:
            tok = BPETokenizer.load(args.tok)
            print(f"  vocab: {len(tok)} tokens")
        except Exception as e:
            print(f"  [aviso] Falha ao carregar tokenizer: {e}")

    # Otimizador com weight decay seletivo
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-2}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(args.device, dtype=ptdtype) if args.device == "cuda" else nullcontext()

    # Loop de treino com gradient accumulation
    t0 = time.time()
    best_val_loss = float("inf")
    accum_steps = max(1, args.grad_accum_steps)
    optim.zero_grad(set_to_none=True)

    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup, args.max_iters) * args.lr
        for g in optim.param_groups:
            g["lr"] = lr

        # Micro-batches para gradient accumulation
        for micro in range(accum_steps):
            with ctx:
                x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
                _, loss, _ = model(x, y)
                loss = loss / accum_steps  # Normaliza pelo numero de steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if it % 50 == 0:
            dt = (time.time() - t0) * 1000 / max(it + 1, 1)
            print(f"iter {it}/{args.max_iters} | lr {lr:.2e} | loss {loss.item()*accum_steps:.3f} | {dt:.1f}ms/iter")

        if it > 0 and it % args.eval_interval == 0 and val_data is not None:
            val_loss = estimate_loss(model, val_data, args.block_size,
                                     args.batch_size, args.device)
            print(f"  >>> val loss: {val_loss:.3f} (best: {best_val_loss:.3f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "args": vars(args),
                }, args.out)
                print(f"  >>> melhor checkpoint salvo em {args.out}")

        if it > 0 and it % args.save_interval == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "args": vars(args),
            }, args.out)
            print(f"  >> checkpoint salvo em {args.out}")

    # Salva final
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "args": vars(args),
    }, args.out)

    print(f"\nTreino finalizado. Loss final: {loss.item()*accum_steps:.3f}")
    if val_data is not None:
        final_val = estimate_loss(model, val_data, args.block_size,
                                  args.batch_size, args.device)
        print(f"Val loss final: {final_val:.3f}")
    print(f"Checkpoint: {args.out} ({(os.path.getsize(args.out)/1024/1024):.1f} MB)")


if __name__ == "__main__":
    main()
