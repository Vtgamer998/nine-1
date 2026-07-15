"""
Script de pre-treinamento do NINE-1.
Uso:
    python -m nine.train --data nine/data/corpus.bin --tok nine/data/nine1-tok.json
                         --out nine/data/nine1-base.pt
"""

from __future__ import annotations
import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from .model import NINE1, NINEConfig
from .data import get_batch


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
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Carrega dataset binario
    if args.data.endswith(".bin"):
        data = np.memmap(args.data, dtype=np.uint16, mode="r")
    else:
        with open(args.data, "r", encoding="utf-8") as f:
            txt = f.read()
        data = np.memmap(
            args.data + ".tmp.bin", dtype=np.uint16, mode="w+",
            shape=(len(txt),),
        )
        for i, ch in enumerate(txt):
            data[i] = ord(ch) % 65536
        data.flush()

    # Split treino/val
    n = len(data)
    val_size = int(n * args.val_split)
    train_data = data[val_size:]
    val_data = data[:val_size] if val_size > 0 else None
    print(f"Tokens no dataset: {n} (treino: {len(train_data)}, val: {val_size})")

    # Configuracao
    cfg = NINEConfig(
        vocab_size=args.vocab,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
    )

    model = NINE1(cfg)
    model.to(args.device)
    n_params = model.num_params() / 1e6
    print(f"NINE-1 (base): {n_params:.2f}M parametros")

    if args.tok:
        print(f"Tokenizer BPE: {args.tok}")
        from .tokenizer import BPETokenizer
        tok = BPETokenizer.load(args.tok)
        print(f"  vocab: {len(tok)} tokens")

    # Otimizador
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

    # Loop
    t0 = time.time()
    best_val_loss = float("inf")
    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup, args.max_iters) * args.lr
        for g in optim.param_groups:
            g["lr"] = lr

        with ctx:
            x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
            _, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if it % 50 == 0:
            dt = (time.time() - t0) * 1000 / max(it + 1, 1)
            print(f"iter {it}/{args.max_iters} | lr {lr:.2e} | loss {loss.item():.3f} | {dt:.1f}ms/iter")

        if it > 0 and it % args.eval_interval == 0 and val_data is not None:
            val_loss = estimate_loss(model, val_data, args.block_size,
                                     args.batch_size, args.device)
            print(f"  >>> val loss: {val_loss:.3f} (best: {best_val_loss:.3f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
                print(f"  >>> melhor checkpoint salvo em {args.out}")

        if it > 0 and it % args.save_interval == 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
            print(f"  >> checkpoint salvo em {args.out}")

    # Salva final
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
    print(f"\nTreino finalizado. Loss final: {loss.item():.3f}")
    if val_data is not None:
        final_val = estimate_loss(model, val_data, args.block_size,
                                  args.batch_size, args.device)
        print(f"Val loss final: {final_val:.3f}")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
