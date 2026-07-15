"""
CLI em PT-BR para o NINE-1, com modo chat interativo.

Modos:
  base     - continua texto livre
  instruct - formato tarefa/solucao (fine-tuned)
  chat     - chat interativo multi-turno

Uso:
    python -m nine.cli "escreva fibonacci" --mode instruct --tokens 120
    python -m nine.cli --mode chat --tokens 200
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, Tuple

_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch

from .tokenizer import BPETokenizer, BOS_TOKEN, EOS_TOKEN
from .fuse import load_fused_model


PROMPT_TEMPLATES = {
    "base": "{prompt}\n",
    "instruct": "# tarefa: {prompt}\n# solucao:\n",
    "chat": "{prompt}",
}

# Histórico do chat é mantido como string acumulada
CHAT_SYSTEM_PROMPT = "Voce e uma IA de programacao em portugues chamada NINE-1. Responda com codigo Python quando apropriado."


def _load_tokenizer(tok_path: Optional[str], block_size: int) -> Tuple[BPETokenizer, bool]:
    if tok_path and os.path.exists(tok_path):
        try:
            return BPETokenizer.load(tok_path), True
        except Exception as e:
            print(f"[nine] aviso: falhou carregar tokenizer BPE ({e}); usando codepoint fallback.", file=sys.stderr)
    return None, False


def _encode(text: str, tok: Optional[BPETokenizer], block_size: int, add_bos: bool = True):
    """Codifica texto: BPE se disponivel, fallback codepoint."""
    torch = _get_torch()
    if tok is not None:
        ids = tok.encode(text, add_bos=add_bos)
        if not add_bos and BOS_TOKEN in tok.token_to_id:
            ids = [tok.token_to_id[BOS_TOKEN]] + ids
        ids = ids[-block_size:]
        return torch.tensor([ids], dtype=torch.long)
    ids = [ord(c) & 0xFFFF for c in text[:block_size]]
    return torch.tensor([ids], dtype=torch.long)


def _decode(ids, tok: Optional[BPETokenizer]) -> str:
    if tok is not None:
        return tok.decode(ids[0].tolist())
    out = []
    for i in ids[0].tolist():
        if 0 < i <= 0x10FFFF:
            try:
                out.append(chr(i))
            except ValueError:
                continue
    return "".join(out)


def _stop_condition(next_id: int, tok: Optional[BPETokenizer]) -> bool:
    if tok is not None and EOS_TOKEN in tok.token_to_id:
        return next_id == tok.token_to_id[EOS_TOKEN]
    return next_id in (0, 1, 2, 3)


def generate_tokens(model, ids, args, decode_fn, tok):
    """Gera tokens com streaming, retorna o prompt + gerado completo."""
    torch = _get_torch()
    output_ids = ids.clone()
    device = next(model.parameters()).device
    ids = ids.to(device)

    for _ in range(args.tokens):
        if ids.size(1) > model.cfg.block_size:
            ids = ids[:, -model.cfg.block_size:]

        logits, _, _ = model(ids)
        logits = logits[:, -1, :] / max(args.temp, 1e-6)

        if args.top_k is not None:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if args.top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = sorted_logits.softmax(-1).cumsum(-1)
            sorted_logits[cum > args.top_p] = float("-inf")
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = logits.softmax(-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        output_ids = torch.cat([output_ids, next_id.to(output_ids.device)], dim=1)

        chunk = decode_fn(next_id)
        sys.stdout.write(chunk)
        sys.stdout.flush()

        if _stop_condition(next_id.item(), tok):
            break

    return output_ids


def main():
    p = argparse.ArgumentParser(
        prog="nine",
        description="NINE-1 - IA de programacao em PT-BR",
    )
    p.add_argument("prompt", type=str, nargs="?", help="prompt em linguagem natural")
    p.add_argument("--ckpt", type=str, default="nine/data/nine1-base.pt",
                   help="checkpoint base (.pt)")
    p.add_argument("--lora", type=str, default=None,
                   help="checkpoint LoRA (.pt) opcional")
    p.add_argument("--tok", type=str, default="nine/data/nine1-tok.json",
                   help="arquivo BPE tokenizer JSON")
    p.add_argument("--mode", choices=["base", "instruct", "chat"], default="base",
                   help="modo de prompt")
    p.add_argument("--tokens", type=int, default=120, help="tokens a gerar")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    torch = _get_torch()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    if not os.path.exists(args.ckpt):
        print(f"[nine] erro: checkpoint nao encontrado: {args.ckpt}", file=sys.stderr)
        sys.exit(2)

    model, tok_bpe = load_fused_model(
        base_path=args.ckpt,
        lora_path=args.lora,
        tokenizer_path=args.tok,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        verbose=args.verbose,
    )

    if args.mode == "chat":
        print("=== NINE-1 Chat (PT-BR) ===")
        print(f"Digite 'sair' ou Ctrl+C para encerrar.\n")
        chat_history = CHAT_SYSTEM_PROMPT

        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nAte logo!")
                break
            if not user_input or user_input.lower() in ("sair", "quit", "exit"):
                break

            chat_history += f"\nUsuario: {user_input}\nNINE-1:"
            ids = _encode(chat_history, tok_bpe, model.cfg.block_size, add_bos=True)

            print("--- NINE-1 ---")
            generated = generate_tokens(model, ids, args,
                                        lambda x: _decode(x, tok_bpe), tok_bpe)
            print("\n--------------\n")

            # Adiciona geracao ao historico
            new_text = _decode(generated, tok_bpe)
            chat_history += new_text.split("NINE-1:", 1)[-1] if "NINE-1:" in new_text else new_text

    else:
        if not args.prompt:
            if args.mode == "instruct":
                args.prompt = "escreva uma funcao fibonacci iterativa em python"
            else:
                args.prompt = "def fala_oi():\n    "

        template = PROMPT_TEMPLATES[args.mode].format(prompt=args.prompt)
        ids = _encode(template, tok_bpe, model.cfg.block_size)

        print("\n--- NINE-1 ---")
        print(template, end="")
        generate_tokens(model, ids, args,
                        lambda x: _decode(x, tok_bpe), tok_bpe)
        print("\n--------------")


if __name__ == "__main__":
    main()
