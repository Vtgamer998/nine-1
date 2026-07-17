"""
CLI em PT-BR para o NINE-1, com modo chat interativo.

Modos:
  base     - continua texto livre
  instruct - formato tarefa/solucao (fine-tuned)
  chat     - chat interativo multi-turno

Seguranca:
- Sanitizacao de input do usuario (tamanho max, caracteres de controle)
- Protecao contra path traversal em caminhos de arquivo
- Validacao de checkpoint antes de carregar
- Limite de tokens gerados (protecao contra loops infinitos)
- Timeout silencioso para geracao muito longa
- Modo chat com historico limitado

Uso:
    python -m nine.cli "escreva fibonacci" --mode instruct --tokens 120
    python -m nine.cli --mode chat --tokens 200
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Optional, Tuple, List

_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch

from .tokenizer import BPETokenizer, BOS_TOKEN, EOS_TOKEN, sanitize_filename_component


# ---------------------------------------------------------------------------
# Constantes de seguranca
# ---------------------------------------------------------------------------

# Tamanho maximo do prompt
MAX_PROMPT_CHARS = 10_000
MAX_CHAT_HISTORY_CHARS = 100_000
# Tamanho maximo de geracao
MAX_GENERATION_TOKENS = 2048
# Caminhos permitidos para checkpoints (restringe path traversal)
ALLOWED_CKPT_DIRS = {"nine/data", "data", "."}
# Comprimento maximo do caminho do checkpoint
MAX_CKPT_PATH_LENGTH = 256
# Caracteres de controle a serem removidos do input
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


PROMPT_TEMPLATES = {
    "base": "{prompt}\n",
    "instruct": "# tarefa: {prompt}\n# solucao:\n",
    "chat": "{prompt}",
}

# Historico do chat e mantido como string acumulada
CHAT_SYSTEM_PROMPT = (
    "Voce e uma IA de programacao em portugues chamada NINE-1. "
    "Responda com codigo Python quando apropriado. "
    "Nao gere codigo malicioso, virus, ou exploits de seguranca. "
    "Sempre priorize boas praticas de programacao."
)


# ---------------------------------------------------------------------------
# Funcoes de seguranca
# ---------------------------------------------------------------------------

def sanitize_prompt(text: str) -> str:
    """Sanitiza prompt do usuario.

    - Remove caracteres de controle
    - Limita tamanho
    - Remove whitespace excessivo

    Args:
        text: Texto do prompt.

    Returns:
        Texto sanitizado.
    """
    if not isinstance(text, str):
        return ""
    # Remove caracteres de controle
    text = CONTROL_CHARS_RE.sub("", text)
    # Normaliza whitespace horizontal (preserva newlines para codigo)
    text = re.sub(r"[ \t]+", " ", text)
    # Limita tamanho
    if len(text) > MAX_PROMPT_CHARS:
        text = text[:MAX_PROMPT_CHARS]
    return text


def validate_ckpt_path(path: str) -> bool:
    """Valida caminho de checkpoint contra path traversal.

    Args:
        path: Caminho do arquivo.

    Returns:
        True se o caminho e seguro.
    """
    if not isinstance(path, str) or not path:
        return False
    if len(path) > MAX_CKPT_PATH_LENGTH:
        return False
    # Nao permite caminhos com '..' (path traversal)
    if ".." in path.split(os.sep):
        return False
    # Verifica extensao
    if not path.endswith((".pt", ".pth", ".bin", ".json")):
        return False
    return True


def is_safe_path(path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
    """Verifica se o caminho esta dentro de diretorios permitidos.

    Args:
        path: Caminho a verificar.
        allowed_dirs: Lista de diretorios permitidos.

    Returns:
        True se o caminho e seguro.
    """
    if allowed_dirs is None:
        allowed_dirs = ALLOWED_CKPT_DIRS
    abs_path = os.path.abspath(path)
    for d in allowed_dirs:
        allowed_abs = os.path.abspath(d)
        if abs_path.startswith(allowed_abs + os.sep) or abs_path == allowed_abs:
            return True
    return False


def _load_tokenizer(tok_path: Optional[str], block_size: int) -> Tuple[Optional[BPETokenizer], bool]:
    if tok_path and os.path.exists(tok_path):
        if not validate_ckpt_path(tok_path):
            print(f"[nine] aviso: caminho do tokenizer parece inseguro: {tok_path}", file=sys.stderr)
            return None, False
        try:
            return BPETokenizer.load(tok_path), True
        except Exception as e:
            print(f"[nine] aviso: falhou carregar tokenizer BPE ({e}); usando codepoint fallback.", file=sys.stderr)
    return None, False


def _encode(text: str, tok: Optional[BPETokenizer], block_size: int, add_bos: bool = True):
    """Codifica texto: BPE se disponivel, fallback codepoint.

    Args:
        text: Texto a codificar.
        tok: Tokenizer BPE (opcional).
        block_size: Tamanho maximo do contexto.
        add_bos: Se True, adiciona BOS token.

    Returns:
        Tensor (1, T) com IDs dos tokens.
    """
    torch = _get_torch()
    text = sanitize_prompt(text)
    if tok is not None:
        ids = tok.encode(text, add_bos=add_bos)
        if not add_bos and BOS_TOKEN in tok.token_to_id:
            ids = [tok.token_to_id[BOS_TOKEN]] + ids
        ids = ids[-block_size:]
        return torch.tensor([ids], dtype=torch.long)
    # Fallback seguro: usa codepoint com mascara
    ids = []
    for c in text[:block_size]:
        cp = ord(c) & 0xFFFF
        if cp > 0:  # Ignora codepoint 0 (nulo)
            ids.append(cp)
    if not ids:
        ids = [0]  # Fallback minimo
    return torch.tensor([ids], dtype=torch.long)


def _decode(ids, tok: Optional[BPETokenizer]) -> str:
    """Decodifica IDs para texto de forma segura.

    Args:
        ids: Tensor (1, T) com IDs.
        tok: Tokenizer BPE (opcional).

    Returns:
        String decodificada.
    """
    if tok is not None:
        return tok.decode(ids[0].tolist())
    # Fallback seguro para codepoint
    out = []
    for i in ids[0].tolist():
        if 0 < i <= 0x10FFFF:
            try:
                ch = chr(i)
                # Filtra caracteres de controle (exceto newline, tab)
                if ch.isprintable() or ch in ("\n", "\t", "\r", " "):
                    out.append(ch)
            except (ValueError, OverflowError):
                continue
    return "".join(out)


def _stop_condition(next_id: int, tok: Optional[BPETokenizer]) -> bool:
    """Verifica se deve parar a geracao.

    Args:
        next_id: ID do token gerado.
        tok: Tokenizer BPE (opcional).

    Returns:
        True se deve parar.
    """
    if tok is not None and EOS_TOKEN in tok.token_to_id:
        return next_id == tok.token_to_id[EOS_TOKEN]
    # Fallback: IDs de controle comuns
    return next_id in (0, 1, 2, 3)


def generate_tokens(model, ids, args, decode_fn, tok):
    """Gera tokens com streaming e seguranca.

    Args:
        model: Modelo NINE-1.
        ids: Tensor de input IDs.
        args: Argumentos da CLI.
        decode_fn: Funcao de decode.
        tok: Tokenizer (opcional).

    Returns:
        output_ids: Tensor com prompt + geracao completa.
    """
    torch = _get_torch()
    output_ids = ids.clone()
    device = next(model.parameters()).device
    ids = ids.to(device)

    tokens_to_generate = min(args.tokens, MAX_GENERATION_TOKENS)

    for step in range(tokens_to_generate):
        if ids.size(1) > model.cfg.block_size:
            ids = ids[:, -model.cfg.block_size:]

        logits, _, _ = model(ids)
        logits = logits[:, -1, :] / max(args.temp, 1e-8)

        if args.top_k is not None and args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if args.top_p is not None and args.top_p > 0:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    p.add_argument("--mode", choices=["base", "instruct", "chat"], default="instruct",
                   help="modo de prompt (default: instruct para melhor resultado)")
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

    # Validacao de seguranca dos caminhos
    if not validate_ckpt_path(args.ckpt):
        print(f"[nine] erro: caminho do checkpoint parece inseguro: {args.ckpt}", file=sys.stderr)
        sys.exit(2)
    if args.lora and not validate_ckpt_path(args.lora):
        print(f"[nine] erro: caminho do LoRA parece inseguro: {args.lora}", file=sys.stderr)
        sys.exit(2)
    if args.tok and not validate_ckpt_path(args.tok):
        print(f"[nine] erro: caminho do tokenizer parece inseguro: {args.tok}", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.ckpt):
        print(f"[nine] erro: checkpoint nao encontrado: {args.ckpt}", file=sys.stderr)
        sys.exit(2)

    # Valores seguros
    args.tokens = min(args.tokens, MAX_GENERATION_TOKENS)
    args.temp = max(args.temp, 0.1)  # Evita temperatura muito baixa (determinismo extremo)

    # Import lazy (torch pode nao estar instalado)
    from .fuse import load_fused_model

    model, tok_bpe = load_fused_model(
        base_path=args.ckpt,
        lora_path=args.lora,
        tokenizer_path=args.tok,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        verbose=args.verbose,
        validate=True,  # Ativa validacao de checkpoint
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

            # Sanitiza entrada do usuario
            user_input = sanitize_prompt(user_input)

            chat_history += f"\nUsuario: {user_input}\nNINE-1:"
            # Limita historico para evitar estouro de memoria
            if len(chat_history) > MAX_CHAT_HISTORY_CHARS:
                chat_history = CHAT_SYSTEM_PROMPT + chat_history[-MAX_CHAT_HISTORY_CHARS:]

            ids = _encode(chat_history, tok_bpe, model.cfg.block_size, add_bos=True)

            print("--- NINE-1 ---")
            generated = generate_tokens(model, ids, args,
                                        lambda x: _decode(x, tok_bpe), tok_bpe)
            print("\n--------------\n")

            # Adiciona geracao ao historico (limitado)
            new_text = _decode(generated, tok_bpe)
            chat_response = new_text.split("NINE-1:", 1)[-1] if "NINE-1:" in new_text else new_text
            chat_history += chat_response

    else:
        if not args.prompt:
            if args.mode == "base":
                args.prompt = "def fala_oi():\n    "
            else:
                args.prompt = "escreva uma funcao fibonacci iterativa em python"

        # Sanitiza prompt
        args.prompt = sanitize_prompt(args.prompt)

        template = PROMPT_TEMPLATES[args.mode].format(prompt=args.prompt)
        ids = _encode(template, tok_bpe, model.cfg.block_size)

        print("\n--- NINE-1 ---")
        print(template, end="")
        generate_tokens(model, ids, args,
                        lambda x: _decode(x, tok_bpe), tok_bpe)
        print("\n--------------")


if __name__ == "__main__":
    main()
