"""
Tokenizer BPE (Byte Pair Encoding) implementado do zero em Python puro.
Usa codificacao byte-level (GPT-2 style) para suporte completo a Unicode,
incluindo acentos PT-BR, emojis, etc.

Inspirado em: Sennrich et al. 2016, GPT-2 tokenizer, nanoGPT (Karpathy).
Vocabulario base: 256 bytes + special tokens + merges aprendidos.
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Regex GPT-2 style para tokenizacao pre-BPE
# Usa ranges Unicode explicitos (suportado em Python puro)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z\xC0-\xFF]+| ?[0-9]+| ?[^\w\s]+|\s+(?!\S)|\s+"""


def _bytes_to_unicode():
    """
    Mapeia cada byte (0-255) para um caractere Unicode visivel.
    Bytes 0-31 viram chars nos ranges 256-287 para evitar conflito com controle.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs)), dict(zip(cs, bs))


_byte_encoder, _byte_decoder = _bytes_to_unicode()


def _get_pairs(word: Tuple[str, ...]) -> Counter:
    """Retorna pares adjacentes de uma palavra (como tupla de strings)."""
    pairs = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


class BPETokenizer:
    """
    Tokenizer BPE byte-level (GPT-2 style).
    - Vocabulario base: 256 bytes mapeados para Unicode visivel
    - Merges BPE aprendidos
    - Encode/decode com suporte total a Unicode
    """

    def __init__(self, vocab_size: int = 8192, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.pat = re.compile(PAT)
        self._byte_encoder = _byte_encoder
        self._byte_decoder = _byte_decoder

    # ---------- Treino ----------
    def train(self, text: str, verbose: bool = False):
        if not text:
            raise ValueError("Texto vazio para treino.")

        # Vocab inicial: bytes 0-255 mapeados para Unicode visivel
        self.vocab = {i: self._byte_encoder[i] for i in range(256)}

        # Adiciona special tokens
        next_id = 256
        for st in self.special_tokens:
            if st not in self.vocab.values():
                self.vocab[next_id] = st
                next_id += 1

        # Tokeniza texto em "palavras" (regex GPT-2 style)
        words_raw = self.pat.findall(text)
        word_freq: Counter = Counter()
        for w in words_raw:
            # Converte para bytes, mapeia para Unicode visivel
            byte_list = list(w.encode("utf-8"))
            word = tuple(self._byte_encoder[b] for b in byte_list)
            word_freq[word] += 1

        target_merges = self.vocab_size - len(self.vocab)
        if target_merges < 1:
            raise ValueError("vocab_size muito pequeno.")

        for i in range(target_merges):
            pairs = Counter()
            for word, freq in word_freq.items():
                pairs.update(_get_pairs(word))
            if not pairs:
                break
            pair = pairs.most_common(1)[0][0]
            tok_str = pair[0] + pair[1]
            self.vocab[next_id] = tok_str
            self.merges[pair] = tok_str
            next_id += 1

            new_word_freq = Counter()
            for word, freq in word_freq.items():
                new_word = self._apply_merge(word, pair, tok_str)
                new_word_freq[new_word] += freq
            word_freq = new_word_freq

            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{target_merges}: {pair} -> {tok_str!r}")

        self.vocab = {k: v for k, v in self.vocab.items() if k < self.vocab_size}
        self.token_to_id = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Tokenizer treinado: {len(self.vocab)} tokens, {len(self.merges)} merges")
        return self

    @staticmethod
    def _apply_merge(word: Tuple[str, ...], pair: Tuple[str, str], new_token: str):
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ---------- Encode ----------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not self.vocab:
            raise RuntimeError("Tokenizer nao foi treinado ou carregado.")
        ids: List[int] = []
        if add_bos and BOS_TOKEN in self.token_to_id:
            ids.append(self.token_to_id[BOS_TOKEN])

        for chunk in self.pat.findall(text):
            if not chunk:
                continue
            # Converte para bytes, mapeia para Unicode visivel
            byte_list = list(chunk.encode("utf-8"))
            word = tuple(self._byte_encoder[b] for b in byte_list)

            # Aplica merges BPE
            for pair in sorted(self.merges.keys(),
                               key=lambda p: self.token_to_id.get(self.merges[p], 0)):
                merged = self._apply_merge(word, pair, self.merges[pair])
                if len(merged) < len(word):
                    word = merged

            for token in word:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])

        if add_eos and EOS_TOKEN in self.token_to_id:
            ids.append(self.token_to_id[EOS_TOKEN])
        return ids

    # ---------- Decode ----------
    def decode(self, ids: List[int]) -> str:
        out_bytes = []
        for i in ids:
            tok = self.vocab.get(i)
            if tok is None:
                continue
            if tok in self.special_tokens:
                if tok == EOS_TOKEN:
                    break
                continue
            # Converte de Unicode visivel para byte
            for ch in tok:
                out_bytes.append(self._byte_decoder[ch])
        return bytes(out_bytes).decode("utf-8", errors="replace")

    # ---------- Persistencia ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serial_merges = {f"{p[0]}\u241F{p[1]}": t for p, t in self.merges.items()}
        payload = {
            "vocab_size": self.vocab_size,
            "merges": serial_merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            vocab_size=payload["vocab_size"],
            special_tokens=payload["special_tokens"],
        )
        tok.vocab = {int(k): v for k, v in payload["vocab"].items()}
        tok.token_to_id = {v: k for k, v in tok.vocab.items()}
        tok.merges = {}
        for key, token in payload["merges"].items():
            sep = "\u241F"
            assert sep in key, f"formato de merges invalido: {key!r}"
            a, b = key.split(sep, 1)
            tok.merges[(a, b)] = token
        return tok

    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    sample = (
        "def fibonacci(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
    )
    bt = BPETokenizer(vocab_size=512)
    bt.train(sample * 100, verbose=True)
    print(f"Vocab: {len(bt)} tokens")
    ids = bt.encode(sample)
    print("ids:", ids[:40])
    decoded = bt.decode(ids)
    print("decode:", repr(decoded))
    assert decoded == sample, f"round-trip falhou: {decoded!r} != {sample!r}"
    print("[ok] round-trip perfeito!")

    # Teste com acentos PT-BR
    texto_pt = "função ção coração órgão à ação ç"
    ids2 = bt.encode(texto_pt)
    decoded2 = bt.decode(ids2)
    print(f"PT-BR: {texto_pt!r} -> ids={ids2[:10]}... -> {decoded2!r}")
    assert decoded2 == texto_pt, f"PT-BR round-trip falhou: {decoded2!r} != {texto_pt!r}"
    print("[ok] PT-BR round-trip perfeito!")

    bt.save("/tmp/_test_tok.json")
    bt2 = BPETokenizer.load("/tmp/_test_tok.json")
    assert len(bt) == len(bt2)
    os.unlink("/tmp/_test_tok.json")
    print("[ok] persistencia OK!")
