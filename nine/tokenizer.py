"""
Tokenizer BPE (Byte Pair Encoding) implementado do zero em Python puro.
Usa codificacao byte-level (GPT-2 style) para suporte completo a Unicode,
incluindo acentos PT-BR, emojis, etc.

Inspirado em: Sennrich et al. 2016, GPT-2 tokenizer, nanoGPT (Karpathy).
Vocabulario base: 256 bytes + special tokens + merges aprendidos.

Seguranca:
- Validacao de tokens no decode (previne chr() fora de range)
- Protecao contra decode de IDs maliciosos
- Limite de tamanho de texto no encode
- Tratamento seguro de bytes/Unicode
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional, Set

# ---------------------------------------------------------------------------
# Constantes seguras
# ---------------------------------------------------------------------------

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

MAX_ENCODE_CHARS = 1_000_000  # Limite de seguranca para encode
MAX_DECODE_IDS = 100_000      # Limite de seguranca para decode

# Regex GPT-2 style para PT-BR (compativel com re padrao do Python).
# Usa ranges Unicode explicitos sem \\p{L} (nao suportado por re).
# Cobre: Latin-1 Supplement (\\xC0-\\xFF) e Latin Extended-A (\\u0100-\\u017F)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z_\xC0-\xFF\u0100-\u017F]+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+"""

# Pattern alternativo com Unicode properties (usado se regex lib estiver disponivel)
# NOTA: \p{L} NAO inclui underscore, entao adicionamos _ explicitamente
PAT_UNICODE = r""""(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|//[^\n]*|#[^\n]*|'(?:[sdmt]|ll|ve|re)| ?[\p{L}_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""


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


# ---------------------------------------------------------------------------
# Validacao de tokens
# ---------------------------------------------------------------------------

def validate_token_ids(ids: List[int], vocab_size: int) -> bool:
    """Valida se todos os IDs estao no range valido [0, vocab_size).

    Args:
        ids: Lista de IDs de tokens.
        vocab_size: Tamanho do vocabulario.

    Returns:
        True se todos os IDs sao validos.
    """
    if not ids:
        return False
    return all(0 <= i < vocab_size for i in ids)


def sanitize_filename_component(text: str) -> str:
    """Remove caracteres perigosos de nomes de arquivo para evitar path traversal.

    Args:
        text: Texto a sanitizar.

    Returns:
        Texto seguro para uso em nomes de arquivo.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", text)


# ---------------------------------------------------------------------------
# Tokenizer BPE
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Tokenizer BPE byte-level (GPT-2 style).

    - Vocabulario base: 256 bytes mapeados para Unicode visivel
    - Merges BPE aprendidos
    - Encode/decode com suporte total a Unicode
    - Validacao de seguranca nos tokens

    Uso:
        tok = BPETokenizer(vocab_size=4096)
        tok.train(corpus_text)
        ids = tok.encode("texto exemplo")
        text = tok.decode(ids)
        tok.save("tokenizer.json")
        tok2 = BPETokenizer.load("tokenizer.json")
    """

    def __init__(self, vocab_size: int = 8192, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        # Tenta usar regex com suporte a Unicode properties; fallback para re padrao
        self._pat_fallback = False
        try:
            import regex as _regex
            self.pat = _regex.compile(PAT_UNICODE)
        except ImportError:
            self.pat = re.compile(PAT)
            self._pat_fallback = True
        self._byte_encoder = _byte_encoder
        self._byte_decoder = _byte_decoder

    def _get_regex(self) -> re.Pattern:
        """Retorna regex compilado (com fallback para ambientes sem suporte a \\p{L})."""
        return self.pat

    # ---------- Treino ----------
    def train(self, text: str, verbose: bool = False):
        """Treina o tokenizer BPE a partir de um texto.

        Args:
            text: Texto de treinamento (codigo fonte, preferencialmente Python).
            verbose: Se True, exibe progresso.

        Raises:
            ValueError: Se texto vazio ou vocab_size muito pequeno.
        """
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
        pat = self._get_regex()
        words_raw = pat.findall(text)
        if not words_raw:
            raise ValueError("Nao foi possivel tokenizar o texto (regex pode ser incompativel).")

        word_freq: Counter = Counter()
        for w in words_raw:
            byte_list = list(w.encode("utf-8"))
            word = tuple(self._byte_encoder[b] for b in byte_list)
            word_freq[word] += 1

        target_merges = self.vocab_size - len(self.vocab)
        if target_merges < 1:
            raise ValueError(f"vocab_size {self.vocab_size} muito pequeno (min {len(self.vocab) + 1}).")

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

        # Trunca vocab ao tamanho definido
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
        """Codifica texto para IDs de tokens.

        Args:
            text: Texto a codificar.
            add_bos: Se True, adiciona token BOS no inicio.
            add_eos: Se True, adiciona token EOS no final.

        Returns:
            Lista de IDs de tokens.

        Raises:
            RuntimeError: Se tokenizer nao foi treinado/carregado.
            ValueError: Se texto excede limite de seguranca.
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer nao foi treinado ou carregado.")

        if len(text) > MAX_ENCODE_CHARS:
            raise ValueError(f"Text excede limite maximo de {MAX_ENCODE_CHARS} caracteres.")

        ids: List[int] = []
        if add_bos and BOS_TOKEN in self.token_to_id:
            ids.append(self.token_to_id[BOS_TOKEN])

        pat = self._get_regex()
        for chunk in pat.findall(text):
            if not chunk:
                continue
            # Converte para bytes, mapeia para Unicode visivel
            byte_list = list(chunk.encode("utf-8"))
            word = tuple(self._byte_encoder[b] for b in byte_list)

            # Aplica merges BPE
            for pair_key in sorted(self.merges.keys(),
                                   key=lambda p: self.token_to_id.get(self.merges[p], 0)):
                merged = self._apply_merge(word, pair_key, self.merges[pair_key])
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
        """Decodifica IDs de tokens para texto.

        Args:
            ids: Lista de IDs de tokens.

        Returns:
            String decodificada.

        Nota:
            A decodificacao para no token EOS se encontrado.
            Tokens desconhecidos sao ignorados silenciosamente.
        """
        if len(ids) > MAX_DECODE_IDS:
            ids = ids[:MAX_DECODE_IDS]

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
                byte_val = self._byte_decoder.get(ch)
                if byte_val is not None:
                    out_bytes.append(byte_val)

        if not out_bytes:
            return ""
        return bytes(out_bytes).decode("utf-8", errors="replace")

    # ---------- Utilitarios ----------
    def __len__(self):
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Retorna a string do token para um ID, ou None se invalido."""
        return self.vocab.get(token_id)

    # ---------- Persistencia ----------
    def save(self, path: str):
        """Salva tokenizer em arquivo JSON.

        Args:
            path: Caminho do arquivo .json.

        Raises:
            OSError: Se nao for possivel escrever.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serial_merges = {f"{p[0]}\u241F{p[1]}": t for p, t in self.merges.items()}
        payload = {
            "vocab_size": self.vocab_size,
            "merges": serial_merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
            "pat_fallback": self._pat_fallback,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Carrega tokenizer de arquivo JSON.

        Args:
            path: Caminho do arquivo .json.

        Returns:
            Instancia carregada do BPETokenizer.

        Raises:
            FileNotFoundError: Se arquivo nao existe.
            ValueError: Se formato do arquivo for invalido.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo tokenizer nao encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Arquivo tokenizer corrompido: {e}")

        required_keys = {"vocab_size", "merges", "vocab", "special_tokens"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Arquivo tokenizer faltando chaves: {missing}")

        tok = cls(
            vocab_size=payload["vocab_size"],
            special_tokens=payload["special_tokens"],
        )
        # Restaura fallback flag
        tok._pat_fallback = payload.get("pat_fallback", False)

        tok.vocab = {int(k): v for k, v in payload["vocab"].items()}
        tok.token_to_id = {v: k for k, v in tok.vocab.items()}
        tok.merges = {}
        for key, token in payload["merges"].items():
            sep = "\u241F"
            if sep not in key:
                raise ValueError(f"Formato de merges invalido: {key!r}")
            a, b = key.split(sep, 1)
            tok.merges[(a, b)] = token
        return tok


if __name__ == "__main__":
    # Teste basico
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
    texto_pt = "função ção coração órgão à ação ç ã õ á é í ó ú"
    ids2 = bt.encode(texto_pt)
    decoded2 = bt.decode(ids2)
    print(f"PT-BR: {texto_pt!r} -> ids={ids2[:10]}... -> {decoded2!r}")
    assert decoded2 == texto_pt, f"PT-BR round-trip falhou: {decoded2!r} != {texto_pt!r}"
    print("[ok] PT-BR round-trip perfeito!")

    # Teste de validacao
    assert validate_token_ids([0, 1, 2, 255], 512), "validacao falhou"
    assert not validate_token_ids([-1, 999], 512), "validacao de IDs ruins falhou"
    assert not validate_token_ids([], 512), "validacao de lista vazia falhou"
    print("[ok] Validacao de tokens OK!")

    # Teste de seguranca: decode seguro com IDs maliciosos
    safe = bt.decode([-1, 99999, 0])
    assert isinstance(safe, str), f"decode seguro falhou: {type(safe)}"
    print(f"[ok] Decode seguro com IDs ruins: {safe!r}")

    bt.save("/tmp/_test_tok.json")
    bt2 = BPETokenizer.load("/tmp/_test_tok.json")
    assert len(bt) == len(bt2)
    ids3 = bt2.encode(sample)
    assert ids == ids3, "persistencia mudou encode!"
    os.unlink("/tmp/_test_tok.json")
    print("[ok] Persistencia OK!")
    print("\n[Todos os testes do tokenizer passaram]")
