"""Tests do NINE-1"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nine.tokenizer import BPETokenizer
from nine.model import NINE1, tiny_config


def test_tokenizer_basic():
    bt = BPETokenizer(vocab_size=300)
    bt.train("def soma a b ;\n    return a b + ;\n" * 30, verbose=False)
    assert len(bt) > 0
    ids = bt.encode("def soma a,b; return a+b")
    assert len(ids) > 0
    text = bt.decode(ids)
    assert isinstance(text, str)
    bt.save(tempfile.NamedTemporaryFile(delete=False, suffix=".json").name)


def test_tokenizer_roundtrip():
    bt = BPETokenizer(vocab_size=300)
    bt.train("hello world\nola mundo\nfunção coração\n" * 30, verbose=False)
    ids = bt.encode("ola mundo função")
    text = bt.decode(ids)
    assert "ola" in text and "mundo" in text
    assert "função" in text


def test_tokenizer_ptbr_accents():
    bt = BPETokenizer(vocab_size=300)
    bt.train("coração órgão à ação ç ã õ á é í ó ú\n" * 30, verbose=False)
    texto = "coração à ação"
    ids = bt.encode(texto)
    text = bt.decode(ids)
    assert texto == text, f"PT-BR round-trip falhou: {text!r} != {texto!r}"


def test_model_forward():
    cfg = tiny_config(vocab_size=512, block_size=128)
    m = NINE1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = m(x, y)
    assert logits.shape == (2, 32, 512)
    assert loss.item() > 0


def test_model_generate():
    cfg = tiny_config(vocab_size=512, block_size=128)
    m = NINE1(cfg)
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out = m.generate(x, max_new_tokens=20, temperature=1.0, top_k=10)
    assert out.shape == (1, 36)


def test_model_generate_with_cache():
    cfg = tiny_config(vocab_size=512, block_size=128)
    m = NINE1(cfg)
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out = m.generate(x, max_new_tokens=20, temperature=1.0, top_k=10, use_cache=True)
    assert out.shape == (1, 36)


def test_model_kv_cache():
    """KV Cache produz mesmo resultado que sem cache."""
    cfg = tiny_config(vocab_size=512, block_size=128)
    m = NINE1(cfg)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out_no_cache = m.generate(x, max_new_tokens=10, temperature=0.1, top_k=1, use_cache=False)

    torch.manual_seed(42)
    x2 = torch.randint(0, cfg.vocab_size, (1, 16))
    out_cache = m.generate(x2, max_new_tokens=10, temperature=0.1, top_k=1, use_cache=True)

    assert torch.equal(out_no_cache, out_cache), "KV cache mudou a geracao!"


def test_model_param_count():
    cfg = tiny_config()
    m = NINE1(cfg)
    n = m.num_params()
    assert 1_000_000 < n < 100_000_000


if __name__ == "__main__":
    test_tokenizer_basic()
    test_tokenizer_roundtrip()
    test_tokenizer_ptbr_accents()
    test_model_forward()
    test_model_generate()
    test_model_generate_with_cache()
    test_model_kv_cache()
    test_model_param_count()
    print("[ok] Todos os testes basicos passaram.")
