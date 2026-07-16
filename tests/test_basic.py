"""Tests do NINE-1 (v0.3.0 - seguranca)"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nine.tokenizer import BPETokenizer, validate_token_ids, sanitize_filename_component


# ============================================================================
# Tests do Tokenizer (NAO requerem torch)
# ============================================================================

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


def test_tokenizer_encode_empty():
    bt = BPETokenizer(vocab_size=300)
    bt.train("abc def ghi\n" * 10, verbose=False)
    ids = bt.encode("")
    assert isinstance(ids, list)
    ids_bos = bt.encode("", add_bos=True)
    assert isinstance(ids_bos, list)


def test_tokenizer_validate_ids():
    assert validate_token_ids([0, 1, 2], 512), "IDs validos falharam"
    assert not validate_token_ids([-1], 512), "ID negativo passou"
    assert not validate_token_ids([999999], 512), "ID muito grande passou"
    assert not validate_token_ids([], 512), "Lista vazia passou"


def test_tokenizer_decode_seguro():
    bt = BPETokenizer(vocab_size=300)
    bt.train("abc def\n" * 10, verbose=False)
    safe = bt.decode([-1, 999999, 0])
    assert isinstance(safe, str), f"decode seguro falhou: {type(safe)}"


def test_sanitize_filename():
    assert sanitize_filename_component("normal.txt") == "normal.txt"
    assert "/" not in sanitize_filename_component("a/b/c")
    assert "\\" not in sanitize_filename_component("a\\b")


def test_tokenizer_encode_limits():
    bt = BPETokenizer(vocab_size=300)
    bt.train("test " * 100, verbose=False)
    from nine.tokenizer import MAX_ENCODE_CHARS
    texto_grande = "x" * (MAX_ENCODE_CHARS + 1)
    try:
        bt.encode(texto_grande)
        assert False, "Deveria ter levantado ValueError"
    except ValueError:
        pass


# ============================================================================
# Tests do Modelo (requerem torch)
# ============================================================================

def _run_model_tests(has_torch: bool):
    """Roda testes do modelo apenas se torch estiver instalado."""
    if not has_torch:
        print("  [skip] torch nao instalado — todos os testes de modelo foram pulados")
        return

    import torch
    from nine.model import NINE1, tiny_config, small_config, validate_checkpoint_state
    from nine.dataset import TextDataset

    # ---- Model forward ----
    cfg = tiny_config(vocab_size=512, block_size=128)
    m = NINE1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = m(x, y)
    assert logits.shape == (2, 32, 512)
    assert loss.item() > 0
    print("[ok] test_model_forward")

    # ---- Model generate ----
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out = m.generate(x, max_new_tokens=20, temperature=1.0, top_k=10)
    assert out.shape == (1, 36)
    print("[ok] test_model_generate")

    # ---- Generate with cache ----
    out_cached = m.generate(x.clone(), max_new_tokens=20, temperature=1.0, top_k=10, use_cache=True)
    assert out_cached.shape == (1, 36)
    print("[ok] test_model_generate_with_cache")

    # ---- KV cache determinism ----
    torch.manual_seed(42)
    x1 = torch.randint(0, cfg.vocab_size, (1, 16))
    out_no_cache = m.generate(x1, max_new_tokens=10, temperature=0.1, top_k=1, use_cache=False)
    torch.manual_seed(42)
    x2 = torch.randint(0, cfg.vocab_size, (1, 16))
    out_cache = m.generate(x2, max_new_tokens=10, temperature=0.1, top_k=1, use_cache=True)
    assert torch.equal(out_no_cache, out_cache), "KV cache mudou a geracao!"
    print("[ok] test_model_kv_cache")

    # ---- Param count ----
    n = m.num_params()
    assert 1_000_000 < n < 100_000_000
    print("[ok] test_model_param_count")

    # ---- Configs ----
    for name, cfg_fn in [("tiny", tiny_config), ("small", small_config)]:
        c = cfg_fn()
        model = NINE1(c)
        inp = torch.randint(0, c.vocab_size, (1, 16))
        log, _, _ = model(inp)
        assert log is not None
        print(f"  {name}: {model.num_params()/1e6:.2f}M params OK")
    print("[ok] test_model_configs")

    # ---- Validate input ----
    assert m.validate_input_ids(torch.tensor([[0, 1, 2, 511]]))
    assert not m.validate_input_ids(torch.tensor([[-1, 0, 1]]))
    assert not m.validate_input_ids(torch.tensor([[0, 512, 1000]]))
    assert not m.validate_input_ids(torch.tensor([[0, 1]], dtype=torch.float32))
    print("[ok] test_model_validate_input")

    # ---- Temperature clamp ----
    out = m.generate(x.clone(), max_new_tokens=5, temperature=0.0, top_k=1)
    assert out.shape[1] >= 16
    print("[ok] test_model_generate_temperature_clamp")

    # ---- Checkpoint validation ----
    state = {"model": m.state_dict(), "cfg": cfg.__dict__}
    issues = validate_checkpoint_state(state["model"], cfg)
    assert not issues, f"Issues encontrados: {issues}"
    bad_state = dict(state["model"])
    for k in list(bad_state.keys())[:1]:
        bad_state[k] = torch.full_like(bad_state[k], float("nan"))
    bad_issues = validate_checkpoint_state(bad_state, cfg)
    nan_issues = [i for i in bad_issues if "NaN" in i]
    assert len(nan_issues) > 0, "Deveria ter detectado NaN"
    print("[ok] test_model_checkpoint_validation")

    # ---- Max tokens limit ----
    out1 = m.generate(x.clone(), max_new_tokens=-10, temperature=1.0, top_k=1)
    assert out1 is not None
    out2 = m.generate(x.clone(), max_new_tokens=9999, temperature=1.0, top_k=1)
    assert out2.shape[1] <= 16 + cfg.block_size
    print("[ok] test_model_generate_max_tokens_limit")

    # ---- Dataset ----
    import numpy as np
    fake_data = np.array(list(range(1000)), dtype=np.uint16)
    ds = TextDataset(fake_data, block_size=32)
    assert len(ds) > 0
    x_ds, y_ds = ds[0]
    assert x_ds.shape == (32,)
    assert y_ds.shape == (32,)
    print("[ok] test_text_dataset_basic")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=== NINE-1 Tests (v0.3.0) ===\n")

    # Tokenizer tests (NAO requerem torch)
    test_tokenizer_basic()
    print("[ok] test_tokenizer_basic")
    test_tokenizer_roundtrip()
    print("[ok] test_tokenizer_roundtrip")
    test_tokenizer_ptbr_accents()
    print("[ok] test_tokenizer_ptbr_accents")
    test_tokenizer_encode_empty()
    print("[ok] test_tokenizer_encode_empty")
    test_tokenizer_validate_ids()
    print("[ok] test_tokenizer_validate_ids")
    test_tokenizer_decode_seguro()
    print("[ok] test_tokenizer_decode_seguro")
    test_sanitize_filename()
    print("[ok] test_sanitize_filename")
    test_tokenizer_encode_limits()
    print("[ok] test_tokenizer_encode_limits")

    # Model tests (requerem torch)
    print()
    try:
        import torch  # noqa: F401
        _run_model_tests(True)
    except ImportError:
        _run_model_tests(False)

    print("\n=== Todos os testes disponiveis passaram! ===")
