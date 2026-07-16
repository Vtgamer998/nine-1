# 🐉 NINE-1

> **IA de programação em PT-BR, construída do zero.**
> Do tokenizer BPE ao Transformer decoder — tudo implementado em PyTorch puro.

![status](https://img.shields.io/badge/status-v0.4.0--alpha-orange)
![python](https://img.shields.io/badge/python-3.10+-blue)
![license](https://img.shields.io/badge/license-MIT-green)
[![Test](https://github.com/Vtgamer998/nine-1/actions/workflows/test.yml/badge.svg)](https://github.com/Vtgamer998/nine-1/actions/workflows/test.yml)

---

## 📋 Índice

- [O que é](#o-que-é)
- [Quickstart](#quickstart)
- [Arquitetura](#arquitetura)
- [API](#api)
- [Segurança](#segurança)
- [Roadmap](#roadmap)
- [Desenvolvimento](#desenvolvimento)
- [Créditos](#créditos)

---

## 🎯 O que é

**NINE-1** é uma IA de geração de código **construída do zero** — do tokenizador BPE byte-level ao Transformer decoder — focada em **Python** e descrições em **português brasileiro**.

### Filosofia

- **Didático**: Código limpo, comentado, sem dependências mágicas
- **Leve**: Cabe em qualquer celular moderno (~10-50M params)
- **PT-BR**: Tokenizador com suporte total a acentos, ç, ã, õ, etc.
- **Seguro**: Validação de checkpoint, sanitização de input, proteção contra path traversal

### Exemplo

```bash
$ python -m nine.cli "escreva uma funcao fibonacci" --mode instruct --tokens 100

--- NINE-1 ---
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
--------------
```

---

## 🚀 Quickstart

### Setup local (Termux/Linux/Mac)

```bash
git clone https://github.com/Vtgamer998/nine-1
cd nine-1
pip install -r requirements.txt
```

### Processar dados seed + treinar BPE tokenizer

```bash
python -m nine.prep_data \
    --paths nine/data/seed \
    --out nine/data/corpus.txt \
    --train_bpe --vocab 4096 \
    --bin_out nine/data/corpus.bin \
    --tok_out nine/data/nine1-tok.json
```

### Treinar modelo (recomendado: Google Colab)

Abra o notebook: [`notebooks/train_nine1.ipynb`](notebooks/train_nine1.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vtgamer998/nine-1/blob/main/notebooks/train_nine1.ipynb)

Ou, se tiver GPU:

```bash
python -m nine.train \
    --data nine/data/corpus.bin \
    --tok nine/data/nine1-tok.json \
    --out nine/data/nine1-base.pt \
    --vocab 4096 --block_size 256 \
    --n_layer 6 --n_head 6 --n_kv_heads 4 --n_embd 384 \
    --batch_size 64 --max_iters 2000
```

### CLI

```bash
# Modo livre
python -m nine.cli "def soma(a, b):" --ckpt nine/data/nine1-base.pt --tokens 80

# Modo instruct (fine-tuned)
python -m nine.cli "crie uma classe Pilha" --mode instruct --ckpt nine/data/nine1-base.pt

# Modo chat interativo
python -m nine.cli --mode chat --ckpt nine/data/nine1-base.pt
```

### Web UI

```bash
pip install gradio
bash run_webui.sh --share
```

---

## 🏗️ Arquitetura

### Visão Geral

```
                 +-----------+
                 |  Tokens   |
                 +-----+-----+
                       |
                 +-----v-----+
                 |  WTE (emb)|
                 +-----+-----+
                       |
            +----------v----------+
            |   Dropout            |
            +----------+----------+
                       |
          +------------v------------+
          |  Block x N              |
          |  +------------------+   |
          |  | RMSNorm -> Attn  |   |
          |  | +-> RMSNorm->MLP |   |
          |  +------------------+   |
          +------------+------------+
                       |
                 +-----v-----+
                 |  RMSNorm   |
                 +-----+-----+
                       |
                 +-----v-----+
                 |  LM Head   |
                 +-----+-----+
                       |
                   logits
```

### Tokenizer BPE

- **Base**: 256 bytes mapeados para Unicode visível (GPT-2 style)
- **Special tokens**: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`
- **Regex**: GPT-2 style adaptado para PT-BR (acentos, ç, ã, õ)
- **Treino**: Algoritmo BPE padrão com contagem de pares
- **Segurança**: Limite de tamanho no encode/decode, validação de IDs

### Transformer Decoder

| Componente | Detalhes |
|------------|----------|
| **Positional Encoding** | RoPE (Rotary Position Embeddings) — default; fallback para Learned WPE |
| **Normalização** | RMSNorm (pré-norm) — mais estável que LayerNorm |
| **Atenção** | GQA (Grouped Query Attention) — Multi-head causal self-attention |
| **MLP** | Linear → GELU → Linear (expansão 4x) |
| **KV Cache** | Sim — geração O(n) em vez de O(n²) |
| **FlashAttention** | Automático via `torch.nn.functional.scaled_dot_product_attention` |

### Grouped Query Attention (GQA)

A NINE-1 usa **GQA** (Ainslie et al., 2023) para reduzir o tamanho do KV Cache sem perder qualidade:

- `n_head` queries, `n_kv_heads ≤ n_head` keys/values
- Cada head K/V é compartilhada por `n_head / n_kv_heads` queries
- KV Cache reduzido por fator de `n_head / n_kv_heads`

### LoRA (Fine-tuning)

Implementação própria de Low-Rank Adaptation (Hu et al., 2021):
- Congela pesos do modelo base
- Adiciona pares A/B low-rank nas matrizes Q, K, V (target: "qkv")
- `LoRALinear` com dropout e scaling `alpha / r`

---

## 📖 API

### Módulos

| Módulo | Descrição |
|--------|-----------|
| `nine.model` | `NINE1`, `NINEConfig`, `tiny_config()`, `small_config()`, `medium_config()` |
| `nine.tokenizer` | `BPETokenizer` — treino, encode, decode, save/load |
| `nine.train` | Pré-treinamento do zero |
| `nine.finetune` | Fine-tuning LoRA em dataset instrucional |
| `nine.dpo_train` | Alinhamento com Direct Preference Optimization |
| `nine.cli` | Interface de linha de comando (modos: base, instruct, chat) |
| `nine.webui` | Interface Gradio web |
| `nine.fuse` | Fusão do modelo base + LoRA para inferência |
| `nine.dataset` | `TextDataset`, `get_batch` — carregamento de dados |
| `nine.prep_data` | Pré-processamento de corpus e treino do tokenizer |
| `nine.to_gguf` | Conversor para GGUF + export SafeTensors |
| `nine.quantize` | Quantização float16 e int8 |

### NINEConfig

```python
@dataclass
class NINEConfig:
    vocab_size: int = 512
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_heads: int = 0       # 0 = MHA (mesmo que n_head)
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    mlp_ratio: float = 4.0
    use_rope: bool = True
```

### Configs pré-definidas

| Config | Params | n_layer | n_head | n_kv | n_embd | ctx |
|--------|--------|---------|--------|------|--------|-----|
| `tiny_config()` | ~10M | 6 | 6 | 4 | 384 | 256 |
| `small_config()` | ~30M | 10 | 8 | 4 | 512 | 512 |
| `medium_config()` | ~100M | 16 | 12 | 4 | 768 | 1024 |

---

## 🛡️ Segurança

A NINE-1 implementa várias camadas de segurança:

### Checkpoint
- `validate_checkpoint_state()` — detecta NaN/Inf em tensores
- `safe_load_checkpoint()` — usa `weights_only=True` do PyTorch
- Verificação de chaves mínimas e dimensões esperadas

### Input
- `sanitize_prompt()` — remove caracteres de controle, limita tamanho
- `validate_input_ids()` — previne tokens fora do vocabulário
- `validate_ckpt_path()` — proteção contra path traversal

### Tokenizer
- `validate_token_ids()` — range check de IDs
- `sanitize_filename_component()` — remove caracteres perigosos
- Limites de tamanho em encode (`MAX_ENCODE_CHARS`) e decode (`MAX_DECODE_IDS`)

### Geração
- Clamping de temperatura (evita divisão por zero)
- Limite de `max_new_tokens` por `block_size`
- Repetition penalty configurável

---

## 🗺️ Roadmap

### Concluído ✅

- [x] Tokenizer BPE byte-level com suporte PT-BR
- [x] Transformer decoder com RoPE + RMSNorm
- [x] KV Cache para geração eficiente
- [x] GQA (Grouped Query Attention)
- [x] LoRA (implementação própria)
- [x] CLI em PT-BR (modos base, instruct, chat)
- [x] Validação de segurança em checkpoints e inputs
- [x] Gradient accumulation no treino
- [x] Interface Gradio web
- [x] Conversor GGUF + export SafeTensors
- [x] DPO (Direct Preference Optimization)
- [x] Google Colab notebook
- [x] CI via GitHub Actions

### Próximos 🔄

- [ ] Avaliação em HumanEval-pt
- [ ] Suporte a loading de checkpoints GGUF
- [ ] Attention Sink / Sliding Window para contexto maior
- [ ] Dataset de instruções PT-BR maior
- [ ] App Android nativo (via Chaquopy ou Termux)

### Futuro 🔮

- [ ] MoE (Mixture of Experts) leve
- [ ] Quantização INT4/INT8 on-the-fly
- [ ] Fine-tuning com QLoRA
- [ ] Suporte a mais linguagens de programação

---

## 🔧 Desenvolvimento

### Estrutura do projeto

```
nine-1/
├── nine/
│   ├── __init__.py         # v0.4.0
│   ├── _pat.py             # Regex fallback patterns
│   ├── model.py            # NINE1, NINEConfig, GQA, RoPE
│   ├── tokenizer.py        # BPETokenizer byte-level
│   ├── train.py            # Pré-treinamento
│   ├── finetune.py         # LoRA fine-tuning
│   ├── dpo_train.py        # DPO alignment
│   ├── cli.py              # CLI PT-BR
│   ├── webui.py            # Gradio interface
│   ├── fuse.py             # Model fusion (base + LoRA)
│   ├── dataset.py          # TextDataset, get_batch
│   ├── prep_data.py        # Corpus preprocessing
│   ├── to_gguf.py          # GGUF converter
│   ├── quantize.py         # Quantization
│   └── data/
│       ├── seed/           # Seed datasets (31 arquivos)
│       ├── instruct_seed.py
│       └── synthetic_examples.py
├── notebooks/
│   └── train_nine1.ipynb   # Colab training notebook
├── tests/
│   └── test_basic.py       # 18 testes (8 sem torch)
├── .github/workflows/
│   └── test.yml            # CI
├── requirements.txt
├── requirements-web.txt
├── run_webui.sh
└── README.md
```

### Testes

```bash
# Todos os testes (requer torch)
python -m pytest tests/test_basic.py -v

# Só tokenizer (sem torch)
python tests/test_basic.py

# Verificação de sintaxe
for f in nine/*.py tests/test_basic.py; do python -m py_compile "$f"; done
```

### CI/CD

GitHub Actions roda em todo push para `main`:
- Python 3.10, 3.11, 3.12
- Testes do tokenizer
- Verificação de sintaxe de todos os módulos

---

## 📜 Créditos

- **nanoGPT** (Andrej Karpathy) — inspiração para arquitetura e loop de treino
- **GPT-2** (Radford et al.) — BPE byte-level e pré-norm Transformer
- **LoRA** (Hu et al., Microsoft Research) — low-rank adaptation
- **GQA** (Ainslie et al., Google) — grouped query attention
- **RoPE** (Su et al.) — rotary position embeddings

## 📄 Licença

MIT — use, modifique, distribua. Atribuição apreciada.
