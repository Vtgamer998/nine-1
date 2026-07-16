"""
NINE-1 - IA de Programacao em PT-BR, construida do zero.

v0.4.0 - GQA & CI & Documentacao
  - GQA (Grouped Query Attention) para KV Cache 2-3x menor
  - GitHub Actions CI com testes em Python 3.10-3.12
  - Documentacao completa da arquitetura, API e seguranca
  - Compatibilidade retroativa com checkpoints pre-GQA (c_attn)

v0.3.0 - Seguranca & RoPE
  - RoPE (Rotary Position Embeddings) para melhor generalizacao posicional
  - Validacao de checkpoint com deteccao de NaN/Inf
  - Carregamento seguro com weights_only
  - Sanitizacao de input do usuario na CLI
  - Protecao contra path traversal em operacoes de arquivo
  - Gradient accumulation no treino
  - Limites de seguranca em decode/encode do tokenizer
  - Testes de seguranca adicionados

v0.2.0:
  - Tokenizacao BPE byte-level (suporte completo a Unicode/PT-BR)
  - KV Cache na geracao (desempenho O(n) vs O(n^2))
  - LoRA seletivo (Q+KV apenas, target="qkv")
  - Modo chat interativo
  - Val split e metricas durante treino
  - Pipeline de tokenizacao unificado (BPE em treino + fine-tune)
"""

__version__ = "0.4.0"
__author__ = "NINE-1 project"
