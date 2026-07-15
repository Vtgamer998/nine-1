"""
NINE-1 - IA de Programacao em PT-BR, construida do zero.

Melhorias v0.2:
  - Tokenizacao BPE byte-level (suporte completo a Unicode/PT-BR)
  - KV Cache na geracao (desempenho O(n) vs O(n^2))
  - LoRA seletivo (Q+KV apenas, target="qkv")
  - Modo chat interativo
  - Val split e metricas durante treino
  - Pipeline de tokenizacao unificado (BPE em treino + fine-tune)
"""

__version__ = "0.2.0"
__author__ = "NINE-1 project"
