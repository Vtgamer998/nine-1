"""
Dataset loader para o NINE-1.
Le arquivos de binario (memmap) ou texto puro, cria batches (input, target).

Seguranca:
- Validacao de tamanho e integridade dos dados
- Protecao contra indices invalidos
- Verificacao de dtype
"""

from __future__ import annotations
import os
from typing import Optional, Union

# Imports lazy: numpy e torch sao importados apenas quando usados
# (evita ModuleNotFoundError ao importar nine.data.seed sem torch/numpy)
_np = None
_torch = None


def _get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


# Constantes
MAX_DATASET_SIZE = 10_000_000_000  # 10B tokens max
MIN_CHUNK_SIZE = 16                # block_size minimo


class TextDataset:
    """
    Dataset sequencial deterministico sobre memmap .bin.
    Cada index retorna um chunk de block_size+1 tokens consecutivos.
    """

    def __init__(self, data, block_size: int, vocab_size: Optional[int] = None):
        from torch.utils.data import Dataset  # lazy

        if block_size < MIN_CHUNK_SIZE:
            raise ValueError(f"block_size {block_size} muito pequeno (min {MIN_CHUNK_SIZE})")

        self.block_size = block_size
        np = _get_np()

        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Arquivo de dados nao encontrado: {data}")
            file_size = os.path.getsize(data)
            if file_size == 0:
                raise ValueError(f"Arquivo de dados vazio: {data}")

            if data.endswith(".bin"):
                self.data = np.memmap(data, dtype=np.uint16, mode="r")
            else:
                with open(data, "r", encoding="utf-8") as f:
                    txt = f.read()
                if len(txt) > MAX_DATASET_SIZE:
                    raise ValueError(f"Texto muito grande: {len(txt)} chars")
                self.data = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        else:
            self.data = data

        # Validacao de tamanho
        if len(self.data) < block_size + 2:
            raise ValueError(
                f"Dataset muito pequeno: {len(self.data)} elementos "
                f"(precisa de pelo menos {block_size + 2})"
            )

        self.length = max(0, len(self.data) - block_size - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = idx % self.length
        if i < 0 or i + self.block_size + 1 > len(self.data):
            i = 0
        np = _get_np()
        torch = _get_torch()
        chunk = np.array(self.data[i: i + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_batch(data, block_size: int, batch_size: int, device: str = "cpu"):
    """Amostra um batch aleatorio do dataset binario.

    Args:
        data: np.memmap com os tokens.
        block_size: Tamanho do contexto.
        batch_size: Numero de exemplos no batch.
        device: Dispositivo alvo.

    Returns:
        (x, y): Tensores (batch_size, block_size).
    """
    np = _get_np()
    torch = _get_torch()
    n = len(data)
    max_start = n - block_size - 1
    if max_start < 1:
        raise ValueError(f"Dataset muito pequeno: {n} tokens, block_size={block_size}")

    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
