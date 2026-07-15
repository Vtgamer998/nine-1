"""
Dataset loader para o NINE-1.
Le arquivos de binario (memmap) ou texto puro, cria batches (input, target).
"""

from __future__ import annotations
import os
import random
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class TextDataset(Dataset):
    """
    Dataset sequencial deterministico sobre memmap .bin.
    Cada index retorna um chunk de block_size+1 tokens consecutivos.
    """

    def __init__(self, data: Union[np.memmap, np.ndarray, str],
                 block_size: int, vocab_size: Optional[int] = None):
        self.block_size = block_size
        if isinstance(data, str):
            if data.endswith(".bin"):
                self.data = np.memmap(data, dtype=np.uint16, mode="r")
            else:
                with open(data, "r", encoding="utf-8") as f:
                    txt = f.read()
                self.data = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        else:
            self.data = data
        self.length = max(0, len(self.data) - block_size - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = idx % self.length
        chunk = np.array(self.data[i : i + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str = "cpu"):
    """Amostra um batch aleatorio do dataset binario."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
