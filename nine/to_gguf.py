"""
NINE-1 to GGUF Converter

Converte checkpoint .pt do NINE-1 para formato GGUF, compatível com
llama.cpp para inferência em C++.

Formato GGUF (segundo spec do llama.cpp/ggml):
[header] [metadata KV pairs] [tensor data (aligned)]

Uso:
    python -m nine.to_gguf nine/data/nine1-base.pt --out model.gguf
    python -m nine.to_gguf nine/data/nine1-base.pt --out model-q4.gguf --quantize q4_0

Dependencias opcionais para quantizacao:
    pip install numpy
"""

from __future__ import annotations
import argparse
import json
import os
import struct
import sys
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any

import torch

# ---------------------------------------------------------------------------
# Constantes GGUF (little-endian)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" em little-endian
GGUF_VERSION = 3

# Tipos de dados GGUF
class GGUFType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# Tipos de tensores GGUF (quantizacao)
class GGUFTensorType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9

# Block sizes e tamanhos para quantizacao
QUANT_BLOCK_SIZES = {
    GGUFTensorType.F32: 1,
    GGUFTensorType.F16: 1,
    GGUFTensorType.Q4_0: 32,
    GGUFTensorType.Q4_1: 32,
    GGUFTensorType.Q5_0: 32,
    GGUFTensorType.Q5_1: 32,
    GGUFTensorType.Q8_0: 32,
}

QUANT_TYPE_SIZES = {
    GGUFTensorType.F32: 4,
    GGUFTensorType.F16: 2,
    GGUFTensorType.Q4_0: 18,   # 2 bytes scale + 16 bytes quants
    GGUFTensorType.Q4_1: 20,   # 2+2 bytes scales + 16 bytes quants
    GGUFTensorType.Q5_0: 22,   # 2 bytes scale + 20 bytes quants
    GGUFTensorType.Q5_1: 24,   # 2+2 bytes scales + 20 bytes quants
    GGUFTensorType.Q8_0: 34,   # 2 bytes scale + 32 bytes quants
}

# Arquitetura custom "nne1" para NINE-1
# NOTA: Isso requer adicionar suporte no llama.cpp
# Alternativa melhor: exportar pra SafeTensors e usar convert_hf_to_gguf.py
ARCHITECTURE = "nne1"

# Mapeamento de nomes de tensores NINE-1 -> nomes normalizados
TENSOR_NAME_MAP = {
    "transformer.wte.weight": "token_embd.weight",
    "transformer.ln_f.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def _tensor_name_gguf(name: str, n_layer: int) -> str:
    """Mapeia nome interno do NINE-1 para nome GGUF.

    Arquitetura NINE-1 (nne1):
    - MLP simples: ffn_expand -> GELU -> ffn_contract
      (diferente de Llama que tem ffn_gate/ffn_up/ffn_down)
    """
    if name in TENSOR_NAME_MAP:
        return TENSOR_NAME_MAP[name]

    # Blocos transformer
    for i in range(n_layer):
        prefix = f"transformer.h.{i}."
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            blk_prefix = f"blk.{i}."
            if suffix == "attn.c_attn.weight":
                return blk_prefix + "attn_qkv.weight"
            if suffix == "attn.c_proj.weight":
                return blk_prefix + "attn_output.weight"
            if suffix == "mlp.c_fc.weight":
                # MLP expand (sem gating, diferente de Llama)
                return blk_prefix + "ffn_expand.weight"
            if suffix == "mlp.c_proj.weight":
                return blk_prefix + "ffn_contract.weight"
            if suffix == "ln_1.weight":
                return blk_prefix + "attn_norm.weight"
            if suffix == "ln_2.weight":
                return blk_prefix + "ffn_norm.weight"
            # Bias (se houver)
            if suffix == "attn.c_attn.bias":
                return blk_prefix + "attn_qkv.bias"
            if suffix == "attn.c_proj.bias":
                return blk_prefix + "attn_output.bias"
            if suffix == "mlp.c_fc.bias":
                return blk_prefix + "ffn_expand.bias"
            if suffix == "mlp.c_proj.bias":
                return blk_prefix + "ffn_contract.bias"
    return name


# ---------------------------------------------------------------------------
# Funcoes de serializacao GGUF
# ---------------------------------------------------------------------------

def _write_padding(f, alignment: int):
    """Escreve padding zero para alinhamento."""
    pos = f.tell()
    pad = (alignment - (pos % alignment)) % alignment
    if pad > 0:
        f.write(b"\x00" * pad)


def _write_uint8(f, val: int):
    f.write(struct.pack("<B", val & 0xFF))


def _write_int32(f, val: int):
    f.write(struct.pack("<i", val))


def _write_uint32(f, val: int):
    f.write(struct.pack("<I", val & 0xFFFFFFFF))


def _write_uint64(f, val: int):
    f.write(struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))


def _write_float32(f, val: float):
    f.write(struct.pack("<f", val))


def _write_str(f, s: str):
    encoded = s.encode("utf-8")
    _write_uint64(f, len(encoded))
    f.write(encoded)


def _write_kv(f, key: str, value, gguf_type: int):
    """Escreve um par chave-valor de metadado GGUF."""
    _write_str(f, key)
    _write_int32(f, gguf_type)
    if gguf_type == GGUFType.UINT32:
        _write_uint32(f, value)
    elif gguf_type == GGUFType.INT32:
        _write_int32(f, value)
    elif gguf_type == GGUFType.UINT64:
        _write_uint64(f, value)
    elif gguf_type == GGUFType.FLOAT32:
        _write_float32(f, value)
    elif gguf_type == GGUFType.STRING:
        _write_str(f, value)
    elif gguf_type == GGUFType.BOOL:
        _write_uint8(f, 1 if value else 0)
    elif gguf_type == GGUFType.ARRAY:
        # Array: type, length, elements
        _write_int32(f, value.get("type", GGUFType.UINT32))
        elements = value.get("elements", [])
        _write_uint64(f, len(elements))
        for el in elements:
            _write_uint64(f, el & 0xFFFFFFFFFFFFFFFF)
    else:
        raise ValueError(f"Tipo GGUF nao suportado: {gguf_type}")


def _quantize_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantizacao Q4_0: 1 scale f16 + 16 int4 quants por bloco de 32."""
    arr = tensor.detach().cpu().float().numpy().flatten()
    n = len(arr)
    block_size = 32
    out = bytearray()

    for i in range(0, n, block_size):
        block = arr[i:i + block_size]
        amax = max(abs(float(x)) for x in block)
        if amax == 0:
            scale = 0.0
            quants = [0] * block_size
        else:
            scale = amax / 7.0  # Q4 range: [-7, 7]
            quants = [max(-7, min(7, int(round(x / scale)))) for x in block]
        out += struct.pack("<e", float(scale))  # half precision
        # Empacota 2 int4 por byte
        for j in range(0, block_size, 2):
            val = ((quants[j + 1] & 0x0F) << 4) | (quants[j] & 0x0F)
            out.append(val)
    return bytes(out)


def _quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """Quantizacao Q8_0: 1 scale f16 + 32 int8 quants por bloco de 32."""
    arr = tensor.detach().cpu().float().numpy().flatten()
    n = len(arr)
    block_size = 32
    out = bytearray()

    for i in range(0, n, block_size):
        block = arr[i:i + block_size]
        amax = max(abs(float(x)) for x in block)
        if amax == 0:
            scale = 0.0
            quants = [0] * block_size
        else:
            scale = amax / 127.0
            quants = [max(-127, min(127, int(round(x / scale)))) for x in block]
        out += struct.pack("<e", float(scale))
        for q in quants:
            out.append(q & 0xFF)
    return bytes(out)


# ---------------------------------------------------------------------------
# Exportacao alternativa: SafeTensors (recomendado)
# ---------------------------------------------------------------------------

def export_to_safetensors(
    state: Dict[str, torch.Tensor],
    metadata: List,
    output_path: str,
    verbose: bool = False,
):
    """Exporta para formato SafeTensors (Hugging Face).

    Este e o formato RECOMENDADO para conversao para GGUF:
    1. Exportar para .safetensors + config.json
    2. Usar convert_hf_to_gguf.py do llama.cpp

    Args:
        state: State dict do modelo.
        metadata: Lista de metadados (para config.json).
        output_path: Caminho de saida (.safetensors).
        verbose: Logs detalhados.
    """
    try:
        from safetensors.torch import save_file as st_save
        HAS_SAFETENSORS = True
    except ImportError:
        HAS_SAFETENSORS = False

    if not HAS_SAFETENSORS:
        print("[gguf] AVISO: safetensors nao instalado.")
        print("[gguf]   pip install safetensors")
        print("[gguf] Salvando como .pt mesmo...")
        torch.save(state, output_path.replace(".safetensors", ".pt"))
        return

    if verbose:
        print(f"[gguf] Exportando {len(state)} tensores para SafeTensors...")

    # Converte para float16 se possivel
    state_f16 = {}
    for name, tensor in state.items():
        if tensor.dtype == torch.float32 and tensor.numel() > 1000:
            state_f16[name] = tensor.half().contiguous()
        else:
            state_f16[name] = tensor.contiguous()

    # Salva SafeTensors
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    st_save(state_f16, output_path)

    # Cria config.json para o modelo
    cfg_out = output_path.replace(".safetensors", "")
    cfg_dict = {
        "architectures": ["NINE1ForCausalLM"],
        "model_type": "nne1",
        "hidden_size": None,
        "num_hidden_layers": None,
        "num_attention_heads": None,
        "vocab_size": None,
        "max_position_embeddings": None,
        "hidden_act": "gelu",
        "use_rope": True,
    }
    for key, val, typ in metadata:
        key_clean = key.split(".", 1)[-1] if "." in key else key
        if "context_length" in key:
            cfg_dict["max_position_embeddings"] = val
        elif "embedding_length" in key:
            cfg_dict["hidden_size"] = val
        elif "block_count" in key:
            cfg_dict["num_hidden_layers"] = val
        elif "head_count" in key and "kv" not in key:
            cfg_dict["num_attention_heads"] = val
        elif "vocab_size" in key:
            cfg_dict["vocab_size"] = val

    with open(cfg_out + "_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[gguf] SafeTensors salvo: {output_path} ({size_mb:.2f} MB)")
    print(f"[gguf] Config: {cfg_out}_config.json")
    print(f"[gguf] Para converter para GGUF:")
    print(f"[gguf]   git clone https://github.com/ggerganov/llama.cpp")
    print(f"[gguf]   cd llama.cpp && pip install -r requirements.txt")
    print(f"[gguf]   python convert_hf_to_gguf.py {cfg_out} --outfile model.gguf")


# ---------------------------------------------------------------------------
# Conversor principal
# ---------------------------------------------------------------------------

def convert_to_gguf(
    model_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
    quantize: Optional[str] = None,
    verbose: bool = False,
):
    """Converte checkpoint NINE-1 para formato GGUF.

    Args:
        model_path: Caminho do checkpoint .pt.
        output_path: Caminho de saida .gguf.
        tokenizer_path: Caminho opcional do tokenizer BPE .json.
        quantize: Tipo de quantizacao ('q4_0', 'q8_0', None=F16).
        verbose: Logs detalhados.
    """
    if verbose:
        print(f"[gguf] Carregando checkpoint: {model_path}")

    # Carrega checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", ckpt.get("config", {}))
    state = ckpt.get("model", ckpt)

    # Extrai config
    vocab_size = cfg_dict.get("vocab_size", 512)
    block_size = cfg_dict.get("block_size", 256)
    n_layer = cfg_dict.get("n_layer", 6)
    n_head = cfg_dict.get("n_head", 6)
    n_embd = cfg_dict.get("n_embd", 384)
    use_rope = cfg_dict.get("use_rope", True)

    if verbose:
        print(f"[gguf] Arquitetura: {n_layer} camadas, {n_head} cabecas, {n_embd} dim")
        print(f"[gguf] Vocab: {vocab_size}, Contexto: {block_size}")

    # Determina dtype base dos tensores
    sample_tensor = next(iter(state.values()))
    base_dtype = GGUFTensorType.F16 if sample_tensor.dtype == torch.float16 else GGUFTensorType.F32

    # Define tipo de quantizacao
    if quantize:
        qtype_map = {
            "q4_0": GGUFTensorType.Q4_0,
            "q8_0": GGUFTensorType.Q8_0,
        }
        tensor_type = qtype_map.get(quantize.lower(), base_dtype)
        if verbose:
            print(f"[gguf] Quantizacao: {quantize.upper()}")
    else:
        tensor_type = base_dtype
        if verbose:
            print(f"[gguf] Dtype: {'F16' if base_dtype == GGUFTensorType.F16 else 'F32'}")

    # Ordena tensores (consistencia)
    tensor_names = sorted(state.keys())
    n_tensors = len(tensor_names)

    # Prepara metadados
    metadata: List[Tuple[str, Any, int]] = [
        ("general.name", "NINE-1", GGUFType.STRING),
        ("general.version", "0.3.0", GGUFType.STRING),
        ("general.architecture", ARCHITECTURE, GGUFType.STRING),
        ("general.file_type", int(tensor_type), GGUFType.UINT32),
        (f"{ARCHITECTURE}.context_length", block_size, GGUFType.UINT32),
        (f"{ARCHITECTURE}.embedding_length", n_embd, GGUFType.UINT32),
        (f"{ARCHITECTURE}.block_count", n_layer, GGUFType.UINT32),
        (f"{ARCHITECTURE}.head_count", n_head, GGUFType.UINT32),
        (f"{ARCHITECTURE}.head_count_kv", n_head, GGUFType.UINT32),
        (f"{ARCHITECTURE}.vocab_size", vocab_size, GGUFType.UINT32),
        (f"{ARCHITECTURE}.rope.dimension_count", n_embd // n_head, GGUFType.UINT32),
    ]

    if use_rope:
        metadata.append((f"{ARCHITECTURE}.rope.freq_base", 10000.0, GGUFType.FLOAT32))
    if cfg_dict.get("bias", False):
        metadata.append((f"{ARCHITECTURE}.has_bias", True, GGUFType.BOOL))

    # Carrega tokenizer se disponivel
    tokenizer_data = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
            if verbose:
                n_tokens = len(tokenizer_data.get("vocab", {}))
                print(f"[gguf] Tokenizer carregado: {n_tokens} tokens")
        except Exception as e:
            print(f"[gguf] Aviso: falha ao ler tokenizer: {e}")

    # -----------------------------------------------------------------------
    # Escrita do arquivo GGUF
    # -----------------------------------------------------------------------
    alignment = 32  # Alinhamento padrao GGUF

    # AVISO sobre o formato custom
    print(f"[gguf] AVISO: Usando arquitetura custom '{ARCHITECTURE}'.")
    print(f"[gguf] Para usar com llama.cpp, exporte para SafeTensors:")
    print(f"[gguf]   python -m nine.to_gguf {model_path} --out model")
    print(f"[gguf]   cd llama.cpp && python convert_hf_to_gguf.py ./model --outfile model.gguf")

    if quantize is None:
        # Modo padrao: exporta SafeTensors (recomendado)
        safetensors_path = output_path.replace(".gguf", ".safetensors")
        export_to_safetensors(state, metadata, safetensors_path, verbose)
        return

    # Modo GGUF direto (experimental, requer suporte no llama.cpp)
    with open(output_path, "wb") as f:
        # --- HEADER ---
        f.write(struct.pack("<I", GGUF_MAGIC))
        _write_uint32(f, GGUF_VERSION)
        _write_uint64(f, n_tensors)
        _write_uint64(f, len(metadata))

        # --- METADATA KV ---
        for key, value, gguf_type in metadata:
            _write_kv(f, key, value, gguf_type)

        # Adiciona info do tokenizer (simplificado)
        if tokenizer_data:
            token_list = [
                tokenizer_data["vocab"][str(tid)]
                for tid in sorted(int(k) for k in tokenizer_data.get("vocab", {}).keys())
            ]
            _write_kv(f, "tokenizer.ggml.model", "gpt2", GGUFType.STRING)
            if verbose:
                print(f"[gguf] Tokenizer: {len(token_list)} tokens")

        # --- TENSOR INFO + DATA ---
        if verbose:
            print(f"[gguf] Escrevendo {n_tensors} tensores...")

        # Calcula info dos tensores
        tensor_info_offsets = []
        for name in tensor_names:
            mapped_name = _tensor_name_gguf(name, n_layer)
            tensor = state[name]
            shape = list(tensor.shape)
            n_elements = tensor.numel()
            n_dims = len(shape)

            # Calcula tamanho do tensor
            if tensor_type == GGUFTensorType.Q4_0:
                n_blocks = (n_elements + 31) // 32
                data_size = n_blocks * 18
            elif tensor_type == GGUFTensorType.Q8_0:
                n_blocks = (n_elements + 31) // 32
                data_size = n_blocks * 34
            elif tensor_type == GGUFTensorType.F16:
                data_size = n_elements * 2
            else:
                data_size = n_elements * 4

            tensor_info_offsets.append((mapped_name, n_dims, shape, tensor_type, data_size))

        # Escreve info de cada tensor, rastreando posicoes dos offsets
        offset_positions = []
        for name, n_dims, shape, tt, _ in tensor_info_offsets:
            _write_str(f, name)
            _write_uint32(f, n_dims)
            if n_dims == 1:
                _write_uint64(f, shape[0])
            elif n_dims == 2:
                _write_uint64(f, shape[1])
                _write_uint64(f, shape[0])
            else:
                _write_uint64(f, shape[0])
            _write_uint32(f, int(tt))
            offset_positions.append(f.tell())
            _write_uint64(f, 0)  # Placeholder offset

        # Alinha ao comeco dos dados
        _write_padding(f, alignment)
        data_start = f.tell()

        # Prepara e escreve dados
        current_offset = 0
        for i, name in enumerate(tensor_names):
            tensor = state[name]
            if tensor_type == GGUFTensorType.Q4_0:
                data_bytes = _quantize_q4_0(tensor)
            elif tensor_type == GGUFTensorType.Q8_0:
                data_bytes = _quantize_q8_0(tensor)
            elif tensor_type == GGUFTensorType.F16:
                data_bytes = tensor.detach().half().numpy().tobytes()
            else:
                data_bytes = tensor.detach().float().numpy().tobytes()

            # Atualiza offset placeholder
            f.seek(offset_positions[i])
            _write_uint64(f, current_offset)

            # Escreve dados
            f.seek(data_start + current_offset)
            f.write(data_bytes)
            current_offset += len(data_bytes)
            # Alinhamento
            current_offset = (current_offset + alignment - 1) // alignment * alignment

        file_size = f.tell()

    if verbose:
        size_mb = file_size / 1024 / 1024
        print(f"\n[gguf] Arquivo salvo: {output_path}")
        print(f"[gguf] Tamanho: {size_mb:.2f} MB")
        print(f"[gguf] Formato: {tensor_type.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="NINE-1 to GGUF Converter",
    )
    p.add_argument("model_path", type=str, help="Checkpoint .pt do NINE-1")
    p.add_argument("--out", type=str, default=None, help="Arquivo .gguf de saida")
    p.add_argument("--tok", type=str, default=None, help="Tokenizer BPE .json")
    p.add_argument("--quantize", type=str, choices=["q4_0", "q8_0", None],
                   default=None, help="Tipo de quantizacao")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.model_path))[0]
        qsuffix = f"-{args.quantize}" if args.quantize else ""
        args.out = f"{base}{qsuffix}.gguf"

    convert_to_gguf(
        model_path=args.model_path,
        output_path=args.out,
        tokenizer_path=args.tok,
        quantize=args.quantize,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
