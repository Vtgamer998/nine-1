#!/bin/bash
# NINE-1 Web UI - Script de inicializacao
# Uso: bash run_webui.sh [--share] [--port 7860]

CKPT="nine/data/nine1-base.pt"
TOK="nine/data/nine1-tok.json"
LORA="nine/data/nine1-instruct.pt"

# Verifica dependencias
if ! python -c "import gradio" 2>/dev/null; then
    echo "[setup] Instalando dependencias web (gradio)..."
    pip install gradio 2>&1 || {
        echo "[erro] Falha ao instalar gradio. Tente manualmente:"
        echo "  pip install gradio"
        exit 1
    }
fi

# Verifica dependencias opcionais (falha silenciosa)
if ! python -c "import safetensors" 2>/dev/null; then
    echo "[info] safetensors nao instalado (opcional - necessario para exportacao GGUF)"
fi

# Verifica checkpoint
if [ ! -f "$CKPT" ]; then
    echo "[aviso] Checkpoint base nao encontrado: $CKPT"
    echo "[aviso] Use --ckpt para especificar o caminho"
fi

echo "=== NINE-1 Web UI ==="
echo "Checkpoint: $CKPT"
echo "Tokenizer:  $TOK"
echo "LoRA:       $LORA"
echo ""

python -m nine.webui \
    --ckpt "$CKPT" \
    --tok "$TOK" \
    --lora "$LORA" \
    "$@"
