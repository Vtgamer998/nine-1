#!/bin/bash
# NINE-1 Web UI - Script de inicializacao
# Uso: bash run_webui.sh [--share] [--port 7860]
#
# Nota: Requer torch + gradio instalados.
# No Termux (Android), use o Google Colab em vez disso:
#   https://colab.research.google.com/github/Vtgamer998/nine-1/blob/main/notebooks/train_nine1.ipynb

set -e

# Verifica dependencias criticas
MISSING=""
for mod in torch gradio; do
    if ! python -c "import $mod" 2>/dev/null; then
        MISSING="$MISSING $mod"
    fi
done

if [ -n "$MISSING" ]; then
    echo "[erro] Dependencias faltando:$MISSING"
    echo ""
    echo "Para rodar localmente (Linux/desktop):"
    echo "  pip install torch numpy gradio"
    echo ""
    echo "Para rodar no Google Colab (recomendado no celular):"
    echo "  Abra o notebook: notebooks/train_nine1.ipynb"
    echo "  Ou: https://colab.research.google.com/github/Vtgamer998/nine-1/blob/main/notebooks/train_nine1.ipynb"
    exit 1
fi

CKPT="nine/data/nine1-base.pt"
TOK="nine/data/nine1-tok.json"
LORA="nine/data/nine1-instruct.pt"

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
