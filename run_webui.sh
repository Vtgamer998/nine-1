#!/bin/bash
# NINE-1 Web UI - Script de inicializacao
# Uso: bash run_webui.sh [--simple|--gradio] [--port 7860] [--share] [args...]
#
# --simple (padrao): Usa webui_simple.py (stdlib Python, zero dependencias)
# --gradio:         Usa webui.py (Gradio, requer pip install gradio)
#
# No Termux (Android), a opcao --simple funciona sem instalacao extra!

set -e

MODE="simple"
CUSTOM_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --simple) MODE="simple" ;;
        --gradio) MODE="gradio" ;;
        *) CUSTOM_ARGS+=("$arg") ;;
    esac
done

# Auto-detect checkpoint (prefere -g grande)
CKPT_BASE="nine/data/nine1-base.pt"
CKPT_G="nine/data/nine1-base-g.pt"
if [ -f "$CKPT_G" ] && [ $(stat -c%s "$CKPT_G" 2>/dev/null || echo 0) -gt 50000000 ]; then
    CKPT="$CKPT_G"
else
    CKPT="$CKPT_BASE"
fi

TOK_BASE="nine/data/nine1-tok.json"
TOK_G="nine/data/nine1-tok-g.json"
[ -f "$TOK_G" ] && TOK="$TOK_G" || TOK="$TOK_BASE"

LORA_BASE="nine/data/nine1-instruct.pt"
LORA_G="nine/data/nine1-lora-g.pt"
[ -f "$LORA_G" ] && LORA="$LORA_G" || LORA="$LORA_BASE"
[ ! -f "$LORA" ] && LORA=""

echo "=== NINE-1 Web UI ==="
echo "Modo:       $MODE"
echo "Checkpoint: $CKPT"
echo "Tokenizer:  $TOK"
[ -n "$LORA" ] && echo "LoRA:       $LORA"
echo ""

if [ "$MODE" = "gradio" ]; then
    if ! python -c "import gradio" 2>/dev/null; then
        echo "[setup] Instalando gradio..."
        pip install gradio 2>&1 | tail -3
    fi
    echo "[Dica] Use prompts no formato # tarefa / # solucao"
    echo ""
    exec python -m nine.webui \
        --ckpt "$CKPT" --tok "$TOK" --lora "$LORA" \
        "${CUSTOM_ARGS[@]}"
else
    echo "[Dica] Use --gradio para interface Gradio (requer instalacao extra)"
    echo ""
    exec python -m nine.webui_simple \
        --ckpt "$CKPT" --tok "$TOK" \
        ${LORA:+--lora "$LORA"} \
        "${CUSTOM_ARGS[@]}"
fi
