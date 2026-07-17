"""
NINE-1 Web UI Simples — servidor HTTP standalone, ZERO dependencias.
Usa apenas a biblioteca padrao do Python (http.server, json) + HTML/CSS/JS embutido.

Uso:
    python -m nine.webui_simple --ckpt nine/data/nine1-base.pt --tok nine/data/nine1-tok.json
    python -m nine.webui_simple --ckpt nine/data/nine1-base.pt --port 8080 --host 0.0.0.0

Requisitos: apenas Python 3.8+ e torch (ja instalado para o modelo).
Nao precisa de gradio, fastapi, flask ou qualquer outra dependencia.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import torch

from .tokenizer import BPETokenizer, EOS_TOKEN
from .fuse import load_fused_model


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 2048
MAX_NEW_TOKENS = 512
VERSION = "NINE-1 v0.4.0"


# ---------------------------------------------------------------------------
# Engine de geracao (reutiliza logica do webui.py)
# ---------------------------------------------------------------------------

class NINE1Engine:
    """Wrapper do modelo NINE-1 para API HTTP."""

    def __init__(self, ckpt_path: str, lora_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 device: str = "cpu", verbose: bool = False):
        self.device = device
        self.model, self.tokenizer = load_fused_model(
            base_path=ckpt_path, lora_path=lora_path,
            tokenizer_path=tokenizer_path, device=device,
            verbose=verbose,
        )
        self.model.eval()
        self.use_bpe = self.tokenizer is not None
        if verbose:
            print(f"  Modelo: {self.model.num_params()/1e6:.1f}M params")

    def encode(self, text: str) -> torch.Tensor:
        if self.use_bpe:
            ids = self.tokenizer.encode(text, add_bos=True)
            ids = ids[-MAX_CONTEXT_TOKENS:]
            return torch.tensor([ids], dtype=torch.long, device=self.device)
        ids = [min(ord(c), 65535) for c in text[-MAX_CONTEXT_TOKENS:]]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, token_id: int) -> str:
        if self.use_bpe:
            return self.tokenizer.decode([token_id])
        return chr(token_id) if 0 < token_id < 0x10FFFF else ""

    def is_eos(self, token_id: int) -> bool:
        if self.use_bpe and EOS_TOKEN in self.tokenizer.token_to_id:
            return token_id == self.tokenizer.token_to_id[EOS_TOKEN]
        return token_id in (0, 1, 2, 3)

    def generate(self, prompt: str, temperature: float = 0.4,
                 top_k: int = 20, max_tokens: int = 256) -> str:
        prompt = prompt.strip()
        if not prompt:
            return ""

        formatted = f"# tarefa: {prompt}\n# solucao:\n"
        input_ids = self.encode(formatted)
        if input_ids.numel() == 0:
            return ""

        ids = input_ids.clone()
        tokens_generated = 0
        max_t = min(max_tokens, MAX_NEW_TOKENS)
        output_parts = []

        for _ in range(max_t):
            with torch.no_grad():
                if ids.size(1) > MAX_CONTEXT_TOKENS:
                    ids = ids[:, -MAX_CONTEXT_TOKENS:]

                logits, _, _ = self.model(ids)
                logits = logits[:, -1, :] / max(temperature, 0.01)

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = logits.softmax(-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)
            token_str = self.decode(next_id.item())
            if token_str:
                output_parts.append(token_str)

            tokens_generated += 1
            if self.is_eos(next_id.item()):
                break

        return "".join(output_parts)


# ---------------------------------------------------------------------------
# HTML embutido (interface completa)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NINE-1 — IA de Programacao em PT-BR</title>
<style>
  :root {
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #252540;
    --primary: #667eea;
    --primary-dim: #4a5ec9;
    --accent: #764ba2;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --success: #4ade80;
    --border: #2a2a45;
    --radius: 12px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    padding: 1.2rem 2rem;
    text-align: center;
  }
  .header h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
  .header p { opacity: 0.85; font-size: 0.95rem; margin-top: 0.25rem; }
  .header small { opacity: 0.65; font-size: 0.8rem; }
  .container {
    max-width: 900px; margin: 1.5rem auto; padding: 0 1rem;
    flex: 1; display: flex; flex-direction: column;
  }
  .chat-box {
    flex: 1; background: var(--surface); border-radius: var(--radius);
    border: 1px solid var(--border); padding: 1rem;
    overflow-y: auto; min-height: 350px; max-height: 500px;
    display: flex; flex-direction: column; gap: 0.75rem;
  }
  .msg { padding: 0.7rem 1rem; border-radius: 10px; max-width: 85%;
          line-height: 1.5; font-size: 0.9rem; word-wrap: break-word; }
  .msg.user { background: linear-gradient(135deg, var(--primary), var(--accent));
              color: white; align-self: flex-end; }
  .msg.bot { background: var(--surface2); color: var(--text);
             align-self: flex-start; border: 1px solid var(--border); }
  .msg.bot code { background: #1a1a2e; padding: 1px 5px; border-radius: 4px;
                  font-size: 0.85em; font-family: 'Cascadia Code', 'Fira Code', monospace; }
  .msg.bot pre { background: #0f0f1a !important; border: 1px solid var(--border);
                 border-radius: 8px; padding: 0.8rem; margin: 0.5rem 0;
                 overflow-x: auto; font-size: 0.85rem; }
  .msg.bot pre code { background: transparent; padding: 0; }
  .empty-msg { color: var(--text-dim); text-align: center; padding: 3rem 1rem;
               font-style: italic; }
  .input-area { display: flex; gap: 0.5rem; margin-top: 0.75rem; }
  .input-area textarea {
    flex: 1; padding: 0.75rem 1rem; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); font-size: 0.9rem; resize: none; outline: none;
    min-height: 50px; max-height: 120px; transition: border-color 0.2s;
  }
  .input-area textarea:focus { border-color: var(--primary); }
  .input-area button {
    padding: 0.75rem 1.5rem; border-radius: var(--radius);
    border: none; background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white; font-weight: 600; cursor: pointer; font-size: 0.9rem;
    transition: transform 0.15s, opacity 0.15s;
  }
  .input-area button:hover { transform: scale(1.02); opacity: 0.9; }
  .input-area button:active { transform: scale(0.98); }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .params { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;
            padding: 0.75rem; background: var(--surface); border-radius: var(--radius);
            border: 1px solid var(--border); }
  .param-group { display: flex; flex-direction: column; gap: 0.2rem; flex: 1; min-width: 120px; }
  .param-group label { font-size: 0.75rem; color: var(--text-dim); font-weight: 500; }
  .param-group input[type="range"] {
    width: 100%; accent-color: var(--primary); height: 4px;
    background: var(--surface2); border-radius: 2px;
  }
  .param-group .val { font-size: 0.8rem; color: var(--primary); font-weight: 600; }
  .examples { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
  .examples button {
    padding: 0.35rem 0.75rem; border-radius: 20px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text-dim); cursor: pointer;
    font-size: 0.8rem; transition: all 0.2s;
  }
  .examples button:hover { border-color: var(--primary); color: var(--primary); background: var(--surface2); }
  .status-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 1rem; margin-top: 1rem;
    background: var(--surface); border-radius: var(--radius);
    border: 1px solid var(--border); font-size: 0.75rem; color: var(--text-dim);
  }
  .status-bar .dot { display: inline-block; width: 8px; height: 8px;
                     border-radius: 50%; margin-right: 6px; }
  .dot.online { background: var(--success); }
  .dot.offline { background: #ef4444; }
  .footer { text-align: center; padding: 1rem; font-size: 0.75rem; color: var(--text-dim); }
  .footer a { color: var(--primary); text-decoration: none; }
  .footer a:hover { text-decoration: underline; }
  .typing-dots::after { content: '...'; animation: dots 1.5s steps(4, end) infinite; }
  @keyframes dots { 0%, 20% { content: ''; } 40% { content: '.'; } 60% { content: '..'; } 80%, 100% { content: '...'; } }
  @media (max-width: 600px) {
    .header h1 { font-size: 1.4rem; }
    .params { flex-direction: column; gap: 0.5rem; }
    .msg { max-width: 95%; font-size: 0.85rem; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>🐉 NINE-1</h1>
  <p>IA de Programacao em Portugues Brasileiro</p>
  <small>Construida do zero em PyTorch</small>
</div>
<div class="container">
  <div class="chat-box" id="chatBox">
    <div class="empty-msg">Digite um prompt em portugues para gerar codigo Python!</div>
  </div>
  <div class="params" id="params">
    <div class="param-group">
      <label>Temperatura <span class="val" id="tempVal">0.4</span></label>
      <input type="range" id="temperature" min="0.1" max="1.5" step="0.05" value="0.4">
    </div>
    <div class="param-group">
      <label>Top-K <span class="val" id="topkVal">20</span></label>
      <input type="range" id="topK" min="1" max="100" step="1" value="20">
    </div>
    <div class="param-group">
      <label>Max tokens <span class="val" id="maxTokVal">200</span></label>
      <input type="range" id="maxTokens" min="16" max="512" step="16" value="200">
    </div>
  </div>
  <div class="examples">
    <button onclick="useExample('escreva uma funcao fibonacci')">Fibonacci</button>
    <button onclick="useExample('crie uma classe Pilha em python')">Pilha</button>
    <button onclick="useExample('bubble sort em python')">Bubble Sort</button>
    <button onclick="useExample('escreva uma funcao que valida email')">Validar Email</button>
    <button onclick="useExample('calcule o fatorial de um numero')">Fatorial</button>
    <button onclick="useExample('crie um gerador de numeros primos')">Primos</button>
  </div>
  <div class="input-area">
    <textarea id="promptInput" placeholder="Ex: escreva uma funcao que calcula fibonacci" rows="2"></textarea>
    <button id="sendBtn" onclick="sendPrompt()">Gerar</button>
  </div>
  <div class="status-bar">
    <span><span class="dot online" id="statusDot"></span><span id="statusText">Online</span></span>
    <span id="modelInfo">NINE-1 | Aguardando...</span>
  </div>
</div>
<div class="footer">
  Feito do zero com PyTorch &bull; <a href="https://github.com/Vtgamer998/nine-1" target="_blank">GitHub</a>
</div>
<script>
const chatBox = document.getElementById('chatBox');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');

// Parametros
const temperature = document.getElementById('temperature');
const topK = document.getElementById('topK');
const maxTokens = document.getElementById('maxTokens');
const tempVal = document.getElementById('tempVal');
const topkVal = document.getElementById('topkVal');
const maxTokVal = document.getElementById('maxTokVal');

temperature.oninput = () => tempVal.textContent = parseFloat(temperature.value).toFixed(2);
topK.oninput = () => topkVal.textContent = topK.value;
maxTokens.oninput = () => maxTokVal.textContent = maxTokens.value;

promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
});

function useExample(text) {
  promptInput.value = text;
  promptInput.focus();
}

function addMessage(text, role) {
  const empty = chatBox.querySelector('.empty-msg');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = 'msg ' + role;

  if (role === 'bot') {
    // Renderiza codigo com syntax highlight basico
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Detecta blocos de codigo (```...```)
    html = html.replace(/```(\w*)\\n?([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Detecta codigo inline (`...`)
    html = html.replace(/\`([^`]+)\`/g, '<code>$1</code>');
    // Detecta linhas com def (funcoes Python)
    html = html.replace(/^def /gm, '<strong>def </strong>');
    // Detecta linhas com class
    html = html.replace(/^class /gm, '<strong>class </strong>');
    // Quebras de linha
    html = html.replace(/\n/g, '<br>');

    div.innerHTML = html;
  } else {
    div.textContent = text;
  }

  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  chatBox.scrollTop = chatBox.scrollHeight;
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  sendBtn.textContent = loading ? 'Gerando...' : 'Gerar';
}

async function sendPrompt() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  addMessage(prompt, 'user');
  promptInput.value = '';
  setLoading(true);
  statusText.textContent = 'Gerando...';
  statusDot.className = 'dot offline';

  const tempDiv = document.createElement('div');
  tempDiv.className = 'msg bot';
  chatBox.appendChild(tempDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        temperature: parseFloat(temperature.value),
        top_k: parseInt(topK.value),
        max_tokens: parseInt(maxTokens.value),
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      tempDiv.textContent = 'Erro: ' + (err.error || resp.statusText);
      setLoading(false);
      statusText.textContent = 'Online';
      statusDot.className = 'dot online';
      return;
    }

    const data = await resp.json();
    const text = data.text || '';
    let displayText = text;

    // Renderiza HTML
    let html = displayText
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    html = html.replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/\`([^`]+)\`/g, '<code>$1</code>');
    html = html.replace(/^def /gm, '<strong>def </strong>');
    html = html.replace(/^class /gm, '<strong>class </strong>');
    html = html.replace(/\\n/g, '<br>');

    tempDiv.innerHTML = html || '(vazio)';
    chatBox.scrollTop = chatBox.scrollHeight;

  } catch (err) {
    tempDiv.textContent = 'Erro de conexao: ' + err.message;
  }

  setLoading(false);
  statusText.textContent = 'Online';
  statusDot.className = 'dot online';
}

// Carrega info do modelo ao iniciar
fetch('/api/info')
  .then(r => r.json())
  .then(data => {
    document.getElementById('modelInfo').textContent =
      `NINE-1 | ${(data.params / 1e6).toFixed(1)}M params | ${data.vocab} vocab`;
  })
  .catch(() => {
    document.getElementById('modelInfo').textContent = 'NINE-1 | Offline';
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------

class NINE1Handler(BaseHTTPRequestHandler):
    """Handler HTTP para API e interface web."""

    engine: Optional[NINE1Engine] = None  # Setado externamente

    def do_GET(self):
        if self.path == '/api/info':
            self._send_json({
                "status": "ok",
                "model": "NINE-1",
                "version": VERSION,
                "params": self.engine.model.num_params() if self.engine else 0,
                "vocab": self.engine.model.cfg.vocab_size if self.engine else 0,
                "device": self.engine.device if self.engine else "unknown",
            })
        else:
            # Serve a pagina HTML
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(HTML_PAGE)))
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/generate':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error("Corpo da requisicao vazio")
                return

            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_error("JSON invalido")
                return

            prompt = data.get('prompt', '').strip()
            if not prompt:
                self._send_error("Prompt vazio")
                return

            temperature = max(0.1, min(2.0, float(data.get('temperature', 0.4))))
            top_k = max(0, min(100, int(data.get('top_k', 20))))
            max_tokens = max(16, min(MAX_NEW_TOKENS, int(data.get('max_tokens', 256))))

            try:
                t0 = time.time()
                text = self.engine.generate(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - t0

                self._send_json({
                    "status": "ok",
                    "text": text,
                    "tokens": len(text),
                    "time_seconds": round(elapsed, 2),
                })
            except Exception as e:
                self._send_error(f"Erro na geracao: {e}")
        else:
            self._send_error("Rota desconhecida", status=404)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400):
        self._send_json({"status": "error", "error": msg}, status=status)

    def log_message(self, format, *args):
        # Log mais limpo
        if '/api/info' not in str(args):
            print(f"[webui] {self.address_string()} - {format % args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NINE-1 Web UI (standalone, zero dependencias)")

    # Auto-detect: prefere checkpoint grande
    default_ckpt = "nine/data/nine1-base.pt"
    for c in ["nine/data/nine1-base-g.pt", "nine/data/nine1-base.pt"]:
        if os.path.exists(c) and os.path.getsize(c) > 10_000_000:
            default_ckpt = c
            break
    default_tok = "nine/data/nine1-tok.json"
    if os.path.exists("nine/data/nine1-tok-g.json"):
        default_tok = "nine/data/nine1-tok-g.json"

    p.add_argument("--ckpt", type=str, default=default_ckpt)
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--tok", type=str, default=default_tok)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[webui] ERRO: checkpoint nao encontrado: {args.ckpt}", file=sys.stderr)
        print(f"[webui] Use --ckpt <caminho> ou verifique se o arquivo existe.", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("  🐉 NINE-1 Web UI (standalone)")
    print("  Zero dependencias — apenas Python + torch")
    print("=" * 50)
    print(f"\n  Inicializando...")

    # Carrega modelo
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Tokenizer:  {args.tok}")
    if args.lora and os.path.exists(args.lora):
        print(f"  LoRA:       {args.lora}")
    else:
        args.lora = None

    engine = NINE1Engine(
        ckpt_path=args.ckpt, lora_path=args.lora,
        tokenizer_path=args.tok, device=args.device,
        verbose=args.verbose,
    )
    NINE1Handler.engine = engine

    # Inicia servidor
    server = HTTPServer((args.host, args.port), NINE1Handler)
    url = f"http://{args.host}:{args.port}"

    print(f"\n  🚀 Servidor rodando!")
    print(f"  URL: {url}")
    print(f"  Device: {args.device}")
    print(f"  Pressione Ctrl+C para parar.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Servidor encerrado.")
        server.server_close()


if __name__ == "__main__":
    main()
