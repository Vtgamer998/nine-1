import os
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="NINE-1 Cloud API")

# Define a estrutura de dados que o OpenCode envia
class ChatInput(BaseModel):
    model: str
    messages: list

@app.get("/")
def home():
    return {"status": "NINE-1 na nuvem do Render está online e pronta!"}

@app.post("/v1/chat/completions")
def chat(input_data: ChatInput):
    # Captura a última mensagem enviada pelo usuário no terminal
    prompt = input_data.messages[-1]["content"]
    
    # Monta o comando para chamar o CLI original da NINE-1
    # O script usa o modo 'instruct' focado em comandos em português
    cmd = [
        "python", "-m", "nine.cli", prompt, 
        "--mode", "instruct", 
        "--ckpt", "nine/data/nine1-base.pt", 
        "--tokens", "120"
    ]
    
    # Executa o modelo em segundo plano de forma segura
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Remove as bordas estéticas textuais ("--- NINE-1 ---") 
    # para que o OpenCode exiba apenas o código limpo no terminal
    clean_output = result.stdout.replace("--- NINE-1 ---", "").replace("--------------", "").strip()
    
    # Se ocorrer algum erro interno no script, repassa para o terminal
    if not clean_output and result.stderr:
        clean_output = f"Erro na inferência do modelo: {result.stderr}"

    # Retorna o JSON no formato exato que o SDK do OpenCode (padrão OpenAI) exige
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": clean_output
            }
        }]
  }
  
