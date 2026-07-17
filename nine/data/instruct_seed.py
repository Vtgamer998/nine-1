"""Gera dataset instrucional em PT-BR para fine-tuning LoRA do NINE-1.
Expansao massiva: 80+ exemplos cobrindo matematica, strings, estruturas,
algoritmos, OOP, manipulacao de arquivos, e utilitarios.
"""

import json
import os

OUT_DIR = "nine/data"
os.makedirs(OUT_DIR, exist_ok=True)

INSTRUCTIONS = [
    # ========== MATEMATICA BASICA ==========
    {"instruction": "escreva uma funcao fibonacci recursiva em python",
     "output": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"instruction": "escreva uma funcao fibonacci iterativa",
     "output": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    {"instruction": "escreva uma funcao que calcula o fatorial de um numero",
     "output": "def fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n-1)"},
    {"instruction": "escreva uma funcao que verifica se um numero e primo",
     "output": "def primo(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
    {"instruction": "escreva uma funcao que calcula o MDC de dois numeros",
     "output": "def mdc(a, b):\n    while b:\n        a, b = b, a % b\n    return a"},
    {"instruction": "escreva uma funcao que calcula o MMC de dois numeros",
     "output": "def mmc(a, b):\n    return a * b // mdc(a, b)"},
    {"instruction": "escreva uma funcao que retorna os numeros primos ate n",
     "output": "def primos_ate(n):\n    return [i for i in range(2, n+1) if all(i % d for d in range(2, int(i**0.5)+1))]"},
    {"instruction": "escreva uma funcao que calcula o n-esimo numero triangular",
     "output": "def triangular(n):\n    return n * (n + 1) // 2"},

    # ========== ESTATISTICA ==========
    {"instruction": "escreva uma funcao que calcula a media de uma lista",
     "output": "def media(lista):\n    return sum(lista) / len(lista) if lista else 0"},
    {"instruction": "escreva uma funcao que calcula a mediana",
     "output": "def mediana(lista):\n    if not lista:\n        return 0\n    ord = sorted(lista)\n    n = len(ord)\n    if n % 2 == 1:\n        return ord[n // 2]\n    return (ord[n // 2 - 1] + ord[n // 2]) / 2"},
    {"instruction": "escreva uma funcao que calcula o desvio padrao",
     "output": "import math\n\ndef desvio_padrao(lista):\n    if len(lista) < 2:\n        return 0\n    m = sum(lista) / len(lista)\n    var = sum((x - m) ** 2 for x in lista) / (len(lista) - 1)\n    return math.sqrt(var)"},
    {"instruction": "escreva uma funcao que calcula a moda de uma lista",
     "output": "from collections import Counter\n\ndef moda(lista):\n    if not lista:\n        return None\n    freq = Counter(lista)\n    max_freq = max(freq.values())\n    return [k for k, v in freq.items() if v == max_freq]"},
    {"instruction": "escreva uma funcao que calcula a variancia de uma lista",
     "output": "def variancia(lista):\n    if len(lista) < 2:\n        return 0\n    m = sum(lista) / len(lista)\n    return sum((x - m) ** 2 for x in lista) / (len(lista) - 1)"},

    # ========== LISTAS / ARRAYS ==========
    {"instruction": "escreva uma funcao que soma todos os elementos de uma lista",
     "output": "def soma_lista(lista):\n    return sum(lista)"},
    {"instruction": "escreva uma funcao que retorna o maior elemento de uma lista",
     "output": "def maior(lista):\n    return max(lista)"},
    {"instruction": "escreva uma funcao que retorna o menor elemento de uma lista",
     "output": "def menor(lista):\n    return min(lista)"},
    {"instruction": "escreva uma funcao que inverte uma lista",
     "output": "def inverter(lista):\n    return lista[::-1]"},
    {"instruction": "escreva uma funcao que remove duplicatas de uma lista",
     "output": "def remover_duplicatas(lista):\n    return list(set(lista))"},
    {"instruction": "escreva uma funcao que remove duplicatas mantendo a ordem",
     "output": "def remover_duplicatas_ordenado(lista):\n    vistos = set()\n    resultado = []\n    for item in lista:\n        if item not in vistos:\n            vistos.add(item)\n            resultado.append(item)\n    return resultado"},
    {"instruction": "escreva uma funcao que concatena duas listas",
     "output": "def concatenar(a, b):\n    return a + b"},
    {"instruction": "escreva uma funcao que intercala duas listas",
     "output": "def intercalar(a, b):\n    resultado = []\n    for x, y in zip(a, b):\n        resultado.append(x)\n        resultado.append(y)\n    resultado.extend(a[len(b):])\n    resultado.extend(b[len(a):])\n    return resultado"},
    {"instruction": "escreva uma funcao que rotaciona uma lista",
     "output": "def rotacionar(lista, k):\n    n = len(lista)\n    if n == 0:\n        return lista\n    k = k % n\n    return lista[-k:] + lista[:-k]"},
    {"instruction": "escreva uma funcao que particiona uma lista em partes",
     "output": "def particionar(lista, tamanho):\n    return [lista[i:i+tamanho] for i in range(0, len(lista), tamanho)]"},

    # ========== STRINGS ==========
    {"instruction": "escreva uma funcao que conta vogais em uma string",
     "output": "def contar_vogais(texto):\n    vogais = set('aeiouAEIOU')\n    return sum(1 for c in texto if c in vogais)"},
    {"instruction": "escreva uma funcao que conta consoantes em uma string",
     "output": "def contar_consoantes(texto):\n    consoantes = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')\n    return sum(1 for c in texto if c in consoantes)"},
    {"instruction": "escreva uma funcao que verifica se uma string e palindromo",
     "output": "def palindromo(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]"},
    {"instruction": "escreva uma funcao que conta palavras em um texto",
     "output": "def contar_palavras(texto):\n    return len(texto.split())"},
    {"instruction": "escreva uma funcao que encontra a palavra mais frequente",
     "output": "from collections import Counter\n\ndef palavra_mais_frequente(texto):\n    palavras = texto.lower().split()\n    if not palavras:\n        return ''\n    return Counter(palavras).most_common(1)[0][0]"},
    {"instruction": "escreva uma funcao que capitaliza a primeira letra de cada palavra",
     "output": "def capitalizar_palavras(texto):\n    return ' '.join(p.capitalize() for p in texto.split())"},
    {"instruction": "escreva uma funcao que inverte as palavras de uma frase",
     "output": "def inverter_palavras(frase):\n    return ' '.join(frase.split()[::-1])"},
    {"instruction": "escreva uma funcao que remove caracteres especiais de uma string",
     "output": "import re\n\ndef limpar_texto(texto):\n    return re.sub(r'[^a-zA-Z0-9\\s]', '', texto)"},
    {"instruction": "escreva uma funcao que conta a frequencia de cada caractere",
     "output": "from collections import Counter\n\ndef frequencia_caracteres(texto):\n    return dict(Counter(texto))"},

    # ========== ORDENACAO / BUSCA ==========
    {"instruction": "escreva bubble sort em python",
     "output": "def bubble_sort(lista):\n    n = len(lista)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lista[j] > lista[j+1]:\n                lista[j], lista[j+1] = lista[j+1], lista[j]\n    return lista"},
    {"instruction": "escreva selection sort em python",
     "output": "def selection_sort(lista):\n    n = len(lista)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if lista[j] < lista[min_idx]:\n                min_idx = j\n        lista[i], lista[min_idx] = lista[min_idx], lista[i]\n    return lista"},
    {"instruction": "escreva busca linear em python",
     "output": "def busca_linear(lista, alvo):\n    for i, item in enumerate(lista):\n        if item == alvo:\n            return i\n    return -1"},
    {"instruction": "escreva busca binaria em python",
     "output": "def busca_binaria(lista, alvo):\n    esq, dir = 0, len(lista) - 1\n    while esq <= dir:\n        meio = (esq + dir) // 2\n        if lista[meio] == alvo:\n            return meio\n        elif lista[meio] < alvo:\n            esq = meio + 1\n        else:\n            dir = meio - 1\n    return -1"},

    # ========== CLASSES / OOP ==========
    {"instruction": "crie uma classe Pilha em python",
     "output": "class Pilha:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        if not self.items:\n            return None\n        return self.items.pop()\n    def topo(self):\n        if not self.items:\n            return None\n        return self.items[-1]\n    def vazia(self):\n        return len(self.items) == 0\n    def tamanho(self):\n        return len(self.items)"},
    {"instruction": "crie uma classe Fila em python",
     "output": "class Fila:\n    def __init__(self):\n        self.items = []\n    def enfileirar(self, item):\n        self.items.append(item)\n    def desenfileirar(self):\n        if not self.items:\n            return None\n        return self.items.pop(0)\n    def vazia(self):\n        return len(self.items) == 0"},
    {"instruction": "crie uma classe ContaBancaria em python",
     "output": "class ContaBancaria:\n    def __init__(self, titular, saldo=0):\n        self.titular = titular\n        self.saldo = saldo\n    def depositar(self, valor):\n        self.saldo += valor\n    def sacar(self, valor):\n        if valor <= self.saldo:\n            self.saldo -= valor\n            return True\n        return False\n    def extrato(self):\n        return f'Conta de {self.titular}: R$ {self.saldo:.2f}'"},
    {"instruction": "crie uma classe Ponto2D em python",
     "output": "import math\n\nclass Ponto2D:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def distancia(self, outro):\n        return math.sqrt((self.x-outro.x)**2 + (self.y-outro.y)**2)\n    def __add__(self, outro):\n        return Ponto2D(self.x+outro.x, self.y+outro.y)\n    def __str__(self):\n        return f'({self.x}, {self.y})'"},
    {"instruction": "crie uma classe Agenda em python",
     "output": "class Agenda:\n    def __init__(self):\n        self.contatos = {}\n    def adicionar(self, nome, telefone):\n        self.contatos[nome] = telefone\n    def remover(self, nome):\n        return self.contatos.pop(nome, None)\n    def buscar(self, nome):\n        return self.contatos.get(nome)\n    def listar(self):\n        return list(self.contatos.keys())"},
    {"instruction": "crie uma classe Produto com nome e preco",
     "output": "class Produto:\n    def __init__(self, nome, preco):\n        self.nome = nome\n        self.preco = preco\n    def aplicar_desconto(self, percentual):\n        self.preco *= (1 - percentual/100)\n    def __str__(self):\n        return f'{self.nome}: R$ {self.preco:.2f}'"},
    {"instruction": "crie uma classe Carro em python",
     "output": "class Carro:\n    def __init__(self, marca, modelo, ano):\n        self.marca = marca\n        self.modelo = modelo\n        self.ano = ano\n        self.velocidade = 0\n    def acelerar(self, incremento):\n        self.velocidade += incremento\n    def frear(self, decremento):\n        self.velocidade = max(0, self.velocidade - decremento)\n    def __str__(self):\n        return f'{self.marca} {self.modelo} ({self.ano}) - {self.velocidade} km/h'"},

    # ========== ALGORITMOS CLASSICOS ==========
    {"instruction": "escreva uma funcao que implementa o crivo de Eratostenes",
     "output": "def crivo_eratostenes(n):\n    if n < 2:\n        return []\n    primo = [True] * (n+1)\n    primo[0] = primo[1] = False\n    for i in range(2, int(n**0.5)+1):\n        if primo[i]:\n            for j in range(i*i, n+1, i):\n                primo[j] = False\n    return [i for i, p in enumerate(primo) if p]"},
    {"instruction": "escreva uma funcao que fatora um numero em primos",
     "output": "def fatorar(n):\n    fatores = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            fatores.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        fatores.append(n)\n    return fatores"},
    {"instruction": "escreva uma funcao que calcula a sequencia de Collatz",
     "output": "def collatz(n):\n    seq = [n]\n    while n != 1:\n        if n % 2 == 0:\n            n //= 2\n        else:\n            n = 3 * n + 1\n        seq.append(n)\n    return seq"},
    {"instruction": "escreva uma funcao que gera numeros perfeitos ate n",
     "output": "def numeros_perfeitos(n):\n    return [i for i in range(2, n+1)\n            if sum(d for d in range(1, i) if i % d == 0) == i]"},
    {"instruction": "escreva uma funcao que calcula o enesimo termo de uma PA",
     "output": "def pa_termo(a1, razao, n):\n    return a1 + (n-1) * razao"},
    {"instruction": "escreva uma funcao que calcula a soma dos termos de uma PA",
     "output": "def pa_soma(a1, an, n):\n    return (a1 + an) * n // 2"},

    # ========== CONVERSOES ==========
    {"instruction": "escreva uma funcao que converte celsius para fahrenheit",
     "output": "def celsius_para_fahrenheit(c):\n    return (c * 9/5) + 32"},
    {"instruction": "escreva uma funcao que converte fahrenheit para celsius",
     "output": "def fahrenheit_para_celsius(f):\n    return (f - 32) * 5/9"},
    {"instruction": "escreva uma funcao que converte km para milhas",
     "output": "def km_para_milhas(km):\n    return km * 0.621371"},
    {"instruction": "escreva uma funcao que converte milhas para km",
     "output": "def milhas_para_km(milhas):\n    return milhas / 0.621371"},
    {"instruction": "escreva uma funcao que converte segundos para hh:mm:ss",
     "output": "def segundos_para_hms(segundos):\n    h = segundos // 3600\n    m = (segundos % 3600) // 60\n    s = segundos % 60\n    return f'{h:02d}:{m:02d}:{s:02d}'"},
    {"instruction": "escreva uma funcao que converte binario para decimal",
     "output": "def binario_para_decimal(binario):\n    return int(str(binario), 2)"},
    {"instruction": "escreva uma funcao que converte decimal para binario",
     "output": "def decimal_para_binario(n):\n    return bin(n)[2:]"},

    # ========== VALIDACAO ==========
    {"instruction": "escreva uma funcao que valida email",
     "output": "import re\n\ndef email_valido(email):\n    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(padrao, email))"},
    {"instruction": "escreva uma funcao que valida CPF",
     "output": "def cpf_valido(cpf):\n    cpf = ''.join(c for c in cpf if c.isdigit())\n    if len(cpf) != 11 or cpf == cpf[0]*11:\n        return False\n    soma = sum(int(cpf[i]) * (10-i) for i in range(9))\n    d1 = (soma * 10 % 11) % 10\n    if d1 != int(cpf[9]):\n        return False\n    soma = sum(int(cpf[i]) * (11-i) for i in range(10))\n    d2 = (soma * 10 % 11) % 10\n    return d2 == int(cpf[10])"},
    {"instruction": "escreva uma funcao que valida telefone brasileiro",
     "output": "import re\n\ndef telefone_valido(numero):\n    numero = re.sub(r'\\D', '', numero)\n    return len(numero) in (10, 11) and numero.isdigit()"},
    {"instruction": "escreva uma funcao que valida CEP brasileiro",
     "output": "import re\n\ndef cep_valido(cep):\n    cep = re.sub(r'\\D', '', cep)\n    return len(cep) == 8 and cep.isdigit()"},

    # ========== DICIONARIOS ==========
    {"instruction": "escreva uma funcao que ordena um dicionario por valor",
     "output": "def ordenar_por_valor(d):\n    return dict(sorted(d.items(), key=lambda x: x[1]))"},
    {"instruction": "escreva uma funcao que ordena um dicionario por chave",
     "output": "def ordenar_por_chave(d):\n    return dict(sorted(d.items()))"},
    {"instruction": "escreva uma funcao que junta dois dicionarios",
     "output": "def juntar_dicionarios(d1, d2):\n    resultado = d1.copy()\n    resultado.update(d2)\n    return resultado"},
    {"instruction": "escreva uma funcao que inverte um dicionario",
     "output": "def inverter_dicionario(d):\n    return {v: k for k, v in d.items()}"},
    {"instruction": "escreva uma funcao que agrupa uma lista por chave",
     "output": "def agrupar_por(lista, funcao_chave):\n    resultado = {}\n    for item in lista:\n        chave = funcao_chave(item)\n        if chave not in resultado:\n            resultado[chave] = []\n        resultado[chave].append(item)\n    return resultado"},

    # ========== ARQUIVOS ==========
    {"instruction": "escreva uma funcao que le um arquivo e retorna as linhas",
     "output": "def ler_arquivo(caminho):\n    with open(caminho, 'r', encoding='utf-8') as f:\n        return f.readlines()"},
    {"instruction": "escreva uma funcao que escreve texto em um arquivo",
     "output": "def escrever_arquivo(caminho, texto):\n    with open(caminho, 'w', encoding='utf-8') as f:\n        f.write(texto)"},
    {"instruction": "escreva uma funcao que le um arquivo CSV",
     "output": "import csv\n\ndef ler_csv(caminho):\n    with open(caminho, 'r', encoding='utf-8') as f:\n        return list(csv.DictReader(f))"},
    {"instruction": "escreva uma funcao que le um arquivo JSON",
     "output": "import json\n\ndef ler_json(caminho):\n    with open(caminho, 'r', encoding='utf-8') as f:\n        return json.load(f)"},
    {"instruction": "escreva uma funcao que salva dados em JSON",
     "output": "import json\n\ndef salvar_json(caminho, dados):\n    with open(caminho, 'w', encoding='utf-8') as f:\n        json.dump(dados, f, ensure_ascii=False, indent=2)"},

    # ========== GERADORES / ITERADORES ==========
    {"instruction": "escreva um gerador de numeros pares",
     "output": "def pares(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i"},
    {"instruction": "escreva um gerador de numeros impares",
     "output": "def impares(n):\n    for i in range(n):\n        if i % 2 == 1:\n            yield i"},
    {"instruction": "escreva um gerador que le um arquivo linha por linha",
     "output": "def ler_linhas(caminho):\n    with open(caminho, 'r', encoding='utf-8') as f:\n        for linha in f:\n            yield linha.strip()"},
    {"instruction": "escreva um gerador que percorre uma arvore em ordem",
     "output": "def em_ordem(no):\n    if no:\n        yield from em_ordem(no.esquerda)\n        yield no.valor\n        yield from em_ordem(no.direita)"},

    # ========== DECORATORS ==========
    {"instruction": "escreva um decorator que mede tempo de execucao",
     "output": "import time\n\ndef temporizador(func):\n    def wrapper(*args, **kwargs):\n        inicio = time.time()\n        resultado = func(*args, **kwargs)\n        fim = time.time()\n        print(f'{func.__name__} executou em {fim-inicio:.4f}s')\n        return resultado\n    return wrapper"},
    {"instruction": "escreva um decorator que loga chamadas de funcao",
     "output": "def log_chamadas(func):\n    def wrapper(*args, **kwargs):\n        print(f'Chamando {func.__name__}')\n        return func(*args, **kwargs)\n    return wrapper"},
    {"instruction": "escreva um decorator que repete a funcao n vezes",
     "output": "def repetir(n):\n    def decorador(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(n):\n                resultado = func(*args, **kwargs)\n            return resultado\n        return wrapper\n    return decorador"},

    # ========== RECURSAO ==========
    {"instruction": "escreva uma funcao recursiva que calcula a soma de 1 a n",
     "output": "def soma_recursiva(n):\n    if n <= 1:\n        return n\n    return n + soma_recursiva(n-1)"},
    {"instruction": "escreva uma funcao recursiva que calcula a potencia",
     "output": "def potencia_recursiva(base, exp):\n    if exp == 0:\n        return 1\n    if exp % 2 == 0:\n        metade = potencia_recursiva(base, exp//2)\n        return metade * metade\n    return base * potencia_recursiva(base, exp-1)"},
    {"instruction": "escreva uma funcao recursiva que inverte uma string",
     "output": "def inverter_recursivo(s):\n    if len(s) <= 1:\n        return s\n    return inverter_recursivo(s[1:]) + s[0]"},
    {"instruction": "escreva uma funcao recursiva que calcula o resto da divisao",
     "output": "def resto_divisao(dividendo, divisor):\n    if dividendo < divisor:\n        return dividendo\n    return resto_divisao(dividendo - divisor, divisor)"},

    # ========== UTILITARIOS ==========
    {"instruction": "escreva uma funcao que calcula o IMC",
     "output": "def calcular_imc(peso, altura):\n    imc = peso / (altura ** 2)\n    if imc < 18.5:\n        return imc, 'Abaixo do peso'\n    elif imc < 25:\n        return imc, 'Peso normal'\n    elif imc < 30:\n        return imc, 'Sobrepeso'\n    return imc, 'Obesidade'"},
    {"instruction": "escreva uma funcao que calcula juros compostos",
     "output": "def juros_compostos(principal, taxa, meses):\n    return principal * (1 + taxa) ** meses"},
    {"instruction": "escreva uma funcao que calcula o valor de parcela",
     "output": "def calcular_parcela(valor_total, parcelas, juros=0.02):\n    return valor_total * (juros * (1+juros)**parcelas) / ((1+juros)**parcelas - 1)"},
    {"instruction": "escreva uma funcao que gera senhas aleatorias",
     "output": "import random\nimport string\n\ndef gerar_senha(tamanho=12):\n    caracteres = string.ascii_letters + string.digits + '!@#$%&*'\n    return ''.join(random.choice(caracteres) for _ in range(tamanho))"},
    {"instruction": "escreva uma funcao que embaralha uma lista",
     "output": "import random\n\ndef embaralhar(lista):\n    random.shuffle(lista)\n    return lista"},
]


def main():
    path = os.path.join(OUT_DIR, "instruct.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in INSTRUCTIONS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Salvo {len(INSTRUCTIONS)} exemplos instrucionais em {path}")
    total_chars = sum(len(item["instruction"]) + len(item["output"]) for item in INSTRUCTIONS)
    print(f"Total de caracteres: {total_chars:,}")
    print(f"Categorias: matematica, estatistica, listas, strings, ordenacao, OOP, algoritmos, conversoes, validacao, dicionarios, arquivos, geradores, decorators, recursao, utilitarios")


if __name__ == "__main__":
    main()
