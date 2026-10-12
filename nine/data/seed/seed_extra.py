def inverter(s):
    return s[::-1]


def palindromo(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]


def contar_vogais(s):
    return sum(1 for c in s.lower() if c in 'aeiouáéíóúâêôãõ')


def contar_palavras(texto):
    return len(texto.split())


def maior(lista):
    m = lista[0]
    for x in lista:
        if x > m:
            m = x
    return m


def menor(lista):
    m = lista[0]
    for x in lista:
        if x < m:
            m = x
    return m


def sem_duplicados(lista):
    visto = set()
    saida = []
    for x in lista:
        if x not in visto:
            visto.add(x)
            saida.append(x)
    return saida


def pares(lista):
    return [x for x in lista if x % 2 == 0]


def impares(lista):
    return [x for x in lista if x % 2 != 0]


def somar(lista):
    total = 0
    for x in lista:
        total += x
    return total


def quadrados(lista):
    return [x ** 2 for x in lista]


def achatar(listas):
    return [x for sub in listas for x in sub]


def dividir(lista, n):
    return [lista[i:i+n] for i in range(0, len(lista), n)]


def frequencia(lista):
    contagem = {}
    for x in lista:
        contagem[x] = contagem.get(x, 0) + 1
    return contagem


def mais_comum(lista):
    contagem = {}
    for x in lista:
        contagem[x] = contagem.get(x, 0) + 1
    return max(contagem, key=contagem.get)


def bubble_sort(lista):
    lista = lista[:]
    n = len(lista)
    for i in range(n):
        for j in range(n - i - 1):
            if lista[j] > lista[j+1]:
                lista[j], lista[j+1] = lista[j+1], lista[j]
    return lista


def selection_sort(lista):
    lista = lista[:]
    for i in range(len(lista)):
        m = i
        for j in range(i + 1, len(lista)):
            if lista[j] < lista[m]:
                m = j
        lista[i], lista[m] = lista[m], lista[i]
    return lista


def insertion_sort(lista):
    lista = lista[:]
    for i in range(1, len(lista)):
        chave = lista[i]
        j = i - 1
        while j >= 0 and lista[j] > chave:
            lista[j+1] = lista[j]
            j -= 1
        lista[j+1] = chave
    return lista


def merge_sort(lista):
    if len(lista) <= 1:
        return lista
    meio = len(lista) // 2
    esq = merge_sort(lista[:meio])
    dir = merge_sort(lista[meio:])
    saida = []
    i = j = 0
    while i < len(esq) and j < len(dir):
        if esq[i] <= dir[j]:
            saida.append(esq[i])
            i += 1
        else:
            saida.append(dir[j])
            j += 1
    return saida + esq[i:] + dir[j:]


def quick_sort(lista):
    if len(lista) <= 1:
        return lista
    p = lista[len(lista) // 2]
    menores = [x for x in lista if x < p]
    iguais = [x for x in lista if x == p]
    maiores = [x for x in lista if x > p]
    return quick_sort(menores) + iguais + quick_sort(maiores)


def busca_linear(lista, alvo):
    for i, x in enumerate(lista):
        if x == alvo:
            return i
    return -1


def potencia(base, exp):
    resultado = 1
    for _ in range(exp):
        resultado *= base
    return resultado


def celsius_para_fahrenheit(c):
    return c * 9 / 5 + 32


def fahrenheit_para_celsius(f):
    return (f - 32) * 5 / 9


def para_binario(n):
    if n == 0:
        return '0'
    bits = ''
    while n > 0:
        bits = str(n % 2) + bits
        n //= 2
    return bits


def binario_para_decimal(b):
    total = 0
    for c in b:
        total = total * 2 + int(c)
    return total


def soma_digitos(n):
    return sum(int(d) for d in str(abs(n)))


def inverter_numero(n):
    return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)


def perfeito(n):
    return n > 1 and sum(i for i in range(1, n) if n % i == 0) == n


def potencia_de_dois(n):
    return n > 0 and (n & (n - 1)) == 0


def bissexto(ano):
    return ano % 4 == 0 and (ano % 100 != 0 or ano % 400 == 0)


def divisores(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def fib_lista(n):
    seq = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def tabuada(n):
    return [f'{n} x {i} = {n*i}' for i in range(1, 11)]


def anagrama(a, b):
    return sorted(a.replace(' ', '').lower()) == sorted(b.replace(' ', '').lower())


def titulo(texto):
    return ' '.join(p.capitalize() for p in texto.split())


def limpar_espacos(s):
    return ' '.join(s.split())


def contar_caractere(s, c):
    return s.count(c)


def palavra_mais_longa(texto):
    return max(texto.split(), key=len)


def so_digitos(s):
    return s.isdigit()


def inverter_caixa(s):
    return s.swapcase()


def balanceado(expr):
    pilha = []
    pares = {')': '(', ']': '[', '}': '{'}
    for c in expr:
        if c in '([{':
            pilha.append(c)
        elif c in ')]}':
            if not pilha or pilha.pop() != pares[c]:
                return False
    return not pilha


def validar_cpf(cpf):
    n = [int(c) for c in cpf if c.isdigit()]
    if len(n) != 11 or len(set(n)) == 1:
        return False
    for i in (9, 10):
        s = sum(n[j] * (i + 1 - j) for j in range(i))
        d = (s * 10 % 11) % 10
        if d != n[i]:
            return False
    return True


import re

def validar_email(email):
    return re.fullmatch(r'[\w.+-]+@[\w-]+\.[\w.-]+', email) is not None


import random
import string

def gerar_senha(n=12):
    chars = string.ascii_letters + string.digits + '!@#$%'
    return ''.join(random.choice(chars) for _ in range(n))


def juntar(a, b):
    return {**a, **b}


def inverter_dict(d):
    return {v: k for k, v in d.items()}


def ordenar_por_valor(d):
    return dict(sorted(d.items(), key=lambda kv: kv[1]))


def interseccao(a, b):
    return list(set(a) & set(b))


def uniao(a, b):
    return list(set(a) | set(b))


def transpor(m):
    return [list(l) for l in zip(*m)]


def somar_matrizes(a, b):
    return [[x + y for x, y in zip(l1, l2)] for l1, l2 in zip(a, b)]


def multiplicar_matrizes(a, b):
    return [[sum(x * y for x, y in zip(l, c)) for c in zip(*b)] for l in a]


from statistics import mean, median, mode

def estatisticas(lista):
    return mean(lista), median(lista), mode(lista)


def desvio_padrao(lista):
    m = sum(lista) / len(lista)
    return (sum((x - m) ** 2 for x in lista) / len(lista)) ** 0.5


def distancia(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


import math

def area_circulo(r):
    return math.pi * r ** 2


def area_triangulo(base, altura):
    return base * altura / 2


def imc(peso, altura):
    v = peso / altura ** 2
    if v < 18.5:
        return v, 'abaixo do peso'
    if v < 25:
        return v, 'normal'
    if v < 30:
        return v, 'sobrepeso'
    return v, 'obesidade'


def juros_compostos(capital, taxa, tempo):
    return capital * (1 + taxa) ** tempo


def aplicar_desconto(preco, percentual):
    return preco * (1 - percentual / 100)


from datetime import date

def idade(ano_nascimento):
    return date.today().year - ano_nascimento


from datetime import date

def dias_ate(ano, mes, dia):
    return (date(ano, mes, dia) - date.today()).days


def formatar_real(valor):
    s = f'{valor:,.2f}'
    return 'R$ ' + s.replace(',', 'X').replace('.', ',').replace('X', '.')


def ler_linhas(caminho):
    with open(caminho, encoding='utf-8') as f:
        return f.read().splitlines()


def salvar_lista(lista, caminho):
    with open(caminho, 'w', encoding='utf-8') as f:
        for item in lista:
            f.write(str(item) + '\n')


import json

def ler_json(caminho):
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


import json

def salvar_json(dados, caminho):
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


import csv

def ler_csv(caminho):
    with open(caminho, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def contar_linhas(caminho):
    with open(caminho, encoding='utf-8') as f:
        return sum(1 for _ in f)


import requests

def buscar_json(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


import time
from functools import wraps

def cronometro(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        print(f'{func.__name__} levou {time.time() - inicio:.4f}s')
        return resultado
    return wrapper


from functools import lru_cache

@lru_cache(maxsize=None)
def fib_cache(n):
    if n < 2:
        return n
    return fib_cache(n - 1) + fib_cache(n - 2)


import time

def tentar(func, tentativas=3, espera=1):
    for i in range(tentativas):
        try:
            return func()
        except Exception:
            if i == tentativas - 1:
                raise
            time.sleep(espera)


def gerar_primos():
    n = 2
    while True:
        if all(n % i for i in range(2, int(n ** 0.5) + 1)):
            yield n
        n += 1


def segundo_maior(lista):
    unicos = sorted(set(lista))
    return unicos[-2] if len(unicos) >= 2 else None


def rotacionar(lista, n):
    if not lista:
        return lista
    n %= len(lista)
    return lista[-n:] + lista[:-n]


def soma_alvo(lista, alvo):
    vistos = {}
    for i, x in enumerate(lista):
        if alvo - x in vistos:
            return vistos[alvo - x], i
        vistos[x] = i
    return None


def soma_acumulada(lista):
    total = 0
    saida = []
    for x in lista:
        total += x
        saida.append(total)
    return saida


def comprimir(s):
    if not s:
        return ''
    saida = []
    atual, cont = s[0], 1
    for c in s[1:]:
        if c == atual:
            cont += 1
        else:
            saida.append(f'{atual}{cont}')
            atual, cont = c, 1
    saida.append(f'{atual}{cont}')
    return ''.join(saida)


def cesar(texto, chave):
    saida = ''
    for c in texto:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            saida += chr((ord(c) - base + chave) % 26 + base)
        else:
            saida += c
    return saida


def para_romano(n):
    tabela = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    saida = ''
    for valor, s in tabela:
        while n >= valor:
            saida += s
            n -= valor
    return saida


import unicodedata

def remover_acentos(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


import re
import unicodedata

def slugify(s):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


import math

def bhaskara(a, b, c):
    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return None
    x1 = (-b + math.sqrt(delta)) / (2 * a)
    x2 = (-b - math.sqrt(delta)) / (2 * a)
    return x1, x2


def busca_binaria_rec(lista, alvo, esq=0, dir=None):
    if dir is None:
        dir = len(lista) - 1
    if esq > dir:
        return -1
    meio = (esq + dir) // 2
    if lista[meio] == alvo:
        return meio
    if lista[meio] < alvo:
        return busca_binaria_rec(lista, alvo, meio + 1, dir)
    return busca_binaria_rec(lista, alvo, esq, meio - 1)


def hanoi(n, origem='A', destino='C', aux='B'):
    if n == 0:
        return []
    return hanoi(n-1, origem, aux, destino) + [(origem, destino)] + hanoi(n-1, aux, destino, origem)


def permutacoes(lista):
    if len(lista) <= 1:
        return [lista]
    saida = []
    for i, x in enumerate(lista):
        for p in permutacoes(lista[:i] + lista[i+1:]):
            saida.append([x] + p)
    return saida


def subconjuntos(lista):
    saida = [[]]
    for x in lista:
        saida += [s + [x] for s in saida]
    return saida


def pascal(n):
    linhas = [[1]]
    for _ in range(n - 1):
        ant = linhas[-1]
        linhas.append([1] + [ant[i] + ant[i+1] for i in range(len(ant) - 1)] + [1])
    return linhas


def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print('FizzBuzz')
        elif i % 3 == 0:
            print('Fizz')
        elif i % 5 == 0:
            print('Buzz')
        else:
            print(i)


import random

def lancar_dado(lados=6):
    return random.randint(1, lados)


import random

def embaralhar(lista):
    copia = lista[:]
    random.shuffle(copia)
    return copia


from collections import Counter

def top3(texto):
    return Counter(texto.lower().split()).most_common(3)


from collections import defaultdict

def agrupar_por_tamanho(palavras):
    grupos = defaultdict(list)
    for p in palavras:
        grupos[len(p)].append(p)
    return dict(grupos)


def todos_unicos(lista):
    return len(lista) == len(set(lista))


def amplitude(lista):
    return max(lista) - min(lista)


def ordenada(lista):
    return all(lista[i] <= lista[i+1] for i in range(len(lista) - 1))


def fatorial_iterativo(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def armstrong(n):
    d = str(n)
    return n == sum(int(c) ** len(d) for c in d)


def formatar_tempo(seg):
    h, resto = divmod(seg, 3600)
    m, s = divmod(resto, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


from datetime import date

def dia_da_semana(ano, mes, dia):
    nomes = ['segunda', 'terca', 'quarta', 'quinta', 'sexta', 'sabado', 'domingo']
    return nomes[date(ano, mes, dia).weekday()]


class Pilha:
    def __init__(self):
        self.itens = []

    def push(self, x):
        self.itens.append(x)

    def pop(self):
        if self.vazia():
            raise IndexError('pilha vazia')
        return self.itens.pop()

    def vazia(self):
        return not self.itens


from collections import deque

class Fila:
    def __init__(self):
        self.itens = deque()

    def enfileirar(self, x):
        self.itens.append(x)

    def desenfileirar(self):
        if not self.itens:
            raise IndexError('fila vazia')
        return self.itens.popleft()

    def vazia(self):
        return not self.itens


class ContaBancaria:
    def __init__(self, titular, saldo=0):
        self.titular = titular
        self.saldo = saldo

    def depositar(self, valor):
        if valor <= 0:
            raise ValueError('valor invalido')
        self.saldo += valor

    def sacar(self, valor):
        if valor > self.saldo:
            raise ValueError('saldo insuficiente')
        self.saldo -= valor


class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade

    def apresentar(self):
        return f'Ola, eu sou {self.nome} e tenho {self.idade} anos'


class Retangulo:
    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura

    def area(self):
        return self.largura * self.altura

    def perimetro(self):
        return 2 * (self.largura + self.altura)


class Animal:
    def __init__(self, nome):
        self.nome = nome

    def falar(self):
        return '...'

class Cachorro(Animal):
    def falar(self):
        return 'Au au!'


class No:
    def __init__(self, valor):
        self.valor = valor
        self.proximo = None

class ListaLigada:
    def __init__(self):
        self.inicio = None

    def adicionar(self, valor):
        novo = No(valor)
        if not self.inicio:
            self.inicio = novo
            return
        atual = self.inicio
        while atual.proximo:
            atual = atual.proximo
        atual.proximo = novo

    def listar(self):
        saida, atual = [], self.inicio
        while atual:
            saida.append(atual.valor)
            atual = atual.proximo
        return saida


class No:
    def __init__(self, valor):
        self.valor = valor
        self.esq = None
        self.dir = None

class Arvore:
    def __init__(self):
        self.raiz = None

    def inserir(self, valor):
        if not self.raiz:
            self.raiz = No(valor)
            return
        atual = self.raiz
        while True:
            if valor < atual.valor:
                if not atual.esq:
                    atual.esq = No(valor)
                    return
                atual = atual.esq
            else:
                if not atual.dir:
                    atual.dir = No(valor)
                    return
                atual = atual.dir

    def buscar(self, valor):
        atual = self.raiz
        while atual:
            if valor == atual.valor:
                return True
            atual = atual.esq if valor < atual.valor else atual.dir
        return False


class Contador:
    def __init__(self):
        self.valor = 0

    def incrementar(self):
        self.valor += 1

    def decrementar(self):
        self.valor -= 1

    def resetar(self):
        self.valor = 0


class Produto:
    def __init__(self, nome, preco):
        self.nome = nome
        self.preco = preco

    def com_desconto(self, percentual):
        return self.preco * (1 - percentual / 100)


class Carrinho:
    def __init__(self):
        self.itens = []

    def adicionar(self, nome, preco, qtd=1):
        self.itens.append((nome, preco, qtd))

    def total(self):
        return sum(preco * qtd for _, preco, qtd in self.itens)


class Tarefas:
    def __init__(self):
        self.tarefas = []

    def adicionar(self, texto):
        self.tarefas.append({'texto': texto, 'feita': False})

    def concluir(self, i):
        self.tarefas[i]['feita'] = True

    def pendentes(self):
        return [t['texto'] for t in self.tarefas if not t['feita']]


from dataclasses import dataclass

@dataclass
class Ponto:
    x: float
    y: float

    def distancia_origem(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
