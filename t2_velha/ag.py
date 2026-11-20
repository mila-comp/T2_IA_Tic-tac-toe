"""Algoritmo genético para evoluir os pesos da rede (cromossomo = lista de reais).

Operadores:
- população inicial: pesos aleatórios reais em [-1, 1]
- seleção: elitismo (os melhores passam direto) + torneio de tamanho k
- cruzamento para valores reais: BLX-alfa (mistura) ou aritmético
- mutação aleatória: cada gene, com probabilidade `taxa_mutacao`, recebe um novo valor ou um ruído aleatório
"""
import random

LIMITE_PESO = 5.0     # mantém os pesos em uma faixa razoável após cruzamentos


def populacao_inicial(n, n_pesos, rng):
    return [[rng.uniform(-1, 1) for _ in range(n_pesos)] for _ in range(n)]


def torneio(populacao, aptidoes, k, rng):
    """Sorteia k indivíduos e devolve o de maior aptidão."""
    candidatos = rng.sample(range(len(populacao)), k)
    melhor = max(candidatos, key=lambda i: aptidoes[i])
    return populacao[melhor]


def elite(populacao, aptidoes, n_elite):
    ordem = sorted(range(len(populacao)), key=lambda i: aptidoes[i], reverse=True)
    return [list(populacao[i]) for i in ordem[:n_elite]]


def cruzamento_blx(p1, p2, alfa, rng):
    """BLX-alfa: cada gene do filho é sorteado em [min - alfa*d, max + alfa*d], d = |p1 - p2|."""
    filho = []
    for a, b in zip(p1, p2):
        lo, hi = min(a, b), max(a, b)
        d = hi - lo
        g = rng.uniform(lo - alfa * d, hi + alfa * d)
        filho.append(max(-LIMITE_PESO, min(LIMITE_PESO, g)))
    return filho


def cruzamento_aritmetico(p1, p2, rng):
    """Combinação aritmética: filho = r*p1 + (1-r)*p2, com r sorteado (um por cromossomo)."""
    r = rng.random()
    return [r * a + (1 - r) * b for a, b in zip(p1, p2)]


def mutacao(c, taxa, rng, tipo="aleatoria", sigma=0.2):
    """Mutação aleatória: cada gene, com probabilidade `taxa`, é alterado.
    tipo "aleatoria": o gene recebe um valor novo sorteado em [-1, 1] (troca total)
    tipo "ruido":     o gene recebe um ruído aleatório pequeno (+/- sigma, distribuição normal)"""
    if tipo == "aleatoria":
        return [rng.uniform(-1, 1) if rng.random() < taxa else g for g in c]
    return [max(-LIMITE_PESO, min(LIMITE_PESO, g + rng.gauss(0, sigma))) if rng.random() < taxa else g for g in c]


def nova_geracao(populacao, aptidoes, cfg, rng):
    """Monta a próxima população: elite + filhos (torneio -> cruzamento -> mutação)."""
    nova = elite(populacao, aptidoes, cfg["elite"])
    while len(nova) < len(populacao):
        p1 = torneio(populacao, aptidoes, cfg["torneio_k"], rng)
        p2 = torneio(populacao, aptidoes, cfg["torneio_k"], rng)
        if rng.random() < cfg["taxa_cruzamento"]:
            if cfg["cruzamento"] == "blx":
                filho = cruzamento_blx(p1, p2, cfg["alfa"], rng)
            else:
                filho = cruzamento_aritmetico(p1, p2, rng)
        else:
            filho = list(p1)
        nova.append(mutacao(filho, cfg["taxa_mutacao"], rng, cfg["tipo_mutacao"], cfg["sigma_mutacao"]))
    return nova
