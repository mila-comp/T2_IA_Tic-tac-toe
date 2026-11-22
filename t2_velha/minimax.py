"""Minimax (formulação negamax) com dois modos de jogo.

- difícil: sempre usa o Minimax (determinístico: o resultado é sempre o mesmo).
- médio:   a cada jogada, com probabilidade `taxa_minimax` (50%) usa o Minimax;
           nas demais joga em uma casa livre aleatória. O algoritmo em si não muda.
"""
import random
from jogo import aplicar, jogadas_validas, resultado, vencedor

_cache = {}


def _valor(t, jogador):
    """Valor do tabuleiro para quem vai jogar `jogador`: >0 ganha, 0 empata, <0 perde.
    Vitórias mais rápidas valem mais (e derrotas mais lentas perdem menos)."""
    chave = (t, jogador)
    if chave in _cache:
        return _cache[chave]
    if vencedor(t) != 0:                       # o jogador anterior já venceu
        v = -(1 + len(jogadas_validas(t)))
    else:
        livres = jogadas_validas(t)
        if not livres:
            v = 0
        else:
            v = max(-_valor(aplicar(t, p, jogador), -jogador) for p in livres)
    _cache[chave] = v
    return v


def jogadas_otimas(t, jogador):
    """Todas as decisões de valor máximo; aceita empates entre melhores jogadas."""
    if resultado(t) is not None:
        return []
    valores = {p: -_valor(aplicar(t, p, jogador), -jogador) for p in jogadas_validas(t)}
    melhor = max(valores.values())
    return [p for p, valor in valores.items() if valor == melhor]


def melhor_jogada(t, jogador):
    """Minimax puro; desempate determinístico pela primeira casa."""
    opcoes = jogadas_otimas(t, jogador)
    return opcoes[0] if opcoes else None


def jogada_minimax(t, jogador, modo, rng=None, taxa_minimax=0.5):
    """Escolhe a jogada do Minimax no modo 'dificil' ou 'medio'."""
    rng = rng or random
    if modo == "dificil":
        return melhor_jogada(t, jogador)
    if modo == "medio":
        if rng.random() < taxa_minimax:
            return melhor_jogada(t, jogador)
        return rng.choice(jogadas_validas(t))
    raise ValueError("modo deve ser 'dificil' ou 'medio'")
