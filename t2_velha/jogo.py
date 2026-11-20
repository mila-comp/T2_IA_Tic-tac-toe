"""Regras do jogo da velha (controle algorítmico do estado do jogo).

Tabuleiro: tupla de 9 inteiros. 0 = vazio, 1 = X, -1 = O.
Posições:   0 1 2
            3 4 5
            6 7 8
"""

LINHAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]

X, O = 1, -1


def tabuleiro_vazio():
    return (0,) * 9


def vencedor(t):
    """Devolve 1 (X venceu), -1 (O venceu) ou 0 (ninguém venceu ainda)."""
    for a, b, c in LINHAS:
        if t[a] != 0 and t[a] == t[b] == t[c]:
            return t[a]
    return 0


def jogadas_validas(t):
    return [i for i in range(9) if t[i] == 0]


def aplicar(t, pos, jogador):
    """Devolve um novo tabuleiro com a jogada feita (não altera o original)."""
    l = list(t)
    l[pos] = jogador
    return tuple(l)


def resultado(t):
    """None se ainda há jogo; senão 1 (X venceu), -1 (O venceu) ou 0 (empate)."""
    v = vencedor(t)
    if v != 0:
        return v
    if not jogadas_validas(t):
        return 0
    return None


def mostrar(t):
    simbolo = {0: ".", 1: "X", -1: "O"}
    return "\n".join(" ".join(simbolo[t[3 * l + c]] for c in range(3)) for l in range(3))
