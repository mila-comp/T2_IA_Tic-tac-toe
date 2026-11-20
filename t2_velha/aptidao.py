"""Função de aptidão: mede o desempenho de um cromossomo jogando contra o Minimax.

Recompensas e penalidades (configuráveis em CONFIG_APTIDAO):
- cada jogada VÁLIDA da rede ........ +1   (a primeira coisa a aprender é jogar em casa livre)
- jogada INVÁLIDA (casa ocupada) .... -10  e a partida é encerrada na hora
- vitória da rede ................... +10
- empate ............................ +5
- derrota ........................... -5
A aptidão do cromossomo é a média de pontos por partida. Metade das partidas começa
pela rede e metade pelo Minimax. No modo difícil basta 1 partida por lado (o Minimax é
determinístico); no médio há aleatoriedade, então jogamos várias.
"""
import random
from jogo import aplicar, jogadas_validas, resultado, tabuleiro_vazio
from minimax import jogada_minimax

CONFIG_APTIDAO = dict(
    pontos_jogada_valida=1,
    pontos_jogada_invalida=-10,
    pontos_vitoria=10,
    pontos_empate=5,
    pontos_derrota=-5,
    partidas_medio=10,      # partidas por cromossomo no modo médio (metade começa pela rede)
    partidas_dificil=2,     # no difícil o resultado é sempre o mesmo: 1 partida por lado
    taxa_minimax_medio=0.5,
)


def jogar_partida(rede, modo, rede_comeca, rng, cfg=CONFIG_APTIDAO):
    """Joga uma partida rede x Minimax. A rede usa X se começa e O se não começa.
    Devolve dict com resultado ('vitoria','empate','derrota','invalida'), pontos e contagem de jogadas."""
    eu = 1 if rede_comeca else -1
    t = tabuleiro_vazio()
    vez = 1                                        # X sempre começa
    validas = 0
    while True:
        if vez == eu:
            pos = rede.jogar(t, eu)
            if t[pos] != 0:                        # casa ocupada: penaliza e encerra
                pontos = validas * cfg["pontos_jogada_valida"] + cfg["pontos_jogada_invalida"]
                return dict(resultado="invalida", pontos=pontos, validas=validas, total=validas + 1)
            validas += 1
        else:
            pos = jogada_minimax(t, vez, modo, rng, cfg["taxa_minimax_medio"])
        t = aplicar(t, pos, vez)
        r = resultado(t)
        if r is not None:
            if r == 0:
                res, fim = "empate", cfg["pontos_empate"]
            elif r == eu:
                res, fim = "vitoria", cfg["pontos_vitoria"]
            else:
                res, fim = "derrota", cfg["pontos_derrota"]
            pontos = validas * cfg["pontos_jogada_valida"] + fim
            return dict(resultado=res, pontos=pontos, validas=validas, total=validas)
        vez = -vez


def n_partidas(modo, cfg=CONFIG_APTIDAO):
    return cfg["partidas_dificil"] if modo == "dificil" else cfg["partidas_medio"]


def aptidao(rede, cromossomo, modos, semente, cfg=CONFIG_APTIDAO):
    """Aptidão = média de pontos nas partidas de todos os `modos` informados."""
    rede.carregar(cromossomo)
    rng = random.Random(semente)           # mesma semente para todos da geração: comparação justa
    pontos, partidas = 0.0, 0
    for modo in modos:
        n = n_partidas(modo, cfg)
        for k in range(n):
            p = jogar_partida(rede, modo, rede_comeca=(k % 2 == 0), rng=rng, cfg=cfg)
            pontos += p["pontos"]
            partidas += 1
    return pontos / partidas


def avaliar(rede, modo, n=100, semente=0, cfg=CONFIG_APTIDAO):
    """Estatísticas da rede (sem mudar nada): usada ao final do treino para medir a acurácia."""
    rng = random.Random(semente)
    cont = dict(vitoria=0, empate=0, derrota=0, invalida=0)
    validas = total = 0
    for k in range(n):
        p = jogar_partida(rede, modo, rede_comeca=(k % 2 == 0), rng=rng, cfg=cfg)
        cont[p["resultado"]] += 1
        validas += p["validas"]
        total += p["total"]
    cont.update(partidas=n, jogadas_validas=validas, jogadas=total,
                acuracia_jogadas=validas / total if total else 0.0,
                partidas_sem_derrota=(cont["vitoria"] + cont["empate"]) / n)
    return cont
