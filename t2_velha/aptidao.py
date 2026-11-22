"""Função de aptidão: mede o desempenho de um cromossomo jogando contra o Minimax.

A pontuação soma um pequeno bônus por jogada válida ao resultado final.
Uma proposta inválida encerra a partida, sem acumular a penalidade de derrota.
Os valores são escolhas do projeto, não constantes exigidas pelo enunciado.
"""
import random
from jogo import aplicar, jogadas_validas, resultado, tabuleiro_vazio
from minimax import jogada_minimax, jogadas_otimas

CONFIG_APTIDAO = dict(
    pontos_jogada_valida=0.05,
    pontos_jogada_invalida=-5,
    pontos_vitoria=3,
    pontos_empate=1,
    pontos_derrota=-2,
    partidas_medio=10,      # partidas por cromossomo no modo médio (metade começa pela rede)
    partidas_dificil=2,     # no difícil o resultado é sempre o mesmo: 1 partida por lado
    taxa_minimax_medio=0.5,
)


def jogar_partida(rede, modo, rede_comeca, rng, cfg=CONFIG_APTIDAO, medir=False):
    """Joga uma partida rede x Minimax. A rede usa X se começa e O se não começa.
    Devolve dict com resultado ('vitoria','empate','derrota','invalida'), pontos e contagem de jogadas."""
    eu = 1 if rede_comeca else -1
    t = tabuleiro_vazio()
    vez = 1                                        # X sempre começa
    validas = otimas = 0
    while True:
        if vez == eu:
            pos = rede.jogar(t, eu)
            if medir and pos in jogadas_otimas(t, eu):
                otimas += 1
            if pos not in jogadas_validas(t):                        # casa ocupada: penaliza e encerra
                pontos = validas * cfg["pontos_jogada_valida"] + cfg["pontos_jogada_invalida"]
                return dict(resultado="invalida", pontos=pontos, validas=validas, total=validas + 1, otimas=otimas)
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
            return dict(resultado=res, pontos=pontos, validas=validas, total=validas, otimas=otimas)
        vez = -vez


def n_partidas(modo, cfg=CONFIG_APTIDAO):
    return cfg["partidas_dificil"] if modo == "dificil" else cfg["partidas_medio"]


def aptidao(rede, cromossomo, modos, semente, cfg=CONFIG_APTIDAO, detalhar=False):
    """Médias por modo, ponderadas sem repetir partidas difíceis idênticas.

    Sem pesos explícitos mantém a média por partida dos modelos antigos.
    O cache vale só para este cromossomo e não altera a decisão da MLP.
    """
    rede.carregar(cromossomo)
    original = rede
    class Decisoes:
        def __init__(self):
            self.cache = {}

        def __getattr__(self, nome):
            return getattr(original, nome)

        def jogar(self, t, jogador):
            chave = (tuple(t), jogador)
            if chave not in self.cache:
                self.cache[chave] = original.jogar(t, jogador)
            return self.cache[chave]
    rede = Decisoes()
    partidas = 0
    soma_ponderada = soma_pesos = 0.0
    cont = dict(vitoria=0, empate=0, derrota=0, invalida=0)
    inicio = 0
    for modo in modos:
        n = n_partidas(modo, cfg)
        pontos_modo = 0.0
        for k in range(n):
            p = jogar_partida(rede, modo, rede_comeca=((inicio + k) % 2 == 0), rng=random.Random(f"{semente}:{modo}:{k}"), cfg=cfg)
            if detalhar:
                cont[p["resultado"]] += 1
            pontos_modo += p["pontos"]
            partidas += 1
        peso = cfg.get("pesos_modos", {}).get(modo, n)
        if n and peso > 0:
            soma_ponderada += peso * pontos_modo / n
            soma_pesos += peso
        inicio += n
    if not partidas:
        raise ValueError("Informe partidas para avaliação")
    if not soma_pesos:
        raise ValueError("Informe peso positivo para avaliação")
    valor = soma_ponderada / soma_pesos
    return dict(cont, aptidao=valor, partidas=partidas) if detalhar else valor


def avaliar(rede, modo, n=100, semente=0, cfg=CONFIG_APTIDAO):
    """Estatísticas da rede (sem mudar nada): usada ao final do treino para medir a acurácia."""
    if n < 1:
        raise ValueError("Informe ao menos uma partida")
    cont = dict(vitoria=0, empate=0, derrota=0, invalida=0)
    lados = {"primeiro": dict(partidas=0, vitoria=0, empate=0, derrota=0, invalida=0),
             "segundo": dict(partidas=0, vitoria=0, empate=0, derrota=0, invalida=0)}
    validas = total = otimas = 0
    for k in range(n):
        p = jogar_partida(rede, modo, rede_comeca=(k % 2 == 0),
                          rng=random.Random(f"{semente}:{modo}:{k}"), cfg=cfg, medir=True)
        cont[p["resultado"]] += 1
        lado = lados["primeiro" if k % 2 == 0 else "segundo"]
        lado["partidas"] += 1
        lado[p["resultado"]] += 1
        validas += p["validas"]
        total += p["total"]
        otimas += p["otimas"]
    cont.update(partidas=n, jogadas_validas=validas, jogadas=total, jogadas_otimas=otimas,
                acuracia_decisao=otimas / total if total else 0.0,
                acuracia_jogadas=validas / total if total else 0.0,
                taxa_vitoria=cont["vitoria"] / n, taxa_empate=cont["empate"] / n,
                taxa_derrota=(cont["derrota"] + cont["invalida"]) / n,
                partidas_sem_derrota=(cont["vitoria"] + cont["empate"]) / n,
                por_inicio=lados)
    return cont
