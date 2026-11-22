"""Partidas humanas e métricas, independentes da interface gráfica."""
import random

from jogo import aplicar, resultado, tabuleiro_vazio
from minimax import jogada_minimax, jogadas_otimas


class Sessao:
    def __init__(self):
        self.rede = None
        self.estatisticas = dict(vitoria=0, empate=0, derrota=0, invalida=0,
                                 validas=0, tentativas=0, otimas=0)
        self.nova()

    def usar_rede(self, rede):
        self.rede = rede
        # Métricas de redes diferentes não devem ser misturadas.
        self.estatisticas = dict.fromkeys(self.estatisticas, 0)

    def nova(self, adversario="minimax", modo="dificil", humano=1):
        if adversario not in ("minimax", "rede") or modo not in ("medio", "dificil"):
            raise ValueError("Adversário ou modo inválido")
        if humano not in (1, -1):
            raise ValueError("Humano deve ser X ou O")
        if adversario == "rede" and self.rede is None:
            raise ValueError("Carregue ou treine uma rede primeiro")
        self.adversario, self.modo, self.humano = adversario, modo, humano
        self.tabuleiro, self.vez, self.fim = tabuleiro_vazio(), 1, None
        self.validas = self.tentativas = self.otimas = 0
        self.rng = random.Random()

    def jogar(self, pos=None):
        if self.fim is not None:
            raise ValueError("A partida já terminou")
        if self.vez == self.humano:
            if pos is None:
                raise ValueError("Escolha uma casa")
        elif self.adversario == "rede":
            pos = self.rede.jogar(self.tabuleiro, self.vez)
            self.tentativas += 1
            self.otimas += pos in jogadas_otimas(self.tabuleiro, self.vez)
            if not 0 <= pos < 9 or self.tabuleiro[pos] != 0:
                self._encerrar("invalida")
                return
            self.validas += 1
        else:
            pos = jogada_minimax(self.tabuleiro, self.vez, self.modo, self.rng)
        self.tabuleiro = aplicar(self.tabuleiro, pos, self.vez)
        r = resultado(self.tabuleiro)
        if r is not None:
            self._encerrar("empate" if r == 0 else
                           "derrota" if r == self.humano else "vitoria")
        else:
            self.vez = -self.vez

    def _encerrar(self, fim):
        # Os resultados são sempre da perspectiva da IA.
        self.fim = fim
        if self.adversario == "rede":
            self.estatisticas[fim] += 1
            self.estatisticas["otimas"] += self.otimas
            self.estatisticas["validas"] += self.validas
            self.estatisticas["tentativas"] += self.tentativas

    def metricas(self):
        e = self.estatisticas
        n = sum(e[k] for k in ("vitoria", "empate", "derrota", "invalida"))
        return dict(e, partidas=n,
                    acuracia_decisao=e["otimas"] / e["tentativas"] if e["tentativas"] else None,
                    acuracia_jogadas=e["validas"] / e["tentativas"] if e["tentativas"] else None,
                    sem_derrota=(e["vitoria"] + e["empate"]) / n if n else None)
