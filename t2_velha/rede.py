"""Rede neural MLP de 2 camadas (oculta + saída), só propagação (sem backpropagation).

Estrutura de dados: cada camada é uma lista de neurônios, e cada neurônio é uma lista
de pesos [w1, w2, ..., wn, bias].
Entrada: o tabuleiro visto por quem joga (meu símbolo = 1, do adversário = -1, vazio = 0).
Saída: 9 valores, um por casa. A rede joga na casa de maior valor (mesmo que ocupada:
isso é uma jogada inválida, penalizada na aptidão).
"""
import math


class Rede:
    def __init__(self, n_oculta=9, n_saida=9, codificacao="sinal"):
        self.codificacao = codificacao
        n_entrada = 18 if codificacao == "dois_planos" else 9
        self.n_entrada, self.n_oculta, self.n_saida = n_entrada, n_oculta, n_saida
        self.oculta = [[0.0] * (n_entrada + 1) for _ in range(n_oculta)]
        self.saida = [[0.0] * (n_oculta + 1) for _ in range(n_saida)]

    @property
    def n_pesos(self):
        return self.n_oculta * (self.n_entrada + 1) + self.n_saida * (self.n_oculta + 1)

    def carregar(self, cromossomo):
        """Copia a lista de pesos (cromossomo) para os neurônios, camada por camada."""
        if len(cromossomo) != self.n_pesos:
            raise ValueError(f"cromossomo com {len(cromossomo)} pesos; a rede precisa de {self.n_pesos}")
        i = 0
        for camada in (self.oculta, self.saida):
            for neuronio in camada:
                n = len(neuronio)
                neuronio[:] = cromossomo[i:i + n]
                i += n

    def cromossomo(self):
        return [w for camada in (self.oculta, self.saida) for n in camada for w in n]

    @staticmethod
    def _camada(camada, entradas, ativacao):
        saidas = []
        for n in camada:
            soma = n[-1] + sum(w * x for w, x in zip(n, entradas))
            saidas.append(ativacao(soma))
        return saidas

    def propagar(self, entrada):
        """Propagação: entrada -> camada oculta (tanh) -> camada de saída (linear)."""
        h = self._camada(self.oculta, entrada, math.tanh)
        return self._camada(self.saida, h, lambda s: s)

    def codificar(self, tabuleiro, jogador):
        """Transforma o tabuleiro (visto por quem joga) na entrada da rede.
        sinal:        9 entradas: meu símbolo = 1, adversário = -1, vazio = 0
        ocupacao:     9 entradas: vazio = 0, meu símbolo = 1, adversário = 2
        dois_planos:  18 entradas: 9 'minhas casas' (0/1) + 9 'casas do adversário' (0/1)"""
        meu = [c * jogador for c in tabuleiro]
        if self.codificacao == "sinal":
            return meu
        if self.codificacao == "ocupacao":
            return [0 if c == 0 else (1 if c == 1 else 2) for c in meu]
        return [1 if c == 1 else 0 for c in meu] + [1 if c == -1 else 0 for c in meu]

    def jogar(self, tabuleiro, jogador):
        """Recebe o tabuleiro e devolve a casa escolhida (0 a 8)."""
        y = self.propagar(self.codificar(tabuleiro, jogador))
        return max(range(len(y)), key=lambda i: y[i])
