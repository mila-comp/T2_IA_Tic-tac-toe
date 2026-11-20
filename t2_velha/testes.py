"""Testes rápidos dos módulos: python3 testes.py"""
import random
import ag
from aptidao import CONFIG_APTIDAO, aptidao, jogar_partida
from jogo import aplicar, jogadas_validas, resultado, tabuleiro_vazio, vencedor
from minimax import jogada_minimax, melhor_jogada
from rede import Rede


def partida(modo_x, modo_o, rng):
    t, vez = tabuleiro_vazio(), 1
    while resultado(t) is None:
        modo = modo_x if vez == 1 else modo_o
        pos = rng.choice(jogadas_validas(t)) if modo == "aleatorio" else jogada_minimax(t, vez, modo, rng)
        t = aplicar(t, pos, vez)
        vez = -vez
    return resultado(t)


def test_regras():
    assert vencedor((1, 1, 1, 0, 0, 0, 0, 0, 0)) == 1
    assert vencedor((-1, 0, 0, 0, -1, 0, 0, 0, -1)) == -1
    assert resultado(tabuleiro_vazio()) is None
    assert resultado((1, -1, 1, 1, -1, -1, -1, 1, 1)) == 0          # empate


def test_minimax():
    rng = random.Random(0)
    assert partida("dificil", "dificil", rng) == 0                      # difícil x difícil empata
    for _ in range(100):                                                # difícil nunca perde para aleatório
        assert partida("dificil", "aleatorio", rng) in (0, 1)
        assert partida("aleatorio", "dificil", rng) in (0, -1)
    assert melhor_jogada((1, 1, 0, -1, -1, 0, 0, 0, 0), 1) == 2         # completa a linha e vence
    assert melhor_jogada((1, 1, 0, -1, -1, 0, 0, 0, 0), -1) == 5      # O vence completando a linha 3-4-5


def test_rede():
    for cod, n_in in [("sinal", 9), ("ocupacao", 9), ("dois_planos", 18)]:
        r = Rede(n_oculta=9, codificacao=cod)
        assert r.n_pesos == 9 * (n_in + 1) + 9 * (9 + 1)
        c = [random.uniform(-1, 1) for _ in range(r.n_pesos)]
        r.carregar(c)
        assert r.cromossomo() == c
        assert 0 <= r.jogar(tabuleiro_vazio(), 1) <= 8
    assert Rede(n_oculta=9, codificacao="sinal").n_pesos == 180          # 9-9-9 com bias


def test_ag():
    rng = random.Random(1)
    pop = ag.populacao_inicial(10, 20, rng)
    assert all(-1 <= g <= 1 for c in pop for g in c)
    filho = ag.cruzamento_blx(pop[0], pop[1], 0.3, rng)
    assert len(filho) == 20 and all(abs(g) <= ag.LIMITE_PESO for g in filho)
    apts = list(range(10))
    assert ag.elite(pop, apts, 2) == [pop[9], pop[8]]
    cfg = dict(elite=2, torneio_k=3, taxa_cruzamento=0.9, cruzamento="blx", alfa=0.3,
               taxa_mutacao=0.05, tipo_mutacao="ruido", sigma_mutacao=0.2)
    nova = ag.nova_geracao(pop, apts, cfg, rng)
    assert len(nova) == 10 and nova[0] == pop[9]                         # elitismo preservou o melhor


def test_aptidao():
    rng = random.Random(0)
    r = Rede(n_oculta=9)
    r.carregar([0.0] * r.n_pesos)               # rede "morta": sempre joga na casa 0
    p = jogar_partida(r, "dificil", True, rng)
    assert p["resultado"] == "invalida"          # 2ª jogada repete a casa 0 -> invalida
    assert p["pontos"] == CONFIG_APTIDAO["pontos_jogada_valida"] + CONFIG_APTIDAO["pontos_jogada_invalida"]
    assert aptidao(r, [0.0] * r.n_pesos, ["dificil"], 1) < 0


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok ", nome)
