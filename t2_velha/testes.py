"""Testes rápidos dos módulos: python3 testes.py"""
import random
import tempfile
from pathlib import Path
import ag
from aptidao import CONFIG_APTIDAO, aptidao, jogar_partida
from jogo import aplicar, jogadas_validas, resultado, tabuleiro_vazio, vencedor
from minimax import jogada_minimax, melhor_jogada
from rede import Rede
from sessao import Sessao
from treino import CONFIG, carregar_rede, escolher_melhor, salvar, treinar


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
    assert melhor_jogada((1, 1, 0, -1, -1, 0, 1, 0, 0), -1) == 5      # O vence completando a linha 3-4-5


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


def test_sessao_humana():
    s = Sessao()
    assert s.metricas()["acuracia_jogadas"] is None
    s.usar_rede(Rede())  # pesos zero: escolhe sempre a casa 0
    s.nova("rede", humano=-1)
    s.jogar()           # rede X joga 0
    s.jogar(1)          # humano O joga 1
    s.jogar()           # rede repete 0
    m = s.metricas()
    assert s.fim == "invalida" and m["partidas"] == 1
    assert m["acuracia_jogadas"] == 0.5 and m["sem_derrota"] == 0
    s.nova("rede")
    s.jogar(4)
    s.nova("rede")      # abandonar partida não altera métricas concluídas
    assert s.metricas() == m
    s.usar_rede(Rede())
    assert s.metricas()["partidas"] == 0


def test_treino_e_persistencia():
    cfg = dict(CONFIG, N=6, paciencia=2, fases=[
        dict(modos=["medio"], geracoes=3), dict(modos=["dificil"], geracoes=3)])
    observados = []
    hist, _ = treinar(cfg, log=lambda _: None, ao_gerar=observados.append)
    repetido, _ = treinar(cfg, log=lambda _: None)
    assert hist == repetido and hist == observados
    assert {h["fase"] for h in hist} == {1, 2}
    assert all(len(h["aptidoes"]) == 6 for h in hist)
    assert hist[-1]["parada"] in ("convergencia", "limite_geracoes")
    melhor, _ = escolher_melhor(hist, cfg, n_partidas=4)
    with tempfile.TemporaryDirectory() as pasta:
        salvar(hist, melhor, cfg, pasta)
        assert carregar_rede(Path(pasta, "melhor_rede.json")).cromossomo() == melhor
        assert Path(pasta, "config.json").is_file()


def test_convergencia_avaliacao_fixa():
    # Um limiar de melhoria inalcançável força a parada por paciência antes do limite.
    cfg = dict(CONFIG, N=3, elite=2, taxa_cruzamento=0, taxa_mutacao=0,
               paciencia=1, min_geracoes=1, eps=100, fases=[dict(modos=["medio"], geracoes=20)])
    hist, _ = treinar(cfg, log=lambda _: None)
    assert len(hist) == 2 and hist[-1]["parada"] == "convergencia"


def test_bloqueio_jogada_ilegal():
    for t, p, jogador in [(tabuleiro_vazio(), 0, -1),
                          (tabuleiro_vazio(), 0.5, 1),
                          (tabuleiro_vazio(), -1, 1),
                          ((1, 0, 0, 0, 0, 0, 0, 0, 0), 0, -1),
                          ((1, 1, 1, -1, -1, 0, 0, 0, 0), 5, -1)]:
        try:
            aplicar(t, p, jogador)
        except ValueError:
            continue
        raise AssertionError("Aceitou jogada ilegal")


def test_decisoes_otimas_e_invalida():
    from aptidao import avaliar
    from minimax import jogadas_otimas
    class Perfeita:
        def jogar(self, t, jogador):
            return melhor_jogada(t, jogador)
    assert len(jogadas_otimas(tabuleiro_vazio(), 1)) == 9
    r = avaliar(Perfeita(), "dificil", n=8)
    assert r["acuracia_decisao"] == 1 and r["empate"] == 8
    r = avaliar(Rede(), "dificil", n=4)
    assert r["invalida"] == 4 and r["taxa_derrota"] == 1
    assert r["acuracia_decisao"] <= r["acuracia_jogadas"]
    assert r["por_inicio"]["primeiro"]["partidas"] == 2


def test_propagacao_e_validacao_modelo():
    import math
    import json
    r = Rede()
    pesos = [0.0] * 180
    pesos[0] = 2  # primeira entrada do primeiro neurônio oculto
    pesos[90] = 3  # primeira conexão da saída
    pesos[99] = 0.5  # bias da primeira saída
    r.carregar(pesos)
    assert math.isclose(r.propagar([1] + [0]*8)[0], 3*math.tanh(2) + 0.5)
    assert r.codificar((1, -1, 0, 0, 0, 0, 0, 0, 0), -1)[:3] == [-1, 1, 0]
    for invalido in ([0]*8, [float("nan")]*9):
        try:
            r.propagar(invalido)
        except ValueError:
            pass
        else:
            raise AssertionError("Aceitou entrada inválida")
    for valor in (float("inf"), float("nan"), "peso"):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta, "rede.json")
            caminho.write_text(json.dumps(dict(n_oculta=9, pesos=[valor]*180)))
            try:
                carregar_rede(caminho)
            except ValueError:
                pass
            else:
                raise AssertionError("Aceitou peso inválido")


def test_minimax_exaustivo():
    # Explora TODAS as respostas legais ao difícil, de ambos os lados.
    from functools import lru_cache
    @lru_cache(None)
    def visitar(t, vez, eu):
        fim = resultado(t)
        if fim is not None:
            assert fim in (0, eu)
            return
        opcoes = [melhor_jogada(t, vez)] if vez == eu else jogadas_validas(t)
        for pos in opcoes:
            visitar(aplicar(t, pos, vez), -vez, eu)
    for eu in (1, -1):
        visitar(tabuleiro_vazio(), 1, eu)


def test_sementes_independentes_por_partida():
    # Consumir aleatoriedade numa partida não deve mudar a próxima.
    from unittest.mock import patch
    import aptidao as modulo
    coletadas = []
    def simular(rede, modo, rede_comeca, rng, cfg):
        coletadas.append((modo, rede_comeca, rng.random()))
        for _ in range(rede.consumo):
            rng.random()
        return dict(pontos=0)
    r = Rede()
    with patch.object(modulo, "jogar_partida", simular):
        r.consumo = 1
        aptidao(r, r.cromossomo(), ["medio", "dificil"], 42)
        primeira = list(coletadas)
        coletadas.clear()
        r.consumo = 15
        aptidao(r, r.cromossomo(), ["medio", "dificil"], 42)
    assert coletadas == primeira


def test_curriculo_interrupcao_csv():
    from treino import criar_fases
    import csv
    fases = criar_fases(7)
    assert sum(f["geracoes"] for f in fases) == 7
    assert [f["pesos_modos"]["dificil"] for f in fases] == [.25, .5, .75]
    assert all(f["partidas_medio"] == 32 and f["partidas_dificil"] == 2 for f in fases)
    cfg = dict(CONFIG, N=3, fases=fases)
    registros = []
    hist, melhor = treinar(cfg, log=lambda _: None, ao_gerar=registros.append,
                           continuar=lambda: len(registros) < 2)
    assert len(hist) == 2 and melhor is not None
    with tempfile.TemporaryDirectory() as pasta:
        salvar(hist, melhor, cfg, pasta)
        with Path(pasta, "historico.csv").open() as arquivo:
            assert len(list(csv.DictReader(arquivo))) == 2


def test_minimo_antes_convergencia():
    cfg = dict(CONFIG, N=3, elite=2, taxa_cruzamento=0, taxa_mutacao=0,
               paciencia=1, min_geracoes=4, eps=100,
               fases=[dict(modos=["dificil"], geracoes=10)])
    hist, _ = treinar(cfg, log=lambda _: None)
    assert len(hist) == 4 and hist[-1]["parada"] == "convergencia"


def test_medio_escolhe_entre_busca_e_aleatorio():
    from unittest.mock import Mock
    t = tabuleiro_vazio()
    rng = Mock()
    rng.random.return_value = 0.49
    assert jogada_minimax(t, 1, "medio", rng) == melhor_jogada(t, 1)
    rng.choice.assert_not_called()
    rng.random.return_value = 0.50
    rng.choice.return_value = 8
    assert jogada_minimax(t, 1, "medio", rng) == 8
    rng.choice.assert_called_once_with(jogadas_validas(t))


def test_ponderacao_modos_independe_de_repeticoes():
    from unittest.mock import patch
    import aptidao as modulo
    r = Rede()
    def simular(rede, modo, rede_comeca, rng, cfg):
        return dict(pontos=3 if modo == "medio" else -2, resultado="vitoria")
    with patch.object(modulo, "jogar_partida", simular):
        for quantidade in (2, 20):
            cfg = dict(CONFIG_APTIDAO, partidas_medio=quantidade,
                       pesos_modos={"medio": .5, "dificil": .5})
            assert aptidao(r, r.cromossomo(), ["medio", "dificil"], 42, cfg) == .5


def test_cache_preserva_partidas_e_invalida_entre_cromossomos():
    from aptidao import avaliar
    r = Rede()
    cfg = dict(CONFIG_APTIDAO, partidas_medio=10, partidas_dificil=2)
    for bias in (0, 1):
        c = [0.0] * r.n_pesos
        c[109] = bias  # bias da segunda saída; troca a casa escolhida
        cont = aptidao(r, c, ["medio", "dificil"], 77, cfg, detalhar=True)
        amostras = [jogar_partida(r, modo, k % 2 == 0,
                    random.Random(f"77:{modo}:{k}"), cfg)
                    for modo, n in (("medio", 10), ("dificil", 2)) for k in range(n)]
        assert abs(cont["aptidao"] - sum(p["pontos"] for p in amostras)/12) < 1e-12
        for resultado_partida in ("vitoria", "empate", "derrota", "invalida"):
            assert cont[resultado_partida] == sum(p["resultado"] == resultado_partida for p in amostras)


def test_selecao_final_equilibra_os_modos():
    from unittest.mock import patch
    import treino as modulo
    cfg = dict(CONFIG)
    c1, c2 = [0.0]*180, [1.0]*180
    hist = [dict(fase=1, melhor=0, validacao=0, melhor_cromossomo=c) for c in (c1, c2)]
    def pontuar(rede, c, modos, semente, cfg):
        assert semente == 999
        assert cfg["pesos_modos"] == {"medio": .5, "dificil": .5}
        return 1 if c == c2 else 0
    with patch.object(modulo, "aptidao", pontuar):
        assert escolher_melhor(hist, cfg)[0] == c2


def test_inicializacao_local_aleatoria_e_configuravel():
    from treino import validar_config
    pop = ag.populacao_local(5, random.Random(42))
    assert pop == ag.populacao_local(5, random.Random(42))
    assert len({tuple(c) for c in pop}) == 5
    assert all(len(c) == 180 and all(-5 <= p <= 5 for p in c) for c in pop)
    for c in pop:
        r = Rede(codificacao="ocupacao")
        r.carregar(c)
        assert r.cromossomo() == c
    for incorreta in (dict(CONFIG, codificacao="sinal"),
                       dict(CONFIG, n_oculta=10),
                       dict(CONFIG, fases=[dict(modos=["medio", "dificil"], geracoes=1,
                            pesos_modos={"medio": 1, "dificil": float("nan")})])):
        try:
            validar_config(incorreta)
        except ValueError:
            pass
        else:
            raise AssertionError("Aceitou configuração incompatível")


def test_desempenho_modelo_padrao():
    from aptidao import avaliar
    modelo = Path(__file__).resolve().parent / "resultados" / "melhor_rede.json"
    rede = carregar_rede(modelo)
    medio = avaliar(rede, "medio", n=200, semente=123)
    dificil = avaliar(rede, "dificil", n=2, semente=123)
    assert medio["vitoria"] >= 60 and medio["invalida"] <= 5
    assert dificil["empate"] == 2


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok ", nome)
