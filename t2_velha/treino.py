"""Ciclo de treino: o AG evolui a população; cada cromossomo joga contra o Minimax.

Uso:  python3 treino.py                 (treina e salva resultados_execucao/)
      python3 treino.py --passo         (pausa a cada geração para acompanhar passo a passo)
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import random
import statistics
import time
from pathlib import Path

import ag
from aptidao import CONFIG_APTIDAO, aptidao, avaliar
from rede import Rede

def criar_fases(total):
    """Currículo 50%/25%/25%; pesos por modo 75/25 → 50/50 → 25/75."""
    if total < 3:
        raise ValueError("Informe pelo menos 3 gerações para os três estágios")
    primeira = max(1, total // 2)
    segunda = max(1, total // 4)
    return [dict(modos=["medio", "dificil"], geracoes=n,
                 partidas_medio=32, partidas_dificil=2,
                 pesos_modos={"medio": 1-w, "dificil": w})
            for n, w in ((primeira, .25), (segunda, .5), (total-primeira-segunda, .75))]


CONFIG = dict(
    N=120,                      # tamanho da população (configurável)
    n_oculta=9,                 # neurônios da camada oculta
    codificacao="ocupacao",     # vazio=0, própria marca=1, adversário=2
    inicializacao="local",     # aleatória, com viés de ocupação; alternativa: "uniforme"
    partidas_fixas=True,        # lote de treino fixo; validação usa outras sementes
    elite=2,                    # quantos melhores passam direto
    torneio_k=3,
    taxa_cruzamento=0.3,
    cruzamento="aritmetico",    # "blx" ou "aritmetico"
    alfa=0.3,
    taxa_mutacao=0.025,
    tipo_mutacao="ruido",       # "aleatoria" (troca o gene) ou "ruido" (soma ruído pequeno)
    sigma_mutacao=0.2,
    fases=criar_fases(600),
    min_geracoes=100,           # mínimo por fase antes de parar por estagnação
    paciencia=100,              # convergência: gerações seguidas sem melhora do melhor
    eps=0.01,                   # melhora mínima que conta
    semente=99,
)


def validar_config(cfg):
    """Falha antes de começar, em vez de aceitar um treino sem sentido."""
    if cfg.get("inicializacao", "uniforme") not in ("uniforme", "local"):
        raise ValueError("Inicialização desconhecida")
    if cfg.get("inicializacao") == "local" and (cfg["n_oculta"] != 9 or cfg["codificacao"] != "ocupacao"):
        raise ValueError("Inicialização local exige topologia 9–9–9 e codificação ocupacao")
    if not isinstance(cfg.get("min_geracoes", 30), int) or cfg.get("min_geracoes", 30) < 1:
        raise ValueError("min_geracoes deve ser inteiro positivo")
    for chave in ("N", "n_oculta", "paciencia", "torneio_k"):
        if not isinstance(cfg[chave], int) or cfg[chave] < 1:
            raise ValueError(f"{chave} deve ser inteiro positivo")
    if not 0 <= cfg["elite"] < cfg["N"] or cfg["torneio_k"] > cfg["N"]:
        raise ValueError("Exija 0 <= elite < N e torneio_k <= N")
    for chave in ("taxa_cruzamento", "taxa_mutacao"):
        if not 0 <= cfg[chave] <= 1:
            raise ValueError(f"{chave} deve estar entre 0 e 1")
    if cfg["cruzamento"] not in ("blx", "aritmetico"):
        raise ValueError("Cruzamento desconhecido")
    if cfg["tipo_mutacao"] not in ("aleatoria", "ruido"):
        raise ValueError("Mutação desconhecida")
    if cfg["eps"] < 0 or cfg["alfa"] < 0 or cfg["sigma_mutacao"] < 0:
        raise ValueError("eps, alfa e sigma devem ser não negativos")
    if not cfg["fases"]:
        raise ValueError("Informe ao menos uma fase")
    for fase in cfg["fases"]:
        if not isinstance(fase["geracoes"], int) or fase["geracoes"] < 1:
            raise ValueError("Cada fase precisa de gerações inteiras positivas")
        if not fase["modos"] or any(m not in ("medio", "dificil") for m in fase["modos"]):
            raise ValueError("Modos permitidos: medio e dificil")
        for chave in ("partidas_medio", "partidas_dificil"):
            if chave in fase and (not isinstance(fase[chave], int) or fase[chave] < 1):
                raise ValueError("Partidas por modo devem ser inteiros positivos")
        pesos = fase.get("pesos_modos", {})
        if pesos:
            if (set(pesos) != set(fase["modos"]) or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                        for v in pesos.values())):
                raise ValueError("Informe um peso finito positivo para cada modo da fase")


def treinar(cfg=CONFIG, passo=False, log=print, ao_gerar=None, continuar=None):
    validar_config(cfg)
    rng = random.Random(cfg["semente"])
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"])
    pop = (ag.populacao_local(cfg["N"], rng) if cfg.get("inicializacao") == "local"
           else ag.populacao_inicial(cfg["N"], rede.n_pesos, rng))
    historico, geracao = [], 0
    melhor_global, apt_global = None, float("-inf")
    for f, fase in enumerate(cfg["fases"]):
        sem_melhora, melhor_fase = 0, float("-inf")
        campeao_fase = None
        cfg_apt = dict(CONFIG_APTIDAO, **{k: fase[k] for k in
                       ("partidas_medio", "partidas_dificil", "pesos_modos") if k in fase})
        for indice in range(fase["geracoes"]):
            if continuar is not None and not continuar():
                if historico:
                    historico[-1]["parada"] = "interrompido"
                return historico, melhor_global
            geracao += 1
            semente = (cfg["semente"] if cfg.get("partidas_fixas", False)
                       else rng.randrange(10**9))
            apts = [aptidao(rede, c, fase["modos"], semente, cfg_apt) for c in pop]
            i_melhor = max(range(len(pop)), key=lambda i: apts[i])
            reg = dict(semente=semente, geracao=geracao, fase=f + 1, modos=fase["modos"], partidas_medio=cfg_apt["partidas_medio"],
                       partidas_dificil=cfg_apt["partidas_dificil"],
                       pesos_modos=cfg_apt.get("pesos_modos"),
                       melhor=apts[i_melhor], media=statistics.mean(apts), pior=min(apts),
                       desvio=statistics.pstdev(apts), melhor_cromossomo=list(pop[i_melhor]),
                       aptidoes=list(apts))
            reg["desempenho"] = aptidao(rede, pop[i_melhor], fase["modos"], semente, cfg_apt, detalhar=True)
            historico.append(reg)
            # Compara candidato e campeão nas mesmas partidas de validação.
            # A validação usa outra semente, inclusive quando o treino tem lote fixo.
            validacao = aptidao(rede, pop[i_melhor], fase["modos"], 2026, cfg_apt)
            reg["validacao"] = validacao
            if apts[i_melhor] > apt_global:
                melhor_global, apt_global = list(pop[i_melhor]), apts[i_melhor]
            log(f"ger {geracao:3d} | fase {f+1} {'+'.join(fase['modos']):14s} | "
                f"melhor {reg['melhor']:6.2f} | média {reg['media']:6.2f} | pior {reg['pior']:6.2f}")
            if validacao > melhor_fase + cfg["eps"]:
                melhor_fase, sem_melhora = validacao, 0
                campeao_fase = list(pop[i_melhor])
            else:
                sem_melhora += 1
            convergiu = sem_melhora >= cfg["paciencia"] and indice + 1 >= cfg.get("min_geracoes", 30)
            reg["melhor_acumulado"] = apt_global
            reg["parada"] = ("convergencia" if convergiu else
                             "limite_geracoes" if indice + 1 == fase["geracoes"] else None)
            if ao_gerar is not None:
                ao_gerar(reg)
            if passo:
                input("   [Enter] próxima geração ... ")
            if convergiu:
                log(f"   convergiu na fase {f+1} (sem melhora há {cfg['paciencia']} gerações)")
                break
            if indice + 1 < fase["geracoes"]:
                pop = ag.nova_geracao(pop, apts, cfg, rng)
                # Evita perder o campeão validado por ruído do adversário médio.
                if campeao_fase is not None:
                    pop[-1] = list(campeao_fase)
        # Leva o campeão validado para a fase seguinte sem descartar a população.
        pop[0] = campeao_fase
    return historico, melhor_global


def escolher_melhor(historico, cfg, n_partidas=100, n_candidatos=10):
    """A aptidão de uma geração é ruidosa (o modo médio é aleatório), então o 'melhor' de uma geração
    pode ter só dado sorte. Reavalia os melhores das últimas gerações com mais partidas e fica com o melhor."""
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"])
    if not historico or n_partidas < 1 or n_candidatos < 1:
        raise ValueError("Histórico e quantidades de avaliação devem ser positivos")
    candidatos = [h["melhor_cromossomo"] for h in historico[-n_candidatos:]]
    # Inclui também o campeão validado de cada fase, evitando esquecer fases anteriores.
    for fase in sorted({h["fase"] for h in historico}):
        campeao = max((h for h in historico if h["fase"] == fase),
                      key=lambda h: h.get("validacao", h["melhor"]))
        candidatos.append(campeao["melhor_cromossomo"])
    # Mais amostras no médio reduzem ruído, sem diluir o difícil na seleção.
    cfg_apt = dict(CONFIG_APTIDAO, partidas_medio=n_partidas,
                   pesos_modos={"medio": 0.5, "dificil": 0.5})
    pontuacao = [aptidao(rede, c, ["medio", "dificil"], 999, cfg_apt) for c in candidatos]
    i = max(range(len(candidatos)), key=lambda k: pontuacao[k])
    return candidatos[i], pontuacao[i]


def exportar_csv(historico, caminho):
    campos = ["geracao", "fase", "melhor", "media", "pior", "desvio", "validacao", "melhor_acumulado", "parada"]
    with Path(caminho).open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(historico)


def salvar(historico, melhor, cfg, pasta=Path(__file__).resolve().parent / "resultados_execucao", medidas=None):
    """Mantém o formato antigo legível e acrescenta metadados e CSV."""
    Path(pasta).mkdir(parents=True, exist_ok=True)
    Path(pasta, "config.json").write_text(json.dumps(
        {"treino": cfg, "aptidao": CONFIG_APTIDAO}, indent=2), encoding="utf-8")
    if medidas is not None:
        Path(pasta, "avaliacao.json").write_text(json.dumps(
            {"semente_avaliacao": 123, "resultados": medidas}, indent=2), encoding="utf-8")
    exportar_csv(historico, Path(pasta, "historico.csv"))
    leve = [{k: v for k, v in h.items() if k not in ("melhor_cromossomo", "aptidoes")} for h in historico]
    Path(pasta, "historico.json").write_text(json.dumps(leve, indent=1))
    Path(pasta, "populacoes.json").write_text(json.dumps(
        [{"geracao": h["geracao"], "aptidoes": h["aptidoes"], "melhor_cromossomo": h["melhor_cromossomo"]}
         for h in historico]))
    Path(pasta, "melhor_rede.json").write_text(json.dumps(
        {"versao": 2, "data_treino": datetime.now(timezone.utc).isoformat(),
         "topologia": [18 if cfg["codificacao"] == "dois_planos" else 9, cfg["n_oculta"], 9],
         "avaliacao": medidas, "config": cfg, "n_oculta": cfg["n_oculta"], "codificacao": cfg["codificacao"], "pesos": melhor}))


def carregar_rede(caminho=Path(__file__).resolve().parent / "resultados" / "melhor_rede.json"):
    d = json.loads(Path(caminho).read_text())
    rede = Rede(n_oculta=d["n_oculta"], codificacao=d.get("codificacao", "sinal"))
    if "topologia" in d and d["topologia"] != [rede.n_entrada, rede.n_oculta, rede.n_saida]:
        raise ValueError("Topologia incompatível com os parâmetros")
    rede.carregar(d["pesos"])
    return rede


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passo", action="store_true")
    ap.add_argument("--semente", type=int, default=CONFIG["semente"])
    ap.add_argument("--N", type=int, default=CONFIG["N"])
    ap.add_argument("--geracoes", type=int, default=sum(f["geracoes"] for f in CONFIG["fases"]), help="limite total dividido nos três estágios")
    ap.add_argument("--paciencia", type=int, default=CONFIG["paciencia"])
    ap.add_argument("--saida", type=Path, default=Path(__file__).resolve().parent / "resultados_execucao",
                    help="pasta da execução; preserva o modelo de referência em resultados")
    a = ap.parse_args()
    cfg = dict(CONFIG, semente=a.semente, N=a.N, paciencia=a.paciencia,
               fases=criar_fases(a.geracoes))
    t0 = time.time()
    hist, _ = treinar(cfg, passo=a.passo)
    melhor, _ = escolher_melhor(hist, cfg)
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"]); rede.carregar(melhor)
    print(f"\nTreino: {len(hist)} gerações em {time.time()-t0:.1f}s")
    medidas = {}
    for modo, n in [("dificil", 2), ("medio", 200)]:
        r = avaliar(rede, modo, n=n, semente=123)
        medidas[modo] = r
        print(f"{modo:8s} {n:3d} partidas: {r['vitoria']} vit, {r['empate']} emp, {r['derrota']} der, "
              f"{r['invalida']} inválidas | decisões ótimas {r['acuracia_decisao']:.1%} | jogadas válidas {r['acuracia_jogadas']:.1%} | sem derrota {r['partidas_sem_derrota']:.1%}")
    salvar(hist, melhor, cfg, a.saida, medidas)
