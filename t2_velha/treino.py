"""Ciclo de treino: o AG evolui a população; cada cromossomo joga contra o Minimax.

Uso:  python3 treino.py                 (treina e salva resultados/)
      python3 treino.py --passo         (pausa a cada geração para acompanhar passo a passo)
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import ag
from aptidao import CONFIG_APTIDAO, aptidao, avaliar
from rede import Rede

# Ordem dos modos do Minimax durante o aprendizado (decisão do grupo, comparada em experimentos.py)
FASES_MEDIO_DEPOIS_DIFICIL = [
    {"modos": ["medio"], "geracoes": 400},
    {"modos": ["medio", "dificil"], "geracoes": 400},
]

CONFIG = dict(
    N=200,                      # tamanho da população (configurável)
    n_oculta=9,                 # neurônios da camada oculta
    codificacao="ocupacao",     # entrada da rede: "sinal", "ocupacao" ou "dois_planos" (ver rede.py)
    elite=2,                    # quantos melhores passam direto
    torneio_k=3,
    taxa_cruzamento=0.9,
    cruzamento="blx",           # "blx" ou "aritmetico"
    alfa=0.3,
    taxa_mutacao=0.05,
    tipo_mutacao="ruido",       # "aleatoria" (troca o gene) ou "ruido" (soma ruído pequeno)
    sigma_mutacao=0.2,
    fases=FASES_MEDIO_DEPOIS_DIFICIL,
    paciencia=60,               # convergência: gerações seguidas sem melhora do melhor
    eps=0.05,                   # melhora mínima que conta
    semente=42,
)


def treinar(cfg=CONFIG, passo=False, log=print):
    rng = random.Random(cfg["semente"])
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"])
    pop = ag.populacao_inicial(cfg["N"], rede.n_pesos, rng)
    historico, geracao = [], 0
    melhor_global, apt_global = None, float("-inf")
    for f, fase in enumerate(cfg["fases"]):
        sem_melhora, melhor_fase = 0, float("-inf")
        for _ in range(fase["geracoes"]):
            geracao += 1
            semente = rng.randrange(10**9)            # mesmas partidas para todos nesta geração
            apts = [aptidao(rede, c, fase["modos"], semente) for c in pop]
            i_melhor = max(range(len(pop)), key=lambda i: apts[i])
            reg = dict(geracao=geracao, fase=f + 1, modos=fase["modos"],
                       melhor=apts[i_melhor], media=statistics.mean(apts), pior=min(apts),
                       desvio=statistics.pstdev(apts), melhor_cromossomo=list(pop[i_melhor]),
                       aptidoes=list(apts))
            historico.append(reg)
            if apts[i_melhor] > apt_global:
                melhor_global, apt_global = list(pop[i_melhor]), apts[i_melhor]
            log(f"ger {geracao:3d} | fase {f+1} {'+'.join(fase['modos']):14s} | "
                f"melhor {reg['melhor']:6.2f} | média {reg['media']:6.2f} | pior {reg['pior']:6.2f}")
            if passo:
                input("   [Enter] próxima geração ... ")
            # convergência: o melhor não melhora há `paciencia` gerações
            if reg["melhor"] > melhor_fase + cfg["eps"]:
                melhor_fase, sem_melhora = reg["melhor"], 0
            else:
                sem_melhora += 1
            if sem_melhora >= cfg["paciencia"]:
                log(f"   convergiu na fase {f+1} (sem melhora há {cfg['paciencia']} gerações)")
                break
            pop = ag.nova_geracao(pop, apts, cfg, rng)
    return historico, melhor_global


def escolher_melhor(historico, cfg, n_partidas=100, n_candidatos=10):
    """A aptidão de uma geração é ruidosa (o modo médio é aleatório), então o 'melhor' de uma geração
    pode ter só dado sorte. Reavalia os melhores das últimas gerações com mais partidas e fica com o melhor."""
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"])
    candidatos = [h["melhor_cromossomo"] for h in historico[-n_candidatos:]]
    ultima = historico[-1]
    ordem = sorted(range(len(ultima["aptidoes"])), key=lambda i: ultima["aptidoes"][i], reverse=True)
    cfg_apt = dict(CONFIG_APTIDAO, partidas_medio=n_partidas)
    pontuacao = [aptidao(rede, c, ["medio", "dificil"], 999, cfg_apt) for c in candidatos]
    i = max(range(len(candidatos)), key=lambda k: pontuacao[k])
    return candidatos[i], pontuacao[i]


def salvar(historico, melhor, cfg, pasta="resultados"):
    Path(pasta).mkdir(exist_ok=True)
    leve = [{k: v for k, v in h.items() if k not in ("melhor_cromossomo", "aptidoes")} for h in historico]
    Path(pasta, "historico.json").write_text(json.dumps(leve, indent=1))
    Path(pasta, "populacoes.json").write_text(json.dumps(
        [{"geracao": h["geracao"], "aptidoes": h["aptidoes"], "melhor_cromossomo": h["melhor_cromossomo"]}
         for h in historico]))
    Path(pasta, "melhor_rede.json").write_text(json.dumps(
        {"n_oculta": cfg["n_oculta"], "codificacao": cfg["codificacao"], "pesos": melhor}))


def carregar_rede(caminho="resultados/melhor_rede.json"):
    d = json.loads(Path(caminho).read_text())
    rede = Rede(n_oculta=d["n_oculta"], codificacao=d.get("codificacao", "sinal"))
    rede.carregar(d["pesos"])
    return rede


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passo", action="store_true")
    ap.add_argument("--semente", type=int, default=CONFIG["semente"])
    ap.add_argument("--N", type=int, default=CONFIG["N"])
    a = ap.parse_args()
    cfg = dict(CONFIG, semente=a.semente, N=a.N)
    t0 = time.time()
    hist, _ = treinar(cfg, passo=a.passo)
    melhor, _ = escolher_melhor(hist, cfg)
    salvar(hist, melhor, cfg)
    rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"]); rede.carregar(melhor)
    print(f"\nTreino: {len(hist)} gerações em {time.time()-t0:.1f}s")
    for modo, n in [("dificil", 2), ("medio", 200)]:
        r = avaliar(rede, modo, n=n, semente=123)
        print(f"{modo:8s} {n:3d} partidas: {r['vitoria']} vit, {r['empate']} emp, {r['derrota']} der, "
              f"{r['invalida']} inválidas | jogadas válidas {r['acuracia_jogadas']:.1%} | sem derrota {r['partidas_sem_derrota']:.1%}")
