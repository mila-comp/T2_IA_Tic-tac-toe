"""Avaliação reproduzível de modelos prontos; não treina nem seleciona pesos."""
import argparse
import json
from pathlib import Path

from aptidao import avaliar
from treino import carregar_rede


def medir(caminho, n=2000, semente=8675309):
    rede = carregar_rede(caminho)
    return {modo: avaliar(rede, modo, n=quantidade, semente=semente)
            for modo, quantidade in (("medio", n), ("dificil", 2))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rede", type=Path, default=Path(__file__).resolve().parent /
                    "resultados" / "melhor_rede.json")
    ap.add_argument("--comparar", type=Path)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--semente", type=int, default=8675309)
    ap.add_argument("--saida", type=Path)
    args = ap.parse_args()
    modelos = {"nova": args.rede}
    if args.comparar:
        modelos["anterior"] = args.comparar
    dados = dict(semente_avaliacao=args.semente, modelos={})
    for nome, caminho in modelos.items():
        medidas = medir(caminho, args.n, args.semente)
        dados["modelos"][nome] = dict(arquivo=str(caminho), resultados=medidas)
        for modo, r in medidas.items():
            print(f"{nome} / {modo}: {r['vitoria']} vitórias, {r['empate']} empates, "
                  f"{r['derrota']} derrotas, {r['invalida']} inválidas; "
                  f"decisões ótimas {r['acuracia_decisao']:.1%}")
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps(dados, indent=2), encoding="utf-8")
