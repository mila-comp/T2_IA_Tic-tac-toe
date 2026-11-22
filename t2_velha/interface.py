"""Front end mínimo em Tkinter: python3 t2_velha/interface.py."""
import queue
import random
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from aptidao import avaliar
from jogo import aplicar, resultado, tabuleiro_vazio, jogadas_validas
from minimax import jogada_minimax
from sessao import Sessao
from treino import CONFIG, carregar_rede, escolher_melhor, salvar, treinar, validar_config, exportar_csv, criar_fases
from rede import Rede

BASE = Path(__file__).resolve().parent


class Interface:
    def __init__(self, raiz):
        self.raiz, self.sessao = raiz, Sessao()
        self.fila, self.avancar = queue.Queue(), threading.Event()
        self.fechado, self.treinando = False, False
        self.pausar = threading.Event()
        self.interromper = threading.Event()
        self.demonstracao = 0
        raiz.title("T2 IA — Rede neural + AG + Minimax")
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)
        self.adversario = tk.StringVar(value="minimax")
        self.modo = tk.StringVar(value="dificil")
        self.simbolo = tk.StringVar(value="X")
        controles = ttk.Frame(raiz, padding=10)
        controles.pack(fill="x")
        for texto, variavel, valores in (
            ("Adversário", self.adversario, ("minimax", "rede")),
            ("Minimax", self.modo, ("medio", "dificil")),
            ("Você", self.simbolo, ("X", "O")),
        ):
            ttk.Label(controles, text=texto).pack(side="left")
            ttk.Combobox(controles, textvariable=variavel, values=valores,
                         state="readonly", width=9).pack(side="left", padx=5)
        ttk.Button(controles, text="Nova partida", command=self.nova).pack(side="left")
        ttk.Button(raiz, text="Carregar rede JSON", command=self.carregar).pack()
        tabuleiro = ttk.Frame(raiz, padding=8)
        tabuleiro.pack()
        self.casas = []
        for i in range(9):
            b = ttk.Button(tabuleiro, text=" ", width=8, command=lambda p=i: self.jogar(p))
            b.grid(row=i // 3, column=i % 3, ipadx=6, ipady=12)
            self.casas.append(b)
        self.estado, self.metricas = tk.StringVar(), tk.StringVar()
        ttk.Label(raiz, textvariable=self.estado).pack()
        ttk.Label(raiz, textvariable=self.metricas, wraplength=800).pack(pady=5)
        treino = ttk.LabelFrame(raiz, text="Rede aprendendo com Minimax", padding=8)
        treino.pack(fill="both", expand=True)
        self.campos = {}
        for i, (chave, valor, rotulo) in enumerate((
            ("N", CONFIG["N"], "População N"), ("geracoes", sum(f["geracoes"] for f in CONFIG["fases"]), "Gerações totais"),
            ("paciencia", CONFIG["paciencia"], "Paciência"), ("semente", CONFIG["semente"], "Semente"),
        )):
            ttk.Label(treino, text=rotulo).grid(row=0, column=i * 2)
            v = tk.StringVar(value=str(valor))
            ttk.Entry(treino, textvariable=v, width=6).grid(row=0, column=i * 2 + 1)
            self.campos[chave] = v
        self.passo = tk.BooleanVar(value=True)
        ttk.Checkbutton(treino, text="Pausar a cada geração", variable=self.passo, command=self.definir_pausa).grid(row=1, column=0, columnspan=3)
        self.botao_treino = ttk.Button(treino, text="Treinar nova rede", command=self.treinar)
        self.botao_treino.grid(row=1, column=3, columnspan=2)
        ttk.Button(treino, text="Próxima geração", command=self.proxima).grid(row=1, column=5, columnspan=3)
        comandos = ttk.Frame(treino)
        comandos.grid(row=5, column=0, columnspan=8)
        for texto, comando in (("Executar automático", self.automatico),
                                ("Pausar", self.pausar_treino), ("Interromper", self.parar),
                                ("Demonstrar melhor selecionado", self.demonstrar),
                                ("Exportar CSV", self.exportar)):
            ttk.Button(comandos, text=texto, command=comando).pack(side="left", padx=2)
        self.grafico = tk.Canvas(treino, width=760, height=130, background="white")
        self.grafico.grid(row=6, column=0, columnspan=8, sticky="ew")
        self.tabela = ttk.Treeview(treino, columns=("g", "f", "melhor", "media", "pior", "val"),
                                  show="headings", height=7)
        for chave, titulo in zip(self.tabela["columns"], ("Geração", "Fase", "Melhor", "Média", "Pior", "Validação")):
            self.tabela.heading(chave, text=titulo)
            self.tabela.column(chave, width=110)
        self.tabela.grid(row=2, column=0, columnspan=8, sticky="nsew")
        self.registros = {}
        self.tabela.bind("<<TreeviewSelect>>", self.detalhar)
        self.detalhes = tk.Text(treino, height=4, width=95, state="disabled", wrap="word")
        self.detalhes.grid(row=3, column=0, columnspan=8)
        self.progresso = tk.StringVar(value="32 partidas médias + 2 difíceis; pesos 75/25 → 50/50 → 25/75. Selecione uma geração para ver a população.")
        ttk.Label(treino, textvariable=self.progresso, wraplength=800).grid(row=4, column=0, columnspan=8)
        padrao = BASE / "resultados" / "melhor_rede.json"
        if padrao.exists():
            try:
                self.sessao.usar_rede(carregar_rede(padrao))
            except (OSError, ValueError, KeyError, TypeError) as e:
                self.progresso.set(f"Não foi possível carregar a rede padrão: {e}")
        self.nova()
        raiz.after(100, self.processar)

    def nova(self):
        self.demonstracao += 1
        try:
            self.sessao.nova(self.adversario.get(), self.modo.get(), 1 if self.simbolo.get() == "X" else -1)
            if self.sessao.vez != self.sessao.humano:
                self.sessao.jogar()
            self.atualizar()
        except ValueError as e:
            messagebox.showerror("Partida", str(e))

    def jogar(self, pos):
        try:
            self.sessao.jogar(pos)
            if self.sessao.fim is None:
                self.sessao.jogar()
            self.atualizar()
        except ValueError as e:
            messagebox.showerror("Jogada", str(e))

    def atualizar(self):
        s = self.sessao
        for i, b in enumerate(self.casas):
            b.configure(text={0: " ", 1: "X", -1: "O"}[s.tabuleiro[i]],
                        state="normal" if s.fim is None and s.tabuleiro[i] == 0 else "disabled")
        mensagens = {"vitoria": "A IA venceu.", "derrota": "Você venceu.",
                     "empate": "Empate.", "invalida": "A rede escolheu casa ocupada: penalidade e fim da partida."}
        self.estado.set(mensagens.get(s.fim, "Sua vez."))
        m = s.metricas()
        percentual = lambda v: "—" if v is None else f"{v:.1%}"
        self.metricas.set(f"Rede × pessoa (partidas concluídas): {m['partidas']} | "
                         f"V/E/D/Inv: {m['vitoria']}/{m['empate']}/{m['derrota']}/{m['invalida']} | "
                         f"Decisões ótimas: {percentual(m['acuracia_decisao'])} | "
                         f"Jogadas válidas: {percentual(m['acuracia_jogadas'])} | "
                         f"Sem derrota: {percentual(m['sem_derrota'])}")

    def carregar(self):
        if self.treinando:
            return
        arquivo = filedialog.askopenfilename(initialdir=BASE / "resultados", filetypes=[("Rede", "*.json")])
        if arquivo:
            try:
                self.sessao.usar_rede(carregar_rede(arquivo))
                self.adversario.set("rede")
                self.nova()
            except (OSError, ValueError, KeyError, TypeError) as e:
                messagebox.showerror("Carregar rede", str(e))

    def treinar(self):
        if self.treinando:
            return
        try:
            valores = {k: int(v.get()) for k, v in self.campos.items()}
            geracoes = valores.pop("geracoes")
            cfg = dict(CONFIG, **valores, fases=criar_fases(geracoes))
            validar_config(cfg)
        except ValueError as e:
            messagebox.showerror("Configuração", str(e))
            return
        self.treinando = True
        self.botao_treino.configure(state="disabled")
        self.tabela.delete(*self.tabela.get_children())
        self.registros.clear()
        self.interromper.clear()
        self.definir_pausa()
        self.desenhar_grafico()
        self.progresso.set("Treinando…")

        def ao_gerar(reg):
            if self.fechado:
                raise InterruptedError("Janela fechada")
            self.avancar.clear()
            self.fila.put(("geracao", reg))
            while (self.pausar.is_set() and not self.interromper.is_set()
                   and not (reg["parada"] and reg["fase"] == len(cfg["fases"]))):
                if self.avancar.wait(0.1):
                    break
            if self.fechado:
                raise InterruptedError("Janela fechada")

        def executar():
            try:
                hist, _ = treinar(cfg, log=lambda _: None, ao_gerar=ao_gerar,
                                  continuar=lambda: not self.interromper.is_set())
                if not hist:
                    raise InterruptedError("Treino interrompido antes da primeira geração")
                pesos, _ = escolher_melhor(hist, cfg)
                rede = Rede(n_oculta=cfg["n_oculta"], codificacao=cfg["codificacao"])
                rede.carregar(pesos)
                pasta = BASE / "resultados_interface"
                medidas = {m: avaliar(rede, m, n=n, semente=123) for m, n in (("medio", 200), ("dificil", 2))}
                salvar(hist, pesos, cfg, pasta, medidas)
                self.fila.put(("fim", (rede, medidas)))
            except Exception as e:
                self.fila.put(("erro", str(e)))
        threading.Thread(target=executar, daemon=True).start()

    def detalhar(self, _=None):
        selecionado = self.tabela.selection()
        if selecionado:
            r = self.registros[selecionado[0]]
            self.detalhes.configure(state="normal")
            self.detalhes.delete("1.0", "end")
            self.detalhes.insert("end", f"Partidas médio/difícil: {r.get('partidas_medio')}/{r.get('partidas_dificil')} | Parada: {r['parada'] or 'continua'}\n"
                                 f"Pesos por modo: {r.get('pesos_modos') or 'proporcionais às partidas'}\n"
                                 f"Melhor indivíduo: {r.get('desempenho', {})}\n"
                                 f"Aptidão de cada um dos {len(r['aptidoes'])} indivíduos:\n" +
                                 ", ".join(f"{i+1}: {a:.2f}" for i, a in enumerate(r['aptidoes'])))
            self.detalhes.configure(state="disabled")

    def processar(self):
        try:
            while True:
                tipo, dado = self.fila.get_nowait()
                if tipo == "geracao":
                    item = self.tabela.insert("", "end", values=(dado["geracao"], dado["fase"],
                        *(f"{dado[k]:.2f}" for k in ("melhor", "media", "pior", "validacao"))))
                    self.registros[item] = dado
                    self.tabela.see(item)
                    self.tabela.selection_set(item)
                    self.detalhar()
                    self.desenhar_grafico()
                    self.progresso.set(f"Geração {dado['geracao']} | Recorde {dado['melhor_acumulado']:.2f} — {dado['parada'] or 'em evolução'}. "
                                       "Use Próxima geração se a pausa estiver ativa.")
                else:
                    self.treinando = False
                    self.botao_treino.configure(state="normal")
                    if tipo == "erro":
                        messagebox.showerror("Treino", dado)
                    else:
                        rede, medidas = dado
                        self.sessao.usar_rede(rede)
                        self.adversario.set("rede")
                        self.nova()
                        self.progresso.set("Rede salva em resultados_interface/. " + " | ".join(
                            f"{m}: decisões ótimas {r['acuracia_decisao']:.1%}, válidas {r['acuracia_jogadas']:.1%}, sem derrota {r['partidas_sem_derrota']:.1%}"
                            for m, r in medidas.items()))
        except queue.Empty:
            pass
        if not self.fechado:
            self.raiz.after(100, self.processar)

    def definir_pausa(self):
        if self.passo.get():
            self.pausar.set()
        else:
            self.automatico()

    def pausar_treino(self):
        self.pausar.set()
        self.passo.set(True)

    def proxima(self):
        self.pausar.set()
        self.passo.set(True)
        self.avancar.set()

    def automatico(self):
        self.pausar.clear()
        self.passo.set(False)
        self.avancar.set()

    def parar(self):
        self.interromper.set()
        self.avancar.set()
        self.progresso.set("Interrompendo após a geração atual; resultados concluídos serão salvos.")

    def exportar(self):
        if not self.registros:
            return
        arquivo = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if arquivo:
            try:
                exportar_csv(list(self.registros.values()), arquivo)
            except OSError as e:
                messagebox.showerror("Exportar", str(e))

    def desenhar_grafico(self):
        c = self.grafico
        c.delete("all")
        c.create_text(10, 10, anchor="nw", text="Aptidão por geração: melhor (azul), média (laranja)")
        registros = list(self.registros.values())
        if not registros:
            return
        valores = [r[k] for r in registros for k in ("melhor", "media")]
        lo, hi = min(valores), max(valores)
        c.create_text(10, 35, anchor="nw", text=f"{hi:.2f}")
        c.create_text(10, 110, anchor="nw", text=f"{lo:.2f}")
        for chave, cor in (("melhor", "royalblue"), ("media", "darkorange")):
            pontos = [(65 + i * 680 / max(1, len(registros) - 1),
                       112 - (r[chave] - lo) * 75 / max(1e-9, hi - lo))
                      for i, r in enumerate(registros)]
            if len(pontos) > 1:
                c.create_line(*[v for ponto in pontos for v in ponto], fill=cor, width=2)
            else:
                x, y = pontos[0]
                c.create_oval(x-2, y-2, x+2, y+2, fill=cor)

    def demonstrar(self):
        selecionado = self.tabela.selection()
        if not selecionado:
            return
        registro = self.registros[selecionado[0]]
        rede = Rede(n_oculta=CONFIG["n_oculta"], codificacao=CONFIG["codificacao"])
        rede.carregar(registro["melhor_cromossomo"])
        self.demonstracao += 1
        identificador = self.demonstracao
        t, vez = tabuleiro_vazio(), 1
        eu = 1 if self.simbolo.get() == "X" else -1
        modo = self.modo.get()
        rng = random.Random(42)

        def passo():
            nonlocal t, vez
            if self.fechado or identificador != self.demonstracao:
                return
            pos = rede.jogar(t, vez) if vez == eu else jogada_minimax(t, vez, modo, rng)
            if pos not in jogadas_validas(t):
                self.estado.set("Demonstração: rede propôs casa ocupada; derrota por jogada inválida.")
                return
            t = aplicar(t, pos, vez)
            for i, botao in enumerate(self.casas):
                botao.configure(text={0: " ", 1: "X", -1: "O"}[t[i]], state="disabled")
            fim = resultado(t)
            self.estado.set(f"Demonstração da geração {registro['geracao']} — " +
                            ("empate" if fim == 0 else "rede venceu" if fim == eu else
                             "Minimax venceu" if fim is not None else "em andamento"))
            if fim is None:
                vez = -vez
                self.raiz.after(500, passo)
        passo()

    def fechar(self):
        self.fechado = True
        self.interromper.set()
        self.avancar.set()
        self.raiz.destroy()


if __name__ == "__main__":
    raiz = tk.Tk()
    Interface(raiz)
    raiz.mainloop()
