# T2 — Jogo da Velha: MLP + AG + Minimax

MLP própria, com pesos evoluídos exclusivamente por algoritmo genético. Python 3
com Tkinter; não precisa instalar NumPy, frameworks de IA ou outras bibliotecas
para executar a aplicação. Mantivemos os módulos e nomes em português do projeto.

## Executar

Na raiz do repositório:

```sh
python3 t2_velha/interface.py
python3 t2_velha/testes.py
```

Para usar pytest, se instalado: `python3 -m pytest -q t2_velha/testes.py`.
Se a instalação do Python não oferecer Tkinter, utilize uma com suporte a Tk.

## Demonstrar

1. **Humano × Minimax:** escolha médio/difícil e seu símbolo; clique em
   **Nova partida**. X começa sempre; escolhendo O, o adversário começa.
2. **Treino:** clique em **Treinar nova rede**. Com a pausa marcada, avance uma
   geração por clique. **Executar automático**, **Pausar** e **Interromper**
   controlam a execução; a pausa e a interrupção valem entre gerações.
3. Selecione uma geração para consultar as aptidões de toda a população e os
   resultados do melhor indivíduo. Use **Demonstrar melhor selecionado** para
   assistir a uma partida no tabuleiro. Seu símbolo define o lado da rede nessa
   demonstração; a dificuldade selecionada define o Minimax. A demonstração não
   entra no fitness. **Nova partida** encerra a demonstração.
4. **Humano × rede:** após o treino, a rede selecionada é carregada. Também é
   possível usar **Carregar rede JSON**, incluindo os modelos antigos preservados.
5. **Exportar CSV** salva a tabela de gerações. O gráfico mostra melhor aptidão
   e média da população; os detalhes mostram vitórias, empates, derrotas e inválidas
   do melhor indivíduo. As métricas contra pessoas incluem somente partidas
   concluídas com a mesma rede; abandonar uma partida não conta.

O treino salva em `t2_velha/resultados_interface/`. Após terminar ou interromper,
**Treinar nova rede** reinicia com uma nova população e substitui essa pasta.

## Treinar no terminal

```sh
python3 t2_velha/treino.py
python3 t2_velha/treino.py --N 10 --geracoes 6 --passo --saida /tmp/t2-demo
python3 t2_velha/treino.py --N 200 --geracoes 600 --paciencia 60 --saida /tmp/t2-experimento
```

`--geracoes` é o limite **total**, dividido nos três estágios, e deve ser ≥ 3.
O padrão usa N=120, até 600 gerações, semente 99 e 34 partidas por indivíduo:
32 médias e duas difíceis (uma por lado). A contribuição médio/difícil para a
aptidão muda de 75/25 para 50/50 e 25/75. São 17 partidas começando de cada lado.

Sem `--saida`, salva em `t2_velha/resultados_execucao/`; repetir substitui essa
pasta. O modelo de referência em `resultados/` fica preservado.
Para jogar com o novo treino do terminal, use **Carregar rede JSON** e selecione
`resultados_execucao/melhor_rede.json`.

## Resultados salvos

A única pasta de resultados preservada é `t2_velha/resultados/`:

- `melhor_rede.json`: rede padrão carregada pela interface.
- `config.json`, `historico.json`, `historico.csv` e `populacoes.json`: configuração
  e registros do treino usado no relatório.
- `avaliacao.json`, `avaliacao_inicial.json` e `comparacao_independente.json`:
  evidências das métricas apresentadas.
- `rede_anterior.json`: modelo de referência para reproduzir a comparação.

Os experimentos original e intermediário foram removidos, assim como os registros
redundantes do treino anterior. As pastas `resultados_interface/` e
`resultados_execucao/` só são criadas ao executar novos treinos e são ignoradas pelo
Git, preservando os resultados de referência.

## Rede melhorada e resultados medidos

A janela carrega automaticamente `t2_velha/resultados/melhor_rede.json`.
Para jogar com ela, selecione **rede** e clique em **Nova partida**.
O treino padrão terminou por estagnação dos estágios após **397 gerações**.

| Teste, semente 123 | Rede anterior | Rede melhorada |
|---|---:|---:|
| Vitórias no médio (200 partidas) | 11 | **93** |
| Empates no médio | 0 | **67** |
| Derrotas legais no médio | 115 | **35** |
| Partidas encerradas por jogada inválida no médio | 74 | **5** |
| Decisões ótimas no médio | 39,15% | **85,00%** |
| Difícil, uma partida por lado | 2 derrotas | **2 empates** |

Mantivemos a MLP 9–9–9 com 180 pesos, só propagação, e aprendizado pelo AG.
As mudanças são codificação de ocupação, inicialização aleatória com viés de
ocupação, cruzamento menos frequente, mutação mais localizada, preservação do
campeão validado e avaliação final com peso igual para os dois modos.
O viés inicial favorece evitar casas ocupadas; não contém regras de vitória ou
bloqueio. Todos os pesos evoluem e propostas inválidas continuam penalizadas.

Uma avaliação adicional, sem novos ajustes, usou semente 8675309 e **2.000 partidas
médias** para cada modelo. A nova rede teve 780 vitórias, 771 empates, 391 derrotas
e 58 inválidas; a anterior teve 171 vitórias, nenhum empate, 1.041 derrotas e 788
inválidas. A rede melhorou, mas ainda perde com frequência. Os dois empates no
difícil valem para o adversário determinístico implementado, sem provar domínio
de todas as posições ou de partidas humanas.

## Estudo de robustez com quatro sementes de treino

Para não depender de uma única inicialização, treinamos a mesma configuração com
as sementes 42, 7, 99 e 12345 (só a semente de treino muda). Cada modelo foi avaliado
com 2.000 partidas médias e 2 difíceis, com a semente de avaliação 8675309 (a semente 31415
dá resultados quase idênticos). Os quatro empataram as duas partidas do modo difícil.

| Semente de treino | Vitórias | Empates | Derrotas | Inválidas | Decisões ótimas |
|---|---:|---:|---:|---:|---:|
| 42 | 674 | 509 | 782 | 35 | 69,9% |
| 7 | 614 | 709 | 646 | 31 | 74,6% |
| **99 (modelo entregue)** | **780** | **771** | **391** | 58 | **81,7%** |
| 12345 | 799 | 508 | 693 | 0 | 76,8% |

A variação entre sementes é real: de 69,9% a 81,7% de decisões ótimas e de 391 a 782
derrotas. Escolhemos a semente 99 por ter as menos derrotas e mais decisões ótimas, mas
ela também tem mais partidas encerradas por jogada inválida (58 de 2.000, 2,9%) que a 12345 (0). A escolha foi feita olhando
as próprias sementes de avaliação 8675309 e 31415; por isso esses números são otimistas para
o modelo escolhido e não contam como teste nunca visto. Os modelos e logs das outras
sementes não foram incluídos no repositório. Para repetir:
`python3 t2_velha/treino.py --semente 7 --saida /tmp/s7`.

Reproduzir a comparação, sem treinar:

```sh
python3 t2_velha/avaliacao.py --comparar t2_velha/resultados/rede_anterior.json
```

21 testes passaram; a interface foi verificada carregando o novo modelo e iniciando
uma partida. Números completos, configurações, pesos e históricos estão em
`t2_velha/resultados/`. Acurácia de decisão, legalidade e taxa de vitória
são métricas separadas.

- [Explicação dos algoritmos](t2_velha/README.md)
- [Notas para o relatório, diagramas e resultados](docs/report_notes.md)
- [Requisitos e pendências acadêmicas](ACOMPANHAMENTO.md)
- [Roteiro do vídeo e da demonstração](entrega/roteiro.md)
