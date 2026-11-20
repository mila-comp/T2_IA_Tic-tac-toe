# T2 – Jogo da Velha: rede neural + algoritmo genético + Minimax

Esqueleto em Python puro (sem bibliotecas externas). Ainda **sem front end**: os módulos já funcionam pelo terminal.

## Arquivos
| Arquivo | O que faz | Sugestão de quem cuida |
|---|---|---|
| `jogo.py` | regras: tabuleiro (tupla de 9: 0 vazio, 1 X, -1 O), vencedor, empate, jogadas válidas | Pessoa 2 |
| `minimax.py` | Minimax (negamax com cache) e os modos **difícil** (sempre) e **médio** (50% Minimax, 50% aleatório) | Pessoa 2 |
| `aptidao.py` | função de aptidão (recompensas e penalidades), partida rede × Minimax, `avaliar()` (acurácia) | Pessoa 2 |
| `rede.py` | MLP de 2 camadas, só propagação; cada camada é uma lista de neurônios | Pessoa 1 |
| `ag.py` | população inicial, torneio, elitismo, cruzamento BLX-α (reais), mutação | Pessoa 1 |
| `treino.py` | ciclo de gerações, fases (ordem dos modos), convergência, histórico, salva resultados | Pessoa 1 |
| `testes.py` | testes rápidos (`python3 testes.py`) | todas |

## Como rodar
    python3 testes.py            # testes dos módulos
    python3 treino.py            # treina e salva em resultados/ (cerca de 30 s)
    python3 treino.py --passo    # pausa a cada geração (acompanhar passo a passo)

`resultados/` guarda `historico.json` (melhor, média e pior por geração), `populacoes.json` (aptidão de toda a
população e melhor cromossomo por geração) e `melhor_rede.json` (pesos da rede escolhida).

## Decisões já tomadas (para explicar no relatório e ajustar)
- **Cromossomo:** lista de reais com todos os pesos e bias; população inicial uniforme em [-1, 1].
- **Topologia:** 9 entradas, 9 neurônios na camada oculta (tanh), 9 saídas (linear); a rede joga na casa de maior saída,
  mesmo ocupada (isso é jogada inválida, penalizada). Total de 180 pesos (com bias).
- **Codificação da entrada** (`codificacao` em `treino.py`): `sinal` (meu = 1, adversário = -1, vazio = 0),
  `ocupacao` (vazio = 0, meu = 1, adversário = 2) e `dois_planos` (18 entradas). Com `sinal`, "casa ocupada" não é uma
  função simples da entrada, e a rede ficou em ~85% de jogadas válidas; com `ocupacao`, ~96 a 98%. **Confirmar com a
  professora se mudar a codificação é aceito**; a topologia inicial é a que ela passou em aula.
- **Aptidão:** +1 por jogada válida; jogada inválida -10 e encerra a partida; vitória +10; empate +5; derrota -5.
  Média por partida. No médio, 10 partidas por cromossomo (metade começa pela rede); no difícil, 2 (uma por lado), pois o
  resultado é sempre o mesmo. Todos os cromossomos de uma geração jogam com a mesma semente aleatória.
- **AG:** elitismo (2) + torneio (k = 3); cruzamento BLX-α (α = 0,3), taxa 0,9; mutação aleatória por gene
  (soma de ruído normal, σ = 0,2, taxa 0,05; há também a versão "troca o gene por valor novo"). **Se o Moodle
  indicar outro cruzamento para reais, trocar `cruzamento_blx` em `ag.py`.**
- **Ordem dos modos:** fase 1 só médio (400 gerações no máximo) e fase 2 médio + difícil (400). Cada fase para por
  convergência (60 gerações sem melhora). Quem começa alterna a cada partida.
- **Escolha final:** a aptidão de uma geração é ruidosa; no fim, as últimas candidatas são reavaliadas com 100 partidas.

## Resultado do esqueleto (semente 42; os números variam muito de semente para semente)
- Contra o Minimax **médio** (200 partidas): ~96% das jogadas válidas, 65 vitórias, 110 derrotas, 25 partidas com
  jogada inválida (32,5% sem derrota).
- Contra o **difícil**: jogadas 100% válidas, mas perde (o difícil não erra; a melhor rede possível empata).
- Em testes com N = 200 e 400 + 400 gerações, até 54% das partidas sem derrota no médio. A rede **ainda não aprendeu a
  bloquear e a empatar com o difícil**: é aí que vocês podem ajustar aptidão, topologia, parâmetros e ordem dos modos.

## O que falta
1. Front end (3 modos: você × Minimax, rede aprendendo com o Minimax, você × rede treinada) com acompanhamento das gerações
   e cálculo da acurácia (`aptidao.avaliar` já calcula jogadas válidas e partidas sem derrota).
2. Experimentos para o relatório: comparar a ordem dos modos, topologias e parâmetros do AG.
3. Relatório em PPT e vídeo.
