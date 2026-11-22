# Como a solução funciona

**População de pesos → partidas contra Minimax → aptidão → seleção → cruzamento
→ mutação → nova população.** Implementação própria com biblioteca padrão do
Python, sem backpropagation ou frameworks de aprendizagem.

| Arquivo | Responsabilidade |
|---|---|
| `jogo.py` | Regras, alternância X/O e rejeição de jogadas ilegais |
| `minimax.py` | Negamax completo com cache e conjunto de melhores jogadas |
| `rede.py` | Codificação, propagação e conversão de cromossomos |
| `ag.py` | População real, elitismo, torneio, cruzamento e mutação |
| `aptidao.py` | Partidas reais, pontuação e métricas automáticas |
| `treino.py` | Currículo, parada, seleção final, JSON e CSV |
| `avaliacao.py` | Testar e comparar modelos prontos, sem alterar pesos |
| `sessao.py` | Partidas humanas e suas métricas |
| `interface.py` | Tkinter, tabuleiro, gráfico e controle do treino |
| `testes.py` | 21 testes de comportamento |

## Rede e Minimax

Padrão: 9 entradas → 9 neurônios ocultos com tanh → 9 saídas lineares.
Cada neurônio armazena pesos e bias; são `9×10 + 9×10 = 180` valores reais.
A codificação padrão é `ocupacao`: vazio=0, própria marca=1, adversário=2.
`sinal` (0/1/−1) e `dois_planos` (18 entradas binárias) continuam disponíveis.
Modelos antigos guardam sua própria codificação e permanecem compatíveis.
A saída é argmax das nove casas, sem máscara ou correção de propostas inválidas.
Minimax não participa da decisão da rede durante a partida.

Minimax busca até estados terminais; prefere vitórias rápidas e adia derrotas.
Em empate escolhe a primeira casa de valor máximo. No médio, cada turno tem
50% de chance de usar Minimax e 50% de fazer uma jogada aleatória válida.
No difícil sempre usa Minimax. O teste exaustivo confirma que o difícil não perde
para nenhuma sequência legal, de ambos os lados.

## Aptidão e AG

Pontuação por partida: `0,05 × jogadas válidas + resultado`, com vitória=3,
empate=1, derrota=−2 e proposta inválida=−5. Invalidar encerra a partida sem
somar outra penalidade de derrota. Essas constantes são escolhas do projeto.

N=120, elite=2, torneio de 3; cruzamento aritmético com chance 0,3; mutação
gaussiana por gene com chance 0,025 e σ=0,2; pesos limitados a [−5,5]. BLX-α
e mutação por substituição continuam disponíveis. A mutação altera em média
4,5 genes por filho, contra 18 na configuração anterior.

A inicialização `local` é aleatória, com viés explícito de ocupação: um neurônio
por casa começa com conexão positiva de entrada e negativa para a saída da mesma
casa; as demais conexões também são aleatórias. Isso facilita reconhecer casas
preenchidas, sem incorporar estratégia de vitória, bloqueio ou resposta Minimax.
Todos os 180 pesos continuam livres durante a evolução. Só é compatível com
9–9–9 e `ocupacao`; para outras topologias use `inicializacao="uniforme"` em CONFIG.
A alternativa uniforme em [−1,1] permanece disponível.

São 32 partidas médias e 2 difíceis por indivíduo, alternando 17 inícios de cada
lado. Como rede e difícil são determinísticos, duas partidas bastam para medir os
dois lados desse oponente. A aptidão pondera as **médias por modo**, não quantas
vezes repetimos o difícil. Pesos do médio/difícil: 75/25 → 50/50 → 25/75.
Configurações antigas sem pesos explícitos mantêm a média por partida.

O padrão usa um lote fixo de treino com semente 42. Cada partida tem gerador
independente a partir de `semente:modo:índice`; todos os indivíduos recebem as
mesmas condições. Isso estabiliza a comparação e o elitismo, mas pode causar
sobreajuste. Validação e teste usam sementes diferentes. `partidas_fixas=False`
permite voltar à troca de lote a cada geração. Um cache temporário de decisões
por cromossomo acelera partidas repetidas sem alterar seu resultado.

## Treino e parada

Até 600 gerações: 300/150/150 nos três estágios. Cada estágio para no limite ou
por 100 gerações sem melhora >0,01 na validação com semente 2026, após pelo menos
100 gerações naquele estágio. A contagem reinicia na transição. Estágios menores
que 100 terminam pelo limite. Estagnação não prova convergência matemática.

O campeão validado permanece na população entre gerações e na transição de fase.
A seleção final reavalia campeões das últimas dez gerações e de cada estágio,
com semente 999: 100 partidas médias e duas difíceis, cada modo com peso 50%.
A quantidade maior de amostras médias não dilui a contribuição do difícil.
O recorde bruto da janela combina currículos distintos e não deve ser usado
sozinho para comparar estágios.

## Resultados e métricas

A execução salva em `resultados/` tem 389 gerações. No teste com semente
123: médio=69 vitórias, 44 empates, 84 derrotas e 3 inválidas; difícil=2 empates.
A rede anterior (`resultados/rede_anterior.json`) tinha 11 vitórias e 74 inválidas no médio
e perdia as duas difíceis. O melhor indivíduo da população inicial local,
selecionado pela mesma validação 999, tinha 58 vitórias, 42 empates e 100 derrotas
no médio, e um empate e uma derrota no difícil. Assim, parte do ganho vem do viés
de inicialização e outra parte da evolução pelo AG.

Após concluir os ajustes, a semente adicional 8675309 mediu 2.000 partidas médias:
nova=674 vitórias, 509 empates, 782 derrotas, 35 inválidas; anterior=171 vitórias,
0 empates, 1.041 derrotas, 788 inválidas. Não foram feitos novos ajustes após esse
teste. Há melhora, mas o modelo ainda não domina o jogo. Os dois empates no difícil
não cobrem todas as escolhas ótimas possíveis do adversário.

- Decisões ótimas = decisões no conjunto ótimo do Minimax / tentativas.
- Legalidade = jogadas válidas / tentativas (`acuracia_jogadas`, nome legado).
- Sem derrota = (vitórias + empates) / partidas concluídas.
- Taxa de derrota inclui inválidas; `derrota` e `invalida` são contadores separados.

A sessão humana calcula as métricas sem misturar redes ou partidas abandonadas.
`melhor_rede.json` guarda pesos, topologia, configuração, data UTC e avaliação;
`config.json`, `historico.json`, `historico.csv`, `populacoes.json` e `avaliacao.json`
registram o treino. `avaliacao_inicial.json` mede o melhor indivíduo antes do AG;
`comparacao_independente.json` registra a comparação adicional entre os modelos.
A leitura valida dimensões e pesos finitos. Os arquivos não permitem restaurar
exatamente a população de uma execução interrompida após fechar a aplicação.
