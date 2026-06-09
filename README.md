# ⚽ Copa do Mundo 2026 — Previsor de Resultados

Aplicação Python/Streamlit para prever resultados de jogos da Copa do Mundo 2026
com base no elenco escalado, usando um ensemble de três modelos:
**Elo + Poisson bivariado + XGBoost**.

---

## Instalação

```bash
# 1. Clone ou extraia o projeto
cd world_cup_predictor

# 2. Crie um ambiente virtual (recomendado)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o setup (baixa Elo ratings e treina o XGBoost se dados disponíveis)
python setup.py

# 5. Inicie a aplicação
streamlit run app.py
```

## Dataset histórico (XGBoost)

O modelo XGBoost requer o arquivo `data/results.csv` do dataset Kaggle
**"International football results from 1872 to 2023"**.

Baixe em: <https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>

Salve como `data/results.csv`. Sem ele, a aplicação usa apenas **Elo + Poisson** — ainda precisa e funcional.

> Com as variáveis de ambiente `KAGGLE_USERNAME` e `KAGGLE_KEY` configuradas,
> o `setup.py` tenta baixar automaticamente via kaggle CLI.

---

## Estrutura do projeto

```
world_cup_predictor/
├── app.py                  # Interface Streamlit (3 telas)
├── config.py               # Constantes globais e pesos do modelo
├── setup.py                # Script de setup e treinamento
├── requirements.txt
├── data/
│   ├── league_strength.json   # Coeficientes de força por liga
│   ├── demo_squads.json       # Elencos de demonstração (8 seleções)
│   ├── elo_ratings.csv        # Baixado pelo setup.py
│   └── cache/                 # Cache de scraping (TTL: 24h)
├── src/
│   ├── data_fetcher.py     # FBref, Transfermarkt, Elo, Kaggle
│   ├── player_rater.py     # Nota ponderada por jogador
│   ├── team_builder.py     # Agrega elenco em score do time
│   ├── elo_model.py        # Modelo Elo
│   ├── poisson_model.py    # Poisson bivariado (Dixon & Coles)
│   ├── ml_model.py         # XGBoost + calibração
│   └── ensemble.py         # Combina os três modelos
└── tests/
    ├── test_player_rater.py
    ├── test_elo_model.py
    └── test_ensemble.py    # Inclui casos de teste Copa 2022
```

---

## Telas da aplicação

### 1 — Configurar Jogo
- Seleção de times A e B, fase do torneio
- Editor de elenco (11 titulares com posição, clube, liga, notas)
- Botão para carregar elencos demo pré-populados
- Checkbox por jogador para marcar lesionados/suspensos

### 2 — Resultado da Previsão
- Cards com P(vitória A) / P(empate) / P(vitória B)
- Placar mais provável com probabilidade
- Heatmap de distribuição de placares (0–6 × 0–6)
- Gráfico de pizza com contribuição de cada modelo
- Tabela comparativa de probabilidades por modelo
- Top 5 jogadores por impacto no resultado (via SHAP/contribuição)

### 3 — Análise do Elenco
- Radar chart em 6 dimensões (Ataque, Defesa, Meio, Experiência, Coesão, Forma)
- Tabela completa por jogador com coeficiente de liga, notas e score final
- Avisos de baixa amostra de dados pela seleção (< 5 jogos em 24 meses)
- Gráfico de barras comparativo por setor

---

## Lógica dos modelos

### Nota do jogador

```
player_score = (nota_seleção × 0.65) + (nota_clube × coef_liga × 0.35)
```

- Jogos mais recentes valem mais (decaimento exponencial, meia-vida = 180 dias)
- Se < 5 jogos pela seleção nos últimos 24 meses: pesos mudam para 0.45/0.55
- Coeficiente de liga: Premier League = 1.00, Brasileirão = 0.75, MLS = 0.70, etc.

### Nota do time

```
team_score = soma ponderada por posição × player_score de cada titular
```

Bônus: coesão (+3 % se ≥ 6 titulares do mesmo clube) e experiência (+0.5 % por jogador com > 2 Copas).

### Ensemble final

```
P(vitória A) = 0.40 × P_elo + 0.35 × P_poisson + 0.25 × P_xgboost
```

---

## Fontes de dados

| Fonte | Uso | Acesso |
|---|---|---|
| [eloratings.net](https://www.eloratings.net/) | Ratings Elo atuais | Download TSV |
| [FBref](https://fbref.com/) | Stats individuais de jogadores | Scraping (rate-limited) |
| [Transfermarkt](https://www.transfermarkt.com/) | Valor de mercado | Scraping (delay 2s) |
| [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Histórico de resultados | CSV manual |

> **Cache local**: todos os dados raspados são salvos em `data/cache/` com TTL de 24h,
> evitando requisições repetidas.

---

## Executar testes

```bash
pytest tests/ -v
```

Os testes cobrem:
- Decaimento temporal e cálculo de nota do jogador (`test_player_rater.py`)
- Modelo Elo: probabilidades, update de rating, simetria (`test_elo_model.py`)
- Ensemble: soma de probabilidades, formato de saída, 3 casos do Qatar 2022 (`test_ensemble.py`)

### Casos de validação — Copa 2022

| Jogo | Resultado real | O que o teste verifica |
|---|---|---|
| Argentina × França (Final) | 3–3 (AET), ARG nos pênaltis | Diferença de probabilidades < 25 pp (jogo equilibrado) |
| França × Marrocos (Semi) | 2–0 FRA | França favorita (P_fra > P_mar) |
| Brasil × Croácia (QF) | 1–1 (AET), CRO nos pênaltis | Croácia tem ≥ 12 % de chance de vitória |

---

## Limitações conhecidas

1. **Notas de jogador não são em tempo real**: a aplicação usa notas inseridas manualmente ou dados demo. Integração live com FBref/SofaScore requer tratar rate-limiting e possíveis bloqueios.
2. **XGBoost sem dados históricos**: sem `data/results.csv`, apenas Elo+Poisson são usados (ensemble rebalanceado para 53 %/47 %).
3. **Pênaltis não modelados**: o modelo prevê o resultado no tempo regulamentar (90 min). Fases eliminatórias com empate precisariam de modelo adicional para a prorrogação/pênaltis.
4. **Dados demo estáticos**: os elencos demo refletem um snapshot de 2024/25; atualize `data/demo_squads.json` para novas temporadas.
5. **Transfermarkt scraping**: o site usa medidas anti-bot; em produção considere usar a API Rapid-API/FootballData.

---

## Referências

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics.
- World Football Elo Ratings — <https://www.eloratings.net/>
- FiveThirtyEight Soccer Power Index — metodologia pública
- Kaggle dataset: *International football results from 1872 to 2023*
