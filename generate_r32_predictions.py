"""
Gera previsoes para os 16 jogos da Rodada de 32 (mata-mata) com Elo
atualizado pelos resultados reais da fase de grupos.

O modelo da Copa do Mundo recebe peso maior porque:
  1. Os resultados reais aplicam o fator K=60 (World Cup) no Elo
     vs K=20 (amistosos) que o dataset historico usa em media.
  2. Isso faz o Elo refletir desempenho *neste* torneio, nao apenas
     o historico pre-Copa.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Resultados reais da fase de grupos (ESPN, 28/jun/2026) ───────────────────
# Formato: (mandante, visitante, gols_mandante, gols_visitante)
WC_GROUP_RESULTS = [
    # Grupo A
    ("Mexico",         "South Africa",  2, 0),
    ("South Korea",    "Czechia",        2, 1),
    ("Mexico",         "Czechia",        3, 0),
    ("South Africa",   "South Korea",    1, 0),
    # Grupo B
    ("Bosnia-Herzegovina", "Qatar",      3, 1),
    ("Switzerland",    "Bosnia-Herzegovina", 4, 1),
    ("Canada",         "Qatar",          6, 0),
    ("Switzerland",    "Canada",         2, 1),
    # Grupo C
    ("Brazil",         "Morocco",        1, 1),
    ("Haiti",          "Scotland",       1, 0),
    ("Brazil",         "Haiti",          3, 0),
    ("Morocco",        "Scotland",       3, 0),
    # Grupo D
    ("United States",  "Paraguay",       4, 1),
    ("Australia",      "Türkiye",        2, 0),
    ("Paraguay",       "Australia",      0, 0),
    ("Türkiye",        "United States",  3, 2),
    # Grupo E
    ("Germany",        "Curacao",        7, 1),
    ("Ivory Coast",    "Ecuador",        1, 0),
    ("Ecuador",        "Germany",        2, 1),
    ("Curacao",        "Ivory Coast",    0, 2),
    # Grupo F
    ("Netherlands",    "Sweden",         5, 1),
    ("Japan",          "Tunisia",        4, 0),
    ("Japan",          "Netherlands",    1, 1),
    ("Sweden",         "Tunisia",        5, 1),
    # Grupo G
    ("Belgium",        "Egypt",          1, 1),
    ("Iran",           "New Zealand",    2, 2),
    ("Belgium",        "Iran",           0, 0),
    ("New Zealand",    "Egypt",          1, 3),
    # Grupo H
    ("Spain",          "Cape Verde",     0, 0),
    ("Saudi Arabia",   "Uruguay",        1, 1),
    ("Spain",          "Saudi Arabia",   4, 0),
    ("Uruguay",        "Spain",          0, 1),
    # Grupo I
    ("France",         "Senegal",        3, 1),
    ("Norway",         "Iraq",           4, 1),
    ("France",         "Norway",         3, 0),
    ("Senegal",        "Iraq",           5, 0),
    # Grupo J
    ("Argentina",      "Algeria",        3, 0),
    ("Austria",        "Jordan",         3, 1),
    ("Argentina",      "Austria",        2, 0),
    ("Algeria",        "Jordan",         3, 1),
    # Grupo K
    ("Portugal",       "Uzbekistan",     5, 0),
    ("Colombia",       "Congo DR",       1, 0),
    ("Colombia",       "Portugal",       0, 0),
    ("Congo DR",       "Uzbekistan",     3, 1),
    # Grupo L
    ("England",        "Croatia",        4, 2),
    ("Ghana",          "Panama",         1, 0),
    ("England",        "Ghana",          0, 0),
    ("Croatia",        "Panama",         1, 0),
]

# ── Confrontos confirmados do Mata-Mata ───────────────────────────────────────
R32_MATCHES = [
    {"id": "R32-01", "date": "2026-06-28", "time_brt": "16:00", "team_a": "South Africa",  "team_b": "Canada",             "venue": "SoFi Stadium, Los Angeles"},
    {"id": "R32-02", "date": "2026-06-29", "time_brt": "14:00", "team_a": "Brazil",         "team_b": "Japan",              "venue": "NRG Stadium, Houston"},
    {"id": "R32-03", "date": "2026-06-29", "time_brt": "17:30", "team_a": "Germany",        "team_b": "Paraguay",           "venue": "Gillette Stadium, Boston"},
    {"id": "R32-04", "date": "2026-06-29", "time_brt": "22:00", "team_a": "Netherlands",    "team_b": "Morocco",            "venue": "Estadio BBVA, Monterrey"},
    {"id": "R32-05", "date": "2026-06-30", "time_brt": "14:00", "team_a": "Ivory Coast",    "team_b": "Norway",             "venue": "AT&T Stadium, Dallas"},
    {"id": "R32-06", "date": "2026-06-30", "time_brt": "18:00", "team_a": "France",         "team_b": "Sweden",             "venue": "MetLife Stadium, Nova York"},
    {"id": "R32-07", "date": "2026-06-30", "time_brt": "22:00", "team_a": "Mexico",         "team_b": "Ecuador",            "venue": "Estadio Azteca, Cidade do Mexico"},
    {"id": "R32-08", "date": "2026-07-01", "time_brt": "13:00", "team_a": "England",        "team_b": "Congo DR",           "venue": "Mercedes-Benz Stadium, Atlanta"},
    {"id": "R32-09", "date": "2026-07-01", "time_brt": "17:00", "team_a": "Belgium",        "team_b": "Senegal",            "venue": "Lumen Field, Seattle"},
    {"id": "R32-10", "date": "2026-07-01", "time_brt": "21:00", "team_a": "United States",  "team_b": "Bosnia-Herzegovina", "venue": "Levi's Stadium, Santa Clara"},
    {"id": "R32-11", "date": "2026-07-02", "time_brt": "16:00", "team_a": "Spain",          "team_b": "Austria",            "venue": "SoFi Stadium, Los Angeles"},
    {"id": "R32-12", "date": "2026-07-02", "time_brt": "20:00", "team_a": "Portugal",       "team_b": "Croatia",            "venue": "BMO Field, Toronto"},
    {"id": "R32-13", "date": "2026-07-02", "time_brt": "00:00", "team_a": "Switzerland",    "team_b": "Algeria",            "venue": "BC Place, Vancouver"},
    {"id": "R32-14", "date": "2026-07-03", "time_brt": "15:00", "team_a": "Egypt",          "team_b": "Australia",          "venue": "AT&T Stadium, Dallas"},
    {"id": "R32-15", "date": "2026-07-03", "time_brt": "19:00", "team_a": "Argentina",      "team_b": "Cape Verde",         "venue": "Hard Rock Stadium, Miami"},
    {"id": "R32-16", "date": "2026-07-03", "time_brt": "22:30", "team_a": "Colombia",       "team_b": "Ghana",              "venue": "Arrowhead Stadium, Kansas City"},
]


def build_updated_elo() -> dict[str, float]:
    """Aplica os resultados reais da Copa sobre o Elo base (K=60 por jogo)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data_fetcher import load_elo_ratings
    from src.elo_model import update_elo

    elo = dict(load_elo_ratings())

    for home, away, sh, sa in WC_GROUP_RESULTS:
        elo_h = elo.get(home, 1800)
        elo_a = elo.get(away, 1800)
        new_h, new_a = update_elo(elo_h, elo_a, sh, sa, tournament="World Cup")
        elo[home] = new_h
        elo[away] = new_a

    return elo


def run() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.schedule_loader import get_squad
    from src.team_builder import build_team_score
    from src.ensemble import EnsemblePredictor

    print("Atualizando Elo com os 48 resultados reais da fase de grupos (K=60)...")
    updated_elo = build_updated_elo()

    predictor = EnsemblePredictor()
    predictor._elo_ratings = updated_elo   # injeta Elo atualizado

    predictions = []
    print(f"\nGerando previsoes para {len(R32_MATCHES)} jogos do mata-mata...\n")

    for i, match in enumerate(R32_MATCHES, 1):
        ta = match["team_a"]
        tb = match["team_b"]

        players_a, _, _ = get_squad(ta)
        players_b, _, _ = get_squad(tb)

        team_a = build_team_score(players_a)
        team_b = build_team_score(players_b)

        result = predictor.predict(
            ta, tb, team_a, team_b,
            tournament_phase_name="Oitavas de Final",
        )

        sh, sa = result["most_probable_score"]
        fav = ta if result["win_a"] > result["win_b"] else (
              tb if result["win_b"] > result["win_a"] else "EQUILIBRADO")

        print(
            f"[{i:02d}/16] {match['date']} {match['time_brt']} BRT  "
            f"{ta} vs {tb}  =>  {sh}-{sa}  "
            f"|  {result['win_a']:.0%}/{result['draw']:.0%}/{result['win_b']:.0%}  "
            f"(fav: {fav})"
        )

        predictions.append({
            "id":             match["id"],
            "data":           match["date"],
            "horario_brt":    match["time_brt"],
            "mandante":       ta,
            "visitante":      tb,
            "venue":          match["venue"],
            "placar_previsto": f"{sh}-{sa}",
            "prob_mandante":  result["win_a"],
            "prob_empate":    result["draw"],
            "prob_visitante": result["win_b"],
            "favorito":       fav,
            "elo_mandante":   round(updated_elo.get(ta, 1800), 1),
            "elo_visitante":  round(updated_elo.get(tb, 1800), 1),
        })

    output = {
        "_gerado_em":   datetime.now().isoformat(),
        "_modelo":      "Elo(40%) + Poisson(35%) + XGBoost(25%) — Elo atualizado com 48 resultados reais da fase de grupos (K=60)",
        "_nota":        "Elo reflete desempenho NESTA Copa. Times que surpreenderam (ex: Ecuador, Haiti) sobem; decepcionantes caem.",
        "total_jogos":  len(predictions),
        "jogos":        predictions,
    }

    simple_path  = ROOT / "data" / "previsoes_r32.json"
    simple_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSalvo em {simple_path}")


if __name__ == "__main__":
    run()
