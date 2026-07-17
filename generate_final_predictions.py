"""
Gera previsoes para os 2 ultimos jogos da Copa do Mundo 2026:
  - Disputa do 3o lugar: France vs England (18/jul)
  - Final: Spain vs Argentina (19/jul)

Elo atualizado com 78 resultados reais (grupos + R32 + oitavas + quartas + semis).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ALL_WC_RESULTS = [
    # ── Fase de grupos ──────────────────────────────────────────────────────
    ("Mexico",             "South Africa",       2, 0),
    ("South Korea",        "Czechia",            2, 1),
    ("Mexico",             "Czechia",            3, 0),
    ("South Africa",       "South Korea",        1, 0),
    ("Bosnia-Herzegovina", "Qatar",              3, 1),
    ("Switzerland",        "Bosnia-Herzegovina", 4, 1),
    ("Canada",             "Qatar",              6, 0),
    ("Switzerland",        "Canada",             2, 1),
    ("Brazil",             "Morocco",            1, 1),
    ("Haiti",              "Scotland",           1, 0),
    ("Brazil",             "Haiti",              3, 0),
    ("Morocco",            "Scotland",           3, 0),
    ("United States",      "Paraguay",           4, 1),
    ("Australia",          "Türkiye",            2, 0),
    ("Paraguay",           "Australia",          0, 0),
    ("Türkiye",            "United States",      3, 2),
    ("Germany",            "Curacao",            7, 1),
    ("Ivory Coast",        "Ecuador",            1, 0),
    ("Ecuador",            "Germany",            2, 1),
    ("Curacao",            "Ivory Coast",        0, 2),
    ("Netherlands",        "Sweden",             5, 1),
    ("Japan",              "Tunisia",            4, 0),
    ("Japan",              "Netherlands",        1, 1),
    ("Sweden",             "Tunisia",            5, 1),
    ("Belgium",            "Egypt",              1, 1),
    ("Iran",               "New Zealand",        2, 2),
    ("Belgium",            "Iran",               0, 0),
    ("New Zealand",        "Egypt",              1, 3),
    ("Spain",              "Cape Verde",         0, 0),
    ("Saudi Arabia",       "Uruguay",            1, 1),
    ("Spain",              "Saudi Arabia",       4, 0),
    ("Uruguay",            "Spain",              0, 1),
    ("France",             "Senegal",            3, 1),
    ("Norway",             "Iraq",               4, 1),
    ("France",             "Norway",             3, 0),
    ("Senegal",            "Iraq",               5, 0),
    ("Argentina",          "Algeria",            3, 0),
    ("Austria",            "Jordan",             3, 1),
    ("Argentina",          "Austria",            2, 0),
    ("Algeria",            "Jordan",             3, 1),
    ("Portugal",           "Uzbekistan",         5, 0),
    ("Colombia",           "Congo DR",           1, 0),
    ("Colombia",           "Portugal",           0, 0),
    ("Congo DR",           "Uzbekistan",         3, 1),
    ("England",            "Croatia",            4, 2),
    ("Ghana",              "Panama",             1, 0),
    ("England",            "Ghana",              0, 0),
    ("Croatia",            "Panama",             1, 0),
    # ── Rodada de 32 ────────────────────────────────────────────────────────
    ("Canada",             "South Africa",       1, 0),
    ("Brazil",             "Japan",              2, 1),
    ("Paraguay",           "Germany",            1, 1),
    ("Morocco",            "Netherlands",        1, 1),
    ("Norway",             "Ivory Coast",        2, 1),
    ("France",             "Sweden",             3, 0),
    ("Mexico",             "Ecuador",            2, 0),
    ("England",            "Congo DR",           2, 1),
    ("Belgium",            "Senegal",            3, 2),
    ("United States",      "Bosnia-Herzegovina", 2, 0),
    ("Spain",              "Austria",            3, 0),
    ("Portugal",           "Croatia",            2, 1),
    ("Switzerland",        "Algeria",            2, 0),
    ("Egypt",              "Australia",          1, 1),
    ("Argentina",          "Cape Verde",         3, 2),
    ("Colombia",           "Ghana",              1, 0),
    # ── Oitavas de Final ────────────────────────────────────────────────────
    ("Morocco",            "Canada",             3, 0),
    ("France",             "Paraguay",           1, 0),
    ("Norway",             "Brazil",             2, 0),
    ("England",            "Mexico",             3, 2),
    ("Portugal",           "Spain",              0, 1),
    ("United States",      "Belgium",            1, 4),
    ("Argentina",          "Egypt",              3, 2),
    ("Colombia",           "Switzerland",        0, 0),
    # ── Quartas de Final ────────────────────────────────────────────────────
    ("France",             "Morocco",            2, 0),
    ("Spain",              "Belgium",            2, 1),
    ("Norway",             "England",            1, 2),
    ("Argentina",          "Switzerland",        3, 1),
    # ── Semifinais ──────────────────────────────────────────────────────────
    ("France",             "Spain",              0, 2),
    ("England",            "Argentina",          1, 2),
]

JOGOS_FINAIS = [
    {"id": "3PL",    "fase": "Disputa do 3o Lugar",
     "date": "2026-07-18", "time_brt": "18:00",
     "team_a": "France",    "team_b": "England",
     "venue": "Hard Rock Stadium, Miami"},
    {"id": "FINAL",  "fase": "Final",
     "date": "2026-07-19", "time_brt": "16:00",
     "team_a": "Spain",     "team_b": "Argentina",
     "venue": "MetLife Stadium, Nova York/Nova Jersey"},
]


def build_updated_elo() -> dict[str, float]:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data_fetcher import load_elo_ratings
    from src.elo_model import update_elo

    elo = dict(load_elo_ratings())
    for home, away, sh, sa in ALL_WC_RESULTS:
        eh, ea = update_elo(elo.get(home, 1800), elo.get(away, 1800),
                            sh, sa, tournament="World Cup")
        elo[home] = eh
        elo[away] = ea
    return elo


def run() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.schedule_loader import get_squad
    from src.team_builder import build_team_score
    from src.ensemble import EnsemblePredictor

    n = len(ALL_WC_RESULTS)
    print(f"Atualizando Elo com {n} resultados reais da Copa (K=60)...")
    updated_elo = build_updated_elo()

    print("\nTop 4 finalistas - Elo final da Copa:")
    for team in ["Argentina", "Spain", "France", "England"]:
        print(f"  {team}: {updated_elo.get(team, 1800):.0f}")

    predictor = EnsemblePredictor()
    predictor._elo_ratings = updated_elo

    predictions = []
    print()

    for m in JOGOS_FINAIS:
        ta, tb = m["team_a"], m["team_b"]

        players_a, _, _ = get_squad(ta)
        players_b, _, _ = get_squad(tb)
        team_a = build_team_score(players_a)
        team_b = build_team_score(players_b)

        result = predictor.predict(ta, tb, team_a, team_b,
                                   tournament_phase_name=m["fase"])

        sh, sa = result["most_probable_score"]
        fav = (ta if result["win_a"] > result["win_b"] else
               tb if result["win_b"] > result["win_a"] else "EQUILIBRADO")

        print(
            f"[{m['id']}] {m['fase'].upper()}\n"
            f"  {m['date']} {m['time_brt']} BRT  {ta} vs {tb}\n"
            f"  Previsao: {sh}-{sa}  |  "
            f"{result['win_a']:.0%}/{result['draw']:.0%}/{result['win_b']:.0%}  "
            f"(fav: {fav})\n"
            f"  Elo: {ta}={updated_elo.get(ta,1800):.0f}  "
            f"{tb}={updated_elo.get(tb,1800):.0f}  "
            f"diff={updated_elo.get(ta,1800)-updated_elo.get(tb,1800):+.0f}\n"
        )

        predictions.append({
            "id":              m["id"],
            "fase":            m["fase"],
            "data":            m["date"],
            "horario_brt":     m["time_brt"],
            "mandante":        ta,
            "visitante":       tb,
            "venue":           m["venue"],
            "placar_previsto": f"{sh}-{sa}",
            "prob_mandante":   result["win_a"],
            "prob_empate":     result["draw"],
            "prob_visitante":  result["win_b"],
            "favorito":        fav,
            "elo_mandante":    round(updated_elo.get(ta, 1800), 1),
            "elo_visitante":   round(updated_elo.get(tb, 1800), 1),
            "elo_diff":        round(updated_elo.get(ta, 1800) - updated_elo.get(tb, 1800), 1),
            "jogado":          False,
            "resultado_real":  None,
        })

    output = {
        "_gerado_em":         datetime.now().isoformat(),
        "_modelo":            "Elo(~53%) + Poisson(~47%) — Elo atualizado com 78 resultados reais (toda a Copa)",
        "_resultados_usados": n,
        "jogos":              predictions,
    }

    dest = ROOT / "data" / "previsoes_final.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {dest}")


if __name__ == "__main__":
    run()
