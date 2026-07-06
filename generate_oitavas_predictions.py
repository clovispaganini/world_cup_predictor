"""
Gera previsoes para as 8 Oitavas de Final com Elo atualizado por:
  - 48 resultados da fase de grupos
  - 16 resultados da Rodada de 32
  -  4 resultados das oitavas ja disputadas

Para jogos com prorrogacao/penaltis o Elo usa o placar dos 90+120min
(prorrogacao conta como resultado; penaltis sao ignorados — tratados como empate).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Todos os resultados acumulados ate 06/jul/2026 ───────────────────────────
# Formato: (mandante, visitante, gols_mandante, gols_visitante)
# Penaltis = placar do tempo regulamentar + prorrogacao

ALL_WC_RESULTS = [
    # ── Fase de grupos ──────────────────────────────────────────────────────
    ("Mexico",         "South Africa",       2, 0),
    ("South Korea",    "Czechia",            2, 1),
    ("Mexico",         "Czechia",            3, 0),
    ("South Africa",   "South Korea",        1, 0),
    ("Bosnia-Herzegovina", "Qatar",          3, 1),
    ("Switzerland",    "Bosnia-Herzegovina", 4, 1),
    ("Canada",         "Qatar",              6, 0),
    ("Switzerland",    "Canada",             2, 1),
    ("Brazil",         "Morocco",            1, 1),
    ("Haiti",          "Scotland",           1, 0),
    ("Brazil",         "Haiti",              3, 0),
    ("Morocco",        "Scotland",           3, 0),
    ("United States",  "Paraguay",           4, 1),
    ("Australia",      "Türkiye",            2, 0),
    ("Paraguay",       "Australia",          0, 0),
    ("Türkiye",        "United States",      3, 2),
    ("Germany",        "Curacao",            7, 1),
    ("Ivory Coast",    "Ecuador",            1, 0),
    ("Ecuador",        "Germany",            2, 1),
    ("Curacao",        "Ivory Coast",        0, 2),
    ("Netherlands",    "Sweden",             5, 1),
    ("Japan",          "Tunisia",            4, 0),
    ("Japan",          "Netherlands",        1, 1),
    ("Sweden",         "Tunisia",            5, 1),
    ("Belgium",        "Egypt",              1, 1),
    ("Iran",           "New Zealand",        2, 2),
    ("Belgium",        "Iran",               0, 0),
    ("New Zealand",    "Egypt",              1, 3),
    ("Spain",          "Cape Verde",         0, 0),
    ("Saudi Arabia",   "Uruguay",            1, 1),
    ("Spain",          "Saudi Arabia",       4, 0),
    ("Uruguay",        "Spain",              0, 1),
    ("France",         "Senegal",            3, 1),
    ("Norway",         "Iraq",               4, 1),
    ("France",         "Norway",             3, 0),
    ("Senegal",        "Iraq",               5, 0),
    ("Argentina",      "Algeria",            3, 0),
    ("Austria",        "Jordan",             3, 1),
    ("Argentina",      "Austria",            2, 0),
    ("Algeria",        "Jordan",             3, 1),
    ("Portugal",       "Uzbekistan",         5, 0),
    ("Colombia",       "Congo DR",           1, 0),
    ("Colombia",       "Portugal",           0, 0),
    ("Congo DR",       "Uzbekistan",         3, 1),
    ("England",        "Croatia",            4, 2),
    ("Ghana",          "Panama",             1, 0),
    ("England",        "Ghana",              0, 0),
    ("Croatia",        "Panama",             1, 0),

    # ── Rodada de 32 ────────────────────────────────────────────────────────
    # Penaltis: usa placar do 90+120min (tratado como empate para o Elo)
    ("Canada",         "South Africa",       1, 0),
    ("Brazil",         "Japan",              2, 1),
    ("Paraguay",       "Germany",            1, 1),   # 4-3 pens — empate no tempo
    ("Morocco",        "Netherlands",        1, 1),   # 3-2 pens — empate no tempo
    ("Norway",         "Ivory Coast",        2, 1),
    ("France",         "Sweden",             3, 0),
    ("Mexico",         "Ecuador",            2, 0),
    ("England",        "Congo DR",           2, 1),
    ("Belgium",        "Senegal",            3, 2),   # AET
    ("United States",  "Bosnia-Herzegovina", 2, 0),
    ("Spain",          "Austria",            3, 0),
    ("Portugal",       "Croatia",            2, 1),
    ("Switzerland",    "Algeria",            2, 0),
    ("Egypt",          "Australia",          1, 1),   # 4-2 pens — empate no tempo
    ("Argentina",      "Cape Verde",         3, 2),   # AET
    ("Colombia",       "Ghana",              1, 0),

    # ── Oitavas de Final (ja disputadas) ────────────────────────────────────
    ("Morocco",        "Canada",             3, 0),
    ("France",         "Paraguay",           1, 0),
    ("Norway",         "Brazil",             2, 0),   # MAIOR AZARAO!
    ("England",        "Mexico",             3, 2),
]

# ── 8 Confrontos das Oitavas de Final ────────────────────────────────────────
OITAVAS = [
    # Ja disputadas
    {"id": "R16-01", "date": "2026-07-04", "time_brt": "14:00", "team_a": "Morocco",       "team_b": "Canada",      "venue": "NRG Stadium, Houston",                "played": True,  "score": "3-0"},
    {"id": "R16-02", "date": "2026-07-04", "time_brt": "18:00", "team_a": "France",        "team_b": "Paraguay",    "venue": "Lincoln Financial Field, Filadélfia", "played": True,  "score": "1-0"},
    {"id": "R16-03", "date": "2026-07-05", "time_brt": "17:00", "team_a": "Norway",        "team_b": "Brazil",      "venue": "MetLife Stadium, Nova York",           "played": True,  "score": "2-0"},
    {"id": "R16-04", "date": "2026-07-05", "time_brt": "21:00", "team_a": "England",       "team_b": "Mexico",      "venue": "Estadio Azteca, Cidade do México",     "played": True,  "score": "3-2"},
    # A disputar
    {"id": "R16-05", "date": "2026-07-06", "time_brt": "16:00", "team_a": "Spain",         "team_b": "Portugal",    "venue": "AT&T Stadium, Dallas",                "played": False, "score": None},
    {"id": "R16-06", "date": "2026-07-06", "time_brt": "21:00", "team_a": "United States", "team_b": "Belgium",     "venue": "Lumen Field, Seattle",                "played": False, "score": None},
    {"id": "R16-07", "date": "2026-07-07", "time_brt": "13:00", "team_a": "Argentina",     "team_b": "Egypt",       "venue": "Mercedes-Benz Stadium, Atlanta",      "played": False, "score": None},
    {"id": "R16-08", "date": "2026-07-07", "time_brt": "17:00", "team_a": "Colombia",      "team_b": "Switzerland", "venue": "BC Place, Vancouver",                 "played": False, "score": None},
]


def build_updated_elo() -> dict[str, float]:
    """Aplica todos os resultados da Copa ate hoje (K=60 cada jogo)."""
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

    n_results = len(ALL_WC_RESULTS)
    print(f"Atualizando Elo com {n_results} resultados reais da Copa (K=60)...")
    updated_elo = build_updated_elo()

    predictor = EnsemblePredictor()
    predictor._elo_ratings = updated_elo

    predictions = []
    print(f"\nGerando previsoes para os 8 jogos das Oitavas de Final...\n")

    for m in OITAVAS:
        ta, tb = m["team_a"], m["team_b"]

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

        status = f"RESULTADO REAL: {m['score']}" if m["played"] else "A DISPUTAR"
        print(
            f"[{m['id']}] {m['date']} {m['time_brt']} BRT  "
            f"{ta} vs {tb}\n"
            f"  Previsao: {sh}-{sa}  |  "
            f"{result['win_a']:.0%}/{result['draw']:.0%}/{result['win_b']:.0%}  "
            f"(fav: {fav})\n"
            f"  {status}\n"
        )

        predictions.append({
            "id":              m["id"],
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
            "jogado":          m["played"],
            "resultado_real":  m["score"],
        })

    output = {
        "_gerado_em":    datetime.now().isoformat(),
        "_modelo":       "Elo(~53%) + Poisson(~47%) — Elo atualizado com todos os resultados da Copa ate 05/jul/2026",
        "_resultados_usados": n_results,
        "_nota":         "Jogos ja disputados mostram o resultado real para comparacao com a previsao.",
        "jogos":         predictions,
    }

    dest = ROOT / "data" / "previsoes_oitavas.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {dest}")


if __name__ == "__main__":
    run()
