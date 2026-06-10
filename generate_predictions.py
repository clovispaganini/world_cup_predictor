"""
Gera previsões do modelo ensemble para todos os 72 jogos da fase de grupos
e salva em data/group_predictions.json para auditoria futura.

Uso:
    python generate_predictions.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.schedule_loader import get_all_group_matches, get_squad
from src.team_builder import build_team_score
from src.ensemble import predict_match

OUTPUT_PATH = ROOT / "data" / "group_predictions.json"


def run() -> None:
    matches = get_all_group_matches()
    predictions = []

    print(f"Gerando previsões para {len(matches)} jogos...\n")

    for i, match in enumerate(matches, 1):
        home = match["home"]
        away = match["away"]

        players_h, source_h, _ = get_squad(home)
        players_a, source_a, _ = get_squad(away)

        team_h = build_team_score(players_h)
        team_a = build_team_score(players_a)

        result = predict_match(
            home, away,
            team_h, team_a,
            tournament_phase_name="Fase de Grupos",
        )

        score_h, score_a = result["most_probable_score"]
        exp_h, exp_a     = result["expected_score"]

        pred = {
            "group":    match["group"],
            "matchday": match["matchday"],
            "date":     match["date"],
            "time_brt": match.get("time_brt", ""),
            "venue":    match.get("venue", ""),
            "home":     home,
            "away":     away,
            "squad_source_home": source_h,
            "squad_source_away": source_a,
            # Probabilidades
            "prob_win_home": result["win_a"],
            "prob_draw":     result["draw"],
            "prob_win_away": result["win_b"],
            # Placar previsto
            "predicted_score_home": int(score_h),
            "predicted_score_away": int(score_a),
            # Gols esperados (lambda Poisson)
            "expected_goals_home": round(float(exp_h), 3),
            "expected_goals_away": round(float(exp_a), 3),
            "lambda_home": round(result["lambda_a"], 3),
            "lambda_away": round(result["lambda_b"], 3),
            # Elo
            "elo_home": result["elo_a"],
            "elo_away": result["elo_b"],
            "elo_diff": result["elo_diff"],
            # Score do elenco
            "team_score_home": round(result["team_score_a"], 3),
            "team_score_away": round(result["team_score_b"], 3),
            # Modelo XGBoost usado?
            "ml_trained": result["ml_trained"],
            # Breakdown por modelo
            "breakdown": {
                "elo": {
                    "win_home": result["breakdown"]["elo"]["win_a"],
                    "draw":     result["breakdown"]["elo"]["draw"],
                    "win_away": result["breakdown"]["elo"]["win_b"],
                    "weight":   result["breakdown"]["elo"]["weight"],
                },
                "poisson": {
                    "win_home": result["breakdown"]["poisson"]["win_a"],
                    "draw":     result["breakdown"]["poisson"]["draw"],
                    "win_away": result["breakdown"]["poisson"]["win_b"],
                    "weight":   result["breakdown"]["poisson"]["weight"],
                },
                "xgboost": {
                    "win_home": result["breakdown"]["xgboost"]["win_a"],
                    "draw":     result["breakdown"]["xgboost"]["draw"],
                    "win_away": result["breakdown"]["xgboost"]["win_b"],
                    "weight":   result["breakdown"]["xgboost"]["weight"],
                },
            },
        }

        predictions.append(pred)
        favorite = home if result["win_a"] > result["win_b"] else (away if result["win_b"] > result["win_a"] else "EMPATE")
        print(f"[{i:02d}/72] Grupo {match['group']} - {home} vs {away}  =>  {score_h}-{score_a}  |  {result['win_a']:.0%}/{result['draw']:.0%}/{result['win_b']:.0%}  (fav: {favorite})")

    output = {
        "_generated_at": datetime.now().isoformat(),
        "_model_version": "Elo(40%) + Poisson(35%) + XGBoost(25%)",
        "_note": "Previsões geradas ANTES dos jogos. Use group_results.json para registrar os resultados reais.",
        "total_matches": len(predictions),
        "predictions": predictions,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
