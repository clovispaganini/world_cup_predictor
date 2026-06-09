"""
Ensemble predictor: weighted combination of Elo + Poisson + XGBoost.

Weights (from config):
  Elo     40 %
  Poisson 35 %
  XGBoost 25 %

When the XGBoost model is unavailable (not trained / missing dataset),
the remaining weight is redistributed proportionally to Elo and Poisson.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ENSEMBLE_WEIGHTS, TOURNAMENT_PHASES
from src import elo_model, poisson_model
from src.data_fetcher import get_recent_team_form, load_elo_ratings
from src.ml_model import WorldCupModel, get_model, prepare_features


# ── Probability normaliser ────────────────────────────────────────────────────

def _normalise(win_a: float, draw: float, win_b: float) -> dict[str, float]:
    total = win_a + draw + win_b
    if total <= 0:
        return {"win_a": 1 / 3, "draw": 1 / 3, "win_b": 1 / 3}
    return {
        "win_a": round(win_a / total, 4),
        "draw":  round(draw  / total, 4),
        "win_b": round(win_b / total, 4),
    }


# ── Main predictor ────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Orchestrates all three models and returns a unified prediction dict.
    """

    def __init__(self) -> None:
        self._elo_ratings: dict[str, float] | None = None

    def _elo(self) -> dict[str, float]:
        if self._elo_ratings is None:
            self._elo_ratings = load_elo_ratings()
        return self._elo_ratings

    def predict(
        self,
        team_a: str,
        team_b: str,
        team_result_a: dict,
        team_result_b: dict,
        tournament_phase_name: str = "Fase de Grupos",
        injured_a: list[str] | None = None,
        injured_b: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run all models and return the ensemble prediction.

        Parameters
        ----------
        team_a, team_b          : national team names
        team_result_a/b         : output of team_builder.build_team_score()
        tournament_phase_name   : key from config.TOURNAMENT_PHASES
        injured_a/b             : (unused here; already factored into team_result)

        Returns
        -------
        dict with:
            win_a, draw, win_b              – ensemble final probabilities
            breakdown                        – per-model probabilities + weights
            most_probable_score             – (goals_a, goals_b)
            poisson_score_matrix            – 2-D list for heatmap
            shap_features                   – feature-level importance
            top5_impact_a                   – per-player impact for team_a
            top5_impact_b                   – per-player impact for team_b
            elo_a, elo_b, elo_diff
            lambda_a, lambda_b
        """
        phase = TOURNAMENT_PHASES.get(tournament_phase_name, 1)

        ts_a = team_result_a.get("team_score", 7.0)
        ts_b = team_result_b.get("team_score", 7.0)

        form_a = get_recent_team_form(team_a)
        form_b = get_recent_team_form(team_b)

        elo_ratings = self._elo()

        # ── Elo ───────────────────────────────────────────────────────────────
        elo_out  = elo_model.get_match_probabilities(team_a, team_b, elo_ratings)

        # ── Poisson ───────────────────────────────────────────────────────────
        pois_out = poisson_model.get_match_probabilities(
            team_a, team_b,
            team_score_a=ts_a,
            team_score_b=ts_b,
            form_a=form_a,
            form_b=form_b,
            pos_a=team_result_a.get("position_scores"),   # {FWD, MID, DEF, GK}
            pos_b=team_result_b.get("position_scores"),
        )

        # ── XGBoost ───────────────────────────────────────────────────────────
        ml_model: WorldCupModel = get_model()
        mv_a = team_result_a.get("market_value_total_m", 500)
        mv_b = team_result_b.get("market_value_total_m", 500)
        p5_a = team_result_a.get("pct_top5_league", 0.5)
        p5_b = team_result_b.get("pct_top5_league", 0.5)

        X = prepare_features(
            team_a, team_b,
            team_score_a=ts_a,
            team_score_b=ts_b,
            tournament_phase=phase,
            market_value_a=float(mv_a),
            market_value_b=float(mv_b),
            pct_top5_a=float(p5_a),
            pct_top5_b=float(p5_b),
            elo_ratings=elo_ratings,
        )

        ml_out    = ml_model.predict_proba(X)
        shap_vals = ml_model.feature_importance(X) if ml_model.is_trained else {}

        # ── Weights ───────────────────────────────────────────────────────────
        w_elo  = ENSEMBLE_WEIGHTS["elo"]
        w_pois = ENSEMBLE_WEIGHTS["poisson"]
        w_ml   = ENSEMBLE_WEIGHTS["xgboost"] if ml_model.is_trained else 0.0

        if w_ml == 0.0:
            # Redistribute XGBoost weight proportionally
            total_remaining = w_elo + w_pois
            w_elo  = w_elo  / total_remaining
            w_pois = w_pois / total_remaining

        # ── Ensemble ──────────────────────────────────────────────────────────
        raw_win_a = (
            w_elo  * elo_out["win_a"]  +
            w_pois * pois_out["win_a"] +
            w_ml   * ml_out["win_a"]
        )
        raw_draw  = (
            w_elo  * elo_out["draw"]   +
            w_pois * pois_out["draw"]  +
            w_ml   * ml_out["draw"]
        )
        raw_win_b = (
            w_elo  * elo_out["win_b"]  +
            w_pois * pois_out["win_b"] +
            w_ml   * ml_out["win_b"]
        )

        final = _normalise(raw_win_a, raw_draw, raw_win_b)

        # ── Player impact via SHAP ────────────────────────────────────────────
        top5_a = ml_model.player_impact(
            team_result_a.get("rated_players", []), shap_vals
        )
        top5_b = ml_model.player_impact(
            team_result_b.get("rated_players", []), shap_vals
        )

        # If no SHAP (untrained), use team_builder's built-in impact list
        if not top5_a:
            top5_a = team_result_a.get("top5_impact", [])
        if not top5_b:
            top5_b = team_result_b.get("top5_impact", [])

        return {
            # Final probabilities
            "win_a": final["win_a"],
            "draw":  final["draw"],
            "win_b": final["win_b"],

            # Per-model breakdown
            "breakdown": {
                "elo":     {**elo_out,  "weight": round(w_elo,  3)},
                "poisson": {**pois_out, "weight": round(w_pois, 3)},
                "xgboost": {**ml_out,   "weight": round(w_ml,   3)},
            },

            # Score prediction
            "most_probable_score":  pois_out.get("most_probable_score", (1, 1)),
            "most_probable_prob":   pois_out.get("most_probable_prob",   0.07),
            "expected_score":       pois_out.get("expected_score",       (1.0, 1.0)),
            "top_scorelines":       pois_out.get("top_scorelines",       []),
            "poisson_score_matrix": pois_out.get("score_matrix",         []),

            # Auxiliary stats
            "elo_a":    elo_out.get("elo_a", 1800),
            "elo_b":    elo_out.get("elo_b", 1800),
            "elo_diff": elo_out.get("elo_diff", 0),
            "lambda_a": pois_out.get("lambda_a", 1.35),
            "lambda_b": pois_out.get("lambda_b", 1.35),

            # Player impact
            "top5_impact_a": top5_a,
            "top5_impact_b": top5_b,
            "shap_features": shap_vals,

            # Metadata
            "team_score_a": ts_a,
            "team_score_b": ts_b,
            "ml_trained":   ml_model.is_trained,
        }


# ── Module-level convenience function ────────────────────────────────────────

_predictor: EnsemblePredictor | None = None


def predict_match(
    team_a: str,
    team_b: str,
    team_result_a: dict,
    team_result_b: dict,
    tournament_phase_name: str = "Fase de Grupos",
) -> dict[str, Any]:
    """Convenience wrapper around EnsemblePredictor.predict()."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor.predict(
        team_a, team_b,
        team_result_a, team_result_b,
        tournament_phase_name,
    )
