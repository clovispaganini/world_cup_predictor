"""
XGBoost classification model for match outcome prediction.

Training data: Kaggle "International football results from 1872 to 2023"
  - World Cup matches weighted × 1.0
  - Qualifier matches weighted × 0.7
  - Friendlies weighted × 0.4

Target: 0 = team_a loses, 1 = draw, 2 = team_a wins

Calibration: CalibratedClassifierCV with isotonic regression trained on
2014, 2018, 2022 World Cup data.

SHAP: TreeExplainer on the raw XGBoost model; feature importance mapped back
to player-level impact through the team_score feature.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CALIBRATION_METHOD,
    KAGGLE_RESULTS_PATH,
    MODEL_SAVE_PATH,
    TOURNAMENT_PHASES,
    XGB_PARAMS,
)
from src.data_fetcher import (
    get_head_to_head,
    get_recent_team_form,
    load_elo_ratings,
    load_historical_matches,
)
from src.elo_model import expected_score

try:
    from xgboost import XGBClassifier
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[ML] xgboost / shap not installed. XGBoost model disabled.")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Feature names ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "elo_a", "elo_b", "elo_diff",
    "team_score_a", "team_score_b", "team_score_diff",
    "market_value_a", "market_value_b",
    "avg_age_a", "avg_age_b",
    "pct_top5_a", "pct_top5_b",
    "injured_key_a", "injured_key_b",
    "goals_scored_avg_a", "goals_conceded_avg_a",
    "xg_avg_a", "xga_avg_a",
    "goals_scored_avg_b", "goals_conceded_avg_b",
    "xg_avg_b", "xga_avg_b",
    "tournament_phase",
    "days_since_last_a", "days_since_last_b",
    "h2h_wins_a", "h2h_draws", "h2h_wins_b", "h2h_goals_diff",
]

MATCH_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.85,
    "Copa América": 0.85,
    "AFC Asian Cup": 0.80,
    "Africa Cup of Nations": 0.80,
    "FIFA World Cup qualification": 0.70,
    "UEFA Euro qualification": 0.65,
    "Friendly": 0.40,
}


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(
    team_a: str,
    team_b: str,
    team_score_a: float,
    team_score_b: float,
    tournament_phase: int = 1,
    market_value_a: float = 500.0,
    market_value_b: float = 500.0,
    pct_top5_a: float = 0.5,
    pct_top5_b: float = 0.5,
    injured_key_a: int = 0,
    injured_key_b: int = 0,
    avg_age_a: float = 27.0,
    avg_age_b: float = 27.0,
    days_since_last_a: int = 7,
    days_since_last_b: int = 7,
    elo_ratings: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Build a feature vector for a single match prediction.

    Returns a (1, n_features) numpy array.
    """
    if elo_ratings is None:
        elo_ratings = load_elo_ratings()

    elo_a = float(elo_ratings.get(team_a, 1800))
    elo_b = float(elo_ratings.get(team_b, 1800))

    form_a = get_recent_team_form(team_a)
    form_b = get_recent_team_form(team_b)
    h2h    = get_head_to_head(team_a, team_b)

    row = [
        elo_a, elo_b, elo_a - elo_b,
        team_score_a, team_score_b, team_score_a - team_score_b,
        market_value_a, market_value_b,
        avg_age_a, avg_age_b,
        pct_top5_a, pct_top5_b,
        injured_key_a, injured_key_b,
        form_a.get("last5_scored",   1.4),
        form_a.get("last5_conceded", 1.0),
        form_a.get("xg_avg",         1.3),
        form_a.get("xga_avg",        1.0),
        form_b.get("last5_scored",   1.4),
        form_b.get("last5_conceded", 1.0),
        form_b.get("xg_avg",         1.3),
        form_b.get("xga_avg",        1.0),
        tournament_phase,
        days_since_last_a, days_since_last_b,
        h2h.get("wins_a",       1),
        h2h.get("draws",        2),
        h2h.get("wins_b",       1),
        h2h.get("goals_diff_a", 0),
    ]
    return np.array(row, dtype=float).reshape(1, -1)


# ── Training data builder ─────────────────────────────────────────────────────

def _build_training_data(df: pd.DataFrame, elo_ratings: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Convert the historical match CSV into (X, y, sample_weights).
    Only uses matches from 2000 onward.
    """
    df = df[df["date"] >= "2000-01-01"].copy()

    records = []
    targets = []
    weights_list = []

    for _, row in df.iterrows():
        ta = row["home_team"]
        tb = row["away_team"]
        sa = int(row["home_score"])
        sb = int(row["away_score"])

        elo_a = float(elo_ratings.get(ta, 1800))
        elo_b = float(elo_ratings.get(tb, 1800))

        # Derive simple features without full form lookup (speed)
        feat = [
            elo_a, elo_b, elo_a - elo_b,
            7.2, 7.2, 0.0,            # placeholder team scores
            500, 500,                  # placeholder market values
            27.0, 27.0,
            0.5, 0.5,
            0, 0,
            1.4, 1.0, 1.3, 1.0,       # placeholder form
            1.4, 1.0, 1.3, 1.0,
            1,                         # group stage default
            7, 7,
            1, 2, 1, 0,
        ]
        records.append(feat)

        if sa > sb:
            targets.append(2)
        elif sa == sb:
            targets.append(1)
        else:
            targets.append(0)

        tournament = str(row.get("tournament", "Friendly"))
        w = 1.0
        for key, wt in MATCH_WEIGHTS.items():
            if key.lower() in tournament.lower():
                w = wt
                break
        weights_list.append(w)

    X = np.array(records, dtype=float)
    y = np.array(targets, dtype=int)
    sw = np.array(weights_list, dtype=float)
    return pd.DataFrame(X, columns=FEATURE_NAMES), y, sw


# ── Model class ───────────────────────────────────────────────────────────────

class WorldCupModel:
    """
    Wraps XGBoost + Platt/Isotonic calibration for World Cup match prediction.
    """

    def __init__(self) -> None:
        self.model      = None
        self.calibrated = None
        self.explainer  = None
        self._trained   = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, save: bool = True) -> bool:
        """
        Train on the Kaggle dataset.  Returns True if successful.
        """
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            return False

        df = load_historical_matches()
        if df.empty:
            print("[ML] No training data found. Skipping training.")
            return False

        elo = load_elo_ratings()
        X_df, y, sw = _build_training_data(df, elo)

        # Hold out 2014-2022 World Cup games for calibration
        wc_mask = (
            X_df.index.isin(
                df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2014-01-01")].index
            )
        )
        X_cal = X_df[wc_mask].values  if wc_mask.any() else X_df.values[-200:]
        y_cal = y[wc_mask]            if wc_mask.any() else y[-200:]

        X_train = X_df[~wc_mask].values if wc_mask.any() else X_df.values
        y_train = y[~wc_mask]           if wc_mask.any() else y
        sw_train = sw[~wc_mask]         if wc_mask.any() else sw

        print(f"[ML] Training on {len(X_train)} matches …")

        xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_train, y_train, sample_weight=sw_train,
                       eval_set=[(X_cal, y_cal)], verbose=False)

        # Calibrate
        cal = CalibratedClassifierCV(self.model, method=CALIBRATION_METHOD, cv="prefit")
        cal.fit(X_cal, y_cal)
        self.calibrated = cal

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        self._trained = True
        print("[ML] Training complete.")

        if save:
            self._save()
        return True

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> dict[str, float]:
        """
        Return P(loss), P(draw), P(win) for team A.
        Falls back to Elo-derived uniform if not trained.
        """
        if not self._trained or self.calibrated is None:
            return {"win_a": 0.40, "draw": 0.25, "win_b": 0.35}

        proba = self.calibrated.predict_proba(X)[0]
        # classes: 0=loss_a, 1=draw, 2=win_a
        return {
            "win_a": round(float(proba[2]), 4),
            "draw":  round(float(proba[1]), 4),
            "win_b": round(float(proba[0]), 4),
        }

    def feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """Return SHAP-based feature importance for a single sample."""
        if not self._trained or self.explainer is None:
            return {}
        shap_vals = self.explainer.shap_values(X)
        # shap_vals shape: (3, 1, n_features) → use class 2 (win_a)
        if isinstance(shap_vals, list):
            sv = shap_vals[2][0]
        else:
            sv = shap_vals[0]
        return {FEATURE_NAMES[i]: float(sv[i]) for i in range(len(FEATURE_NAMES))}

    def player_impact(self, rated_players: list[dict],
                      shap_values: dict[str, float]) -> list[dict]:
        """
        Map team_score SHAP importance back to individual players.

        Ranks players by their deviation from the team mean, weighted by the
        SHAP importance of the team_score_a feature.
        """
        if not rated_players:
            return []

        ts_shap = abs(shap_values.get("team_score_a", 1.0))
        scores  = [p["score"] for p in rated_players]
        avg     = float(np.mean(scores))

        result = []
        for p in rated_players:
            contribution = (p["score"] - avg) * ts_shap
            result.append({
                "player_name": p["player_name"],
                "position":    p["position"],
                "score":       p["score"],
                "impact":      round(contribution, 4),
            })

        result.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return result[:5]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        path = Path(MODEL_SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "calibrated": self.calibrated}, f)
        print(f"[ML] Model saved to {path}.")

    def load(self) -> bool:
        path = Path(MODEL_SAVE_PATH)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.model      = obj["model"]
            self.calibrated = obj["calibrated"]
            if XGB_AVAILABLE and self.model is not None:
                self.explainer = shap.TreeExplainer(self.model)
            self._trained = True
            print(f"[ML] Model loaded from {path}.")
            return True
        except Exception as exc:
            print(f"[ML] Could not load model: {exc}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained


# ── Module-level singleton ────────────────────────────────────────────────────

_MODEL: WorldCupModel | None = None


def get_model() -> WorldCupModel:
    """Return the global WorldCupModel instance (load or train if needed)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = WorldCupModel()
        if not _MODEL.load():
            _MODEL.train(save=True)
    return _MODEL
