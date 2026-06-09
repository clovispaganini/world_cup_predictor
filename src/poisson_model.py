"""
Bivariate Negative-Binomial model for match-score prediction.

Lambda computation (core change):
  Instead of using xG × (opponent_xga / global_mean), lambdas are derived
  directly from the **attack vs defence cross-comparison** of position scores:

      att_A = 0.6·FWD_A + 0.4·MID_A   (what A creates)
      def_B = 0.7·DEF_B + 0.3·GK_B    (what B concedes)

      advantage_A = att_A − def_B       (>0 → A's attack > B's defence)
      λ_A = GLOBAL_ATT × exp( SENSITIVITY × min(advantage_A, MAX_ADVANTAGE) )

  This formula:
    • Produces elastic scores (3-0, 4-0) when there is a large quality gap.
    • Falls back to approximately GLOBAL_ATT (1.35) for equal teams.
    • Is capped to prevent unrealistic scores against extreme mismatches.

  When position scores are not available, the model falls back to the
  xG-form approach used previously.

Dixon-Coles correction:
  DIXON_COLES_RHO = 0.0 (disabled) – the previous rho=-0.13 was
  artificially suppressing draws. With the new att/def lambda the draw
  probability is naturally calibrated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ATK_DEF_MAX_ADVANTAGE,
    ATK_DEF_SENSITIVITY,
    DIXON_COLES_RHO,
    LAMBDA_FLOOR,
    MAX_GOALS,
    NB_DISPERSION,
    N_SIMULATIONS,
    TEAM_SCORE_LAMBDA_MAX,
    TEAM_SCORE_LAMBDA_MIN,
    TEAM_SCORE_NEUTRAL,
    TOP_SCORELINES_DISPLAY,
)
from src.data_fetcher import load_historical_matches


# ── Global constants ──────────────────────────────────────────────────────────

_GLOBAL_ATT: float = 1.35   # global average goals per match (World Cup level)
_NEUTRAL_POS: float = 7.0   # neutral position score (all sectors)


# ── Attack / defence estimates from historical data ───────────────────────────

def _estimate_attack_defence(team: str, df) -> tuple[float, float]:
    """
    Return (attack_strength, defence_weakness) for the given team relative to
    the global average.  Falls back to 1.0 / 1.0 if data is insufficient.
    Used only when position-score data is unavailable (fallback path).
    """
    if df.empty:
        return 1.0, 1.0

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    scored   = list(home["home_score"]) + list(away["away_score"])
    conceded = list(home["away_score"]) + list(away["home_score"])

    if len(scored) < 5:
        return 1.0, 1.0

    attack  = (sum(scored)   / len(scored))   / _GLOBAL_ATT
    defence = (sum(conceded) / len(conceded))  / _GLOBAL_ATT
    return float(attack), float(defence)


# ── Team-score → lambda scaling (fallback only) ───────────────────────────────

def _team_score_factor(team_score: float) -> float:
    """
    Map team_score (≈ 6-9 range) to a lambda multiplier in
    [TEAM_SCORE_LAMBDA_MIN, TEAM_SCORE_LAMBDA_MAX].
    Used only in the xG-based fallback path.
    """
    slope  = (TEAM_SCORE_LAMBDA_MAX - TEAM_SCORE_LAMBDA_MIN) / 3.0
    factor = 1.0 + slope * (team_score - TEAM_SCORE_NEUTRAL)
    return float(np.clip(factor, TEAM_SCORE_LAMBDA_MIN, TEAM_SCORE_LAMBDA_MAX))


# ── Attack / defence position composites ─────────────────────────────────────

def _attack_composite(pos: dict) -> float:
    """
    Weighted attack rating from position_scores dict.
    att = 0.6·FWD + 0.4·MID  (both on the 0-10 player-score scale)
    """
    return (0.6 * pos.get("FWD", _NEUTRAL_POS)
          + 0.4 * pos.get("MID", _NEUTRAL_POS))


def _defence_composite(pos: dict) -> float:
    """
    Weighted defence rating from position_scores dict.
    def = 0.7·DEF + 0.3·GK
    """
    return (0.7 * pos.get("DEF", _NEUTRAL_POS)
          + 0.3 * pos.get("GK",  _NEUTRAL_POS))


def _lambda_from_positions(att: float, def_opp: float) -> float:
    """
    Compute expected-goals lambda for one side given:
        att      – attacking composite of THIS team
        def_opp  – defensive composite of the OPPONENT

    Formula:
        advantage = att − def_opp
        λ = GLOBAL_ATT × exp( SENSITIVITY × min(advantage, MAX_ADVANTAGE) )

    Properties:
        • att == def_opp == 7.0  →  λ = GLOBAL_ATT = 1.35  (neutral)
        • att >> def_opp          →  λ grows exponentially (elastic mismatch)
        • att << def_opp          →  λ shrinks (strong defence suppresses attack)
        • advantage capped at ATK_DEF_MAX_ADVANTAGE = 1.5 to prevent unrealistic scores
    """
    advantage = att - def_opp
    capped    = min(advantage, ATK_DEF_MAX_ADVANTAGE)
    lam       = _GLOBAL_ATT * np.exp(ATK_DEF_SENSITIVITY * capped)
    return max(LAMBDA_FLOOR, float(lam))


# ── Negative-Binomial marginals ───────────────────────────────────────────────

def _nb_pmf(lambda_val: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Return the NB probability vector P(X = k) for k = 0 … max_goals.

    Parameterisation:  r = λ / NB_DISPERSION,  p = r / (r + λ)
    → E[X] = λ(1 + NB_DISPERSION),  Var[X] = λ + λ²/r  (overdispersed vs Poisson).

    The vector is renormalised after truncation at max_goals.
    """
    r     = lambda_val / NB_DISPERSION
    p     = r / (r + lambda_val)
    probs = nbinom.pmf(np.arange(max_goals + 1), r, p).astype(float)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


# ── Dixon-Coles τ correction ──────────────────────────────────────────────────

def _dixon_coles_tau(y1: int, y2: int, lam1: float, lam2: float,
                     rho: float = DIXON_COLES_RHO) -> float:
    """
    Dixon-Coles low-score correction factor τ.
    With DIXON_COLES_RHO = 0.0 (default) all τ = 1.0 (no correction).
    Left here for future calibration experiments.
    """
    if rho == 0.0:
        return 1.0
    if y1 == 0 and y2 == 0:
        return 1.0 + lam1 * lam2 * rho
    if y1 == 1 and y2 == 0:
        return 1.0 - lam2 * rho
    if y1 == 0 and y2 == 1:
        return 1.0 - lam1 * rho
    if y1 == 1 and y2 == 1:
        return 1.0 + rho
    return 1.0


# ── Joint score-probability matrix ────────────────────────────────────────────

def score_probability_matrix(lambda_a: float, lambda_b: float,
                              max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    (max_goals+1) × (max_goals+1) joint probability matrix using NB marginals
    with optional Dixon-Coles correction (currently disabled, rho=0.0).
    Row i = goals by team A, column j = goals by team B.
    """
    prob_a = _nb_pmf(lambda_a, max_goals)
    prob_b = _nb_pmf(lambda_b, max_goals)

    n   = max_goals + 1
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tau        = _dixon_coles_tau(i, j, lambda_a, lambda_b)
            mat[i, j]  = prob_a[i] * prob_b[j] * tau

    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ── Derived statistics ────────────────────────────────────────────────────────

def get_top_scorelines(score_matrix: np.ndarray,
                       top_n: int = TOP_SCORELINES_DISPLAY) -> list[dict]:
    """Return the top_n most probable scorelines with probabilities."""
    max_goals = score_matrix.shape[0] - 1
    results: list[dict] = [
        {
            "goals_a":     ga,
            "goals_b":     gb,
            "score":       f"{ga}-{gb}",
            "probability": float(score_matrix[ga, gb]),
            "most_likely": False,
        }
        for ga in range(max_goals + 1)
        for gb in range(max_goals + 1)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    top = results[:top_n]
    if top:
        top[0]["most_likely"] = True
    return top


def get_expected_score(score_matrix: np.ndarray) -> tuple[float, float]:
    """Return the mean (expected) goals for each team."""
    max_goals = score_matrix.shape[0] - 1
    expected_a = float(sum(ga * float(score_matrix[ga, :].sum())
                           for ga in range(max_goals + 1)))
    expected_b = float(sum(gb * float(score_matrix[:, gb].sum())
                           for gb in range(max_goals + 1)))
    return round(expected_a, 1), round(expected_b, 1)


# ── Monte Carlo simulation (validation layer) ─────────────────────────────────

def simulate_match(lambda_a: float, lambda_b: float,
                   n_sims: int = N_SIMULATIONS) -> dict[str, Any]:
    """NB Monte-Carlo simulation for blending with the analytical matrix."""
    r_a = lambda_a / NB_DISPERSION
    r_b = lambda_b / NB_DISPERSION
    p_a = r_a / (r_a + lambda_a)
    p_b = r_b / (r_b + lambda_b)

    rng = np.random.default_rng(42)
    ga  = rng.negative_binomial(r_a, p_a, n_sims)
    gb  = rng.negative_binomial(r_b, p_b, n_sims)

    return {
        "win_a": round(float((ga > gb).mean()), 4),
        "draw":  round(float((ga == gb).mean()), 4),
        "win_b": round(float((ga < gb).mean()), 4),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_match_probabilities(
    team_a: str,
    team_b: str,
    team_score_a: float = TEAM_SCORE_NEUTRAL,
    team_score_b: float = TEAM_SCORE_NEUTRAL,
    form_a: dict | None = None,
    form_b: dict | None = None,
    pos_a: dict | None = None,   # position_scores from team_builder: {FWD, MID, DEF, GK}
    pos_b: dict | None = None,   # position_scores from team_builder: {FWD, MID, DEF, GK}
) -> dict[str, Any]:
    """
    Return NB model probabilities and score distribution for team_a vs team_b.

    Lambda calculation priority:
      1. If pos_a AND pos_b are supplied → attack-vs-defence position model
         (primary path: elastic and quality-differentiated).
      2. Otherwise → xG-form model (fallback, less elastic for mismatches).

    Parameters
    ----------
    team_a, team_b      : national team names
    team_score_a/b      : aggregate squad scores (used in fallback path)
    form_a/b            : dicts with 'xg_avg', 'xga_avg' (fallback path)
    pos_a/b             : position_scores dict {FWD, MID, DEF, GK} — PRIMARY path

    Returns
    -------
    dict: win_a, draw, win_b, lambda_a, lambda_b,
          most_probable_score, most_probable_prob, expected_score,
          top_scorelines, score_matrix
    """
    # ── Lambda computation ────────────────────────────────────────────────────
    if pos_a and pos_b:
        # ── PRIMARY: attack-vs-defence cross-comparison ───────────────────────
        att_a = _attack_composite(pos_a)
        att_b = _attack_composite(pos_b)
        def_a = _defence_composite(pos_a)
        def_b = _defence_composite(pos_b)

        lambda_a = _lambda_from_positions(att_a, def_b)   # A attacks vs B defends
        lambda_b = _lambda_from_positions(att_b, def_a)   # B attacks vs A defends

    else:
        # ── FALLBACK: xG-form approach ────────────────────────────────────────
        df = load_historical_matches()
        att_h, def_h = _estimate_attack_defence(team_a, df)
        att_a_hist, def_b_hist = _estimate_attack_defence(team_b, df)

        _MEAN = _GLOBAL_ATT
        if form_a and form_a.get("xg_avg"):
            opp_def = float(form_b.get("xga_avg", _MEAN)) / _MEAN if form_b else 1.0
            base_a  = float(form_a["xg_avg"]) * opp_def
        else:
            base_a = att_h * def_b_hist * _GLOBAL_ATT

        if form_b and form_b.get("xg_avg"):
            opp_def = float(form_a.get("xga_avg", _MEAN)) / _MEAN if form_a else 1.0
            base_b  = float(form_b["xg_avg"]) * opp_def
        else:
            base_b = att_a_hist * def_h * _GLOBAL_ATT

        fa = _team_score_factor(team_score_a)
        fb = _team_score_factor(team_score_b)
        lambda_a = float(np.clip(base_a * fa, LAMBDA_FLOOR, 6.0))
        lambda_b = float(np.clip(base_b * fb, LAMBDA_FLOOR, 6.0))

    # ── Score matrix (NB + optional DC correction, currently rho=0) ──────────
    mat = score_probability_matrix(lambda_a, lambda_b)

    # Most probable scoreline (mode)
    best_idx    = np.unravel_index(np.argmax(mat), mat.shape)
    most_prob   = (int(best_idx[0]), int(best_idx[1]))
    most_prob_p = float(mat[best_idx])

    top_scores = get_top_scorelines(mat)
    exp_score  = get_expected_score(mat)

    # W/D/L from analytical matrix
    p_win_a = float(np.tril(mat, -1).sum())
    p_draw  = float(np.trace(mat))
    p_win_b = float(np.triu(mat, 1).sum())

    # Blend 70% analytical + 30% NB Monte-Carlo
    sim = simulate_match(lambda_a, lambda_b)

    def blend(analytical: float, simulated: float) -> float:
        return round(0.70 * analytical + 0.30 * simulated, 4)

    raw_win_a = blend(p_win_a, sim["win_a"])
    raw_draw  = blend(p_draw,  sim["draw"])
    raw_win_b = blend(p_win_b, sim["win_b"])

    total = raw_win_a + raw_draw + raw_win_b
    return {
        "win_a":               round(raw_win_a / total, 4),
        "draw":                round(raw_draw  / total, 4),
        "win_b":               round(raw_win_b / total, 4),
        "lambda_a":            round(lambda_a, 3),
        "lambda_b":            round(lambda_b, 3),
        "most_probable_score": most_prob,
        "most_probable_prob":  round(most_prob_p, 4),
        "expected_score":      exp_score,
        "top_scorelines":      top_scores,
        "score_matrix":        mat.tolist(),
    }
