"""
Calculate a single weighted score per player combining national-team
performance and club performance (adjusted for league strength).

Key logic:
  - National team context carries 65 % weight (or 45 % if <5 games in 24 months)
  - Club performance is multiplied by a league-strength coefficient before weighting
  - Both streams use exponential temporal decay (half-life = 180 days)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DECAY_HALF_LIFE_DAYS,
    LEAGUE_STRENGTH,
    MISSING_SOURCE_PENALTY,
    MIN_NATIONAL_GAMES_FOR_HIGH_WEIGHT,
    WEIGHT_CLUB,
    WEIGHT_CLUB_LOW_SAMPLE,
    WEIGHT_NATIONAL_TEAM,
    WEIGHT_NATIONAL_TEAM_LOW_SAMPLE,
)


# ── Decay ─────────────────────────────────────────────────────────────────────

def temporal_decay(game_date: datetime, reference: datetime,
                   half_life: int = DECAY_HALF_LIFE_DAYS) -> float:
    """
    Exponential decay weight.

    A game played exactly half_life days ago receives weight 0.5.
    Games in the future (edge case) receive weight 1.0.
    """
    days = max(0, (reference - game_date).days)
    return 2.0 ** (-days / half_life)


def _weighted_mean(ratings: list[float], dates: list[datetime],
                   reference: datetime) -> float:
    """Decay-weighted average of a list of per-game ratings."""
    weights = [temporal_decay(d, reference) for d in dates]
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(ratings))
    return float(np.average(ratings, weights=weights))


# ── League adjustment ─────────────────────────────────────────────────────────

def league_coefficient(league: str) -> float:
    """Return the strength multiplier for the given league name."""
    return LEAGUE_STRENGTH.get(league, LEAGUE_STRENGTH["Other"])


# ── Core rating function ──────────────────────────────────────────────────────

def calculate_player_score(
    national_games: list[dict],   # [{"date": datetime, "rating": float}, ...]
    club_games:     list[dict],   # [{"date": datetime, "rating": float}, ...]
    league:         str,
    reference_date: datetime | None = None,
) -> dict[str, Any]:
    """
    Compute a single player score on a 0-10 scale.

    Parameters
    ----------
    national_games : list of dicts with 'date' (datetime) and 'rating' (float)
    club_games     : list of dicts with 'date' (datetime) and 'rating' (float)
        Club ratings are multiplied by the league coefficient before weighting.
    league         : name of the player's current club league
    reference_date : today by default; used as the decay anchor

    Returns
    -------
    dict with keys:
        score           – final weighted player score
        national_rating – decay-weighted national-team average (or None)
        club_rating     – decay-weighted club average after league adjustment (or None)
        raw_club_rating – club average before league adjustment
        league_coeff    – league strength coefficient
        low_sample      – True if fewer than MIN_NATIONAL_GAMES_FOR_HIGH_WEIGHT
                          national games in the last 24 months
        n_national_games – count of national games within 24 months
        w_national      – weight applied to national_rating
        w_club          – weight applied to club_rating
    """
    if reference_date is None:
        reference_date = datetime.now()

    lc = league_coefficient(league)

    # ── National team component ───────────────────────────────────────────────
    cutoff_24m = reference_date.timestamp() - 730 * 86400
    recent_nat = [g for g in national_games
                  if g["date"].timestamp() >= cutoff_24m]
    n_nat = len(recent_nat)
    low_sample = n_nat < MIN_NATIONAL_GAMES_FOR_HIGH_WEIGHT

    if recent_nat:
        nat_rating: float | None = _weighted_mean(
            [g["rating"] for g in recent_nat],
            [g["date"]   for g in recent_nat],
            reference_date,
        )
    else:
        nat_rating = None

    # ── Club component ────────────────────────────────────────────────────────
    if club_games:
        raw_club = _weighted_mean(
            [g["rating"] for g in club_games],
            [g["date"]   for g in club_games],
            reference_date,
        )
        club_rating: float | None = raw_club * lc
    else:
        raw_club   = None
        club_rating = None

    # ── Weights ───────────────────────────────────────────────────────────────
    if low_sample:
        w_nat  = WEIGHT_NATIONAL_TEAM_LOW_SAMPLE
        w_club = WEIGHT_CLUB_LOW_SAMPLE
    else:
        w_nat  = WEIGHT_NATIONAL_TEAM
        w_club = WEIGHT_CLUB

    # ── Combine ───────────────────────────────────────────────────────────────
    if nat_rating is not None and club_rating is not None:
        score = nat_rating * w_nat + club_rating * w_club
    elif nat_rating is not None:
        score = nat_rating * (1.0 - MISSING_SOURCE_PENALTY)
        w_nat, w_club = 1.0, 0.0
    elif club_rating is not None:
        score = club_rating * (1.0 - MISSING_SOURCE_PENALTY)
        w_nat, w_club = 0.0, 1.0
    else:
        score = 6.5   # global fallback median

    return {
        "score":           round(score, 3),
        "national_rating": round(nat_rating, 3) if nat_rating is not None else None,
        "club_rating":     round(club_rating, 3) if club_rating is not None else None,
        "raw_club_rating": round(raw_club, 3) if raw_club is not None else None,
        "league_coeff":    lc,
        "low_sample":      low_sample,
        "n_national_games": n_nat,
        "w_national":      w_nat,
        "w_club":          w_club,
    }


# ── Convenience: build from raw dicts (used by team_builder / demo) ───────────

def rate_player_from_stats(player: dict, reference_date: datetime | None = None) -> dict:
    """
    Accept the flat dict format used in demo_squads.json and return a full rating dict.

    Expected keys: rating_national, rating_club, n_national_games, league, name
    Dates are synthesised to preserve the decay logic with a uniform spacing.
    """
    if reference_date is None:
        reference_date = datetime.now()

    from datetime import timedelta

    n_nat = player.get("n_national_games", 0)
    r_nat = player.get("rating_national")
    r_club = player.get("rating_club")
    league = player.get("league", "Other")

    # Synthesise game-level records from aggregate stats
    # Space them ~14 days apart ending at today, varying around the mean ±0.2
    rng = lambda base, n: [base + np.random.uniform(-0.15, 0.15) for _ in range(n)]

    if r_nat is not None and n_nat > 0:
        nat_games = [
            {
                "date":   reference_date - timedelta(days=14 * i + 7),
                "rating": max(1.0, min(10.0, v)),
            }
            for i, v in enumerate(rng(r_nat, min(n_nat, 20)))
        ]
    else:
        nat_games = []

    if r_club is not None:
        n_club = 30
        club_games = [
            {
                "date":   reference_date - timedelta(days=7 * i + 3),
                "rating": max(1.0, min(10.0, v)),
            }
            for i, v in enumerate(rng(r_club, n_club))
        ]
    else:
        club_games = []

    result = calculate_player_score(nat_games, club_games, league, reference_date)
    result["player_name"]      = player.get("name", "Unknown")
    result["club"]             = player.get("club", "Unknown")
    result["league"]           = league
    result["position"]         = player.get("position", "MID")
    result["market_value_m"]   = player.get("market_value_m")
    result["wc_games"]         = player.get("wc_games", 0)
    result["rating_national_raw"] = r_nat
    result["rating_club_raw"]     = r_club
    return result
