"""
Aggregate individual player scores into a single team score.

Bonuses applied on top of the position-weighted average:
  - Cohesion: ≥ COHESION_THRESHOLD starters from the same club → +COHESION_BONUS
  - Experience: each starter with >2 World Cup games → +EXPERIENCE_BONUS_PER_PLAYER
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    COHESION_BONUS,
    COHESION_THRESHOLD,
    EXPERIENCE_BONUS_PER_PLAYER,
    MIN_WORLD_CUP_GAMES,
    POSITION_WEIGHTS,
    SUBSTITUTE_WEIGHT_FACTOR,
    TOP_5_LEAGUES,
)
from src.player_rater import rate_player_from_stats


# ── Position normalisation ────────────────────────────────────────────────────

def _normalise_position(raw: str) -> str:
    raw = raw.upper().strip()
    if raw in ("GK", "G"):
        return "GK"
    if raw in ("DEF", "CB", "RB", "LB", "RWB", "LWB", "D"):
        return "DEF"
    if raw in ("MID", "CM", "DM", "AM", "CDM", "CAM", "RM", "LM", "M"):
        return "MID"
    if raw in ("FWD", "CF", "ST", "LW", "RW", "SS", "F", "ATT"):
        return "FWD"
    return "MID"


# ── Squad-level aggregation ───────────────────────────────────────────────────

def build_team_score(
    players: list[dict],
    injured_names: list[str] | None = None,
    reference_date: datetime | None = None,
) -> dict[str, Any]:
    """
    Compute a team's aggregate score from a list of player dicts.

    Parameters
    ----------
    players       : raw player records (from demo_squads or manual input)
    injured_names : names of players ruled out; their substitute (if available)
                    contributes at SUBSTITUTE_WEIGHT_FACTOR instead
    reference_date: decay anchor (defaults to today)

    Returns
    -------
    dict with:
        team_score        – final weighted score (roughly 6-9 range)
        position_scores   – average score per position group
        cohesion_bonus    – bonus actually applied
        experience_bonus  – bonus actually applied
        rated_players     – list of per-player rating dicts (sorted by score desc)
        top5_impact       – top 5 players ranked by contribution to team_score
        avg_age           – placeholder (set if 'age' field present)
        market_value_total– total squad market value (€M)
        pct_top5_league   – fraction of starters from top-5 leagues
    """
    if reference_date is None:
        reference_date = datetime.now()
    if injured_names is None:
        injured_names = []

    injured_lower = {n.lower() for n in injured_names}

    # Split into starters / substitutes (first 11 = starters)
    starters    = players[:11]
    substitutes = players[11:]

    # Compute individual ratings
    rated: list[dict] = []
    for p in players:
        r = rate_player_from_stats(p, reference_date)
        r["is_starter"] = p in starters
        r["is_injured"] = p["name"].lower() in injured_lower
        rated.append(r)

    rated_starters = [r for r in rated if r["is_starter"]]

    # ── Position-weighted sum ─────────────────────────────────────────────────
    pos_buckets: dict[str, list[float]] = {k: [] for k in POSITION_WEIGHTS}
    for r in rated_starters:
        pos = _normalise_position(r["position"])
        if r["is_injured"]:
            # Try to find a sub of the same position
            sub = _find_sub(pos, substitutes, injured_lower, rated)
            if sub:
                pos_buckets[pos].append(sub["score"] * SUBSTITUTE_WEIGHT_FACTOR)
            # If no sub, skip this slot (slight penalty implicit)
        else:
            pos_buckets[pos].append(r["score"])

    position_scores: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for pos, scores in pos_buckets.items():
        if scores:
            avg = sum(scores) / len(scores)
            position_scores[pos] = round(avg, 3)
            weighted_sum  += avg * POSITION_WEIGHTS[pos]
            total_weight  += POSITION_WEIGHTS[pos]

    base_score = (weighted_sum / total_weight) if total_weight > 0 else 6.5

    # ── Cohesion bonus ────────────────────────────────────────────────────────
    clubs = [p.get("club", "") for p in starters if p["name"].lower() not in injured_lower]
    club_counts = Counter(clubs)
    most_common_club, most_common_count = club_counts.most_common(1)[0] if club_counts else ("", 0)
    cohesion = COHESION_BONUS if most_common_count >= COHESION_THRESHOLD else 0.0

    # ── Experience bonus ──────────────────────────────────────────────────────
    exp_players = [
        p for p in starters
        if p.get("wc_games", 0) > MIN_WORLD_CUP_GAMES
        and p["name"].lower() not in injured_lower
    ]
    experience = len(exp_players) * EXPERIENCE_BONUS_PER_PLAYER

    team_score = base_score * (1.0 + cohesion + experience)

    # ── Top-5-impact ranking ──────────────────────────────────────────────────
    active_rated = [r for r in rated_starters if not r["is_injured"]]
    avg_score    = sum(r["score"] for r in active_rated) / max(len(active_rated), 1)
    for r in active_rated:
        pos = _normalise_position(r["position"])
        r["impact"] = (r["score"] - avg_score) * POSITION_WEIGHTS.get(pos, 0.25)
    top5_impact = sorted(active_rated, key=lambda x: x["impact"], reverse=True)[:5]

    # ── Supplementary stats ───────────────────────────────────────────────────
    mv_total = sum(p.get("market_value_m") or 0 for p in starters)
    top5_pct = sum(
        1 for p in starters
        if p.get("league", "") in TOP_5_LEAGUES and p["name"].lower() not in injured_lower
    ) / max(len(starters), 1)

    rated.sort(key=lambda r: r["score"], reverse=True)

    return {
        "team_score":          round(team_score, 4),
        "base_score":          round(base_score, 4),
        "position_scores":     position_scores,
        "cohesion_bonus":      cohesion,
        "cohesion_club":       most_common_club if most_common_count >= COHESION_THRESHOLD else None,
        "experience_bonus":    round(experience, 4),
        "rated_players":       rated,
        "top5_impact":         top5_impact,
        "market_value_total_m": round(mv_total, 1),
        "pct_top5_league":     round(top5_pct, 3),
        "n_experienced":       len(exp_players),
    }


def _find_sub(position: str, substitutes: list[dict], injured_lower: set,
              all_rated: list[dict]) -> dict | None:
    for p in substitutes:
        if _normalise_position(p.get("position", "")) == position:
            for r in all_rated:
                if r["player_name"].lower() == p["name"].lower():
                    return r
    return None


# ── Six-dimension radar data ──────────────────────────────────────────────────

def radar_dimensions(team_result: dict) -> dict[str, float]:
    """
    Return a dict of 6 radar dimensions, each normalised to 0-100.

    Dimensions: Ataque, Defesa, Meio, Experiência, Coesão, Forma Recente
    """
    ps = team_result.get("position_scores", {})
    base = team_result.get("base_score", 7.0)

    def _scale(v: float, lo: float = 5.5, hi: float = 9.5) -> float:
        return round(max(0.0, min(100.0, (v - lo) / (hi - lo) * 100)), 1)

    return {
        "Ataque":         _scale(ps.get("FWD", base)),
        "Defesa":         _scale(ps.get("DEF", base)),
        "Meio":           _scale(ps.get("MID", base)),
        "Experiência":    round(min(100, team_result.get("n_experienced", 0) * 10), 1),
        "Coesão":         100.0 if team_result.get("cohesion_bonus", 0) > 0 else 30.0,
        "Forma Recente":  _scale(base),
    }
