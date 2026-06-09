"""
Loads and queries the WC 2026 schedule (groups + fixtures + knockout bracket)
and the squads file that covers all 48 qualified teams.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Data file paths ────────────────────────────────────────────────────────────

_SCHEDULE_PATH = ROOT / "data" / "wc2026_schedule.json"
_SQUADS_PATH   = ROOT / "data" / "wc2026_squads.json"
_DEMO_PATH     = ROOT / "data" / "demo_squads.json"

# ── Lazy-loaded caches ────────────────────────────────────────────────────────

_schedule: dict | None = None
_squads:   dict | None = None
_demo:     dict | None = None


def _load_schedule() -> dict:
    global _schedule
    if _schedule is None:
        _schedule = json.loads(_SCHEDULE_PATH.read_text(encoding="utf-8"))
    return _schedule


def _load_squads() -> dict:
    global _squads
    if _squads is None:
        _squads = json.loads(_SQUADS_PATH.read_text(encoding="utf-8"))
    return _squads


def _load_demo() -> dict:
    global _demo
    if _demo is None:
        if _DEMO_PATH.exists():
            _demo = json.loads(_DEMO_PATH.read_text(encoding="utf-8"))
        else:
            _demo = {"squads": {}}
    return _demo


# ── Schedule helpers ──────────────────────────────────────────────────────────

def get_all_groups() -> dict[str, list[str]]:
    """Return {group_id: [team1, team2, team3, team4]}."""
    sched = _load_schedule()
    return {gid: gdata["teams"] for gid, gdata in sched["groups"].items()}


def get_group_of(team: str) -> str | None:
    """Return the group letter (A-L) that the team belongs to, or None."""
    for gid, teams in get_all_groups().items():
        if team in teams:
            return gid
    return None


def get_group_matches(group_id: str) -> list[dict]:
    """Return all 6 matches for a group (sorted by matchday)."""
    sched = _load_schedule()
    return sorted(
        sched["groups"].get(group_id.upper(), {}).get("matches", []),
        key=lambda m: (m["matchday"], m["date"]),
    )


def get_all_group_matches() -> list[dict]:
    """Return all 72 group-stage matches, each annotated with the group_id."""
    sched = _load_schedule()
    matches = []
    for gid, gdata in sched["groups"].items():
        for m in gdata["matches"]:
            matches.append({**m, "group": gid})
    return sorted(matches, key=lambda m: (m["date"], m["group"], m["matchday"]))


def get_knockout_bracket() -> dict:
    """Return the full knockout bracket dict."""
    return _load_schedule()["knockout_bracket"]


def get_r32_slots() -> list[dict]:
    """Return the 16 Round-of-32 matchup slot descriptions."""
    return _load_schedule()["knockout_bracket"]["round_of_32"]


def get_matches_by_date() -> dict[str, list[dict]]:
    """
    Return all 72 group-stage matches grouped by date.
    Each match dict is annotated with 'group' and an estimated 'time_brt'.
    Result: {date_str: [match_dict, ...]} ordered chronologically.
    """
    all_matches = get_all_group_matches()

    by_date: dict[str, list] = {}
    for m in all_matches:
        by_date.setdefault(m["date"], []).append(m)

    _TIME_SLOTS = ["13:00", "16:00", "19:00", "22:00"]
    result: dict[str, list] = {}
    for date_str in sorted(by_date.keys()):
        matches = by_date[date_str]
        annotated = []
        for i, m in enumerate(matches):
            slot_idx = min(i, len(_TIME_SLOTS) - 1)
            annotated.append({**m, "time_brt": m.get("time_brt", _TIME_SLOTS[slot_idx])})
        result[date_str] = annotated
    return result


# ── Squad helpers ─────────────────────────────────────────────────────────────

SQUAD_SOURCE_LABELS = {
    "official":     "✅ Convocação oficial",
    "recent_games": "⚽ Baseado nos últimos jogos",
    "estimated":    "📋 Elenco estimado",
}


def get_squad(team: str) -> tuple[list[dict], str, str | None]:
    """
    Return (players_list, squad_source, announced_date) for a team.

    Priority:
      1. demo_squads.json  (8 detailed teams — highest quality)
      2. wc2026_squads.json teams dict (detailed entries)
      3. wc2026_squads.json _estimated_teams (metadata only → generate placeholder)
    """
    # 1. Demo squads (highest detail)
    demo = _load_demo()
    demo_entry = demo.get("squads", {}).get(team)
    if demo_entry and demo_entry.get("players"):
        # Demo squads are all marked as official for the simulation
        return demo_entry["players"], "official", None

    # 2. wc2026_squads detailed entries
    squads = _load_squads()
    entry = squads["teams"].get(team)
    if entry and entry.get("players"):
        return (
            entry["players"],
            entry.get("squad_source", "estimated"),
            entry.get("announced_date"),
        )

    # 3. Estimated-only metadata
    estimated = squads["teams"].get("_estimated_teams", {}).get(team)
    if estimated:
        players = _generate_placeholder_squad(team, estimated.get("elo", 1700))
        return players, "estimated", None

    # 4. Absolute fallback
    players = _generate_placeholder_squad(team, 1750)
    return players, "estimated", None


def get_squad_source(team: str) -> str:
    """Return the squad_source label for a team."""
    _, source, _ = get_squad(team)
    return SQUAD_SOURCE_LABELS.get(source, source)


def get_squad_elo(team: str) -> int:
    """Return the Elo rating stored in wc2026_squads for the team, or 1750."""
    squads = _load_squads()
    entry = squads["teams"].get(team)
    if entry and "elo" in entry:
        return int(entry["elo"])
    est = squads["teams"].get("_estimated_teams", {}).get(team)
    if est and "elo" in est:
        return int(est["elo"])
    demo = _load_demo()
    demo_entry = demo.get("squads", {}).get(team, {})
    if "elo" in demo_entry:
        return int(demo_entry["elo"])
    return 1750


def get_recent_form(team: str) -> dict:
    """Return recent form dict for the team."""
    squads = _load_squads()
    entry = squads["teams"].get(team, {})
    if "recent_form" in entry:
        return entry["recent_form"]
    demo = _load_demo()
    demo_entry = demo.get("squads", {}).get(team, {})
    form = demo.get("recent_form", {}).get(team)
    if form:
        return form
    return {"last5_scored": 1.4, "last5_conceded": 1.1, "xg_avg": 1.3, "xga_avg": 1.0}


# ── Placeholder squad generator ───────────────────────────────────────────────

_POSITION_TEMPLATE = [
    ("GK", 1), ("DEF", 4), ("MID", 4), ("FWD", 2),
]

def _generate_placeholder_squad(team: str, elo: int) -> list[dict]:
    """
    Create a dummy 11-player squad whose ratings are proportional to the team's
    Elo. Used for lesser-known teams where real player data isn't available.
    """
    # Map Elo (1550–2150) → base_rating (6.2–7.8)
    base = 6.2 + (elo - 1550) / (2150 - 1550) * 1.6
    base = round(max(6.2, min(7.8, base)), 1)

    import random
    rng = random.Random(hash(team) % 2**32)

    players = []
    i = 0
    for pos, count in _POSITION_TEMPLATE:
        for _ in range(count):
            players.append({
                "name":            f"{team} Jogador {i+1}",
                "position":        pos,
                "club":            "—",
                "league":          "Other",
                "rating_national": round(base + rng.uniform(-0.3, 0.3), 1),
                "rating_club":     round(base + rng.uniform(-0.4, 0.2), 1),
                "n_national_games": rng.randint(8, 25),
                "market_value_m":  rng.randint(1, 15),
                "wc_games":        0,
            })
            i += 1
    return players
