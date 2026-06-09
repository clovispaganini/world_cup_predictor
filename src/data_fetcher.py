"""
Data fetching utilities: FBref, Transfermarkt, ELO ratings, Kaggle CSV.
All external requests are cached locally for CACHE_TTL_HOURS hours.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    DEMO_MODE,
    ELO_DATA_URL,
    ELO_CSV_PATH,
    KAGGLE_RESULTS_PATH,
    SCRAPING_DELAY_SECONDS,
    SCRAPING_HEADERS,
)

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    p = Path(CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    safe = key.replace("/", "_").replace(":", "_")
    return p / f"{safe}.json"


def _cache_load(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["_cached_at"])
        if datetime.now() - cached_at < timedelta(hours=CACHE_TTL_HOURS):
            return data["payload"]
    except Exception:
        pass
    return None


def _cache_save(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.write_text(
        json.dumps({"_cached_at": datetime.now().isoformat(), "payload": payload},
                   default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── ELO ratings ───────────────────────────────────────────────────────────────

_FALLBACK_ELO: dict[str, int] = {
    "Argentina": 2142, "Brazil": 2091, "France": 2063, "Spain": 2020,
    "England": 1983, "Germany": 1980, "Netherlands": 1975, "Portugal": 1970,
    "Belgium": 1935, "Croatia": 1935, "Uruguay": 1920, "Italy": 1905,
    "Colombia": 1890, "Morocco": 1878, "Japan": 1865, "USA": 1845,
    "Senegal": 1840, "Switzerland": 1835, "Denmark": 1830, "Mexico": 1825,
    "Poland": 1810, "Ecuador": 1800, "Australia": 1795, "South Korea": 1790,
    "Serbia": 1785, "Chile": 1780, "Peru": 1770, "Canada": 1760,
    "Nigeria": 1755, "Cameroon": 1750, "Ghana": 1745, "Saudi Arabia": 1740,
    "Iran": 1735, "Tunisia": 1730, "Qatar": 1720, "Costa Rica": 1715,
    "Bolivia": 1700, "Paraguay": 1695, "Panama": 1685, "Wales": 1680,
    "Algeria": 1675, "Egypt": 1670, "Ivory Coast": 1665, "Mali": 1660,
}


def load_elo_ratings(force_download: bool = False) -> dict[str, int]:
    """Return {team_name: elo_rating} dict. Downloads from eloratings.net or uses cache."""
    cache_key = "elo_ratings"
    if not force_download:
        cached = _cache_load(cache_key)
        if cached:
            return cached

    if DEMO_MODE:
        _cache_save(cache_key, _FALLBACK_ELO)
        return _FALLBACK_ELO

    csv_path = Path(ELO_CSV_PATH)
    if csv_path.exists() and not force_download:
        try:
            df = pd.read_csv(csv_path, sep="\t", header=None, names=["rank","team","elo","delta"])
            ratings = dict(zip(df["team"].str.strip(), df["elo"].astype(int)))
            _cache_save(cache_key, ratings)
            return ratings
        except Exception:
            pass

    try:
        resp = requests.get(ELO_DATA_URL, headers=SCRAPING_HEADERS, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        ratings: dict[str, int] = {}
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                team = parts[1].strip()
                try:
                    ratings[team] = int(parts[2])
                except ValueError:
                    pass
        if ratings:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ratings.items(), columns=["team","elo"]).to_csv(csv_path, index=False)
            _cache_save(cache_key, ratings)
            return ratings
    except Exception as exc:
        print(f"[ELO] Download failed ({exc}), using fallback values.")

    _cache_save(cache_key, _FALLBACK_ELO)
    return _FALLBACK_ELO


# ── Historical match data (Kaggle) ────────────────────────────────────────────

def load_historical_matches() -> pd.DataFrame:
    """Load results.csv (Kaggle international football results dataset)."""
    path = Path(KAGGLE_RESULTS_PATH)
    if not path.exists():
        return _empty_results_df()

    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _empty_results_df() -> pd.DataFrame:
    cols = ["date","home_team","away_team","home_score","away_score","tournament","country","city","neutral"]
    return pd.DataFrame(columns=cols)


def get_recent_team_form(team: str, n_games: int = 5) -> dict:
    """Return last-n-game averages for a team from the historical dataset."""
    cache_key = f"form_{team}_{n_games}"
    cached = _cache_load(cache_key)
    if cached:
        return cached

    df = load_historical_matches()
    if df.empty:
        return _demo_form(team)

    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_df = df[mask].sort_values("date", ascending=False).head(n_games)

    if team_df.empty:
        return _demo_form(team)

    goals_for, goals_against = [], []
    for _, row in team_df.iterrows():
        if row["home_team"] == team:
            goals_for.append(row["home_score"])
            goals_against.append(row["away_score"])
        else:
            goals_for.append(row["away_score"])
            goals_against.append(row["home_score"])

    result = {
        "last5_scored":    round(sum(goals_for) / len(goals_for), 2),
        "last5_conceded":  round(sum(goals_against) / len(goals_against), 2),
        "xg_avg":          round(sum(goals_for) / len(goals_for) * 0.92, 2),
        "xga_avg":         round(sum(goals_against) / len(goals_against) * 0.92, 2),
    }
    _cache_save(cache_key, result)
    return result


def _demo_form(team: str) -> dict:
    try:
        demo = _load_demo_squads()
        form = demo.get("recent_form", {}).get(team)
        if form:
            return form
    except Exception:
        pass
    # Elo-aware fallback: stronger teams score more and concede less.
    # This prevents all matches from defaulting to 1-1.
    try:
        from src.schedule_loader import get_squad_elo   # safe — schedule_loader doesn't import data_fetcher
        elo = get_squad_elo(team)
    except Exception:
        elo = int(_FALLBACK_ELO.get(team, 1750))

    elo_diff = elo - 1750
    # xg_avg: Elo 2100 → ~1.73, Elo 1750 → 1.30, Elo 1580 → ~1.10
    xg  = round(max(0.55, min(2.30, 1.30 + elo_diff * 0.0012)), 2)
    # xga_avg: strong teams concede fewer expected goals
    xga = round(max(0.55, min(2.30, 1.30 - elo_diff * 0.0008)), 2)
    last5_scored   = round(xg  / 0.92, 2)
    last5_conceded = round(xga / 0.92, 2)
    return {
        "last5_scored":   last5_scored,
        "last5_conceded": last5_conceded,
        "xg_avg":         xg,
        "xga_avg":        xga,
    }


# ── Head-to-head record ───────────────────────────────────────────────────────

def get_head_to_head(team_a: str, team_b: str, last_n: int = 10) -> dict:
    """Return h2h record for the last_n encounters between two teams."""
    cache_key = f"h2h_{team_a}_{team_b}_{last_n}"
    cached = _cache_load(cache_key)
    if cached:
        return cached

    df = load_historical_matches()
    if df.empty:
        result = {"wins_a": 2, "draws": 3, "wins_b": 2, "goals_diff_a": 1}
        _cache_save(cache_key, result)
        return result

    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(last_n)

    wins_a = draws = wins_b = 0
    goals_a = goals_b = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            ga, gb = int(row["home_score"]), int(row["away_score"])
        else:
            ga, gb = int(row["away_score"]), int(row["home_score"])
        goals_a += ga
        goals_b += gb
        if ga > gb:
            wins_a += 1
        elif ga == gb:
            draws += 1
        else:
            wins_b += 1

    result = {
        "wins_a":       wins_a,
        "draws":        draws,
        "wins_b":       wins_b,
        "goals_diff_a": goals_a - goals_b,
    }
    _cache_save(cache_key, result)
    return result


# ── FBref player stats ────────────────────────────────────────────────────────

def get_player_stats_fbref(player_name: str, team: str) -> dict | None:
    """
    Attempt to fetch player stats from FBref.
    Returns a dict with 'rating_national', 'rating_club', 'n_national_games', or None on failure.
    Falls back to demo data in DEMO_MODE.
    """
    if DEMO_MODE:
        return _demo_player_stats(player_name, team)

    cache_key = f"fbref_{player_name}_{team}"
    cached = _cache_load(cache_key)
    if cached:
        return cached

    try:
        search_url = (
            f"https://fbref.com/en/search/search.fcgi?search={requests.utils.quote(player_name)}"
        )
        time.sleep(SCRAPING_DELAY_SECONDS)
        resp = requests.get(search_url, headers=SCRAPING_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        player_link = soup.select_one("div.search-item-url a")
        if not player_link:
            return None

        player_url = "https://fbref.com" + player_link["href"]
        time.sleep(SCRAPING_DELAY_SECONDS)
        resp2 = requests.get(player_url, headers=SCRAPING_HEADERS, timeout=15)
        resp2.raise_for_status()

        tables = pd.read_html(resp2.text)
        if not tables:
            return None

        stats = {"player_name": player_name, "team": team}
        for tbl in tables:
            if "SCA" in tbl.columns or "xG" in str(tbl.columns):
                stats["raw_table"] = tbl.to_dict()
                break

        _cache_save(cache_key, stats)
        return stats

    except Exception as exc:
        print(f"[FBref] Failed for {player_name}: {exc}")
        return None


def _demo_player_stats(player_name: str, team: str) -> dict | None:
    """Look up a player in the bundled demo dataset."""
    demo = _load_demo_squads()
    for squad in demo.get("squads", {}).values():
        for p in squad.get("players", []):
            if p["name"].lower() == player_name.lower():
                return {
                    "player_name":       p["name"],
                    "team":              team,
                    "rating_national":   p["rating_national"],
                    "rating_club":       p["rating_club"],
                    "n_national_games":  p["n_national_games"],
                    "market_value_m":    p["market_value_m"],
                    "wc_games":          p["wc_games"],
                    "club":              p["club"],
                    "league":            p["league"],
                }
    return None


# ── Transfermarkt market values ───────────────────────────────────────────────

def get_market_value(player_name: str, club: str) -> float | None:
    """Return market value in millions EUR. Falls back to positional median if unavailable."""
    if DEMO_MODE:
        demo = _load_demo_squads()
        for squad in demo.get("squads", {}).values():
            for p in squad.get("players", []):
                if p["name"].lower() == player_name.lower():
                    return p.get("market_value_m")
        return None

    cache_key = f"tm_{player_name}"
    cached = _cache_load(cache_key)
    if cached is not None:
        return cached

    try:
        query = requests.utils.quote(f"{player_name}")
        search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={query}"
        time.sleep(SCRAPING_DELAY_SECONDS)
        resp = requests.get(search_url, headers=SCRAPING_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        value_tag = soup.select_one("td.rechts.hauptlink a")
        if not value_tag:
            return None

        text = value_tag.get_text(strip=True).replace("€", "").strip()
        multiplier = 1
        if "m" in text.lower():
            multiplier = 1
            text = text.lower().replace("m", "")
        elif "k" in text.lower():
            multiplier = 0.001
            text = text.lower().replace("k", "")
        value = float(text.replace(",", ".")) * multiplier
        _cache_save(cache_key, value)
        return value

    except Exception as exc:
        print(f"[Transfermarkt] Failed for {player_name}: {exc}")
        return None


# ── Demo squad loader ─────────────────────────────────────────────────────────

_DEMO_CACHE: dict | None = None


def _load_demo_squads() -> dict:
    global _DEMO_CACHE
    if _DEMO_CACHE is not None:
        return _DEMO_CACHE
    path = Path(__file__).resolve().parent.parent / "data" / "demo_squads.json"
    if path.exists():
        _DEMO_CACHE = json.loads(path.read_text(encoding="utf-8"))
    else:
        _DEMO_CACHE = {"squads": {}, "recent_form": {}}
    return _DEMO_CACHE


def get_demo_squad(team: str) -> list[dict]:
    """Return the demo squad list for a given national team."""
    demo = _load_demo_squads()
    squad_data = demo.get("squads", {}).get(team, {})
    return squad_data.get("players", [])


def get_available_demo_teams() -> list[str]:
    """Return the list of teams that have demo squad data."""
    demo = _load_demo_squads()
    return list(demo.get("squads", {}).keys())
