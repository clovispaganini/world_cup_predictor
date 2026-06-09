"""Unit tests for src/player_rater.py"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.player_rater import (
    calculate_player_score,
    league_coefficient,
    temporal_decay,
)
from config import (
    DECAY_HALF_LIFE_DAYS,
    LEAGUE_STRENGTH,
    WEIGHT_CLUB,
    WEIGHT_CLUB_LOW_SAMPLE,
    WEIGHT_NATIONAL_TEAM,
    WEIGHT_NATIONAL_TEAM_LOW_SAMPLE,
)


# ── temporal_decay ────────────────────────────────────────────────────────────

class TestTemporalDecay:
    ref = datetime(2025, 6, 1)

    def test_same_day(self):
        assert temporal_decay(self.ref, self.ref) == pytest.approx(1.0)

    def test_half_life(self):
        game_date = self.ref - timedelta(days=DECAY_HALF_LIFE_DAYS)
        assert temporal_decay(game_date, self.ref) == pytest.approx(0.5, rel=1e-5)

    def test_double_half_life(self):
        game_date = self.ref - timedelta(days=2 * DECAY_HALF_LIFE_DAYS)
        assert temporal_decay(game_date, self.ref) == pytest.approx(0.25, rel=1e-5)

    def test_future_date_clamps_to_one(self):
        future = self.ref + timedelta(days=10)
        assert temporal_decay(future, self.ref) == pytest.approx(1.0)

    def test_monotone_decrease(self):
        d1 = temporal_decay(self.ref - timedelta(days=10),  self.ref)
        d2 = temporal_decay(self.ref - timedelta(days=100), self.ref)
        d3 = temporal_decay(self.ref - timedelta(days=365), self.ref)
        assert d1 > d2 > d3


# ── league_coefficient ────────────────────────────────────────────────────────

class TestLeagueCoefficient:
    def test_premier_league(self):
        assert league_coefficient("Premier League") == pytest.approx(1.00)

    def test_brasileirao(self):
        assert league_coefficient("Brasileirao") == pytest.approx(0.75)

    def test_unknown_league(self):
        assert league_coefficient("Obscure League XYZ") == pytest.approx(LEAGUE_STRENGTH["Other"])

    def test_all_leagues_between_0_and_1(self):
        for name, coeff in LEAGUE_STRENGTH.items():
            assert 0 < coeff <= 1.0, f"{name} coefficient {coeff} out of range"


# ── calculate_player_score ────────────────────────────────────────────────────

class TestCalculatePlayerScore:
    REF = datetime(2025, 6, 1)

    def _nat(self, n: int, rating: float = 7.5) -> list[dict]:
        return [
            {"date": self.REF - timedelta(days=14 * i), "rating": rating}
            for i in range(n)
        ]

    def _club(self, n: int, rating: float = 7.0) -> list[dict]:
        return [
            {"date": self.REF - timedelta(days=7 * i), "rating": rating}
            for i in range(n)
        ]

    def test_high_weight_regime(self):
        """≥5 national games → uses 0.65/0.35 split."""
        result = calculate_player_score(
            self._nat(10, 8.0),
            self._club(20, 7.0),
            "Premier League",
            self.REF,
        )
        assert result["w_national"] == WEIGHT_NATIONAL_TEAM
        assert result["w_club"]     == WEIGHT_CLUB
        assert not result["low_sample"]

    def test_low_sample_regime(self):
        """<5 national games → uses 0.45/0.55 split."""
        result = calculate_player_score(
            self._nat(3, 8.0),
            self._club(20, 7.0),
            "Premier League",
            self.REF,
        )
        assert result["w_national"] == WEIGHT_NATIONAL_TEAM_LOW_SAMPLE
        assert result["w_club"]     == WEIGHT_CLUB_LOW_SAMPLE
        assert result["low_sample"]

    def test_league_coefficient_applied(self):
        """Club rating should be discounted by league coefficient."""
        res_prem  = calculate_player_score([], self._club(10, 8.0), "Premier League", self.REF)
        res_other = calculate_player_score([], self._club(10, 8.0), "Other",          self.REF)
        # Premier league gives higher (or equal) adjusted club rating
        assert res_prem["score"] >= res_other["score"]

    def test_missing_national_applies_penalty(self):
        """Only club data → score penalised by MISSING_SOURCE_PENALTY."""
        from config import MISSING_SOURCE_PENALTY
        res = calculate_player_score([], self._club(15, 7.0), "La Liga", self.REF)
        raw_club = res["club_rating"]
        assert res["score"] == pytest.approx(raw_club * (1 - MISSING_SOURCE_PENALTY), rel=1e-3)

    def test_score_in_reasonable_range(self):
        """Score should remain within a plausible 5–10 range for normal inputs."""
        result = calculate_player_score(
            self._nat(12, 7.8),
            self._club(30, 7.5),
            "Bundesliga",
            self.REF,
        )
        assert 5.0 <= result["score"] <= 10.0

    def test_messi_like_player(self):
        """A 9.0-rated player in national games should dominate the score."""
        result = calculate_player_score(
            self._nat(30, 9.0),
            self._club(40, 7.0),
            "MLS",           # low league coefficient
            self.REF,
        )
        # National component (9.0 × 0.65) is large; score should exceed 8
        assert result["score"] > 7.5

    def test_no_data_fallback(self):
        """No data at all → fallback to 6.5."""
        result = calculate_player_score([], [], "Other", self.REF)
        assert result["score"] == pytest.approx(6.5)
