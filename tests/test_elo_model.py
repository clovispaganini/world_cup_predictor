"""Unit tests for src/elo_model.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.elo_model import (
    expected_score,
    get_match_probabilities,
    goal_diff_multiplier,
    k_factor,
    update_elo,
    win_draw_loss,
)


class TestExpectedScore:
    def test_equal_teams(self):
        assert expected_score(1800, 1800) == pytest.approx(0.5)

    def test_stronger_team(self):
        assert expected_score(2000, 1800) > 0.5

    def test_weaker_team(self):
        assert expected_score(1800, 2000) < 0.5

    def test_home_advantage_shifts_probability(self):
        base  = expected_score(1800, 1800, home_advantage=0)
        home  = expected_score(1800, 1800, home_advantage=100)
        assert home > base


class TestWinDrawLoss:
    def test_probabilities_sum_to_one(self):
        for ea, eb in [(1800, 1800), (2100, 1800), (1600, 2000)]:
            r = win_draw_loss(ea, eb)
            assert sum(r.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_non_negative(self):
        r = win_draw_loss(2000, 1600)
        assert all(v >= 0 for v in r.values())

    def test_better_team_higher_win_prob(self):
        r = win_draw_loss(2100, 1800)
        assert r["win_a"] > r["win_b"]

    def test_symmetric_reversal(self):
        r1 = win_draw_loss(2000, 1800)
        r2 = win_draw_loss(1800, 2000)
        assert r1["win_a"] == pytest.approx(r2["win_b"], abs=0.005)
        assert r1["win_b"] == pytest.approx(r2["win_a"], abs=0.005)
        assert r1["draw"]  == pytest.approx(r2["draw"],  abs=0.005)

    def test_equal_teams_roughly_symmetric(self):
        r = win_draw_loss(1800, 1800)
        assert r["win_a"] == pytest.approx(r["win_b"], abs=0.02)


class TestKFactor:
    def test_world_cup(self):
        from config import K_WORLD_CUP
        assert k_factor("FIFA World Cup") == K_WORLD_CUP

    def test_qualifier(self):
        from config import K_QUALIFIER
        assert k_factor("World Cup qualification") == K_QUALIFIER

    def test_friendly(self):
        from config import K_FRIENDLY
        assert k_factor("Friendly match") == K_FRIENDLY


class TestGoalDiffMultiplier:
    def test_1_goal(self):
        assert goal_diff_multiplier(1) == pytest.approx(1.0)

    def test_2_goals(self):
        assert goal_diff_multiplier(2) == pytest.approx(1.5)

    def test_large_diff(self):
        from config import GOAL_DIFF_MULTIPLIER_MAX
        assert goal_diff_multiplier(7) == pytest.approx(GOAL_DIFF_MULTIPLIER_MAX)


class TestUpdateElo:
    def test_winner_gains_elo(self):
        ea0, eb0 = 1800, 1800
        ea1, eb1 = update_elo(ea0, eb0, score_a=2, score_b=0, tournament="FIFA World Cup")
        assert ea1 > ea0
        assert eb1 < eb0

    def test_elo_transfer_conserved(self):
        ea0, eb0 = 1850, 1750
        ea1, eb1 = update_elo(ea0, eb0, 1, 0)
        assert ea1 + eb1 == pytest.approx(ea0 + eb0, abs=1e-6)

    def test_draw_upsets_weaker_team(self):
        """A draw for the weaker team is an over-performance → gains Elo."""
        ea0, eb0 = 2000, 1700
        ea1, eb1 = update_elo(ea0, eb0, 1, 1, "friendly")
        assert eb1 > eb0


class TestGetMatchProbabilities:
    def test_returns_three_keys(self):
        r = get_match_probabilities("Brazil", "Argentina")
        assert {"win_a", "draw", "win_b"}.issubset(r.keys())

    def test_sum_to_one(self):
        r = get_match_probabilities("France", "England")
        assert r["win_a"] + r["draw"] + r["win_b"] == pytest.approx(1.0, abs=1e-4)

    def test_elo_keys_present(self):
        r = get_match_probabilities("Spain", "Germany")
        assert "elo_a" in r and "elo_b" in r

    def test_override_elo(self):
        elo_override = {"TeamX": 2200, "TeamY": 1700}
        r = get_match_probabilities("TeamX", "TeamY", elo_override=elo_override)
        assert r["win_a"] > r["win_b"]
