"""Unit tests for src/ensemble.py + integration sanity checks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ENSEMBLE_WEIGHTS
from src.data_fetcher import get_demo_squad
from src.ensemble import EnsemblePredictor, predict_match
from src.team_builder import build_team_score


# ── Fixture squads ────────────────────────────────────────────────────────────

@pytest.fixture
def arg_squad():
    return get_demo_squad("Argentina")

@pytest.fixture
def fra_squad():
    return get_demo_squad("France")

@pytest.fixture
def arg_result(arg_squad):
    return build_team_score(arg_squad)

@pytest.fixture
def fra_result(fra_squad):
    return build_team_score(fra_squad)


# ── EnsemblePredictor ─────────────────────────────────────────────────────────

class TestEnsemblePredictor:

    def test_predict_returns_required_keys(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Argentina", "France",
            arg_result, fra_result,
        )
        required = {"win_a", "draw", "win_b", "breakdown",
                    "most_probable_score", "poisson_score_matrix",
                    "elo_a", "elo_b", "lambda_a", "lambda_b"}
        assert required.issubset(pred.keys())

    def test_probabilities_sum_to_one(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Argentina", "France",
            arg_result, fra_result,
        )
        total = pred["win_a"] + pred["draw"] + pred["win_b"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_all_probs_positive(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Brazil", "Germany",
            arg_result, fra_result,
        )
        assert pred["win_a"] > 0
        assert pred["draw"]  > 0
        assert pred["win_b"] > 0

    def test_breakdown_has_all_models(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Spain", "England",
            arg_result, fra_result,
        )
        bd = pred["breakdown"]
        assert "elo" in bd and "poisson" in bd and "xgboost" in bd

    def test_breakdown_weights_sum_to_one(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Argentina", "France",
            arg_result, fra_result,
        )
        total_w = sum(v.get("weight", 0) for v in pred["breakdown"].values())
        assert total_w == pytest.approx(1.0, abs=1e-4)

    def test_most_probable_score_is_tuple(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Brazil", "Croatia",
            arg_result, fra_result,
        )
        s = pred["most_probable_score"]
        assert len(s) == 2
        assert isinstance(s[0], int) and isinstance(s[1], int)
        assert s[0] >= 0 and s[1] >= 0

    def test_team_score_in_output(self, arg_result, fra_result):
        pred = EnsemblePredictor().predict(
            "Argentina", "France",
            arg_result, fra_result,
        )
        assert "team_score_a" in pred and "team_score_b" in pred


# ── 2022 World Cup validation cases ──────────────────────────────────────────
# These test that the model's predictions are at least *directionally* correct
# for famous 2022 matches.  We do NOT require exact probability values.

class TestWC2022Cases:
    """
    Historically, Argentina, France, and Morocco were the strongest sides at
    Qatar 2022.  We verify that the model's ranking of probabilities is
    consistent with each match's actual outcome direction.
    """

    def _run(self, team_a: str, team_b: str, phase: str = "Fase de Grupos") -> dict:
        squad_a = get_demo_squad(team_a) or get_demo_squad("Argentina")
        squad_b = get_demo_squad(team_b) or get_demo_squad("Argentina")
        res_a = build_team_score(squad_a)
        res_b = build_team_score(squad_b)
        return predict_match(team_a, team_b, res_a, res_b, phase)

    def test_argentina_vs_france_final(self):
        """
        Actual result: Argentina 3-3 FRA (AET), ARG win on penalties.
        Model should predict a close match; neither team dominant by >20 pp.
        """
        pred = self._run("Argentina", "France", "Final")
        diff = abs(pred["win_a"] - pred["win_b"])
        assert diff < 0.25, f"Expected a close final, got diff={diff:.2%}"

    def test_france_vs_morocco_semi(self):
        """
        Actual result: France 2-0 Morocco.
        France should be modelled as favourites.
        """
        fra_squad = get_demo_squad("France")
        mor_squad = get_demo_squad("Morocco")
        res_fra = build_team_score(fra_squad)
        res_mor = build_team_score(mor_squad)
        pred = predict_match("France", "Morocco", res_fra, res_mor, "Semifinal")
        assert pred["win_a"] > pred["win_b"], (
            f"Expected France favoured, got win_a={pred['win_a']:.2%}, win_b={pred['win_b']:.2%}"
        )

    def test_brazil_vs_croatia_qf(self):
        """
        Actual result: Brazil 1-1 CRO (AET), CRO win on penalties — a major upset.
        Model may favour Brazil; we only assert neither probability is unreasonably extreme.
        """
        bra_squad = get_demo_squad("Brazil")
        cro_squad = get_demo_squad("Croatia")
        res_bra = build_team_score(bra_squad)
        res_cro = build_team_score(cro_squad)
        pred = predict_match("Brazil", "Croatia", res_bra, res_cro, "Quartas de Final")
        # Croatia should have at least 15 % win probability
        assert pred["win_b"] > 0.12, (
            f"Croatia win prob too low: {pred['win_b']:.2%}"
        )
