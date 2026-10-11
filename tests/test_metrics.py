import numpy as np
import pytest

from src.evaluation.metrics import cost_curve, decision_metrics, ranking_metrics, select_threshold, trivial_costs

Y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.3, 0.2, 0.1])


def test_cost_curve_covers_flag_none_and_flag_all():
    curve = cost_curve(Y, S, 5)
    assert curve.iloc[0][["tp", "fp"]].tolist() == [0, 0] and curve["threshold"].iloc[0] == np.inf
    assert curve.iloc[-1][["tp", "fp"]].tolist() == [3, 5] and curve["threshold"].iloc[-1] == -np.inf
    assert len(curve) == 1 + len(np.unique(S))


def test_selected_threshold_minimises_cost_against_brute_force():
    for r in (1, 2, 3, 5, 10):
        t = select_threshold(Y, S, r)
        best = min(decision_metrics(Y, S, c, r)["cost_per_client"] for c in [-np.inf, *S, np.inf])
        assert decision_metrics(Y, S, t, r)["cost_per_client"] == pytest.approx(best)


def test_higher_cost_ratio_never_raises_threshold():
    ts = [select_threshold(Y, S, r) for r in (1, 2, 3, 5, 10)]
    assert ts == sorted(ts, reverse=True)


def test_ties_prefer_fewer_false_alarms():
    # r=1：报警前两人成本 1（漏 1 个），报警前四人成本也是 1（漏 0、误报 1）……取误报少的
    y = np.array([1, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.5, 0.4])
    m = decision_metrics(y, s, select_threshold(y, s, 1), 1)
    assert (m["fp"], m["fn"]) == (0, 1)


def test_threshold_is_midpoint_and_uses_greater_equal():
    t = select_threshold(Y, S, 1)
    assert t == pytest.approx(0.75)
    assert decision_metrics(Y, S, 0.8, 1)["tp"] == 2


def test_decision_metrics_counts_and_cost():
    m = decision_metrics(Y, S, 0.5, 5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["cost_per_client"] == pytest.approx((5 * 1 + 1) / 8)
    assert np.isnan(decision_metrics(Y, S, np.inf, 5)["precision"])


def test_trivial_costs_and_constant_score_reference():
    t = trivial_costs(Y, 5)
    assert t["flag_none"] == pytest.approx(5 * 3 / 8) and t["flag_all"] == pytest.approx(5 / 8)
    m = ranking_metrics(Y, np.full(len(Y), Y.mean()))
    assert m["brier"] == pytest.approx(m["brier_ref"]) and m["brier_skill"] == pytest.approx(0)
    assert m["roc_auc"] == pytest.approx(0.5) and m["ap"] == pytest.approx(Y.mean())
