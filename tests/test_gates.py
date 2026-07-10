from __future__ import annotations

from trialgate import evaluate_gates


def test_all_gates_pass_produces_qualified_report() -> None:
    report = evaluate_gates(observations=1848, pbo=0.02, deflated_sharpe=0.97)

    assert report.passed is True
    assert [check.passed for check in report.checks] == [True, True, True]
    assert report.summary().endswith("QUALIFIED")


def test_each_gate_fails_independently() -> None:
    thin_data = evaluate_gates(observations=400, pbo=0.02, deflated_sharpe=0.97)
    overfit = evaluate_gates(observations=1848, pbo=0.55, deflated_sharpe=0.97)
    lucky = evaluate_gates(observations=1848, pbo=0.02, deflated_sharpe=0.60)

    assert thin_data.passed is False
    assert [check.passed for check in thin_data.checks] == [False, True, True]
    assert overfit.passed is False
    assert [check.passed for check in overfit.checks] == [True, False, True]
    assert lucky.passed is False
    assert [check.passed for check in lucky.checks] == [True, True, False]
    assert overfit.summary().endswith("NOT QUALIFIED")


def test_boundary_values_pass_exactly_at_thresholds() -> None:
    report = evaluate_gates(observations=1000, pbo=0.05, deflated_sharpe=0.95)

    assert report.passed is True


def test_custom_thresholds_are_respected() -> None:
    report = evaluate_gates(
        observations=500,
        pbo=0.08,
        deflated_sharpe=0.90,
        min_observations=250,
        max_pbo=0.10,
        min_dsr=0.85,
    )

    assert report.passed is True


def test_summary_lists_every_gate_with_verdict() -> None:
    report = evaluate_gates(observations=1848, pbo=0.55, deflated_sharpe=0.97)
    text = report.summary()

    assert "data floor (gate 2)" in text
    assert "PBO (gate 3)" in text
    assert "DSR (gate 4)" in text
    assert "FAIL" in text and "PASS" in text
