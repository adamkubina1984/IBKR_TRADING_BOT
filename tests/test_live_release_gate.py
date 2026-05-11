from __future__ import annotations

from ibkr_trading_bot.core.services.live_release_gate import (
    LiveReleaseGateInputs,
    evaluate_live_release_gate,
)


def test_release_gate_blocks_when_required_rollout_conditions_are_missing():
    result = evaluate_live_release_gate(
        LiveReleaseGateInputs(
            automated_tests_passed=False,
            paper_soak_completed=False,
            paper_soak_days=1,
            audit_trail_complete=False,
            unresolved_reconciliation_failures=2,
            unresolved_safe_stop_events=1,
        )
    )

    assert result.allowed is False
    assert result.blockers == (
        "AUTOMATED_TESTS_FAILED",
        "PAPER_SOAK_NOT_COMPLETED",
        "PAPER_SOAK_DURATION_TOO_SHORT",
        "AUDIT_TRAIL_INCOMPLETE",
        "UNRESOLVED_RECONCILIATION_FAILURES",
        "UNRESOLVED_SAFE_STOP_EVENTS",
    )


def test_release_gate_allows_promotion_after_clean_paper_soak():
    result = evaluate_live_release_gate(
        LiveReleaseGateInputs(
            automated_tests_passed=True,
            paper_soak_completed=True,
            paper_soak_days=4,
            audit_trail_complete=True,
        )
    )

    assert result.allowed is True
    assert result.blockers == ()


def test_release_gate_warns_when_soak_is_only_at_minimum_duration():
    result = evaluate_live_release_gate(
        LiveReleaseGateInputs(
            automated_tests_passed=True,
            paper_soak_completed=True,
            paper_soak_days=3,
            audit_trail_complete=True,
        )
    )

    assert result.allowed is True
    assert result.warnings == ("PAPER_SOAK_AT_MINIMUM_DURATION",)