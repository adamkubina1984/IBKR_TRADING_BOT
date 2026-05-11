from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveReleaseGateInputs:
    automated_tests_passed: bool
    paper_soak_completed: bool
    paper_soak_days: int = 0
    audit_trail_complete: bool = False
    unresolved_reconciliation_failures: int = 0
    unresolved_safe_stop_events: int = 0
    unresolved_emergency_stop_events: int = 0
    min_paper_soak_days: int = 3


@dataclass(frozen=True)
class LiveReleaseGateResult:
    allowed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def evaluate_live_release_gate(inputs: LiveReleaseGateInputs) -> LiveReleaseGateResult:
    blockers: list[str] = []
    warnings: list[str] = []

    if not inputs.automated_tests_passed:
        blockers.append("AUTOMATED_TESTS_FAILED")
    if not inputs.paper_soak_completed:
        blockers.append("PAPER_SOAK_NOT_COMPLETED")
    if int(inputs.paper_soak_days) < int(inputs.min_paper_soak_days):
        blockers.append("PAPER_SOAK_DURATION_TOO_SHORT")
    if not inputs.audit_trail_complete:
        blockers.append("AUDIT_TRAIL_INCOMPLETE")
    if int(inputs.unresolved_reconciliation_failures) > 0:
        blockers.append("UNRESOLVED_RECONCILIATION_FAILURES")
    if int(inputs.unresolved_safe_stop_events) > 0:
        blockers.append("UNRESOLVED_SAFE_STOP_EVENTS")
    if int(inputs.unresolved_emergency_stop_events) > 0:
        blockers.append("UNRESOLVED_EMERGENCY_STOP_EVENTS")

    if inputs.paper_soak_completed and int(inputs.paper_soak_days) == int(inputs.min_paper_soak_days):
        warnings.append("PAPER_SOAK_AT_MINIMUM_DURATION")

    return LiveReleaseGateResult(
        allowed=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )