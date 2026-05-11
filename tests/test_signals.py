import numpy as np

from ibkr_trading_bot.core.services.signal_policy import (
	apply_entry_exit_thresholds,
	apply_live_hysteresis,
	build_live_proposal,
	extract_directional_probabilities,
	evaluate_live_policy,
	normalize_signal_array,
	pick_ternary_direction_from_raw_proba,
	ternary_proba_to_signal,
)


def test_ternary_proba_to_signal_prefers_stronger_side_on_conflict():
	out = ternary_proba_to_signal([0.70, 0.20], [0.80, 0.90], 0.60, 0.60)
	assert out.tolist() == [1, 1]


def test_extract_directional_probabilities_respects_reordered_classes():
	proba = np.array([[0.80, 0.10, 0.10]])
	classes = np.array([2, 1, 0])
	label_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}

	prob_short, prob_hold, prob_long = extract_directional_probabilities(proba, classes, label_map=label_map)

	assert prob_short.tolist() == [0.10]
	assert prob_hold.tolist() == [0.10]
	assert prob_long.tolist() == [0.80]


def test_pick_ternary_direction_from_raw_proba_matches_reordered_classes():
	raw_proba = np.array([0.80, 0.10, 0.10])
	classes = np.array([2, 1, 0])
	label_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}

	direction, confidence = pick_ternary_direction_from_raw_proba(
		raw_proba,
		classes,
		short_threshold=0.50,
		long_threshold=0.50,
		label_map=label_map,
	)

	assert direction == "LONG"
	assert confidence == 0.80


def test_apply_entry_exit_thresholds_flats_low_confidence_predictions():
	raw = np.array([1, -1, 1, 0])
	conf = np.array([0.8, 0.4, 0.55, 0.9])
	out = apply_entry_exit_thresholds(raw, conf, entry_threshold=0.5, exit_threshold=0.6)
	assert out.tolist() == [1, 1, 1, 1]


def test_apply_entry_exit_thresholds_supports_legacy_flat_exit_policy():
	raw = np.array([1, -1, 1, 0])
	conf = np.array([0.8, 0.4, 0.55, 0.9])
	out = apply_entry_exit_thresholds(raw, conf, entry_threshold=0.5, exit_threshold=0.6, exit_policy="legacy_flat_exit")
	assert out.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_build_live_proposal_respects_ma_alignment():
	assert build_live_proposal("LONG", "LONG", True) == "LONG"
	assert build_live_proposal("SHORT", "LONG", True) is None
	assert build_live_proposal("SHORT", "LONG", False) == "LONG"


def test_apply_live_hysteresis_requires_confirmation_for_current_direction():
	assert apply_live_hysteresis("LONG", 0.70, 0, 0.60, 0.50) == "LONG"
	assert apply_live_hysteresis("SHORT", 0.70, 1, 0.60, 0.50) == "SHORT"
	assert apply_live_hysteresis("LONG", 0.40, 1, 0.60, 0.50) == "LONG"
	assert apply_live_hysteresis("LONG", 0.55, 1, 0.60, 0.50) == "LONG"


def test_evaluate_live_policy_returns_specific_exit_reasons():
	decision = evaluate_live_policy("LONG", "LONG", True, 0.40, 1, 0.60, 0.50)
	assert decision.final_signal == "LONG"
	assert decision.reason == "hold_same_signal"
	assert decision.close_reason is None

	decision = evaluate_live_policy("LONG", "SHORT", False, 0.90, 1, 0.60, 0.50)
	assert decision.final_signal == "SHORT"
	assert decision.reason == "flip_confirmed"
	assert decision.close_reason == "opposite_signal"


def test_evaluate_live_policy_legacy_mode_preserves_flat_exit_behavior():
	decision = evaluate_live_policy("LONG", "LONG", True, 0.40, 1, 0.60, 0.50, exit_policy="legacy_flat_exit")
	assert decision.final_signal is None
	assert decision.reason == "exit_low_confidence"
	assert decision.close_reason == "low_confidence"


def test_normalize_signal_array_accepts_text_and_numbers():
	out = normalize_signal_array(["LONG", "short", 0, 2, -3])
	assert out.tolist() == [1.0, -1.0, 0.0, 1.0, -1.0]
