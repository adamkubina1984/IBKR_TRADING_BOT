from __future__ import annotations

from html import escape
from typing import Any


def _fmt_value(value: Any, digits: int = 4) -> str:
	if value is None:
		return ""
	if isinstance(value, float):
		return f"{value:.{digits}f}"
	return str(value)


def evaluation_report_semantics_lines(metrics: dict[str, Any] | None) -> list[str]:
	metrics = metrics if isinstance(metrics, dict) else {}
	lines = ["Net trading metriky jsou preferovane, pokud jsou k dispozici."]
	label_mode = str(metrics.get("label_mode") or "").strip().lower()
	classification_mode = str(metrics.get("classification_mode") or "").strip().lower()
	if classification_mode == "ternary" or label_mode in {"ternary_signed", "ternary_mapped"}:
		lines.append("Ternarni semantika: -1=SHORT, 0=FLAT, 1=LONG.")
	elif classification_mode == "binary" or label_mode in {"binary_01", "binary_signed"}:
		lines.append("Binarni labely jsou pred evaluaci explicitne normalizovane.")
	if label_mode == "ternary_mapped":
		lines.append("Mapped 0/1/2 je jen interni reprezentace short/flat/long pro sklearn flow.")
	if metrics.get("classification_available") is False:
		lines.append("Bez ground truth jsou reportovane jen trading metriky.")
	return lines


def format_evaluation_report_text(
	*,
	model_name: str,
	scope_label: str,
	metrics: dict[str, Any] | None,
	key_metrics: list[tuple[str, Any]],
) -> str:
	metrics = metrics if isinstance(metrics, dict) else {}
	lines = [
		f"Evaluation report: {model_name}",
		f"Scope: {scope_label}",
		"",
		*[f"Note: {line}" for line in evaluation_report_semantics_lines(metrics)],
		"",
		"Key metrics:",
	]
	for label, value in key_metrics:
		lines.append(f"- {label}: {_fmt_value(value)}")

	scalar_items: list[tuple[str, Any]] = []
	for key, value in metrics.items():
		if isinstance(value, (list, tuple, dict, set)):
			continue
		scalar_items.append((str(key), value))
	scalar_items.sort(key=lambda item: item[0])

	lines.extend(["", "All scalar metrics:"])
	for key, value in scalar_items:
		lines.append(f"- {key}: {_fmt_value(value)}")
	lines.append("")
	return "\n".join(lines)


def format_evaluation_report_html(
	*,
	model_name: str,
	scope_label: str,
	metrics: dict[str, Any] | None,
	key_metrics: list[tuple[str, Any]],
) -> str:
	metrics = metrics if isinstance(metrics, dict) else {}
	semantics = "".join(f"<li>{escape(line)}</li>" for line in evaluation_report_semantics_lines(metrics))
	key_items = "".join(
		f"<tr><td>{escape(str(label))}</td><td>{escape(_fmt_value(value))}</td></tr>"
		for label, value in key_metrics
	)
	scalar_items = []
	for key, value in metrics.items():
		if isinstance(value, (list, tuple, dict, set)):
			continue
		scalar_items.append((str(key), value))
	scalar_items.sort(key=lambda item: item[0])
	metric_rows = "".join(
		f"<tr><td>{escape(key)}</td><td>{escape(_fmt_value(value))}</td></tr>"
		for key, value in scalar_items
	)
	return (
		"<!doctype html><html><head><meta charset=\"utf-8\">"
		"<title>Evaluation report</title>"
		"<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}"
		"table{border-collapse:collapse;width:100%;margin-top:12px;}"
		"th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}"
		"th{background:#f6f8fa;}"
		"ul{margin-top:8px;}</style></head><body>"
		f"<h1>{escape(model_name)}</h1>"
		f"<p><strong>Scope:</strong> {escape(scope_label)}</p>"
		f"<h2>Semantics</h2><ul>{semantics}</ul>"
		f"<h2>Key metrics</h2><table><tbody>{key_items}</tbody></table>"
		f"<h2>All scalar metrics</h2><table><tbody>{metric_rows}</tbody></table>"
		"</body></html>"
	)
