import json
import os
import statistics
import hashlib
from typing import Any, Dict, List


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def generate_report(run_file: str) -> int:
    rows = load_jsonl(run_file)
    if not rows:
        print("No rows found.")
        return 1
    report = compute_report(rows)

    out_path = os.path.splitext(run_file)[0] + "_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {out_path}")
    return 0


def compute_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [r.get("duration_sec", 0.0) for r in rows]
    total_tokens = [r.get("total_tokens", 0) for r in rows]
    prompt_tokens = [r.get("prompt_tokens", 0) for r in rows]
    response_tokens = [r.get("response_tokens", 0) for r in rows]
    prompt_chars = [r.get("prompt_chars", 0) for r in rows]
    response_chars = [r.get("response_chars", 0) for r in rows]
    scores = [r.get("verification", {}).get("score", 0.0) for r in rows]
    losses = [r.get("loss", None) for r in rows]
    visible_losses = [r.get("visible_loss", None) for r in rows]
    hidden_losses = [r.get("hidden_loss", None) for r in rows]
    patch_ops = [r.get("patch_ops", 0) for r in rows]
    patch_applied = [r.get("patch_applied_ops", 0) for r in rows]
    patch_chars = [r.get("patch_chars", 0) for r in rows]
    visible_reasons = [
        len(r.get("verification", {}).get("reasons", []) or []) for r in rows
    ]
    hidden_reasons = [
        len(r.get("verification", {}).get("hidden_reasons", []) or []) for r in rows
    ]

    thrash_steps = sum(1 for i in range(1, len(scores)) if scores[i] <= scores[i - 1] + 1e-12)
    hidden_fail_iters = sum(
        1 for r in rows if r.get("verification", {}).get("hidden_reasons")
    )
    visible_pass_hidden_fail = sum(
        1
        for r in rows
        if r.get("verification", {}).get("score", 0.0) == 1.0
        and r.get("verification", {}).get("hidden_reasons")
    )
    raw_hashes = [_sha12(r.get("raw_text", "")) for r in rows]
    repeats = len(raw_hashes) - len(set(raw_hashes))

    duration_ms = [d * 1000.0 for d in durations]
    total_duration = sum(durations) if durations else 0.0
    total_tokens_sum = sum(total_tokens)
    tokens_per_sec = (total_tokens_sum / total_duration) if total_duration > 0 else 0.0
    loss_vals = [l for l in losses if isinstance(l, (int, float))]
    visible_vals = [l for l in visible_losses if isinstance(l, (int, float))]
    hidden_vals = [l for l in hidden_losses if isinstance(l, (int, float))]
    loss_initial = loss_vals[0] if loss_vals else None
    loss_best = min(loss_vals) if loss_vals else None
    loss_end = loss_vals[-1] if loss_vals else None
    chl = None
    if loss_initial and loss_initial > 0:
        target = loss_initial / 2.0
        for idx, lv in enumerate(loss_vals, start=1):
            if lv <= target:
                chl = idx
                break
    thrash_index = thrash_steps / max(1, len(scores) - 1)

    return {
        "iterations": len(rows),
        "score_best": max(scores) if scores else None,
        "score_end": scores[-1] if scores else None,
        "loss_initial": loss_initial,
        "loss_best": loss_best,
        "loss_end": loss_end,
        "avg_visible_loss": sum(visible_vals) / max(1, len(visible_vals)),
        "avg_hidden_loss": sum(hidden_vals) / max(1, len(hidden_vals)),
        "avg_duration_sec": sum(durations) / max(1, len(durations)),
        "total_duration_sec": total_duration,
        "p50_duration_ms": statistics.median(duration_ms) if duration_ms else None,
        "p95_duration_ms": statistics.quantiles(duration_ms, n=20)[-1] if len(duration_ms) >= 20 else None,
        "total_tokens": total_tokens_sum,
        "avg_total_tokens": sum(total_tokens) / max(1, len(total_tokens)),
        "avg_prompt_tokens": sum(prompt_tokens) / max(1, len(prompt_tokens)),
        "avg_response_tokens": sum(response_tokens) / max(1, len(response_tokens)),
        "tokens_per_sec": tokens_per_sec,
        "avg_prompt_chars": sum(prompt_chars) / max(1, len(prompt_chars)),
        "avg_response_chars": sum(response_chars) / max(1, len(response_chars)),
        "thrash_steps": thrash_steps,
        "thrash_index": thrash_index,
        "hidden_fail_iters": hidden_fail_iters,
        "false_completion_rate": visible_pass_hidden_fail / max(1, len(rows)),
        "verifier_pressure_visible": sum(visible_reasons) / max(1, len(visible_reasons)),
        "verifier_pressure_hidden": sum(hidden_reasons) / max(1, len(hidden_reasons)),
        "patch_ops_avg": sum(patch_ops) / max(1, len(patch_ops)),
        "patch_applied_avg": sum(patch_applied) / max(1, len(patch_applied)),
        "patch_chars_avg": sum(patch_chars) / max(1, len(patch_chars)),
        "convergence_half_life": chl,
        "repeat_count": repeats,
    }
