import json
from typing import Any, Dict, Optional

from .types import VerificationResult


def build_patch_spec(
    parsed: Optional[Dict[str, Any]], vr: VerificationResult
) -> Dict[str, Any]:
    fixes = []
    if not parsed:
        fixes.append(
            {"path": "root", "must_include": ["valid JSON", "schema compliance"]}
        )
        return {"fixes": fixes}

    for reason in vr.reasons:
        r = reason.lower()
        if "hypotheses" in r:
            fixes.append(
                {"path": "hypotheses", "must_include": ["exactly 3", "numeric target"]}
            )
        elif "metrics" in r:
            fixes.append(
                {
                    "path": "metrics",
                    "must_include": ["telemetry", "token", "latency"],
                }
            )
        elif "micro_tasks" in r or "micro tasks" in r:
            fixes.append(
                {
                    "path": "micro_tasks",
                    "must_include": [
                        "ablation",
                        "sanity-check",
                        "latency profiling",
                        "token budget",
                        "regression guard",
                    ],
                }
            )
        elif "anti_gaming" in r or "hidden" in r:
            fixes.append(
                {
                    "path": "anti_gaming",
                    "must_include": ["withhold", "secret", "hidden checks"],
                }
            )
        elif "actions" in r:
            fixes.append(
                {
                    "path": "actions",
                    "must_include": [
                        "type",
                        "write_file .md",
                        "note log fields",
                        "note.text",
                    ],
                }
            )
        elif "completion_signal" in r:
            fixes.append(
                {
                    "path": "completion_signal",
                    "must_include": ["done boolean", "reason string"],
                }
            )

    for reason in vr.hidden_reasons or []:
        r = reason.lower()
        if "false completion" in r or "evidence mismatch" in r:
            fixes.append(
                {
                    "path": "metrics",
                    "must_include": [
                        "false completion",
                        "evidence mismatch",
                        "hidden_fail_iters",
                        "iterations",
                        "false_completion_rate",
                    ],
                }
            )

    if not fixes:
        fixes.append({"path": "root", "must_include": ["address verifier feedback"]})

    return {"fixes": fixes}


def format_patch_spec(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False)
