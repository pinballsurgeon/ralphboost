import json
import re
import hashlib
from typing import Any, Dict, Optional, Callable, Tuple, List

from .types import VerificationResult


def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        block = text
    else:
        i = text.find("{")
        j = text.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None
        block = text[i : j + 1]
    try:
        return json.loads(block)
    except Exception:
        return None


def verify_minimal_schema(parsed: Optional[Dict[str, Any]]) -> VerificationResult:
    if not parsed:
        return VerificationResult(
            complete=False,
            score=0.0,
            reasons=["Invalid JSON. Return ONLY JSON matching schema."],
            hidden_reasons=None,
        )

    required = [
        "iteration_summary",
        "hypotheses",
        "metrics",
        "micro_tasks",
        "anti_gaming",
        "actions",
        "completion_signal",
    ]
    missing = [k for k in required if k not in parsed]
    if missing:
        return VerificationResult(
            complete=False,
            score=0.0,
            reasons=[f"Missing required keys: {missing}"],
            hidden_reasons=None,
        )

    return VerificationResult(complete=True, score=1.0, reasons=[], hidden_reasons=None)


def _hidden_pool() -> List[Tuple[str, Callable[[str], bool]]]:
    return [
        ("fcr_metric", lambda blob: "false completion" in blob or "false_completion_rate" in blob),
        ("evidence_mismatch", lambda blob: "evidence mismatch" in blob),
        ("gaming_rate", lambda blob: "gaming rate" in blob),
        ("hidden_fail_metric", lambda blob: "hidden_fail_iters" in blob),
    ]


def _select_hidden_check(hidden_id: Optional[str], rotate_seed: Optional[str]) -> Tuple[str, Callable[[str], bool]]:
    pool = _hidden_pool()
    if hidden_id:
        for pid, fn in pool:
            if pid == hidden_id:
                return pid, fn
    if rotate_seed:
        h = hashlib.sha256(rotate_seed.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % len(pool)
        return pool[idx]
    return pool[0]


def make_strict_verifier(
    hidden_id: Optional[str] = None, rotate_seed: Optional[str] = None
) -> Callable[[Optional[Dict[str, Any]]], VerificationResult]:
    hidden_id_sel, hidden_fn = _select_hidden_check(hidden_id, rotate_seed)

    def _verify(parsed: Optional[Dict[str, Any]]) -> VerificationResult:
        if not parsed:
            return VerificationResult(
                complete=False,
                score=0.0,
                reasons=["Invalid JSON. Return ONLY JSON matching schema."],
                hidden_reasons=None,
            )

        required = [
            "iteration_summary",
            "hypotheses",
            "metrics",
            "micro_tasks",
            "anti_gaming",
            "actions",
            "completion_signal",
        ]
        missing = [k for k in required if k not in parsed]
        if missing:
            return VerificationResult(
                complete=False,
                score=0.0,
                reasons=[f"Missing required keys: {missing}"],
                hidden_reasons=None,
            )

        reasons = []
        checks_ok = 0
        total_checks = 6

        hyps = parsed.get("hypotheses", [])
        ok_h = (
            isinstance(hyps, list)
            and len(hyps) == 3
            and all(isinstance(x, str) and len(x.strip()) >= 25 for x in hyps)
            and all(re.search(r"(\d+%|\b>=\b|\b<=\b|\b\d+\b)", h) for h in hyps)
        )
        if ok_h:
            checks_ok += 1
        else:
            reasons.append("hypotheses must be exactly 3 substantial strings with numeric targets.")

        mets = parsed.get("metrics", [])
        ok_m = (
            isinstance(mets, list)
            and len(mets) == 3
            and all(isinstance(m, dict) for m in mets)
            and all(m.get("name") and m.get("definition") and m.get("formula_or_calc") for m in mets)
        )
        log_terms = [
            "iterations",
            "duration",
            "time",
            "latency",
            "hidden",
            "fail",
            "score",
            "token",
            "budget",
        ]
        ok_m2 = ok_m and sum(
            any(t in str(m["formula_or_calc"]).lower() for t in log_terms) for m in mets
        ) >= 2
        if ok_m2:
            checks_ok += 1
        else:
            reasons.append("metrics must be 3 items and at least 2 formulas reference telemetry fields.")

        tasks = parsed.get("micro_tasks", [])
        ok_t = isinstance(tasks, list) and len(tasks) >= 10 and all(
            isinstance(t, str) and len(t.strip()) >= 12 for t in tasks
        )
        task_blob = " ".join(tasks).lower() if isinstance(tasks, list) else ""
        ok_t2 = ok_t and (
            "ablation" in task_blob or "a/b" in task_blob or "ab test" in task_blob
        ) and "sanity" in task_blob and "latency" in task_blob and "token" in task_blob and "regression" in task_blob
        if ok_t2:
            checks_ok += 1
        else:
            reasons.append("micro_tasks must be >=10 and include A/B ablation and sanity-check.")

        ag = parsed.get("anti_gaming", {})
        ok_a = (
            isinstance(ag, dict)
            and isinstance(ag.get("mechanism"), str)
            and len(ag["mechanism"].strip()) >= 30
            and isinstance(ag.get("hidden_checks"), list)
            and len(ag["hidden_checks"]) >= 2
        )
        ok_a2 = ok_a and any(
            w in ag["mechanism"].lower()
            for w in ["withhold", "secret", "not revealed", "unseen"]
        )
        if ok_a2:
            checks_ok += 1
        else:
            reasons.append("anti_gaming must include >=2 hidden checks and mention withholding/secret.")

        acts = parsed.get("actions", [])
        ok_act = isinstance(acts, list) and any(
            isinstance(a, dict) and a.get("type") == "write_file" and str(a.get("path", "")).endswith(".md")
            for a in acts
        )
        ok_act2 = ok_act and any(
            isinstance(a, dict)
            and a.get("type") == "note"
            and (
                (
                    "latency_ms" in str(a.get("text", "")).lower()
                    and "total_tokens" in str(a.get("text", "")).lower()
                    and "token_budget" in str(a.get("text", "")).lower()
                )
                or (
                    "log" in str(a.get("text", "")).lower()
                    and "latency" in str(a.get("text", "")).lower()
                    and "token" in str(a.get("text", "")).lower()
                )
                or (
                    "telemetry" in str(a.get("text", "")).lower()
                    and "latency" in str(a.get("text", "")).lower()
                    and "token" in str(a.get("text", "")).lower()
                )
            )
            for a in acts
        )
        if ok_act2:
            checks_ok += 1
        else:
            reasons.append("actions must include a .md write_file and a note about log fields.")

        cs = parsed.get("completion_signal", {})
        ok_c = isinstance(cs, dict) and isinstance(cs.get("done", None), bool) and isinstance(
            cs.get("reason", ""), str
        )
        if ok_c:
            checks_ok += 1
        else:
            reasons.append("completion_signal must include boolean done and reason string.")

        blob = json.dumps(parsed, ensure_ascii=False).lower()
        hidden = []
        if not hidden_fn(blob):
            hidden.append(f"Hidden[{hidden_id_sel}]: missing required hidden metric signal.")

        score = checks_ok / total_checks
        complete = (checks_ok == total_checks) and (len(hidden) == 0)
        return VerificationResult(complete, score, reasons, hidden if hidden else None)

    return _verify


def verify_strict_schema(parsed: Optional[Dict[str, Any]]) -> VerificationResult:
    return make_strict_verifier()(parsed)
    if not parsed:
        return VerificationResult(
            complete=False,
            score=0.0,
            reasons=["Invalid JSON. Return ONLY JSON matching schema."],
            hidden_reasons=None,
        )

    required = [
        "iteration_summary",
        "hypotheses",
        "metrics",
        "micro_tasks",
        "anti_gaming",
        "actions",
        "completion_signal",
    ]
    missing = [k for k in required if k not in parsed]
    if missing:
        return VerificationResult(
            complete=False,
            score=0.0,
            reasons=[f"Missing required keys: {missing}"],
            hidden_reasons=None,
        )

    reasons = []
    checks_ok = 0
    total_checks = 6

    hyps = parsed.get("hypotheses", [])
    ok_h = (
        isinstance(hyps, list)
        and len(hyps) == 3
        and all(isinstance(x, str) and len(x.strip()) >= 25 for x in hyps)
        and all(re.search(r"(\d+%|\b>=\b|\b<=\b|\b\d+\b)", h) for h in hyps)
    )
    if ok_h:
        checks_ok += 1
    else:
        reasons.append("hypotheses must be exactly 3 substantial strings with numeric targets.")

    mets = parsed.get("metrics", [])
    ok_m = (
        isinstance(mets, list)
        and len(mets) == 3
        and all(isinstance(m, dict) for m in mets)
        and all(m.get("name") and m.get("definition") and m.get("formula_or_calc") for m in mets)
    )
    log_terms = [
        "iterations",
        "duration",
        "time",
        "latency",
        "hidden",
        "fail",
        "score",
        "token",
        "budget",
    ]
    ok_m2 = ok_m and sum(
        any(t in str(m["formula_or_calc"]).lower() for t in log_terms) for m in mets
    ) >= 2
    if ok_m2:
        checks_ok += 1
    else:
        reasons.append("metrics must be 3 items and at least 2 formulas reference telemetry fields.")

    tasks = parsed.get("micro_tasks", [])
    ok_t = isinstance(tasks, list) and len(tasks) >= 10 and all(
        isinstance(t, str) and len(t.strip()) >= 12 for t in tasks
    )
    task_blob = " ".join(tasks).lower() if isinstance(tasks, list) else ""
    ok_t2 = ok_t and (
        "ablation" in task_blob or "a/b" in task_blob or "ab test" in task_blob
    ) and "sanity" in task_blob and "latency" in task_blob and "token" in task_blob and "regression" in task_blob
    if ok_t2:
        checks_ok += 1
    else:
        reasons.append("micro_tasks must be >=10 and include A/B ablation and sanity-check.")

    ag = parsed.get("anti_gaming", {})
    ok_a = (
        isinstance(ag, dict)
        and isinstance(ag.get("mechanism"), str)
        and len(ag["mechanism"].strip()) >= 30
        and isinstance(ag.get("hidden_checks"), list)
        and len(ag["hidden_checks"]) >= 2
    )
    ok_a2 = ok_a and any(
        w in ag["mechanism"].lower()
        for w in ["withhold", "secret", "not revealed", "unseen"]
    )
    if ok_a2:
        checks_ok += 1
    else:
        reasons.append("anti_gaming must include >=2 hidden checks and mention withholding/secret.")

    acts = parsed.get("actions", [])
    ok_act = isinstance(acts, list) and any(
        isinstance(a, dict) and a.get("type") == "write_file" and str(a.get("path", "")).endswith(".md")
        for a in acts
    )
    ok_act2 = ok_act and any(
        isinstance(a, dict)
        and a.get("type") == "note"
        and "log" in str(a.get("text", "")).lower()
        and "latency" in str(a.get("text", "")).lower()
        and "token" in str(a.get("text", "")).lower()
        for a in acts
    )
    if ok_act2:
        checks_ok += 1
    else:
        reasons.append("actions must include a .md write_file and a note about log fields.")

    cs = parsed.get("completion_signal", {})
    ok_c = isinstance(cs, dict) and isinstance(cs.get("done", None), bool) and isinstance(
        cs.get("reason", ""), str
    )
    if ok_c:
        checks_ok += 1
    else:
        reasons.append("completion_signal must include boolean done and reason string.")

    blob = json.dumps(parsed, ensure_ascii=False).lower()
    hidden = []
    if not any(k in blob for k in ["false completion", "evidence mismatch", "gaming rate"]):
        hidden.append("Hidden: include a metric about false completion or evidence mismatch.")

    score = checks_ok / total_checks
    complete = (checks_ok == total_checks) and (len(hidden) == 0)
    return VerificationResult(complete, score, reasons, hidden if hidden else None)
