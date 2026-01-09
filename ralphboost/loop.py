import json
import time
from typing import Callable, List

from .autopatch import build_patch_spec, format_patch_spec
from .gemini_client import generate_response
from .patch import apply_patch, seed_state
from .types import IterationRecord, LoopResult, VerificationResult
from .verifier import safe_json_load


def _failed_roots(reasons: List[str]) -> List[str]:
    roots = set()
    for r in reasons:
        s = r.lower()
        if "hypotheses" in s:
            roots.add("hypotheses")
        if "metrics" in s or "false completion" in s or "evidence mismatch" in s:
            roots.add("metrics")
        if "micro_tasks" in s or "micro tasks" in s:
            roots.add("micro_tasks")
        if "anti_gaming" in s or "hidden" in s:
            roots.add("anti_gaming")
        if "actions" in s:
            roots.add("actions")
        if "completion_signal" in s:
            roots.add("completion_signal")
        if "iteration_summary" in s:
            roots.add("iteration_summary")
    return sorted(roots)


def _loss(vr: VerificationResult, hidden_weight: float = 0.25) -> tuple[float, float, float]:
    visible = max(0.0, 1.0 - float(vr.score))
    hidden = hidden_weight * float(len(vr.hidden_reasons or []))
    return visible, hidden, visible + hidden


def run_loop(
    base_task: str,
    system_instructions: str,
    verifier: Callable[[dict], VerificationResult],
    settings,
    max_iter: int = 6,
    no_progress_patience: int = 2,
    feedback_mode: str = "full",
    reveal_after: int = 0,
    inject_hidden: bool = False,
    mode: str = "rewrite",
) -> LoopResult:
    history: List[IterationRecord] = []
    best_score = -1.0
    stagnant = 0
    feedback = ""
    hidden_fail_count = 0
    state = seed_state() if mode == "patch" else None
    allowed_roots: List[str] = []

    for i in range(1, max_iter + 1):
        prompt = system_instructions + "\n\nTASK:\n" + base_task.strip()
        if mode == "patch":
            if not allowed_roots:
                allowed_roots = list(seed_state().keys())
            prompt += "\n\nCURRENT_STATE:\n" + json.dumps(state, ensure_ascii=False)
            prompt += "\n\nALLOWED_PATHS:\n" + ", ".join(allowed_roots)
        if feedback:
            prompt += "\n\nVERIFIER FEEDBACK:\n" + feedback

        t0 = time.time()
        resp = generate_response(prompt, settings=settings)
        raw = resp["text"]
        dt = time.time() - t0

        parsed = safe_json_load(raw)
        patch_payload = None
        patch_ops = 0
        patch_applied = 0
        patch_chars = 0
        if mode == "patch":
            patch_payload = parsed if isinstance(parsed, dict) else None
            ops = patch_payload.get("patch", []) if patch_payload else []
            patch_ops = len(ops) if isinstance(ops, list) else 0
            if isinstance(ops, list):
                for op in ops:
                    val = op.get("value")
                    patch_chars += len(json.dumps(val, ensure_ascii=False)) if val is not None else 0
            state, patch_applied, _ = apply_patch(state, ops if isinstance(ops, list) else [], allowed_roots)
            vr = verifier(state)
        else:
            vr = verifier(parsed)
        if vr.hidden_reasons:
            hidden_fail_count += 1

        visible_loss, hidden_loss, loss = _loss(vr)

        history.append(
            IterationRecord(
                iteration=i,
                prompt=prompt,
                raw_text=raw,
                parsed=state if mode == "patch" else parsed,
                verification=vr,
                duration_sec=dt,
                model=resp.get("model"),
                prompt_chars=len(prompt),
                response_chars=len(raw),
                prompt_tokens=resp.get("usage", {}).get("prompt_tokens", 0),
                response_tokens=resp.get("usage", {}).get("response_tokens", 0),
                total_tokens=resp.get("usage", {}).get("total_tokens", 0),
                patch=patch_payload,
                patch_ops=patch_ops,
                patch_applied_ops=patch_applied,
                patch_chars=patch_chars,
                visible_loss=visible_loss,
                hidden_loss=hidden_loss,
                loss=loss,
            )
        )

        if vr.score > best_score + 1e-12:
            best_score = vr.score
            stagnant = 0
        else:
            stagnant += 1

        if vr.complete:
            return LoopResult("verified", best_score, len(history), history)

        if stagnant >= no_progress_patience:
            return LoopResult("no-progress", best_score, len(history), history)

        reasons = list(vr.reasons)
        if inject_hidden or (reveal_after > 0 and hidden_fail_count >= reveal_after):
            reasons += list(vr.hidden_reasons or [])
        allowed_roots = _failed_roots(reasons) if mode == "patch" else []
        patch_spec = build_patch_spec(state if mode == "patch" else parsed, vr) if reasons else None
        if feedback_mode == "top1":
            feedback = reasons[0] if reasons else ""
        elif feedback_mode == "summarize":
            feedback = "; ".join(reasons[:3])
        else:
            feedback = "\n".join(f"- {r}" for r in reasons)
        if patch_spec:
            feedback += "\nPATCH_SPEC:\n" + format_patch_spec(patch_spec)

    return LoopResult("max-iterations", best_score, len(history), history)
