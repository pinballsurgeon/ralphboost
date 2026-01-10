import hashlib
from dataclasses import asdict
from typing import Any, Dict, Optional

from .actions import execute_actions
from .config import load_settings
from .contract import SYSTEM_INSTRUCTIONS
from .contract_patch import PATCH_INSTRUCTIONS
from .loop import run_loop
from .report import compute_report
from .verifier import make_strict_verifier, verify_minimal_schema


def run_task_text(
    task: str,
    mode: str = "patch",
    verify: str = "strict",
    hidden_id: str = "",
    hidden_rotate: bool = False,
    reveal_after: int = 0,
    inject_hidden: bool = False,
    max_iter: int = 6,
    patience: int = 2,
    feedback: str = "full",
    execute: bool = False,
    allow_commands: bool = False,
    action_root: str = ".",
) -> Dict[str, Any]:
    settings = load_settings()
    if verify == "strict":
        seed = hashlib.sha256(task.encode("utf-8")).hexdigest()[:12] if hidden_rotate else None
        verifier = make_strict_verifier(hidden_id=hidden_id or None, rotate_seed=seed)
    else:
        verifier = verify_minimal_schema
    contract = PATCH_INSTRUCTIONS if mode == "patch" else SYSTEM_INSTRUCTIONS
    run = run_loop(
        base_task=task,
        system_instructions=contract,
        verifier=verifier,
        settings=settings,
        max_iter=max_iter,
        no_progress_patience=patience,
        feedback_mode=feedback,
        reveal_after=reveal_after,
        inject_hidden=inject_hidden,
        mode=mode,
    )
    actions_result = None
    if execute and run.history:
        final_state = run.history[-1].parsed or {}
        actions = final_state.get("actions", [])
        if isinstance(actions, list):
            write_count, cmd_count, errors = execute_actions(
                actions, action_root, allow_commands
            )
            actions_result = {
                "write_files": write_count,
                "commands": cmd_count,
                "errors": errors,
            }
    rows = [asdict(r) for r in run.history]
    report = compute_report(rows)
    final = run.history[-1].parsed if run.history else None
    return {
        "status": run.status,
        "best_score": run.best_score,
        "iterations": run.iterations,
        "final": final,
        "report": report,
        "actions": actions_result,
    }


def run_task_simple(task: str) -> Optional[Dict[str, Any]]:
    result = run_task_text(task)
    return result.get("final")


def run_task_fast(
    task: str,
    execute: bool = False,
    allow_commands: bool = False,
    action_root: str = ".",
    **overrides: Any,
) -> Dict[str, Any]:
    params = {
        "mode": "patch",
        "verify": "strict",
        "hidden_rotate": True,
        "reveal_after": 2,
        "max_iter": 6,
        "patience": 2,
        "feedback": "summarize",
        "execute": execute,
        "allow_commands": allow_commands,
        "action_root": action_root,
    }
    params.update(overrides)
    return run_task_text(task, **params)


def build_task(title: str, outputs: list[str], extra: str = "") -> str:
    lines = [
        f"Create {title}.",
        "Requirements:",
        "- Output must follow the JSON schema in the system instructions.",
        "- Exactly 3 hypotheses with numeric targets.",
        '- Exactly 3 metrics; include "False Completion Rate" with formula_or_calc using hidden_fail_iters and iterations.',
        "- micro_tasks list with at least 12 tasks, include A/B ablation, sanity-check, regression guard, latency profiling, token budget analysis.",
        "- anti_gaming must include at least 3 hidden checks and explain withholding.",
        "- actions must include:",
    ]
    for path in outputs:
        lines.append(f'  {{\"type\":\"write_file\",\"path\":\"{path}\",\"content\":\"...\"}}')
    lines.append('  {"type":"note","text":"latency_ms, total_tokens, token_budget, hidden_fail_iters"}')
    if extra.strip():
        lines.append(extra.strip())
    lines.append("Return ONLY valid JSON.")
    return "\n".join(lines)
