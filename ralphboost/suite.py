import glob
import os
import json

from .actions import execute_actions

from .config import load_settings
from .loop import run_loop
from .report import compute_report, load_jsonl
from .telemetry import write_run
from .verifier import verify_minimal_schema, make_strict_verifier


def _resolve_tasks(items: list[str]) -> list[str]:
    paths = []
    for item in items:
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "*.txt")))
        elif any(ch in item for ch in ["*", "?", "[", "]"]):
            paths.extend(glob.glob(item))
        elif os.path.isfile(item):
            paths.append(item)
    return sorted(list(dict.fromkeys(paths)))


def run_suite(args, rewrite_instructions: str, patch_instructions: str) -> int:
    settings = load_settings()
    contract = patch_instructions if args.mode == "patch" else rewrite_instructions
    task_paths = _resolve_tasks(args.tasks)
    if not task_paths:
        print("No task files found.")
        return 1

    per_task = []
    totals = {
        "total_tokens": 0,
        "total_duration_sec": 0.0,
        "iterations": 0,
        "thrash_steps": 0,
        "hidden_fail_iters": 0,
    }

    for path in task_paths:
        with open(path, "r", encoding="utf-8") as f:
            task = f.read()
        base = os.path.splitext(os.path.basename(path))[0]
        run_id = f"{args.run_prefix}_{base}"
        if args.verify == "strict":
            verifier = make_strict_verifier(
                hidden_id=args.hidden_id or None,
                rotate_seed=run_id if args.hidden_rotate else None,
            )
        else:
            verifier = verify_minimal_schema
        run = run_loop(
            base_task=task,
            system_instructions=contract,
            verifier=verifier,
            settings=settings,
            max_iter=args.max_iter,
            no_progress_patience=args.patience,
            feedback_mode=args.feedback,
            reveal_after=args.reveal_after,
            inject_hidden=args.inject_hidden,
            mode=args.mode,
        )
        write_run(run, runs_dir=args.runs_dir, run_id=run_id)
        if args.execute_actions and run.history:
            final_state = run.history[-1].parsed or {}
            actions = final_state.get("actions", [])
            if isinstance(actions, list):
                write_count, cmd_count, errors = execute_actions(
                    actions, args.action_root, args.allow_commands
                )
                if errors:
                    print(f"Action errors for {run_id}: " + "; ".join(errors))
        run_file = os.path.join(args.runs_dir, f"{run_id}.jsonl")
        rows = load_jsonl(run_file)
        report = compute_report(rows)
        per_task.append(
            {
                "task_file": path,
                "run_id": run_id,
                "status": run.status,
                "best_score": run.best_score,
                "iterations": run.iterations,
                "report": report,
            }
        )
        totals["total_tokens"] += report.get("total_tokens", 0)
        totals["total_duration_sec"] += report.get("total_duration_sec", 0.0)
        totals["iterations"] += report.get("iterations", 0)
        totals["thrash_steps"] += report.get("thrash_steps", 0)
        totals["hidden_fail_iters"] += report.get("hidden_fail_iters", 0)

    avg_duration = totals["total_duration_sec"] / max(1, len(per_task))
    avg_tokens = totals["total_tokens"] / max(1, len(per_task))
    avg_iterations = totals["iterations"] / max(1, len(per_task))
    tokens_per_sec = (
        totals["total_tokens"] / totals["total_duration_sec"]
        if totals["total_duration_sec"] > 0
        else 0.0
    )

    suite_report = {
        "tasks": len(per_task),
        "avg_duration_sec": avg_duration,
        "avg_total_tokens": avg_tokens,
        "avg_iterations": avg_iterations,
        "tokens_per_sec": tokens_per_sec,
        "thrash_steps_total": totals["thrash_steps"],
        "hidden_fail_iters_total": totals["hidden_fail_iters"],
        "per_task": per_task,
    }

    out_path = os.path.join(args.runs_dir, f"{args.run_prefix}_suite_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(suite_report, f, indent=2)
    print(json.dumps(suite_report, indent=2))
    print(f"Suite report saved to: {out_path}")
    return 0
