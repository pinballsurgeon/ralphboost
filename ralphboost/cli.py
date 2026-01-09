import argparse
import sys

from .actions import execute_actions
from .config import load_settings
from .contract import SYSTEM_INSTRUCTIONS
from .contract_patch import PATCH_INSTRUCTIONS
from .loop import run_loop
from .report import generate_report
from .suite import run_suite
from .telemetry import write_run
from .verifier import verify_minimal_schema, make_strict_verifier

def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    with open(args.task, "r", encoding="utf-8") as f:
        task = f.read()
    if args.verify == "strict":
        verifier = make_strict_verifier(
            hidden_id=args.hidden_id,
            rotate_seed=args.run_id if args.hidden_rotate else None,
        )
    else:
        verifier = verify_minimal_schema
    contract = PATCH_INSTRUCTIONS if args.mode == "patch" else SYSTEM_INSTRUCTIONS
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
    if args.execute_actions and run.history:
        final_state = run.history[-1].parsed or {}
        actions = final_state.get("actions", [])
        if isinstance(actions, list):
            write_count, cmd_count, errors = execute_actions(
                actions, args.action_root, args.allow_commands
            )
            print(f"Actions executed: write_files={write_count} commands={cmd_count}")
            if errors:
                print("Action errors:", "; ".join(errors))
    runs_dir = write_run(run, runs_dir=args.runs_dir, run_id=args.run_id)
    print(f"Status: {run.status}  iters: {run.iterations}  best: {run.best_score}")
    print(f"Runs saved to: {runs_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ralphboost")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a RalphBoost loop")
    p_run.add_argument("task", help="Path to a task file")
    p_run.add_argument("--max-iter", type=int, default=6)
    p_run.add_argument("--patience", type=int, default=2)
    p_run.add_argument(
        "--feedback", choices=["full", "top1", "summarize"], default="full"
    )
    p_run.add_argument("--verify", choices=["minimal", "strict"], default="minimal")
    p_run.add_argument("--reveal-after", type=int, default=0)
    p_run.add_argument("--inject-hidden", action="store_true")
    p_run.add_argument("--mode", choices=["rewrite", "patch"], default="rewrite")
    p_run.add_argument("--hidden-id", default="")
    p_run.add_argument("--hidden-rotate", action="store_true")
    p_run.add_argument("--execute-actions", action="store_true")
    p_run.add_argument("--allow-commands", action="store_true")
    p_run.add_argument("--action-root", default=".")
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--run-id", default="run")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Generate a metrics report from a run")
    p_report.add_argument("run_file", help="Path to a .jsonl run file")
    p_report.set_defaults(func=lambda args: generate_report(args.run_file))

    p_suite = sub.add_parser("suite", help="Run multiple tasks and emit a combined report")
    p_suite.add_argument("tasks", nargs="+", help="Task files, directories, or glob patterns")
    p_suite.add_argument("--max-iter", type=int, default=6)
    p_suite.add_argument("--patience", type=int, default=2)
    p_suite.add_argument(
        "--feedback", choices=["full", "top1", "summarize"], default="full"
    )
    p_suite.add_argument("--verify", choices=["minimal", "strict"], default="minimal")
    p_suite.add_argument("--reveal-after", type=int, default=0)
    p_suite.add_argument("--inject-hidden", action="store_true")
    p_suite.add_argument("--runs-dir", default="runs")
    p_suite.add_argument("--run-prefix", default="suite")
    p_suite.add_argument("--mode", choices=["rewrite", "patch"], default="rewrite")
    p_suite.add_argument("--hidden-id", default="")
    p_suite.add_argument("--hidden-rotate", action="store_true")
    p_suite.add_argument("--execute-actions", action="store_true")
    p_suite.add_argument("--allow-commands", action="store_true")
    p_suite.add_argument("--action-root", default=".")
    p_suite.set_defaults(func=lambda args: run_suite(args, SYSTEM_INSTRUCTIONS, PATCH_INSTRUCTIONS))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
