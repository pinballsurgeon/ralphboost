
import argparse
import os
import sys

from .config import load_settings
from .contract import CODE_ONLY_INSTRUCTIONS
from .loop import run_loop
from .verifier import verify_code_state
from .telemetry import write_run

def load_state(root: str) -> dict:
    state = {}
    for dirpath, _, filenames in os.walk(root):
        if ".git" in dirpath or "__pycache__" in dirpath or "runs" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".pyc"): continue
            path = os.path.relpath(os.path.join(dirpath, f), root)
            path = path.replace("\\", "/")
            try:
                with open(os.path.join(dirpath, f), "r", encoding="utf-8") as fd:
                    state[path] = fd.read()
            except Exception:
                pass
    return state

def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        with open(args.task, "r", encoding="utf-8") as f:
            task = f.read()
    except FileNotFoundError:
        print(f"Task file not found: {args.task}")
        return 1
        
    initial_state = load_state(args.workdir)
    
    print(f"Running task: {args.task}")
    print(f"Loaded {len(initial_state)} files from {args.workdir}")
    
    run = run_loop(
        base_task=task,
        system_instructions=CODE_ONLY_INSTRUCTIONS,
        verifier=verify_code_state,
        settings=settings,
        max_iter=args.max_iter,
        initial_state=initial_state
    )
    
    runs_dir = write_run(run, runs_dir=args.runs_dir, run_id=args.run_id)
    print(f"Status: {run.status}  iters: {run.iterations}  best: {run.best_score}")
    print(f"Runs saved to: {runs_dir}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ralphboost")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a RalphBoost code loop")
    p_run.add_argument("task", help="Path to a task file")
    p_run.add_argument("--workdir", default=".", help="Directory containing code to patch")
    p_run.add_argument("--max-iter", type=int, default=6)
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--run-id", default="run")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
