import json
import os
import time
from dataclasses import asdict
from typing import Optional

from .types import IterationRecord, LoopResult


def ensure_runs_dir(path: str = "runs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_iteration(rec: IterationRecord, runs_dir: str, run_id: str) -> None:
    path = os.path.join(runs_dir, f"{run_id}.jsonl")
    data = asdict(rec)
    data["logged_at"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_summary(run: LoopResult, runs_dir: str, run_id: str) -> None:
    path = os.path.join(runs_dir, f"{run_id}_summary.json")
    data = {
        "status": run.status,
        "best_score": run.best_score,
        "iterations": run.iterations,
        "saved_at": time.time(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_run(run: LoopResult, runs_dir: Optional[str] = None, run_id: str = "run") -> str:
    runs_dir = ensure_runs_dir(runs_dir or "runs")
    for rec in run.history:
        write_iteration(rec, runs_dir, run_id)
    write_summary(run, runs_dir, run_id)
    return runs_dir
