# RalphBoost

RalphBoost is a lo-fi loop engine, gradient boost inspired

Quick start (Gemini 3 Flash)
- Install deps: `pip install -r requirements.txt`
- Set your key: `setx GEMINI_API_KEY "your_key_here"` (PowerShell: `$env:GEMINI_API_KEY="..."`)
- Optional model override: `setx RALPHBOOST_MODEL "gemini-3.0-flash"`
- Run a task: `python -m ralphboost run path\to\task.txt`
- Strict mode + hidden checks: `python -m ralphboost run tasks\challenge_protocol.txt --verify strict --reveal-after 2`
- Report metrics: `python -m ralphboost report runs\phase1.jsonl`
- Challenge suite: `python -m ralphboost suite tasks\*.txt --verify strict --reveal-after 2 --run-prefix suite1`
- Patch mode (boosting-style): `python -m ralphboost run tasks\challenge_protocol.txt --verify strict --mode patch --reveal-after 2 --run-id patch1`
- Patch suite: `python -m ralphboost suite tasks\*.txt --verify strict --mode patch --reveal-after 2 --run-prefix suite_patch`
- Hidden rotation: `python -m ralphboost suite tasks\*.txt --verify strict --hidden-rotate --run-prefix suite_rot`
- Workflow demo (writes docs): `python -m ralphboost suite tasks\workflow_*.txt --verify strict --mode patch --execute-actions --run-prefix demo1`

Install from GitHub
- `pip install git+https://github.com/pinballsurgeon/ralphboost.git`
- `python -m ralphboost suite tasks/workflow_*.txt --verify strict --mode patch --execute-actions --run-prefix demo`

Python API (no CLI, no task files)
```python
import os
from ralphboost.api import run_task_text, run_task_fast, build_task

os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
os.environ["RALPHBOOST_MODEL"] = "gemini-3-flash-preview"

task = build_task(
    "an Incident Response Playbook + SLO spec + alert config for a Checkout API",
    ["docs/INCIDENT_PLAYBOOK.md", "docs/SLO.md", "configs/alerts.yaml"],
)

result = run_task_fast(task, execute=True)
print(result["status"], result["best_score"])
```
