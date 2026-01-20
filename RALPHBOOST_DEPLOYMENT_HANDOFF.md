# RalphBoost v3.5: Deployment & Colab Handoff

**Date:** 2026-01-20
**Status:** **READY FOR DEPLOYMENT**
**Goal:** Independent validation of the RalphBoost v3.5 Agentic Loop on Google Colab or similar environments.

---

## 1. Git Deployment (Engineering Team)

Push the current codebase to all target branches to ensure synchronization.

```bash
# 1. Stage all changes (Core, Agents, Tests, Examples)
git add .

# 2. Commit
git commit -m "feat(v3.5): RalphBoost Agentic Loop with Gemini Integration"

# 3. Push to upstream branches
git push origin main
git push origin master
git push origin dev
git push origin qa
```

---

## 2. Google Colab / Independent Validation Guide

To validate RalphBoost v3.5 independently, follow these steps in a new Google Colab notebook.

### Step 1: Setup Environment
Clone the repository and install dependencies.

```python
# Clone Repository
!git clone https://github.com/pinballsurgeon/ralphboost.git
%cd ralphboost

# Install Dependencies
!pip install -q -r requirements.txt
```

### Step 2: Download Kepler Data (Kaggle)
Download the `exoTrain.csv` and `exoTest.csv` datasets.

```python
# Setup Kaggle Credentials (Upload kaggle.json to /content first if needed, or set env vars)
import os
# os.environ['KAGGLE_USERNAME'] = "your_username"
# os.environ['KAGGLE_KEY'] = "your_key"

!pip install -q kaggle
!mkdir -p ~/.kaggle
# If you uploaded kaggle.json:
# !cp /content/kaggle.json ~/.kaggle/kaggle.json
# !chmod 600 ~/.kaggle/kaggle.json

# List and Download
!kaggle datasets list -s kepler | head
!kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data -p . --unzip

# Verify
!ls -lh exo*.csv
```

### Step 3: Configure Gemini Agent
Set your Gemini API Key.

```python
import os
from google.colab import userdata

# Option A: Use Colab Secrets (Recommended)
# os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')

# Option B: Direct Input
os.environ["GEMINI_API_KEY"] = "AIza..." # Replace with valid key
```

### Step 4: Run the Agentic Loop Proof-of-Concept
Execute the `kepler_agentic_loop.py` script. This script:
1.  Loads a slice of `exoTest.csv`.
2.  Connects to Gemini (`gemini-flash-latest`).
3.  Runs the **Ralph Wiggum Loop** (Agent Proposal -> Deterministic Refinement).
4.  Outputs the ledger of discovered planets/components.

```python
!python examples/kepler_agentic_loop.py
```

### Step 5: Expected Output
You should see output indicating successful connection, candidate proposals, and loss reduction.

```text
Loading exoTest.csv...
Target Label: 2.0 (2=Planet, 1=Non-Planet)
...
Verifying Gemini Connection...
Connection OK: {'ok': True}
Starting Strategy Loop...
Iteration 0: Agent proposed 3 candidates.
  - periodic_train: {'period': 100.0, ...}
...
--- Final Report ---
Final Loss: 2.4044
Components Found: 1
[0] PERIODIC_TRAIN: {...}
Done.
```

---

## 3. Key Files for Review

*   `ralphboost/core/strategy_loop.py`: The orchestration engine (Layer C).
*   `ralphboost/agents/gemini.py`: The hardened Gemini integration.
*   `ralphboost/domains/transit.py`: The Physics domain with explicit schema context.
*   `examples/kepler_agentic_loop.py`: The runnable proof script.
