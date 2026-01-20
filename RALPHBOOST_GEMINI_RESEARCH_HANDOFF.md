# RalphBoost v3.5: Gemini Integration Research Handoff

**Date:** 2026-01-20
**Target Audience:** AI Research Specialist Team
**Scope:** Deep dive into the "Agentic Boosting" architecture, specifically the Gemini integration layer.

---

## 1. Architectural Overview: The "Ralph Wiggum Loop"

RalphBoost v3.5 transitions from a pure deterministic boosting engine to an **Agent-Directed Scientific Discovery Engine**.
The core philosophy is that **Gemini** acts as the *Strategy Controller* (Ralph Wiggum), while the **Domain Packs** act as the *Physics Engine* (Deterministic Refiners).

### The Loop
1.  **Observation**: The system observes the current `State` (residual signal, current fit, loss).
2.  **Strategy (Gemini)**: The Agent analyzes the residual and proposes a **Blueprint** (JSON).
    *   *Where to look?* (Window/Focus)
    *   *What to try?* (Component Family)
    *   *Constraints?* (Complexity, Depth)
3.  **Refinement (Deterministic)**: The Blueprint is passed to `OptimizationRefiner`, which fits parameters mathematically (no hallucinations allowed here).
4.  **Competition**: Refined candidates compete based on objective improvement ($\Delta Loss$).
5.  **Update**: The best component is added to the ledger, the residual is updated, and the loop repeats.

---

## 2. Current Implementation State (Phase 3.5A)

The following files constitute the current "Agent Wiring" (Fixed and Verified):

### 2.1 `ralphboost/agents/gemini.py`
*   **Class:** `GeminiAgent`
*   **Library:** `google.genai` (Official Google GenAI SDK).
*   **Model:** Defaults to `gemini-flash-latest` (fast, cost-effective reasoning).
*   **Key Features:**
    *   **Connection Validation:** `validate_connection()` performs a minimal API check.
    *   **Robustness:** Uses `types.HttpOptions` to configure timeouts (avoiding default SDK limits).
    *   **Parsing:** `_parse_json_response` handles markdown stripping and JSON extraction robustly.
    *   **Data Handling:** `_coerce_array` ensures consistent input handling.
*   **Key Function:** `propose_blueprints(state, context)`
    *   **Input:** `state.residual` (flattened preview, stats), `context` (iteration, domain hints).
    *   **Prompting:** Constructs a structured prompt with signal stats (mean, std, min, max) and explicit instructions.
    *   **Config:** Uses `types.ThinkingConfig` (budget: 128 tokens) for enhanced reasoning.
    *   **Output:** Enforces a strict JSON Schema (The "Blueprint") with default backfilling for stability.
    *   **Error Handling:** Catches API errors and returns empty blueprint (soft fail) or raises if `fail_hard=True`.

### 2.2 `ralphboost/core/strategy_loop.py`
*   **Class:** `StrategyLoop`
*   **Role:** The orchestration layer (Layer C).
*   **Logic:**
    *   Calls `agent.propose_blueprints()`.
    *   Extracts `candidates`, `focus`, `constraints`.
    *   Passes candidates to `refiner.refine_batch()`.
    *   Evaluates Monotonic Improvement.
    *   Updates `RalphState` with a stable, explainable ledger.

### 2.3 `tests/test_agent_schema.py`
*   **Role:** Integration Integrity Check.
*   **Mechanism:** Makes a **REAL** API call to Gemini (no mocks) to verify that the returned JSON matches the Blueprint schema.
*   **Status:** Passing (verified with real key).

---

## 3. The Blueprint Schema
The interface between Gemini and the Physics Engine is defined by this JSON structure:

```json
{
  "iteration": int,
  "focus": {
    "window_policy": "global" | "local" | "micro",
    "windows": [{"start": int, "end": int}]
  },
  "candidates": [
    {
      "family": "periodic_train" | "local_dip" | "baseline",
      "parameters": { ... }, 
      "rationale": "Explanation of why this component was chosen"
    }
  ],
  "constraints": {
    "max_components": int,
    "complexity_penalty": float
  }
}
```

---

## 4. Research Assignments (Targeted Precision)

The Research Team must investigate the following areas to achieve SOTA performance.

### 4.1 Model Selection & API Precision
*   **Objective:** Identify the optimal Gemini model for scientific time-series reasoning.
*   **Questions:**
    *   Is `gemini-flash-latest` sufficient, or does `gemini-1.5-pro` offer better reasoning for complex "multi-event" structures?
    *   How to handle API versioning (`v1alpha` vs `v1beta`) stability in `google.genai`?
    *   **Task:** Benchmark `flash` vs `pro` on `exoTrain.csv` for "Rationale Quality" vs "Latency".

### 4.2 Context Window & Data Representation
*   **Objective:** Maximize agent insight without token overflow.
*   **Current State:** We send a raw list of the first 256 residual points (`preview_len`).
*   **Research Needed:**
    *   **Tokenization Strategy:** Should we send statistical features (FFT peaks, autocorrelation) instead of raw floats?
    *   **Compression:** Can we define a text-based "ASCII Art" representation of the curve for the LLM?
    *   **RAG:** Should we retrieve similar historical Kepler signals to prompt the agent ("This looks like KOI-123")?

### 4.3 Windowing Strategy (Global/Local/Micro)
*   **Objective:** Enable non-linear, dynamic variance handling.
*   **Research Needed:**
    *   How does the agent decide to switch from "Global" (Trend) to "Local" (Transit)?
    *   Can the agent dynamically define the `windows` list based on residual hotspots (high variance regions)?
    *   **Task:** Develop a prompt strategy that explicitly asks for "Variance Analysis" before component selection.

### 4.4 Schema Evolution
*   **Objective:** Support advanced, multi-dimensional signal interplay.
*   **Research Needed:**
    *   Does the current schema support "Coupled Components" (e.g., Transit + Secondary Eclipse)?
    *   How do we represent "multi-dimensional" constraints (e.g., consistency across multiple wavelengths or sensors)?

### 4.5 Explainability & Verification
*   **Objective:** Ensure the "Rationale" field is not just hallucination.
*   **Research Needed:**
    *   Can we implement a "Verifier Agent" that checks if the `rationale` matches the `fitted` result?
    *   How to visualize the "Agent's Focus" (windows) alongside the "Physics Fit" in the final report?

---

## 5. Immediate Next Steps for Engineering

1.  **Maintain Valid API Key:** Ensure `.env.local` contains a valid key for ongoing tests.
2.  **Benchmark on Kepler:** Run `StrategyLoop` on `exoTest.csv` and measure `PR-AUC` vs `Agent Token Cost`.
