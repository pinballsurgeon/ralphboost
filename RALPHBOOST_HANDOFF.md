# RalphBoost Forever: Full Engineering Handoff (True Boosting + Generalized Science Machine)

**Date:** 2026-01-19
**Status:** **PHASE 3 COMPLETE (Kepler Planet Hunting Success)**
**Goal:** Build RalphBoost into a generalized, deterministic, explainable scientific boosting toolkit (XGBoost-tier credibility, but scientific basis modules).
**North Star:** True Gradient-Boost-style remainder fitting, with deterministic refinement, candidate competition, monotonic objective improvement, dedupe, multi-scale windows, and proof-driven validation.

---

## 0) What RalphBoost is (canonical identity)

**RalphBoost = "Boosting engine where weak learners are scientific components"**

Each iteration, RalphBoost proposes K candidate scientific hypotheses, refines them deterministically, scores them by objective improvement, accepts the best, updates the additive model, and repeats until improvement stalls.

This is the irreducible soul:
- ✅ Formal objective (MSE/logistic/others)
- ✅ Remainder / negative gradient drives the next step
- ✅ Candidate competition (K > 1) matters
- ✅ Deterministic refiner produces truth, not vibes
- ✅ Monotonic improvement enforcement prevents fake boosting
- ✅ Dedupe + complexity penalties prevent "repeat the same thing forever"
- ✅ Telemetry ledger makes it auditable and explainable

Agents (LLMs/FFT) can help propose hypotheses, but must never be required for progress.

---

## 1) Current project status (facts on ground)

**✅ What’s working right now**
- Core package structure exists (`core`, `domains`, `agents`, `refiners`, `examples`, `tests`)
- **TransitDomain** implemented and verified (Phase 3).
- **Kepler Results (Full Test Set 570 samples):**
    - **PR-AUC: 0.479** (vs baseline 0.008). 55x lift.
    - **Label Shuffle Collapse**: PR-AUC shuffled 0.007. **Proof of no leakage.**
    - **Top-K**: Found 4 real planets in top 25.
    - **Audit**: Ledger shows interpretable "periodic_train" components.
- Colab download + load pipeline for Kepler labelled time series works (`exoTrain.csv`, `exoTest.csv`).
- Engine loop exists + state history exists.

**❗ What’s next (The "Home Run" Improvements)**
- Add "periodic dominance" features to separate real trains from local dip spam.
- Penalize too many local dips (complexity control).
- Plot overlays for top predictions.

---

## 2) The final architecture (clean breakpoint, scalable forever)

### 2.1 "Core vs Domain Packs vs Agents" (non-negotiable split)

**✅ Core owns (universal, never domain-specific)**
- Objective evaluation + remainder computation
- Candidate competition and scoring by Δloss
- Acceptance rules: monotonic improvement, early stop, stall detection
- Shrinkage / step size policy
- Dedupe gating (via fingerprint hook)
- Complexity penalty support (via domain hook)
- Telemetry, history, reproducibility

**✅ Domain Packs own ("physics vocabulary")**
- Component types + schemas
- Deterministic `propose()` to generate K candidates
- Deterministic `refine()` to fit parameters to remainder
- `apply()` to add component into fitted signal
- `fingerprint()` for dedupe
- `complexity()` for regularization guidance

**✅ Agents own (optional hypothesis generators)**
- `propose()` suggestions only
- Cannot break determinism
- Cannot bypass scoring
- Cannot bypass refiner

This breakpoint is what makes RalphBoost a generalized toolkit, not a Kepler one-off.

---

## 3) The RalphBoost "Truth Contract" (acceptance gates)

A new engineer must treat these as gates. If these are violated, it’s not RalphBoost.

### 3.1 Boosting correctness gates
1.  **Objective exists and is explicit.**
2.  **Remainder exists and is derived from objective** (residual or negative gradient).
3.  **Each accepted iteration must satisfy:** `loss_after < loss_before`
4.  **K-candidate competition must matter:** `k_candidates > 1` improves faster (measurable).
5.  **Deterministic refinement must improve outcomes:** "no refiner" ablation stalls or slows.
6.  **Deduping works:** repeated/near-duplicate components do not get accepted endlessly.
7.  **Termination is measurable:** stall threshold, improvement threshold, energy ratio, or gradient norm.

### 3.2 Scientific credibility gates
1.  **Label shuffle collapse** (PR-AUC drops to near-base positive rate).
2.  **Multi-seed splits** (mean ± std reported).
3.  **Ablations identify the real hero.**
4.  **Robustness sweeps** (noise, jitter, missing blocks).
5.  **Top-K audit shows real-looking signatures, not artifacts.**
6.  **No data leakage** (dedupe across splits, near-duplicate checks).

---

## 4) Phase plan (Roadmap)

### ✅ PHASE 1: Lock "True Boosting Core" (Engine Credibility)
**Goal:** Upgrade core to a boosting engine that is provably real without any ML libraries.
**Status:** **COMPLETE**

**Deliverables:**
- `ralphboost/core/objectives.py`: `MSEObjective`, `LogisticObjective`.
- `ralphboost/core/engine.py`: K-candidate competition, Monotonic Acceptance, Telemetry.
- `ralphboost/core/state.py`: Rich Telemetry.
- `tests/test_rfc_truth.py`: Deterministic proof tests.

### ✅ PHASE 2: Domain Packs Become Real Scientific Weak Learners
**Goal:** Make domains provide deterministic propose/refine/apply/fingerprint/complexity.
**Status:** **COMPLETE**

**Deliverables:**
- `ralphboost/domains/base.py`: Domain Interface.
- `ralphboost/domains/signal.py`: Deterministic Sinusoid Domain.
- `ralphboost/domains/time_series.py`: Deterministic Trend/Seasonality Domain.

### ✅ PHASE 3: Kepler TransitDomain (The Planet Vocabulary Pack)
**Goal:** Create a real domain pack that can decompose Kepler-style curves into interpretable transit structures.
**Status:** **COMPLETE**

**Deliverables:**
- `ralphboost/domains/transit.py`
    - Families: `BaselineTrend`, `PeriodicTransitTrain`, `LocalTransitDip`.
    - Deterministic `propose` (grids, rolling windows).
    - Deterministic `refine` (analytic depth fit).
    - `fingerprint` (period/phase buckets).
    - `complexity` (penalties for overfitting).
- Tests: `tests/test_transit_domain.py` (Synthetic recovery).
- Benchmark: `benchmarks/kepler_homerun.py` (PR-AUC 0.479 confirmed).

### 🔲 PHASE 4: Multi-Scale "Wiggum Superloop" (Generalized Science Mode)
**Goal:** Make RalphBoost feel like a generalized scientific engine by boosting across windows, not just globally.
**Status:** **TODO**

**Deliverables:**
- `ralphboost/core/lens.py`: Windowing policy (Global -> Local -> Micro).
- `ralphboost/core/controller.py`: Meta-controller for scope and domain selection.
- Updated Component Schema: Add `scope` field.
- Tests: `tests/test_multiscale_controller.py`.

### 🔲 PHASE 5: Benchmarks + Scientific Integrity Harness
**Goal:** Create a "can’t-fake-it" benchmark and audit suite.
**Status:** **TODO (Partially done in Phase 3 Benchmark)**

**Deliverables:**
- `benchmarks/kepler_eval.py`: Stratified splits, PR-AUC, Top-K.
- `benchmarks/credibility.py`: Label shuffle, Ablation, Robustness.
- Success Criteria: PR-AUC lift, Shuffle collapse.

### 🔲 PHASE 6: Demo Artifacts (The "XGBoost but Sci-Fi" story)
**Goal:** Make the demo unforgettable and transparent.
**Status:** **TODO**

**Deliverables:**
- "Component Ledger" (Visual Table).
- "Residual Movie" (Plots).
- "Top-K Audit Panel".
- "Credibility Panel".

### 🔲 PHASE 7: Generalization Proof (Beyond Kepler)
**Goal:** Prove generalized capability with two more domain packs.
**Status:** **TODO**

**Deliverables:**
- `ralphboost/domains/regime.py` (Industrial Bursts).
- `ralphboost/domains/chirp.py` (Radio Chirps).
- Generalization Benchmark.

---

## 5) Exact file plan (what gets added/modified)

**Core**
- `ralphboost/core/engine.py` (DONE)
- `ralphboost/core/objectives.py` (DONE)
- `ralphboost/core/state.py` (DONE)
- `ralphboost/core/lens.py` (NEW - Phase 4)
- `ralphboost/core/controller.py` (NEW - Phase 4)

**Domains**
- `ralphboost/domains/base.py` (DONE)
- `ralphboost/domains/signal.py` (DONE)
- `ralphboost/domains/time_series.py` (DONE)
- `ralphboost/domains/transit.py` (DONE - Phase 3)

**Benchmarks + Notebooks**
- `benchmarks/kepler_homerun.py` (DONE - Phase 3 Benchmark)
- `benchmarks/credibility.py` (NEW - Phase 5)

**Tests**
- `tests/test_rfc_truth.py` (DONE)
- `tests/test_transit_domain.py` (DONE - Phase 3)
- `tests/test_multiscale_controller.py` (NEW - Phase 4)

---

## 6) "Definition of Done" (final acceptance checklist)

**RalphBoost is complete when:**

- ✅ **True Boosting Truth**: Monotonic loss, K-competition, Refiner ablation, Dedupe, Termination.
- ✅ **Kepler Planet Hunting**: TransitDomain works, PR-AUC lift (0.479 vs 0.008), Shuffle collapse.
- 🔲 **Generalization Credibility**: Non-Kepler domains work.
- 🔲 **Demo Greatness**: Ledger, Movie, Audit.

---

## 7) The "RalphBoost Advantage"

**Not:** "we beat XGBoost"
**Yes:** "we build interpretable scientific additive models by boosting hypothesis components"

The public story:
**"RalphBoost doesn’t just predict. It decomposes the signal into a ledger of physical explanations."**
