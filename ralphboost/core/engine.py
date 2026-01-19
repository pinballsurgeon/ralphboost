from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .objectives import MSEObjective, Objective
from .state import RalphState


def _energy(signal) -> float:
    return float(sum(float(x) ** 2 for x in signal))


class RalphResult:
    def __init__(self, state: RalphState):
        self.state = state
        self.history = state.history
        self.components = state.components
        self.final_residual = state.residual
        self.metrics = state.metrics
        self.final_fitted = state.fitted


class RalphEngine:
    def __init__(
        self,
        domain,
        agent=None,
        refiner=None,
        *,
        objective: Optional[Objective] = None,
        learning_rate: float = 0.1,
        k_candidates: int = 10,
        min_improvement: float = 0.0,
        early_stopping_rounds: Optional[int] = None,
        complexity_weight: float = 0.0,
        duplicate_penalty: float = 0.0,
        duplicate_min_improvement: Optional[float] = None,
        verbose: int = 0,
    ):
        self.domain = domain
        self.agent = agent
        self.refiner = refiner
        self.objective = objective or MSEObjective()
        self.learning_rate = float(learning_rate)
        self.k_candidates = int(k_candidates)
        self.min_improvement = float(min_improvement)
        self.early_stopping_rounds = early_stopping_rounds
        self.complexity_weight = float(complexity_weight)
        self.duplicate_penalty = float(duplicate_penalty)
        self.duplicate_min_improvement = duplicate_min_improvement
        self.verbose = int(verbose)

    def fit(self, target, max_iterations: int = 100, min_residual_reduction: Optional[float] = 0.01):
        fitted = self.domain.initialize_fitted(target)
        objective = self.objective

        loss = objective.loss(target, fitted)
        residual = objective.remainder(target, fitted)

        initial_energy = self.domain.energy(residual) if hasattr(self.domain, "energy") else _energy(residual)
        state = RalphState(
            residual=residual,
            fitted=fitted,
            metrics={
                "loss": loss,
                "residual_energy": initial_energy,
                "objective": getattr(objective, "name", "unknown"),
            },
        )

        no_improve_rounds = 0

        for _ in range(int(max_iterations)):
            iteration_start = time.perf_counter()
            loss_before = loss
            residual_before = residual

            context = self.domain.get_context() if hasattr(self.domain, "get_context") else {}
            candidates = []
            if hasattr(self.domain, "propose"):
                candidates = self.domain.propose(
                    state, k=self.k_candidates, agent=self.agent, context=context
                )
            elif self.agent is not None:
                candidates = self.agent.propose(state, k=self.k_candidates, context=context)

            if not candidates:
                break

            refined = candidates
            if self.refiner is not None:
                try:
                    refined = self.refiner.refine_batch(candidates, state, domain=self.domain)
                except TypeError:
                    refined = self.refiner.refine_batch(candidates, state)
            if not refined:
                break

            accepted_fps = set()
            if hasattr(self.domain, "fingerprint"):
                for comp in state.components:
                    fp = self.domain.fingerprint(comp, state)
                    if fp is not None:
                        accepted_fps.add(fp)

            best: Optional[dict] = None
            best_score: Optional[float] = None
            best_delta: float = 0.0
            best_loss_after: float = loss_before
            best_fitted: Any = fitted
            best_duplicate = False
            best_complexity = 0.0
            best_fp = None

            for component in refined:
                applied_component = dict(component)
                applied_component["weight"] = self.learning_rate

                new_fitted = self.domain.apply(applied_component, fitted)
                loss_after = objective.loss(target, new_fitted)
                delta = loss_before - loss_after
                if delta <= 0.0:
                    continue

                fp = (
                    self.domain.fingerprint(applied_component, state)
                    if hasattr(self.domain, "fingerprint")
                    else None
                )
                is_duplicate = fp is not None and fp in accepted_fps
                if is_duplicate and self.duplicate_min_improvement is not None and delta < self.duplicate_min_improvement:
                    continue

                complexity = (
                    self.domain.complexity(applied_component, state)
                    if hasattr(self.domain, "complexity")
                    else 0.0
                )
                score = delta - (self.complexity_weight * complexity) - (self.duplicate_penalty if is_duplicate else 0.0)

                if best_score is None or score > best_score:
                    best = applied_component
                    best_score = score
                    best_delta = delta
                    best_loss_after = loss_after
                    best_fitted = new_fitted
                    best_duplicate = is_duplicate
                    best_complexity = complexity
                    best_fp = fp

            if best is None:
                break

            residual = objective.remainder(target, best_fitted)
            current_energy = self.domain.energy(residual) if hasattr(self.domain, "energy") else _energy(residual)
            variance_explained = 1.0 - (current_energy / initial_energy) if initial_energy > 0.0 else 0.0
            runtime_ms = (time.perf_counter() - iteration_start) * 1000.0

            loss = best_loss_after
            metric_update = {
                "loss": loss,
                "residual_energy": current_energy,
                "variance_explained": variance_explained,
            }
            history_entry = {
                "loss_before": loss_before,
                "loss_after": loss,
                "delta_loss": best_delta,
                "learning_rate": self.learning_rate,
                "k_candidates": self.k_candidates,
                "duplicate": best_duplicate,
                "complexity": best_complexity,
                "runtime_ms": runtime_ms,
                "accepted_component": best,
            }

            state = state.update(best_fitted, residual, best, metric_update=metric_update, history_entry=history_entry)
            fitted = best_fitted

            if best_delta < self.min_improvement:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0

            if best_fp is not None:
                accepted_fps.add(best_fp)

            if (
                self.early_stopping_rounds is not None
                and self.early_stopping_rounds > 0
                and no_improve_rounds >= self.early_stopping_rounds
            ):
                break

            if min_residual_reduction is not None and initial_energy > 0.0:
                if (current_energy / initial_energy) < float(min_residual_reduction):
                    break

        return RalphResult(state)
