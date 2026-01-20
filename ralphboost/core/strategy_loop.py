from __future__ import annotations
import time
from typing import Optional, List, Dict, Any

from .state import RalphState
from .engine import RalphResult, _energy
from .objectives import Objective, MSEObjective

class StrategyLoop:
    """
    Layer C: Agent Strategy Controller (RalphBoost v3.5).
    Orchestrates the boosting process using an Agent to propose 'Blueprints'.
    """
    def __init__(
        self,
        domain,
        agent,
        refiner,
        objective: Optional[Objective] = None,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        min_improvement: float = 0.0,
        verbose: int = 0
    ):
        self.domain = domain
        self.agent = agent
        self.refiner = refiner
        self.objective = objective or MSEObjective()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.verbose = verbose

    def fit(self, target, max_iterations: Optional[int] = None) -> RalphResult:
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # Initialize
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
        
        for i in range(max_iterations):
            iteration_start = time.perf_counter()
            loss_before = loss
            
            # 1. Agent Strategy (Blueprint)
            context = {
                "iteration": i,
                "domain_name": self.domain.__class__.__name__,
                "hint": "Focus on dominant structure."
            }
            # Add domain context
            if hasattr(self.domain, "get_context"):
                context.update(self.domain.get_context())
            
            blueprint = self.agent.propose_blueprints(state, context=context)
            
            # 2. Extract Candidates from Blueprint
            raw_candidates = blueprint.get("candidates", [])
            focus = blueprint.get("focus", {})
            constraints = blueprint.get("constraints", {})
            
            candidates = []
            for c in raw_candidates:
                # Universal Translator for Agent Hallucinations
                # Map 'family'/'model_type'/'component' -> 'type'
                for key in ["family", "model_type", "component"]:
                    if key in c and "type" not in c:
                        c["type"] = c[key]
                
                # Map common parameter hallucinations
                # period_guess -> period, duration_guess -> width
                if "period_guess" in c: c["period"] = c.pop("period_guess")
                if "duration_guess" in c: c["width"] = c.pop("duration_guess")
                if "degree" in c: c["type"] = "baseline" # Polynomial -> Baseline

                # Flatten parameters if nested
                if "parameters" in c:
                    params = c.pop("parameters")
                    if isinstance(params, dict):
                        c.update(params)
                
                candidates.append(c)

            if self.verbose:
                print(f"Iteration {i}: Agent proposed {len(candidates)} candidates.")
                for c in candidates:
                    print(f"  - {c.get('type', 'unknown')}: {c}")

            # Inject focus/window info into candidates if needed
            # (For now, we just pass candidates. Future: handle window slicing)
            if not candidates:
                if self.verbose:
                    print(f"Iteration {i}: No candidates proposed.")
                break

            # 3. Deterministic Refine (Layer B)
            refined = candidates
            if self.refiner:
                # We might need to pass constraints/focus to refiner?
                # Currently refiner signature is (candidates, state, domain)
                try:
                    refined = self.refiner.refine_batch(candidates, state, domain=self.domain)
                except TypeError:
                     refined = self.refiner.refine_batch(candidates, state)

            if not refined:
                if self.verbose:
                    print(f"Iteration {i}: All candidates rejected by refiner.")
                break

            # 4. Score and Select (Layer A Logic)
            best_component = None
            best_score = None
            best_fitted = None
            best_loss_after = None
            best_delta = 0.0

            complexity_weight = constraints.get("complexity_penalty", 0.0)

            for comp in refined:
                comp_applied = dict(comp)
                comp_applied["weight"] = self.learning_rate
                
                new_fitted = self.domain.apply(comp_applied, fitted)
                loss_after = objective.loss(target, new_fitted)
                delta = loss_before - loss_after
                
                # Monotonicity check
                if delta <= 0.0:
                    continue
                    
                # Complexity
                complexity = 0.0
                if hasattr(self.domain, "complexity"):
                    complexity = self.domain.complexity(comp_applied, state)
                
                score = delta - (complexity_weight * complexity)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_component = comp_applied
                    best_fitted = new_fitted
                    best_loss_after = loss_after
                    best_delta = delta

            if best_component is None:
                if self.verbose:
                    print(f"Iteration {i}: No candidate improved objective.")
                break

            # 5. Accept and Update
            residual = objective.remainder(target, best_fitted)
            current_energy = self.domain.energy(residual) if hasattr(self.domain, "energy") else _energy(residual)
            
            runtime_ms = (time.perf_counter() - iteration_start) * 1000.0
            
            history_entry = {
                "iteration": i,
                "loss_before": loss_before,
                "loss_after": best_loss_after,
                "delta_loss": best_delta,
                "blueprint_focus": focus,
                "accepted_component": best_component,
                "runtime_ms": runtime_ms
            }
            
            metric_update = {
                "loss": best_loss_after,
                "residual_energy": current_energy
            }
            
            state = state.update(best_fitted, residual, best_component, metric_update=metric_update, history_entry=history_entry)
            fitted = best_fitted
            loss = best_loss_after
            
            if best_delta < self.min_improvement:
                if self.verbose:
                    print(f"Iteration {i}: Improvement {best_delta} < min {self.min_improvement}")
                break

        return RalphResult(state)
