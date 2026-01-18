from .state import RalphState

class RalphResult:
    def __init__(self, state: RalphState):
        self.state = state
        self.history = state.history
        self.components = state.components
        self.final_residual = state.residual
        self.metrics = state.metrics

class RalphEngine:
    def __init__(self, domain, agent, refiner):
        self.domain = domain
        self.agent = agent
        self.refiner = refiner
    
    def fit(self, target, max_iterations=100, min_residual_reduction=0.01):
        fitted = self.domain.initialize_fitted(target)
        state = RalphState(residual=target, fitted=fitted)
        initial_energy = self.domain.energy(target)
        
        for i in range(max_iterations):
            # 1. Speculative Discovery
            candidates = self.agent.propose(state, k=1)
            if not candidates:
                break
            
            # 2. Refinement
            refined = self.refiner.refine_batch(candidates, state)
            if not refined:
                break
            
            # 3. Selection
            best_component = self.domain.select_best(refined)
            
            # 4. Update
            new_fitted = self.domain.apply(best_component, state.fitted)
            new_residual = self.domain.compute_residual(target, new_fitted)
            
            current_energy = self.domain.energy(new_residual)
            variance_explained = 1 - (current_energy / initial_energy) if initial_energy > 0 else 0
            
            state = state.update(new_fitted, new_residual, best_component, 
                               metric_update={"variance_explained": variance_explained, "residual_energy": current_energy})
            
            # Check termination (if we are close enough)
            # min_residual_reduction here used as threshold for remaining energy? 
            # Or delta? 
            # The feedback said "residual < threshold" or "reduction < threshold".
            # The code I wrote checks variance explained >= (1 - threshold).
            # If min_residual_reduction=0.01 (1%), we stop when we explain 99%.
            
            if min_residual_reduction is not None and (1 - variance_explained) < min_residual_reduction:
                break
                
        return RalphResult(state)
