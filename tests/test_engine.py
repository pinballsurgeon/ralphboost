import pytest
from ralphboost.core.engine import RalphEngine
from ralphboost.core.state import RalphState

class MockAgent:
    def propose(self, state, k=1, context=None):
        # Always propose "adding 1.0" to the signal
        return [{"value": 1.0}]

class MockRefiner:
    def refine_batch(self, candidates, state):
        # Pass through
        return candidates

class MockDomain:
    def initialize_fitted(self, target):
        return [0.0] * len(target)

    def select_best(self, candidates):
        return candidates[0]
    
    def apply(self, component, current_fit, context=None):
        # Add 1.0 to fit
        return [x + 1.0 for x in current_fit]
    
    def compute_residual(self, target, fitted):
        # residual = target - fitted
        if fitted is None:
            return target
        return [t - f for t, f in zip(target, fitted)]
    
    def energy(self, signal):
        return sum(x**2 for x in signal)

def test_wiggum_loop_termination():
    target = [10.0]
    agent = MockAgent()
    refiner = MockRefiner()
    domain = MockDomain()
    
    engine = RalphEngine(domain, agent, refiner)
    
    # Run loop for 5 iterations
    # Each iteration adds 1.0. Target is 10.0.
    # Iter 1: fit=[1.0], res=[9.0]
    # Iter 5: fit=[5.0], res=[5.0]
    result = engine.fit(target, max_iterations=5, min_residual_reduction=0.0)
    
    assert len(result.history) == 5
    assert result.final_residual[0] == 5.0

def test_state_immutability():
    initial = RalphState([10.0])
    updated = initial.update([1.0], [9.0], {"value": 1.0})
    
    assert initial.residual == [10.0]
    assert updated.residual == [9.0]
    assert len(initial.components) == 0
    assert len(updated.components) == 1
