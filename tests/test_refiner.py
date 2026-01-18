import pytest
from ralphboost.refiners.optimization import OptimizationRefiner
from ralphboost.core.state import RalphState

def test_refiner_fallback():
    # Test that refiner works even if we pass dummy data
    refiner = OptimizationRefiner()
    candidates = [{"value": 1.0}]
    state = RalphState([1.0])
    
    refined = refiner.refine_batch(candidates, state)
    
    # Check it returns a list of same length
    assert len(refined) == len(candidates)
