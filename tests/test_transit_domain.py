import math
import pytest
from ralphboost.core.engine import RalphEngine
from ralphboost.core.objectives import MSEObjective
from ralphboost.domains.transit import TransitDomain
from ralphboost.refiners.optimization import OptimizationRefiner

def synth_transit_curve(n=1000, period=50.0, depth=10.0, noise=0.0):
    # Baseline + Periodic Dips
    out = []
    for i in range(n):
        val = 100.0 # Baseline
        # Dip?
        # Center at 25, 75, 125...
        # Phase = 25
        phase = 25.0
        width = 5.0
        
        dist = (i - phase + 0.5 * period) % period - 0.5 * period
        if abs(dist) < 0.5 * width:
            val -= depth
            
        if noise > 0:
            # simple pseudo-random
            val += ((i * 37) % 100 / 100.0 - 0.5) * noise
            
        out.append(val)
    return out

def test_transit_recovery():
    # Test 1: Synthetic periodic recovery
    period = 40.0
    signal = synth_transit_curve(n=400, period=period, depth=20.0, noise=0.1)
    
    domain = TransitDomain(min_period=10, max_period=100)
    refiner = OptimizationRefiner()
    engine = RalphEngine(
        domain=domain,
        refiner=refiner,
        objective=MSEObjective(),
        k_candidates=5,
        learning_rate=1.0, # Full fit
        verbose=1
    )
    
    result = engine.fit(signal, max_iterations=5, min_residual_reduction=None)
    
    # Check monotonic loss
    history = result.history
    for entry in history:
        assert entry["delta_loss"] > 0
        assert entry["loss_after"] < entry["loss_before"]
        
    # Check if we found a periodic train
    types = [c.get("type") for c in result.components]
    assert "periodic_train" in types or "baseline" in types
    
    # Find the periodic component
    periodic_comps = [c for c in result.components if c.get("type") == "periodic_train"]
    assert len(periodic_comps) > 0
    
    # Check period
    # We generated period=40.0
    found_period = periodic_comps[0]["period"]
    assert abs(found_period - period) < 2.0 # Tolerance

def test_dedupe_prevents_infinite_trains():
    # Test 2: Dedupe prevents infinite trains
    # Signal with ONE train
    period = 30.0
    signal = synth_transit_curve(n=300, period=period, depth=10.0)
    
    domain = TransitDomain(min_period=10, max_period=100)
    refiner = OptimizationRefiner()
    # High duplicate penalty
    engine = RalphEngine(
        domain=domain,
        refiner=refiner,
        k_candidates=5,
        duplicate_penalty=1000.0,
        duplicate_min_improvement=1.0
    )
    
    result = engine.fit(signal, max_iterations=10, min_residual_reduction=None)
    
    # We expect maybe 1 baseline, 1 periodic train.
    # Subsequent periodic trains should be rejected if they are duplicates.
    
    periodic_comps = [c for c in result.components if c.get("type") == "periodic_train"]
    # Should not have many periodic trains with same period
    periods = [c["period"] for c in periodic_comps]
    
    # Check if we have duplicates
    # With duplicate_penalty, we might accept if improvement is HUGE, but here residual should be gone.
    # So we expect few components.
    assert len(periodic_comps) <= 2 # Maybe one to fit, one to refine residual errors?

def test_refiner_ablation():
    # Test 3: Refiner ablation proves importance
    # Run WITH refiner
    period = 45.0
    signal = synth_transit_curve(n=400, period=period, depth=10.0, noise=1.0)
    
    domain = TransitDomain(min_period=10, max_period=100)
    
    # Run 1: With Refiner
    engine_ref = RalphEngine(
        domain=domain,
        refiner=OptimizationRefiner(),
        k_candidates=5
    )
    res_ref = engine_ref.fit(signal, max_iterations=3)
    loss_ref = res_ref.metrics["loss"]
    
    # Run 2: Without Refiner (pass None)
    # But wait, OptimizationRefiner delegates to domain.refine.
    # If we pass None, engine skips refinement step.
    # TransitDomain.propose returns candidates with depth=0.0!
    # So without refinement, depth stays 0.0?
    # If depth is 0.0, apply adds nothing (val = -depth*weight).
    # So delta_loss will be 0.0.
    # So engine should reject everything immediately (after baseline).
    # Baseline refinement is also done in refine().
    # Baseline propose returns intercept=0, slope=0.
    # So without refiner, NOTHING works except maybe random chance if propose returned non-zero?
    # TransitDomain.propose returns baseline 0,0 and periodic 0 depth.
    # So unrefined run should fail to improve at all after iteration 1 (if iteration 1 accepts 0,0?)
    # Actually, apply(0,0) does nothing. delta=0. reject.
    # So unrefined run should have 0 components accepted?
    
    engine_no_ref = RalphEngine(
        domain=domain,
        refiner=None,
        k_candidates=5
    )
    res_no_ref = engine_no_ref.fit(signal, max_iterations=3, min_residual_reduction=None)
    
    # Should have no components or only ineffective ones
    # Actually, if refiner is None, candidates are [baseline(0,0), periodic(p, depth=0), ...]
    # apply -> 0 change -> 0 delta -> reject.
    # So 0 components.
    assert len(res_no_ref.components) == 0
    assert len(res_ref.components) > 0
    assert loss_ref < res_no_ref.metrics["loss"]

def test_candidate_competition():
    # Test 4: Candidate competition matters
    # K=1 vs K=10
    # K=1 only proposes Baseline (since baseline is always first in list).
    # So K=1 will fit baseline, but maybe never see periodic train if periodic is candidate #2?
    # TransitDomain.propose returns [baseline, periodic_1, periodic_2 ...]
    # So K=1 -> only baseline.
    # K=5 -> baseline + periodic.
    # So K=5 should win.
    
    period = 50.0
    signal = synth_transit_curve(n=500, period=period, depth=20.0)
    
    domain = TransitDomain(min_period=10, max_period=100)
    refiner = OptimizationRefiner()
    
    # K=1
    engine_k1 = RalphEngine(domain=domain, refiner=refiner, k_candidates=1, learning_rate=1.0)
    res_k1 = engine_k1.fit(signal, max_iterations=3, min_residual_reduction=None)
    
    # K=5
    engine_k5 = RalphEngine(domain=domain, refiner=refiner, k_candidates=5, learning_rate=1.0)
    res_k5 = engine_k5.fit(signal, max_iterations=3, min_residual_reduction=None)
    
    # K1 should only have baseline
    types_k1 = [c["type"] for c in res_k1.components]
    # It might iterate: 1. baseline. 2. baseline (on residual).
    # Since K=1 always takes first proposed (baseline), it will never see periodic.
    assert "periodic_train" not in types_k1
    
    # K5 should have periodic
    types_k5 = [c["type"] for c in res_k5.components]
    assert "periodic_train" in types_k5
    
    assert res_k5.metrics["loss"] < res_k1.metrics["loss"]
