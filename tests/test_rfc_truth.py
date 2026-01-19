import math

from ralphboost.core.engine import RalphEngine
from ralphboost.core.objectives import LogisticObjective, MSEObjective
from ralphboost.domains.signal import SignalDomain
from ralphboost.refiners.optimization import OptimizationRefiner


def _sine(n: int, sample_rate: float, frequency: float, amplitude: float, phase: float):
    dt = 1.0 / sample_rate
    return [amplitude * math.sin(2.0 * math.pi * frequency * (i * dt) + phase) for i in range(n)]


def test_monotonic_loss_reduction_mse_signal_domain():
    n = 400
    sample_rate = 100.0
    target = _sine(n=n, sample_rate=sample_rate, frequency=5.0, amplitude=2.0, phase=0.7)

    domain = SignalDomain(sample_rate=sample_rate)
    engine = RalphEngine(
        domain=domain,
        agent=None,
        refiner=OptimizationRefiner(),
        objective=MSEObjective(),
        learning_rate=0.5,
        k_candidates=10,
    )

    result = engine.fit(target, max_iterations=8, min_residual_reduction=None)

    assert result.history
    for entry in result.history:
        assert entry["delta_loss"] > 0.0
        assert entry["loss_after"] < entry["loss_before"]


def test_k_candidate_selection_beats_k1():
    class ConstantStepDomain:
        def initialize_fitted(self, target):
            return [0.0] * len(target)

        def apply(self, component, current_fit, context=None):
            weight = float(component.get("weight", 1.0))
            value = float(component["value"])
            return [x + weight * value for x in current_fit]

        def energy(self, signal):
            return sum(float(x) ** 2 for x in signal)

        def propose(self, state, k=1, agent=None, context=None):
            candidates = [{"value": 1.0}, {"value": 10.0}]
            return candidates[: int(k)]

    target = [10.0]
    domain = ConstantStepDomain()

    k1 = RalphEngine(domain, agent=None, refiner=None, objective=MSEObjective(), learning_rate=1.0, k_candidates=1)
    k2 = RalphEngine(domain, agent=None, refiner=None, objective=MSEObjective(), learning_rate=1.0, k_candidates=2)

    r1 = k1.fit(target, max_iterations=1, min_residual_reduction=None)
    r2 = k2.fit(target, max_iterations=1, min_residual_reduction=None)

    assert r2.metrics["loss"] < r1.metrics["loss"]


def test_dedupe_skips_duplicate_when_improvement_too_small():
    class DuplicateConstantDomain:
        def initialize_fitted(self, target):
            return [0.0] * len(target)

        def apply(self, component, current_fit, context=None):
            weight = float(component.get("weight", 1.0))
            value = float(component["value"])
            return [x + weight * value for x in current_fit]

        def energy(self, signal):
            return sum(float(x) ** 2 for x in signal)

        def propose(self, state, k=1, agent=None, context=None):
            return [{"value": 1.0}][: int(k)]

        def fingerprint(self, component, state=None):
            return ("const",)

    target = [1.0]
    domain = DuplicateConstantDomain()

    engine = RalphEngine(
        domain,
        agent=None,
        refiner=None,
        objective=MSEObjective(),
        learning_rate=0.1,
        k_candidates=1,
        duplicate_min_improvement=0.02,
    )

    result = engine.fit(target, max_iterations=50, min_residual_reduction=None)

    assert len(result.history) == 9
    assert result.history[-1]["delta_loss"] >= 0.02


def test_ablation_without_refiner_stalls():
    n = 400
    sample_rate = 100.0
    target = _sine(n=n, sample_rate=sample_rate, frequency=5.0, amplitude=2.0, phase=1.0)

    domain = SignalDomain(sample_rate=sample_rate)

    refined = RalphEngine(
        domain=domain,
        agent=None,
        refiner=OptimizationRefiner(),
        objective=MSEObjective(),
        learning_rate=1.0,
        k_candidates=10,
    ).fit(target, max_iterations=3, min_residual_reduction=None)

    unrefined = RalphEngine(
        domain=domain,
        agent=None,
        refiner=None,
        objective=MSEObjective(),
        learning_rate=1.0,
        k_candidates=10,
    ).fit(target, max_iterations=3, min_residual_reduction=None)

    assert refined.metrics["loss"] < 1e-6
    assert unrefined.metrics["loss"] > 0.5


def test_logistic_objective_makes_progress():
    class BiasDomain:
        def initialize_fitted(self, target):
            return [0.0] * len(target)

        def apply(self, component, current_fit, context=None):
            weight = float(component.get("weight", 1.0))
            bias = float(component["bias"])
            return [x + weight * bias for x in current_fit]

        def energy(self, signal):
            return sum(float(x) ** 2 for x in signal)

        def propose(self, state, k=1, agent=None, context=None):
            return [{"bias": -2.0}, {"bias": 2.0}][: int(k)]

    y = [1.0, 1.0, 1.0, 0.0]
    domain = BiasDomain()

    engine = RalphEngine(
        domain=domain,
        agent=None,
        refiner=None,
        objective=LogisticObjective(),
        learning_rate=1.0,
        k_candidates=2,
    )
    result = engine.fit(y, max_iterations=1, min_residual_reduction=None)

    assert result.history
    assert result.history[0]["delta_loss"] > 0.0
