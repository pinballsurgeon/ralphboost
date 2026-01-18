import math
from .base import Domain

class SignalDomain(Domain):
    def __init__(self, sample_rate=100.0):
        self.sample_rate = sample_rate

    def get_context(self):
        return {
            "domain_name": "Signal Decomposition",
            "schema": "Return JSON with {frequency (Hz), amplitude, phase (rad)}",
            "hint": f"Signal sample rate is {self.sample_rate} Hz."
        }

    def initialize_fitted(self, target):
        return [0.0] * len(target)

    def compute_residual(self, target, fitted):
        return [t - f for t, f in zip(target, fitted)]

    def energy(self, signal):
        return sum(x**2 for x in signal)

    def apply(self, component, current_fit, context=None):
        freq = component['frequency']
        amp = component['amplitude']
        phase = component['phase']
        
        n = len(current_fit)
        dt = 1.0 / self.sample_rate
        
        wave = [amp * math.sin(2 * math.pi * freq * (i * dt) + phase) for i in range(n)]
        
        return [c + w for c, w in zip(current_fit, wave)]
