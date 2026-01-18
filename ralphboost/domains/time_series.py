from .base import Domain
import math

class TimeSeriesDomain(Domain):
    def get_context(self):
        return {
            "domain_name": "Time Series Decomposition",
            "schema": "Return JSON with {type: 'trend'|'seasonality'|'cycle'|'holiday', ... params}",
            "hint": "Analyze the residual for temporal patterns."
        }

    def initialize_fitted(self, target):
        return [0.0] * len(target)

    def compute_residual(self, target, fitted):
        return [t - f for t, f in zip(target, fitted)]

    def energy(self, signal):
        return sum(x**2 for x in signal)

    def apply(self, component, current_fit, context=None):
        # Component types: 'trend', 'seasonality', 'cycle', 'holiday'
        ctype = component.get('type', 'seasonality')
        n = len(current_fit)
        
        wave = [0.0] * n
        
        if ctype == 'trend':
            # e.g. exponential: a * (1 + r)^t
            a = component.get('start', 0)
            r = component.get('growth', 0)
            wave = [a * ((1 + r) ** i) for i in range(n)]
            
        elif ctype == 'seasonality' or ctype == 'cycle':
            # sin wave: A * sin(2pi * t / period + phase)
            period = component.get('period', 12)
            amp = component.get('amplitude', 1)
            phase = component.get('phase', 0)
            if period == 0: period = 1 # avoid div zero
            wave = [amp * math.sin(2 * math.pi * i / period + phase) for i in range(n)]
            
        elif ctype == 'holiday':
            # Spikes at specific months
            months = component.get('months', [])
            val = component.get('value', 0)
            # Assuming t is month index?
            wave = [val if (i % 12) in months else 0 for i in range(n)]
            
        return [c + w for c, w in zip(current_fit, wave)]
