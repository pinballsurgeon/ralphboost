from .base import Domain
import math

class TimeSeriesDomain(Domain):
    def __init__(self, max_period: int | None = None):
        self.max_period = max_period

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

    def propose(self, state, k=1, agent=None, context=None):
        k = max(1, int(k))
        residual = getattr(state, "residual", None)
        if residual is None:
            return []

        n = len(residual)
        if n < 2:
            return []

        candidates = [{"type": "linear_trend", "intercept": 0.0, "slope": 0.0, "confidence": 1.0}]

        max_lag = self.max_period if self.max_period is not None else min(60, max(2, n // 2))
        max_lag = max(2, min(max_lag, n - 1))

        lag_scores = []
        max_score = 0.0
        for lag in range(2, max_lag + 1):
            score = 0.0
            for i in range(n - lag):
                score += residual[i] * residual[i + lag]
            score = abs(score)
            if score > max_score:
                max_score = score
            lag_scores.append((score, lag))

        lag_scores.sort(key=lambda x: x[0], reverse=True)

        seen_periods = set()
        for score, lag in lag_scores:
            if score <= 0.0:
                continue
            if lag in seen_periods:
                continue
            seen_periods.add(lag)
            confidence = (score / max_score) if max_score > 0.0 else 0.0
            candidates.append({"type": "seasonality", "period": int(lag), "amplitude": 1.0, "phase": 0.0, "confidence": confidence})
            if len(candidates) >= k:
                break

        if agent is not None and len(candidates) < k:
            agent_candidates = agent.propose(state, k=k, context=context or self.get_context())
            for cand in agent_candidates:
                candidates.append(cand)
                if len(candidates) >= k:
                    break

        return candidates[:k]

    def refine(self, candidate, state):
        residual = getattr(state, "residual", None)
        if residual is None:
            return candidate

        n = len(residual)
        if n == 0:
            return candidate

        ctype = candidate.get("type", "seasonality")
        if ctype == "linear_trend":
            sum_i = (n - 1) * n / 2.0
            sum_i2 = (n - 1) * n * (2 * n - 1) / 6.0
            sum_r = float(sum(residual))
            sum_ir = 0.0
            for i, r in enumerate(residual):
                sum_ir += i * r

            denom = (n * sum_i2) - (sum_i * sum_i)
            if abs(denom) < 1e-12:
                slope = 0.0
                intercept = sum_r / n
            else:
                slope = ((n * sum_ir) - (sum_i * sum_r)) / denom
                intercept = (sum_r - slope * sum_i) / n

            refined = dict(candidate)
            refined.update({"intercept": float(intercept), "slope": float(slope)})
            return refined

        if ctype in {"seasonality", "cycle"}:
            try:
                period = int(candidate.get("period", 12))
            except Exception:
                return candidate
            if period <= 0:
                return candidate

            omega = 2.0 * math.pi / period
            sum_s2 = 0.0
            sum_c2 = 0.0
            sum_sc = 0.0
            sum_rs = 0.0
            sum_rc = 0.0

            for t_idx, r in enumerate(residual):
                angle = omega * t_idx
                s = math.sin(angle)
                c = math.cos(angle)
                sum_s2 += s * s
                sum_c2 += c * c
                sum_sc += s * c
                sum_rs += r * s
                sum_rc += r * c

            det = (sum_s2 * sum_c2) - (sum_sc * sum_sc)
            if abs(det) < 1e-12:
                if sum_c2 <= 0.0:
                    return candidate
                a = 0.0
                b = sum_rc / sum_c2
            else:
                a = (sum_c2 * sum_rs - sum_sc * sum_rc) / det
                b = (-sum_sc * sum_rs + sum_s2 * sum_rc) / det

            amp = math.sqrt(a * a + b * b)
            phase = math.atan2(b, a)

            refined = dict(candidate)
            refined.update({"amplitude": float(amp), "phase": float(phase), "period": int(period)})
            return refined

        return candidate

    def complexity(self, component, state=None) -> float:
        ctype = component.get("type", "seasonality")
        if ctype == "holiday":
            return 2.0
        return 1.0

    def fingerprint(self, component, state=None):
        ctype = component.get("type", "seasonality")
        if ctype in {"seasonality", "cycle"}:
            return (ctype, int(component.get("period", 0)))
        if ctype == "linear_trend":
            return ("linear_trend",)
        return (ctype,)

    def apply(self, component, current_fit, context=None):
        # Component types: 'trend', 'seasonality', 'cycle', 'holiday'
        ctype = component.get('type', 'seasonality')
        n = len(current_fit)
        scale = float(component.get("weight", 1.0))
        if context and "scale" in context:
            scale *= float(context["scale"])
        
        wave = [0.0] * n
        
        if ctype == 'trend':
            # e.g. exponential: a * (1 + r)^t
            a = component.get('start', 0)
            r = component.get('growth', 0)
            wave = [a * ((1 + r) ** i) for i in range(n)]

        elif ctype == "linear_trend":
            intercept = component.get("intercept", 0.0)
            slope = component.get("slope", 0.0)
            wave = [intercept + slope * i for i in range(n)]
            
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

        if scale != 1.0:
            wave = [scale * w for w in wave]

        return [c + w for c, w in zip(current_fit, wave)]
