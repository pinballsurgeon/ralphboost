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

    def propose(self, state, k=1, agent=None, context=None):
        k = max(1, int(k))
        residual = getattr(state, "residual", None)
        if residual is None:
            return []

        n = len(residual)
        if n < 2:
            return []

        dt = 1.0 / self.sample_rate if self.sample_rate else 1.0
        max_freq = (self.sample_rate / 2.0) if self.sample_rate else 0.5

        bins = min(max(32, 8 * k), max(1, n // 2), 256)
        freq_step = max_freq / bins if bins > 0 else max_freq

        scored = []
        max_power = 0.0
        for i in range(0, bins + 1):
            freq = i * freq_step
            omega = 2.0 * math.pi * freq * dt
            rs = 0.0
            rc = 0.0
            for t_idx, r in enumerate(residual):
                angle = omega * t_idx
                rs += r * math.sin(angle)
                rc += r * math.cos(angle)
            power = rs * rs + rc * rc
            if power > max_power:
                max_power = power
            scored.append((power, freq))

        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = []
        seen = set()
        bin_width = (self.sample_rate / n) if self.sample_rate else (1.0 / n)
        bin_width = max(bin_width, 1e-12)

        for power, freq in scored:
            if power <= 0.0:
                continue
            freq_bin = int(round(freq / bin_width))
            if freq_bin in seen:
                continue
            seen.add(freq_bin)
            confidence = (power / max_power) if max_power > 0.0 else 0.0
            candidates.append(
                {"frequency": float(freq), "amplitude": 1.0, "phase": 0.0, "confidence": float(confidence)}
            )
            if len(candidates) >= k:
                break

        if agent is not None:
            agent_candidates = agent.propose(state, k=k, context=context or self.get_context())
            for cand in agent_candidates:
                try:
                    freq = float(cand.get("frequency", 0.0))
                except Exception:
                    continue
                freq_bin = int(round(freq / bin_width))
                if freq_bin in seen:
                    continue
                seen.add(freq_bin)
                candidates.append(cand)
                if len(candidates) >= k:
                    break

        return candidates[:k]

    def refine(self, candidate, state):
        residual = getattr(state, "residual", None)
        if residual is None:
            return candidate

        try:
            base_freq = float(candidate.get("frequency", 0.0))
        except Exception:
            return candidate

        n = len(residual)
        if n == 0:
            return candidate

        nyquist = (self.sample_rate / 2.0) if self.sample_rate else 0.5
        base_freq = max(0.0, min(nyquist, base_freq))
        dt = 1.0 / self.sample_rate if self.sample_rate else 1.0
        resolution = (self.sample_rate / n) if self.sample_rate else (1.0 / n)
        span = max(2.0 * resolution, 0.0)

        def fit_at(freq: float):
            omega = 2.0 * math.pi * freq * dt
            sum_r2 = 0.0
            sum_s2 = 0.0
            sum_c2 = 0.0
            sum_sc = 0.0
            sum_rs = 0.0
            sum_rc = 0.0

            for t_idx, r in enumerate(residual):
                angle = omega * t_idx
                s = math.sin(angle)
                c = math.cos(angle)
                sum_r2 += r * r
                sum_s2 += s * s
                sum_c2 += c * c
                sum_sc += s * c
                sum_rs += r * s
                sum_rc += r * c

            det = (sum_s2 * sum_c2) - (sum_sc * sum_sc)
            if abs(det) < 1e-12:
                if sum_c2 <= 0.0:
                    return None
                a = 0.0
                b = sum_rc / sum_c2
            else:
                a = (sum_c2 * sum_rs - sum_sc * sum_rc) / det
                b = (-sum_sc * sum_rs + sum_s2 * sum_rc) / det

            sse = sum_r2 - (a * sum_rs + b * sum_rc)
            amp = math.sqrt(a * a + b * b)
            phase = math.atan2(b, a)

            return sse, amp, phase

        search_freqs = [base_freq]
        if span > 0.0:
            search_freqs = [
                base_freq - span,
                base_freq - 0.5 * span,
                base_freq,
                base_freq + 0.5 * span,
                base_freq + span,
            ]

        best = None
        for freq in search_freqs:
            freq = max(0.0, min(nyquist, freq))
            fit = fit_at(freq)
            if fit is None:
                continue
            sse, amp, phase = fit
            if best is None or sse < best[0]:
                best = (sse, freq, amp, phase)

        if best is None:
            return candidate

        _, freq, amp, phase = best
        refined = dict(candidate)
        refined.update({"frequency": float(freq), "amplitude": float(amp), "phase": float(phase)})
        return refined

    def complexity(self, component, state=None) -> float:
        return 1.0

    def fingerprint(self, component, state=None):
        try:
            freq = float(component.get("frequency", 0.0))
        except Exception:
            return None

        n = len(getattr(state, "residual", [])) if state is not None else 0
        if n <= 0:
            return ("sinusoid", round(freq, 6))

        bin_width = (self.sample_rate / n) if self.sample_rate else (1.0 / n)
        bin_width = max(bin_width, 1e-12)
        return ("sinusoid", int(round(freq / bin_width)))

    def apply(self, component, current_fit, context=None):
        freq = component['frequency']
        amp = component['amplitude']
        phase = component['phase']
        scale = float(component.get("weight", 1.0))
        if context and "scale" in context:
            scale *= float(context["scale"])
        
        n = len(current_fit)
        dt = 1.0 / self.sample_rate if self.sample_rate else 1.0
        
        amp_scaled = amp * scale
        wave = [amp_scaled * math.sin(2 * math.pi * freq * (i * dt) + phase) for i in range(n)]
        
        return [c + w for c, w in zip(current_fit, wave)]
