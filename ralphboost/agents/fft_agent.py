try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .base import Agent


class FFTSignalAgent(Agent):
    """
    Deterministic agent that proposes dominant sinusoidal components via FFT.
    Requires numpy; install with ralphboost[fast].
    """

    def __init__(self, sample_rate=1.0):
        self.sample_rate = float(sample_rate) if sample_rate else 1.0

    def propose(self, state, k=1, context=None):
        if not NUMPY_AVAILABLE:
            raise ImportError("FFTSignalAgent requires numpy. Install with `pip install ralphboost[fast]`.")

        residual = np.asarray(state.residual, dtype=float)
        n = residual.size
        if n < 2:
            return []

        sample_rate = self.sample_rate if self.sample_rate > 0 else 1.0
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        spectrum = np.fft.rfft(residual)
        mags = np.abs(spectrum)

        if mags.size == 0:
            return []
        mags[0] = 0.0
        if not np.any(mags):
            return []

        k = max(1, int(k))
        k = min(k, mags.size - 1) if mags.size > 1 else 1

        if k == 1:
            idxs = [int(np.argmax(mags))]
        else:
            idxs = np.argpartition(mags, -k)[-k:]
            idxs = idxs[np.argsort(mags[idxs])[::-1]]

        max_mag = mags[idxs[0]] if idxs else 0.0
        candidates = []
        for idx in idxs:
            freq = float(freqs[idx])
            coeff = spectrum[idx]
            amp = float((2.0 * np.abs(coeff)) / n)
            phase = float(np.angle(coeff) + (np.pi / 2.0))
            confidence = float(mags[idx] / max_mag) if max_mag > 0 else 0.0
            candidates.append(
                {
                    "frequency": freq,
                    "amplitude": amp,
                    "phase": phase,
                    "confidence": confidence
                }
            )

        return candidates[:k]
