import pytest

np = pytest.importorskip("numpy")

from ralphboost.agents.fft_agent import FFTSignalAgent
from ralphboost.core.state import RalphState


def test_fft_agent_identifies_peak_frequency():
    sample_rate = 100.0
    n = 1000
    freq = 5.0
    t = np.arange(n) / sample_rate
    signal = 2.0 * np.sin(2 * np.pi * freq * t + 0.3)

    state = RalphState(residual=signal)
    agent = FFTSignalAgent(sample_rate=sample_rate)

    candidates = agent.propose(state, k=1)

    assert candidates
    assert abs(candidates[0]["frequency"] - freq) < 0.2
    assert candidates[0]["amplitude"] > 1.5


def test_fft_agent_handles_zero_signal():
    signal = np.zeros(128)
    state = RalphState(residual=signal)
    agent = FFTSignalAgent(sample_rate=1.0)

    candidates = agent.propose(state, k=1)

    assert candidates == []
