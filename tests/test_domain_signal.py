import pytest
import math
from ralphboost.domains.signal import SignalDomain
from ralphboost.agents.mock import MockSignalAgent

def test_signal_energy():
    domain = SignalDomain()
    signal = [1, 1, 1, 1]
    # energy = 1+1+1+1 = 4
    assert domain.energy(signal) == 4.0

def test_signal_apply():
    domain = SignalDomain(sample_rate=1.0)
    # Component: 1Hz, Amp 1, Phase 0
    # t = 0, 1, 2, 3
    # sin(2*pi*1*t) -> sin(0), sin(2pi), ... -> all 0
    # Let's use 0.25Hz -> period 4
    # t=0: 0
    # t=1: sin(pi/2) = 1
    # t=2: sin(pi) = 0
    # t=3: sin(3pi/2) = -1
    
    component = {"frequency": 0.25, "amplitude": 1.0, "phase": 0.0}
    current_fit = [0.0] * 4
    
    new_fit = domain.apply(component, current_fit)
    
    # Allow float precision
    assert abs(new_fit[0] - 0.0) < 1e-6
    assert abs(new_fit[1] - 1.0) < 1e-6
    assert abs(new_fit[2] - 0.0) < 1e-6
    assert abs(new_fit[3] - (-1.0)) < 1e-6

def test_mock_agent_protocol():
    agent = MockSignalAgent()
    state = None # Mock doesn't use state
    candidates = agent.propose(state)
    
    assert len(candidates) == 1
    c = candidates[0]
    assert "frequency" in c
    assert "amplitude" in c
    assert "phase" in c
