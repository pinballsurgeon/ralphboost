import pytest
from ralphboost.domains.time_series import TimeSeriesDomain

def test_trend_apply():
    domain = TimeSeriesDomain()
    component = {"type": "trend", "start": 100, "growth": 0.1}
    # t=0: 100
    # t=1: 110
    # t=2: 121
    current = [0.0] * 3
    new_fit = domain.apply(component, current)
    
    assert abs(new_fit[0] - 100.0) < 1e-6
    assert abs(new_fit[1] - 110.0) < 1e-6
    assert abs(new_fit[2] - 121.0) < 1e-6

def test_seasonality_apply():
    domain = TimeSeriesDomain()
    # Period 4, Amp 1
    component = {"type": "seasonality", "period": 4, "amplitude": 1.0, "phase": 0.0}
    current = [0.0] * 5
    new_fit = domain.apply(component, current)
    
    # t=0: sin(0) = 0
    # t=1: sin(pi/2) = 1
    # t=4: sin(2pi) = 0
    assert abs(new_fit[0]) < 1e-6
    assert abs(new_fit[1] - 1.0) < 1e-6
    assert abs(new_fit[4]) < 1e-6
