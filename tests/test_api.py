import pytest
from ralphboost import RalphBooster

def test_api_existence():
    assert RalphBooster is not None

def test_initialization():
    model = RalphBooster(
        max_iterations=10,
        learning_rate=0.1,
        thinking_budget=5.0
    )
    assert model.max_iterations == 10
    assert model.learning_rate == 0.1
    assert model.thinking_budget == 5.0

def test_sklearn_compatibility():
    model = RalphBooster()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert hasattr(model, "set_params")

def test_input_validation():
    with pytest.raises(ValueError):
        RalphBooster(max_iterations=0).fit([0.0])

    with pytest.raises(ValueError):
        RalphBooster(learning_rate=0.0).fit([0.0])

    with pytest.raises(ValueError):
        RalphBooster(k_candidates=0).fit([0.0])

    with pytest.raises(ValueError):
        RalphBooster(min_residual_reduction=2.0).fit([0.0])

    with pytest.raises(ValueError):
        RalphBooster(min_improvement=-1.0).fit([0.0])
