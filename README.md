# RalphBoost

Gradient boosting for iterative signal decomposition and basis discovery.
Uses an agentic loop (The Wiggum Loop) to discover components and numerical refinement to optimize them.

Architecture + RFC v1 implementation handoff: `RALPHBOOST_HANDOFF.md`.

## Installation

Core (Zero-Dependency):
```bash
pip install ralphboost
```

With Fast Backend (NumPy/SciPy):
```bash
pip install ralphboost[fast]
```

With Gemini Agent:
```bash
pip install ralphboost[gemini]
```

## Usage

### Signal Decomposition

```python
from ralphboost import RalphBooster
from ralphboost.domains.signal import SignalDomain

# Decompose a signal into frequency components
model = RalphBooster(domain=SignalDomain(sample_rate=100.0), max_iterations=10)
result = model.fit(signal)

print(result.components)
```

### Deterministic FFT Backend (fast)

Install the fast extras for a deterministic FFT-based agent:
```bash
pip install ralphboost[fast]
```

```python
from ralphboost import RalphBooster
from ralphboost.domains.signal import SignalDomain

model = RalphBooster(
    domain=SignalDomain(sample_rate=100.0),
    agent_backend="fft",
    max_iterations=5
)
result = model.fit(signal)
print(result.components)
```

### Time Series Decomposition

```python
from ralphboost import RalphBooster
from ralphboost.domains.time_series import TimeSeriesDomain

# Discover trend, seasonality, and cycles
model = RalphBooster(domain=TimeSeriesDomain())
model.fit(sales_data)
```

## Architecture

- **Core**: Pure Python boosting loop (`ralphboost.core`).
- **Domains**: Pluggable problem definitions (`ralphboost.domains`).
- **Agents**: LLM integration for component proposal (`ralphboost.agents`).
- **Refiners**: Numerical optimization for parameter tuning (`ralphboost.refiners`).

## License

MIT
