# Monte Carlo Pi

Goal
Implement `estimate_pi(samples: int, seed: int) -> float`.

Rules
- Deterministic given seed.
- Use random.Random(seed).

Tests
- samples=10, seed=1 -> approx 3.2 (tolerance 0.8)
- samples=1000, seed=2 -> approx 3.14 (tolerance 0.2)

Hidden checks
- samples=0 returns 0.0
