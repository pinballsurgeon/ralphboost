import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ralphboost.domains.transit import TransitDomain
from ralphboost.agents.gemini import GeminiAgent
from ralphboost.refiners.optimization import OptimizationRefiner
from ralphboost.core.strategy_loop import StrategyLoop

# Load Env
try:
    with open(".env.local") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
except Exception as e:
    print(f"Warning: Could not read .env.local: {e}")

# Load Data
print("Loading exoTest.csv...")
try:
    df = pd.read_csv("exoTest.csv", nrows=5)
except FileNotFoundError:
    print("exoTest.csv not found.")
    sys.exit(1)

target_row = df.iloc[0]
label = target_row["LABEL"]
print(f"Target Label: {label} (2=Planet, 1=Non-Planet)")

# Extract Flux
flux = target_row.filter(like="FLUX").values.astype(float)

# Preprocessing (Robust)
# Fill NaNs
flux = np.nan_to_num(flux)
# Median center
median = np.median(flux)
flux = flux - median
# MAD scale
mad = np.median(np.abs(flux))
if mad > 1e-9:
    flux = flux / mad
    print(f"Signal scaled by MAD={mad:.4f}")
else:
    print("Signal const/zero variance.")

print(f"Flux shape: {flux.shape}")

# Setup Components
domain = TransitDomain(min_period=10, max_period=500)
# Use hardened GeminiAgent
# Note: fail_hard=True to verify connection explicitly first
agent = GeminiAgent(model="gemini-flash-latest", fail_hard=True)
refiner = OptimizationRefiner()

# Verify Connection
print("Verifying Gemini Connection...")
try:
    res = agent.validate_connection()
    print("Connection OK:", res)
except Exception as e:
    print(f"Connection FAILED: {e}")
    # We proceed anyway, StrategyLoop might handle it or fail if agent is critical
    agent.fail_hard = False # Don't crash loop

# Setup Loop
loop = StrategyLoop(
    domain=domain,
    agent=agent,
    refiner=refiner,
    max_iterations=5, # Start small as requested
    min_improvement=0.01,
    verbose=1
)

# Run
print("Starting Strategy Loop...")
result = loop.fit(flux)

# Report
print("\n--- Final Report ---")
loss = result.metrics.get('loss', 0.0)
energy = result.metrics.get('residual_energy', 0.0)
print(f"Final Loss: {loss:.4f}")
print(f"Residual Energy: {energy:.4f}")
print(f"Components Found: {len(result.components)}")

for i, comp in enumerate(result.components):
    ctype = comp.get('type', 'unknown')
    # Format params for readability
    params = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in comp.items() if k not in ['type']}
    print(f"[{i}] {ctype.upper()}: {params}")

print("Done.")
