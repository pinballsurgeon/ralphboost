import sys
import os
import math
sys.path.append(os.getcwd())

from ralphboost import RalphBooster
from ralphboost.domains.time_series import TimeSeriesDomain

def generate_retail_sales():
    """Monthly sales with annual seasonality + holiday spikes + trend"""
    months = list(range(60))  # 5 years of monthly data
    
    # Long-term trend (exponential growth)
    trend = [100 * (1.02 ** m) for m in months]
    
    # Annual seasonality (summer peak, winter trough)
    seasonal = [20 * math.sin(2 * math.pi * m / 12 - math.pi/2) for m in months]
    
    # Holiday spikes (November-December)
    holiday = [50 if m % 12 in [10, 11] else 0 for m in months]
    
    # Economic cycle (3-year business cycle)
    cycle = [15 * math.sin(2 * math.pi * m / 36) for m in months]
    
    sales = [t + s + h + c + (hash(str(m)) % 20 - 10) 
             for t, s, h, c, m in zip(trend, seasonal, holiday, cycle, months)]
    
    return months, sales

if __name__ == "__main__":
    months, sales = generate_retail_sales()

    print("Running UC2: Economic Time Series Decomposition")
    backend = 'gemini' if os.environ.get("GEMINI_API_KEY") else 'mock'
    print(f"Using Backend: {backend}")
    
    model = RalphBooster(
        domain=TimeSeriesDomain(),
        max_iterations=8,
        learning_rate=0.8,
        agent_backend=backend
    )

    result = model.fit(sales)
    
    print(f"Components found: {len(result.components)}")
    for i, c in enumerate(result.components):
        # Pretty print component
        print(f"  {i+1}. {c}")
