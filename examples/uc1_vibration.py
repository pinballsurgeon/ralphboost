import sys
import os
import math
sys.path.append(os.getcwd())

from ralphboost import RalphBooster
from ralphboost.domains.signal import SignalDomain

def generate_engine_signal():
    """Simulates engine with worn bearing (modulated frequency)"""
    t = [i/1000 for i in range(2000)]  # 2 seconds @ 1kHz
    
    # Healthy components
    rotation_freq = 25.0  # 1500 RPM = 25 Hz
    signal = [
        10 * math.sin(2 * math.pi * rotation_freq * ti) +      # Main shaft
        3 * math.sin(2 * math.pi * rotation_freq * 2 * ti) +   # 2nd harmonic
        1.5 * math.sin(2 * math.pi * rotation_freq * 3 * ti)   # 3rd harmonic
        for ti in t
    ]
    
    # Fault signature: Amplitude modulation at bearing frequency
    bearing_fault = 7.3  # Hz (bearing cage frequency)
    for i, ti in enumerate(t):
        modulation = 0.5 * (1 + math.sin(2 * math.pi * bearing_fault * ti))
        signal[i] += 2 * modulation * math.sin(2 * math.pi * 150 * ti)  # Bearing resonance
    
    # Noise
    signal = [s + (hash(str(ti)) % 100 - 50) / 100 for s, ti in zip(signal, t)]
    
    return t, signal

if __name__ == "__main__":
    t, signal = generate_engine_signal()

    print("Running UC1: Fourier-Style Signal Decomposition")
    
    # Use 'mock' agent backend for reliable CI execution, or 'gemini' if key is set
    backend = 'gemini' if os.environ.get("GEMINI_API_KEY") else 'mock'
    print(f"Using Backend: {backend}")
    
    model = RalphBooster(
        domain=SignalDomain(sample_rate=1000),
        max_iterations=10,
        min_residual_reduction=0.01,
        agent_backend=backend
    )

    result = model.fit(signal, sample_rate=1000)
    
    print(f"Components found: {len(result.components)}")
    for i, c in enumerate(result.components):
        print(f"  {i+1}. Freq: {c['frequency']:.2f}, Amp: {c['amplitude']:.2f}")
