import os
import sys
import math

sys.path.append(os.getcwd())

try:
    import numpy as np
except ImportError:
    print("This example requires numpy. Install with `pip install ralphboost[fast]`.")
    raise SystemExit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    print("This example requires scikit-learn. Install with `pip install scikit-learn`.")
    raise SystemExit(1)

from ralphboost import RalphBooster
from ralphboost.domains.signal import SignalDomain


def generate_light_curve(rng, has_planet, length=400, period=50, depth=2.5, noise=0.5):
    t = np.arange(length, dtype=float)
    curve = rng.normal(0.0, noise, size=length)

    trend = 0.2 * np.sin(2 * math.pi * t / length)
    curve += trend

    if has_planet:
        dip_width = 2.0
        for center in range(0, length, period):
            dip = np.exp(-0.5 * ((t - center) / dip_width) ** 2)
            curve -= depth * dip

    return curve


def baseline_features(signal):
    s = np.asarray(signal, dtype=float)
    return np.array(
        [
            float(s.mean()),
            float(s.std()),
            float(s.min()),
            float(s.max()),
            float(np.median(s))
        ],
        dtype=float
    )


def ralph_features(signal, sample_rate=1.0, max_components=3):
    model = RalphBooster(
        domain=SignalDomain(sample_rate=sample_rate),
        agent_backend="fft",
        max_iterations=max_components,
        min_residual_reduction=0.0
    )

    result = model.fit(signal, sample_rate=sample_rate)
    comps = result.components

    freqs = [c["frequency"] for c in comps]
    amps = [c["amplitude"] for c in comps]

    freqs = (freqs + [0.0] * max_components)[:max_components]
    amps = (amps + [0.0] * max_components)[:max_components]

    residual = np.asarray(result.final_residual, dtype=float)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    var = float(np.var(residual))
    kurt = float(np.mean((residual - residual.mean()) ** 4) / (var ** 2 + 1e-8))
    dip_count = int((residual < -2.5).sum())
    dip_min = float(residual.min())

    return np.array(freqs + amps + [rms, kurt, dip_count, dip_min], dtype=float)


def build_dataset(n_samples=400, positive_rate=0.2, seed=42):
    rng = np.random.default_rng(seed)
    curves = []
    labels = []
    for _ in range(n_samples):
        has_planet = rng.random() < positive_rate
        curve = generate_light_curve(rng, has_planet)
        curves.append(curve)
        labels.append(1 if has_planet else 0)
    return np.array(curves, dtype=float), np.array(labels, dtype=int)


def evaluate_features(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    return roc, pr


def main():
    curves, labels = build_dataset()
    r_features = np.array([ralph_features(curve) for curve in curves])
    b_features = np.array([baseline_features(curve) for curve in curves])

    r_roc, r_pr = evaluate_features(r_features, labels)
    b_roc, b_pr = evaluate_features(b_features, labels)

    print("Exoplanet Synthetic Benchmark")
    print(f"Ralph features ROC-AUC: {r_roc:.3f} | PR-AUC: {r_pr:.3f}")
    print(f"Baseline features ROC-AUC: {b_roc:.3f} | PR-AUC: {b_pr:.3f}")


if __name__ == "__main__":
    main()
