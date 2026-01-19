# ============================================================
# 🪐 RalphBoost Kepler Notebook Re-run (TransitDomain Edition)
# ============================================================
import os, sys, json, time, math, random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------
# 0) Install / upgrade RalphBoost
# ---------------------------
# !pip -q install -U git+https://github.com/pinballsurgeon/ralphboost.git

import ralphboost
print("ralphboost version import OK:", ralphboost.__file__)

# ---------------------------
# 1) Locate dataset
# ---------------------------
train_path = "exoTrain.csv"
test_path  = "exoTest.csv"

# Mock data generation if missing (for demo purposes)
def generate_mock_data(path, n=100):
    print(f"Generating mock {path}...")
    # 2=planet, 1=non-planet
    # 3198 columns
    cols = ["LABEL"] + [f"FLUX.{i+1}" for i in range(3197)]
    data = []
    for i in range(n):
        is_planet = (i % 10 == 0)
        label = 2 if is_planet else 1
        flux = np.random.normal(0, 1, 3197)
        if is_planet:
            # Inject dip
            flux[100:105] -= 10.0
        row = [label] + flux.tolist()
        data.append(row)
    pd.DataFrame(data, columns=cols).to_csv(path, index=False)

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    print("[WARN] exoTrain.csv / exoTest.csv not found. Generating MOCK data for demo.")
    generate_mock_data(train_path, n=500)
    generate_mock_data(test_path, n=100)
else:
    print(f"[OK] Found train: {train_path}  test: {test_path}")

# ---------------------------
# 2) Load Kepler data
# ---------------------------
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Train shape:", train.shape, "Test shape:", test.shape)

# labels: 1 = non-planet, 2 = planet  (classic Kepler-labelled dataset)
y_train_raw = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values.astype(np.float32)

y_test_raw = test.iloc[:, 0].values
X_test = test.iloc[:, 1:].values.astype(np.float32)

# map: 2 -> 1 (planet), 1 -> 0 (nonplanet)
y_train = (y_train_raw == 2).astype(int)
y_test  = (y_test_raw == 2).astype(int)

pos_rate = float(y_train.mean())
print("[OK] Positive rate (train):", pos_rate)

# ---------------------------
# 3) Robust scale curves (median / MAD)
# ---------------------------
def robust_scale_rows(X):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + 1e-6
    return (X - med) / mad

X_train_s = robust_scale_rows(X_train)
X_test_s  = robust_scale_rows(X_test)
print("[OK] Robust scaled.")

# ---------------------------
# 4) Build TransitDomain feature extractor using RalphBoost engine
# ---------------------------
from ralphboost import RalphBooster
from ralphboost.domains.transit import TransitDomain

def ralph_transit_features(X, max_iters=12, k_candidates=8, lr=0.8):
    domain = TransitDomain(min_period=10, max_period=200)
    feats = []
    ledgers = []

    for i in range(X.shape[0]):
        curve = X[i].tolist()

        model = RalphBooster(
            domain=domain,
            agent_backend="mock",      # deterministic; agents optional
            objective="mse",           # remainder-driven
            k_candidates=k_candidates,
            learning_rate=lr,
            max_iterations=max_iters,
            min_residual_reduction=None,
        )

        res = model.fit(curve)
        comps = res.components

        # ledger stats (minimal, deterministic, explainable)
        # UPDATED: Use snake_case names as per implementation
        periodic = [c for c in comps if c.get("type") == "periodic_train"]
        local    = [c for c in comps if c.get("type") == "local_dip"]
        trend    = [c for c in comps if c.get("type") == "baseline"]

        # summarize strongest periodic signal if present
        if periodic:
            # assume best periodic is earliest accepted or largest depth
            best_p = max(periodic, key=lambda c: abs(c.get("depth", 0.0)))
            p_period = float(best_p.get("period", 0.0))
            p_depth  = float(best_p.get("depth", 0.0))
            p_width  = float(best_p.get("width", 0.0))
        else:
            p_period = 0.0
            p_depth  = 0.0
            p_width  = 0.0

        # local dips: count + max depth
        if local:
            l_count = float(len(local))
            l_depth = float(max(abs(c.get("depth", 0.0)) for c in local))
        else:
            l_count = 0.0
            l_depth = 0.0

        # trend magnitude
        if trend:
            t_slope = float(trend[-1].get("slope", 0.0))
        else:
            t_slope = 0.0

        # residual energy / loss
        final_loss = float(res.metrics.get("loss", 0.0))
        resid_energy = float(res.metrics.get("residual_energy", 0.0))

        row = [
            p_period, p_depth, p_width,
            l_count, l_depth,
            t_slope,
            final_loss, resid_energy,
            float(len(comps)),
        ]
        feats.append(row)

        # store a mini-ledger for inspection if needed
        ledgers.append(comps[:6])

        if (i+1) % 100 == 0:
            print(f"  ...features baked for {i+1}/{X.shape[0]}")

    feats = np.array(feats, dtype=np.float32)
    return feats, ledgers

print("\n[INFO] Baking RalphBoost TransitDomain features (train/test)...")
t0 = time.time()
# Reduce iterations/size for quick demo
F_train, ledger_train = ralph_transit_features(X_train_s[:100], max_iters=5, k_candidates=5)
F_test,  ledger_test  = ralph_transit_features(X_test_s[:50],  max_iters=5, k_candidates=5)
y_train_sub = y_train[:100]
y_test_sub = y_test[:50]

print("[OK] Feature bake done in", round(time.time() - t0, 2), "sec")
print("Feature matrix:", F_train.shape, F_test.shape)

# ---------------------------
# 5) Train a tiny classifier (just for evaluation)
# ---------------------------
if len(np.unique(y_train_sub)) > 1:
    clf = LogisticRegression(max_iter=2000, class_weight=None)
    clf.fit(F_train, y_train_sub)

    probs = clf.predict_proba(F_test)[:, 1]
    
    try:
        roc = roc_auc_score(y_test_sub, probs)
        pr  = average_precision_score(y_test_sub, probs)
        print("\n[RESULTS] TransitDomain RalphBoost features -> LogisticRegression")
        print("  ROC-AUC:", roc)
        print("  PR-AUC :", pr)
    except Exception as e:
        print("Metrics error:", e)

    # Top-K audit
    order = np.argsort(-probs)
    best_idx = int(order[0])
    print("\n[AUDIT] Best predicted sample index:", best_idx, "True label:", int(y_test_sub[best_idx]), "Score:", float(probs[best_idx]))
    print("Top ledger components (first 6):")
    print(json.dumps(ledger_test[best_idx], indent=2))

else:
    print("Not enough positive samples in subset to train classifier.")
