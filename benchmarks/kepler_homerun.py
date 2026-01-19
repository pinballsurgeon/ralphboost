# ============================================================
# 🪐 RalphBoost Kepler "HOME RUN" Benchmark Cell (TransitDomain)
# - Full bake + real metrics + Top-K audit + label-shuffle gate
# - Caches baked features so you can iterate fast after first run
# ============================================================
import os, time, json, math, random
import numpy as np
import pandas as pd

# ---------------------------
# 0) Config (EDIT THESE)
# ---------------------------
TRAIN_PATH = "exoTrain.csv"
TEST_PATH  = "exoTest.csv"

# Speed controls (set to None for full)
MAX_TRAIN = 2000   # e.g. 2000 for fast-ish, None for all 5087
MAX_TEST  = 570    # 570 is full test set

# RalphBoost controls
MAX_ITERS     = 8
K_CANDIDATES  = 10
LEARNING_RATE = 0.8
MIN_PERIOD    = 10
MAX_PERIOD    = 200

# Caching
CACHE_DIR = "./rb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Determinism
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

print("=== CONFIG ===")
print("MAX_TRAIN:", MAX_TRAIN, " MAX_TEST:", MAX_TEST)
print("MAX_ITERS:", MAX_ITERS, " K:", K_CANDIDATES, " LR:", LEARNING_RATE)
print("Period:", (MIN_PERIOD, MAX_PERIOD))
print("CACHE_DIR:", CACHE_DIR)

# ---------------------------
# 1) Install / upgrade RalphBoost
# ---------------------------
# !pip -q install -U git+https://github.com/pinballsurgeon/ralphboost.git

import ralphboost
print("ralphboost import OK:", ralphboost.__file__)

from ralphboost import RalphBooster
from ralphboost.domains.transit import TransitDomain

# ---------------------------
# 2) Load Kepler dataset
# ---------------------------
if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
    # Mock generation for demo if missing
    print(f"[WARN] {TRAIN_PATH} missing. Generating mock data.")
    def generate_mock(path, n):
        cols = ["LABEL"] + [f"FLUX.{i+1}" for i in range(3197)]
        data = []
        for i in range(n):
            label = 2 if i % 10 == 0 else 1
            row = [label] + np.random.normal(0, 1, 3197).tolist()
            data.append(row)
        pd.DataFrame(data, columns=cols).to_csv(path, index=False)
    generate_mock(TRAIN_PATH, MAX_TRAIN or 500)
    generate_mock(TEST_PATH, MAX_TEST or 100)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape, "Test shape:", test.shape)

y_train_raw = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values.astype(np.float32)

y_test_raw = test.iloc[:, 0].values
X_test = test.iloc[:, 1:].values.astype(np.float32)

# labels: 2=planet, 1=nonplanet -> map to 1/0
y_train = (y_train_raw == 2).astype(int)
y_test  = (y_test_raw  == 2).astype(int)

print("Positive rate train:", float(y_train.mean()))
print("Positive rate test :", float(y_test.mean()))

# optionally subsample for speed
def maybe_subsample(X, y, max_n):
    if max_n is None or X.shape[0] <= max_n:
        return X, y
    idx = np.random.RandomState(SEED).choice(X.shape[0], size=max_n, replace=False)
    return X[idx], y[idx]

X_train, y_train = maybe_subsample(X_train, y_train, MAX_TRAIN)
X_test,  y_test  = maybe_subsample(X_test,  y_test,  MAX_TEST)

print("Using Train:", X_train.shape, "Test:", X_test.shape)

# ---------------------------
# 3) Robust scale curves (median / MAD)
# ---------------------------
def robust_scale_rows(X):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + 1e-6
    return (X - med) / mad

X_train_s = robust_scale_rows(X_train)
X_test_s  = robust_scale_rows(X_test)
print("[OK] Robust scaled curves.")

# ---------------------------
# 4) RalphBoost TransitDomain -> deterministic ledger features
# ---------------------------
def component_features_from_ledger(comps):
    """
    Turn a RalphBoost component ledger into a compact, explainable feature row.
    This is NOT 'feature engineering from scratch' ... it's a summary of the additive model.
    """
    periodic = [c for c in comps if c.get("type") == "periodic_train"]
    local    = [c for c in comps if c.get("type") == "local_dip"]
    base     = [c for c in comps if c.get("type") in ("baseline", "baseline_trend", "baselineTrend")]

    # periodic summaries
    if periodic:
        best_p = max(periodic, key=lambda c: abs(float(c.get("depth", 0.0))))
        p_period = float(best_p.get("period", 0.0))
        p_depth  = float(best_p.get("depth", 0.0))
        p_width  = float(best_p.get("width", 0.0))
        p_count  = float(len(periodic))
        p_depth_sum = float(sum(abs(float(c.get("depth", 0.0))) for c in periodic))
    else:
        p_period = p_depth = p_width = p_count = p_depth_sum = 0.0

    # local summaries
    if local:
        l_count = float(len(local))
        l_depth = float(max(abs(float(c.get("depth", 0.0))) for c in local))
        l_depth_sum = float(sum(abs(float(c.get("depth", 0.0))) for c in local))
    else:
        l_count = l_depth = l_depth_sum = 0.0

    # baseline summaries
    if base:
        # take last accepted slope if present
        t_slope = float(base[-1].get("slope", 0.0))
    else:
        t_slope = 0.0

    return [
        p_period, p_depth, p_width, p_count, p_depth_sum,
        l_count, l_depth, l_depth_sum,
        t_slope
    ]

def bake_ralph_features(X, cache_name):
    """
    Bakes per-curve RalphBoost ledgers + feature summaries.
    Caches to disk so you don't rebake every time.
    """
    feat_path = os.path.join(CACHE_DIR, f"{cache_name}_F.npy")
    y_path    = os.path.join(CACHE_DIR, f"{cache_name}_meta.json")
    led_path  = os.path.join(CACHE_DIR, f"{cache_name}_ledgers.jsonl")

    if os.path.exists(feat_path) and os.path.exists(y_path) and os.path.exists(led_path):
        print(f"[CACHE HIT] Loading baked features: {feat_path}")
        F = np.load(feat_path)
        with open(y_path, "r") as f:
            meta = json.load(f)
        return F, meta

    print(f"[BAKE] Building TransitDomain features for {X.shape[0]} curves -> {cache_name}")
    domain = TransitDomain(min_period=MIN_PERIOD, max_period=MAX_PERIOD)

    F = []
    # write ledgers as JSONL for inspectability
    with open(led_path, "w") as lf:
        t0 = time.time()
        for i in range(X.shape[0]):
            curve = X[i].tolist()

            model = RalphBooster(
                domain=domain,
                agent_backend="mock",         # deterministic only
                objective="mse",
                k_candidates=K_CANDIDATES,
                learning_rate=LEARNING_RATE,
                max_iterations=MAX_ITERS,
                min_residual_reduction=None,
            )

            res = model.fit(curve)
            comps = res.components

            # ledger-derived features
            row = component_features_from_ledger(comps)

            # engine metrics that matter
            final_loss    = float(res.metrics.get("loss", 0.0))
            resid_energy  = float(res.metrics.get("residual_energy", 0.0))
            n_components  = float(len(comps))

            # final row
            row += [final_loss, resid_energy, n_components]

            F.append(row)

            # save the first ~10 components as the "story" per curve
            lf.write(json.dumps({
                "i": int(i),
                "ledger": comps[:10]
            }) + "\n")

            if (i+1) % 100 == 0:
                dt = time.time() - t0
                print(f"  baked {i+1}/{X.shape[0]}  ({dt:.1f}s)")

    F = np.array(F, dtype=np.float32)
    np.save(feat_path, F)

    meta = {
        "cache_name": cache_name,
        "n": int(X.shape[0]),
        "feature_dim": int(F.shape[1]),
        "max_iters": MAX_ITERS,
        "k_candidates": K_CANDIDATES,
        "learning_rate": LEARNING_RATE,
        "min_period": MIN_PERIOD,
        "max_period": MAX_PERIOD,
        "seed": SEED
    }
    with open(y_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Saved cache:", feat_path)
    return F, meta

t0 = time.time()
F_train, meta_train = bake_ralph_features(X_train_s, cache_name=f"train_{len(X_train_s)}")
F_test,  meta_test  = bake_ralph_features(X_test_s,  cache_name=f"test_{len(X_test_s)}")
print("[OK] Baking finished in", round(time.time() - t0, 2), "sec")
print("F_train:", F_train.shape, "F_test:", F_test.shape)

# ---------------------------
# 5) Evaluate (full test) + Top-K audit + credibility checks
# ---------------------------
def topk_report(y_true, probs, ks=(25, 50, 100)):
    order = np.argsort(-probs)
    out = {}
    for k in ks:
        k = min(k, len(order))
        top = order[:k]
        hits = int(y_true[top].sum())
        out[k] = {"hits": hits, "precision": float(hits / k)}
    return out

# Try sklearn if available, otherwise fallback to a simple score heuristic
use_sklearn = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as e:
    use_sklearn = False
    print("[WARN] sklearn not available:", e)

if use_sklearn:
    # Train classifier on ledger features
    clf = LogisticRegression(max_iter=4000, class_weight=None)
    clf.fit(F_train, y_train)
    probs = clf.predict_proba(F_test)[:, 1]

    roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
    pr  = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n=== RESULTS (TransitDomain ledger features -> LogisticRegression) ===")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)

    report = topk_report(y_test, probs, ks=(25, 50, 100, 200))
    print("\n=== TOP-K AUDIT ===")
    for k in sorted(report.keys()):
        print(f"Top-{k}: precision={report[k]['precision']:.3f}  hits={report[k]['hits']}")

    # show best-ranked candidate ledger
    best = int(np.argmax(probs))
    print("\n=== BEST PREDICTED SAMPLE ===")
    print("Index:", best, "True label:", int(y_test[best]), "Score:", float(probs[best]))

    # read its ledger from JSONL cache
    led_path = os.path.join(CACHE_DIR, f"test_{len(X_test_s)}_ledgers.jsonl")
    best_ledger = None
    if os.path.exists(led_path):
        with open(led_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if int(obj["i"]) == best:
                    best_ledger = obj["ledger"]
                    break
    if best_ledger:
        print("Ledger (first 10 comps):")
        print(json.dumps(best_ledger, indent=2))

    # ---------------------------
    # Credibility Gate #1: Label shuffle collapse (quick + brutal)
    # ---------------------------
    y_shuf = np.random.RandomState(SEED).permutation(y_train)
    clf_shuf = LogisticRegression(max_iter=4000, class_weight=None)
    clf_shuf.fit(F_train, y_shuf)
    probs_shuf = clf_shuf.predict_proba(F_test)[:, 1]

    roc_shuf = roc_auc_score(y_test, probs_shuf) if len(np.unique(y_test)) > 1 else float("nan")
    pr_shuf  = average_precision_score(y_test, probs_shuf) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n=== CREDIBILITY: LABEL SHUFFLE COLLAPSE ===")
    print("ROC-AUC shuffled:", roc_shuf)
    print("PR-AUC  shuffled:", pr_shuf)

    # ---------------------------
    # Credibility Gate #2: Multi-seed CV on TRAIN (uses same baked features)
    # ---------------------------
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rocs, prs = [], []
    for fold, (tr, va) in enumerate(skf.split(F_train, y_train), 1):
        m = LogisticRegression(max_iter=4000, class_weight=None)
        m.fit(F_train[tr], y_train[tr])
        p = m.predict_proba(F_train[va])[:, 1]
        rocs.append(roc_auc_score(y_train[va], p))
        prs.append(average_precision_score(y_train[va], p))
    print("\n=== TRAIN CV (5-fold) ===")
    print("ROC-AUC mean±std:", float(np.mean(rocs)), "±", float(np.std(rocs)))
    print("PR-AUC  mean±std:", float(np.mean(prs)),  "±", float(np.std(prs)))

else:
    # fallback: heuristic score without sklearn
    # score = big local dips + periodic depth sum - residual_energy (toy)
    score = (F_test[:, 7] + F_test[:, 4]) - 0.05 * F_test[:, 10]
    # normalize to [0,1] for readability
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    probs = score
    print("\n[NO-SKLEARN MODE] Using heuristic score for ranking.")
    report = topk_report(y_test, probs, ks=(25, 50, 100, 200))
    for k in sorted(report.keys()):
        print(f"Top-{k}: precision={report[k]['precision']:.3f}  hits={report[k]['hits']}")
