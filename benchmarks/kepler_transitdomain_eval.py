import os
import math
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ralphboost.core.estimator import RalphBooster
from ralphboost.domains.transit import TransitDomain

def load_kepler_data(train_path="exoTrain.csv", test_path="exoTest.csv", limit=None):
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Generating synthetic Kepler-like data.")
        return generate_synthetic_data(n_samples=100 if limit is None else limit)
    
    print(f"Loading {train_path}...")
    df_train = pd.read_csv(train_path)
    if limit:
        df_train = df_train.iloc[:limit]
        
    # Standard Kepler format: LABEL (2=planet, 1=non-planet), FLUX.1, FLUX.2...
    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values
    
    # Convert labels to 0/1 (2->1, 1->0)
    y = (y == 2).astype(int)
    
    # Simple normalization
    # Center and scale
    X_norm = []
    for row in X:
        med = np.median(row)
        std = np.std(row)
        if std == 0: std = 1.0
        X_norm.append((row - med) / std)
    
    return np.array(X_norm), y

def generate_synthetic_data(n_samples=100, n_points=1000):
    X = []
    y = []
    for i in range(n_samples):
        is_planet = (i % 10 == 0) # 10% planets
        y.append(1 if is_planet else 0)
        
        # Generate flux
        flux = np.random.normal(0, 0.1, n_points)
        # Add trend
        flux += np.linspace(-1, 1, n_points) * np.random.uniform(-5, 5)
        
        if is_planet:
            period = np.random.uniform(20, 100)
            phase = np.random.uniform(0, period)
            depth = np.random.uniform(2, 10)
            width = 3.0
            
            for t in range(n_points):
                dist = (t - phase + 0.5 * period) % period - 0.5 * period
                if abs(dist) < 0.5 * width:
                    flux[t] -= depth
                    
        X.append(flux)
    return np.array(X), np.array(y)

def evaluate_model(X, y, description="Baseline"):
    print(f"\n--- Evaluating: {description} ---")
    
    # Feature Engineering with RalphBoost
    # We want to extract "Is there a periodic train?" features.
    # Score = sum of depths of accepted periodic trains?
    # Or max depth?
    # Or raw complexity-adjusted score?
    
    scores = []
    
    start_time = time.time()
    
    for idx, signal in enumerate(X):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(X)}...", end='\r')
            
        # Fit RalphBoost
        # Short run for speed in benchmark
        model = RalphBooster(
            domain=TransitDomain(min_period=10, max_period=200),
            agent_backend="mock", # No LLM
            objective="mse",
            k_candidates=5,
            learning_rate=0.5,
            max_iterations=5, # Fast scan
            min_residual_reduction=None, # Force run
            duplicate_penalty=10.0,
            complexity_weight=0.1
        )
        model.fit(signal.tolist())
        
        # Extract feature: "Planet Score"
        # Sum of depths of "periodic_train" components
        planet_score = 0.0
        for comp in model.components:
            if comp["type"] == "periodic_train":
                # Depth is parameter. But 'weight' scales it.
                # Effective depth = param_depth * weight
                d = comp.get("depth", 0.0) * comp.get("weight", 1.0)
                planet_score += d
        
        scores.append(planet_score)
        
    print(f"Finished processing in {time.time() - start_time:.2f}s")
    
    # Metrics
    try:
        prauc = average_precision_score(y, scores)
        rocauc = roc_auc_score(y, scores)
        print(f"PR-AUC: {prauc:.4f}")
        print(f"ROC-AUC: {rocauc:.4f}")
        
        # Top-K Analysis
        # Sort by score desc
        zipped = list(zip(scores, y))
        zipped.sort(key=lambda x: x[0], reverse=True)
        
        k = 10
        top_k = zipped[:k]
        hits = sum(1 for s, label in top_k if label == 1)
        print(f"Top-{k} Hits: {hits}/{k}")
        
    except Exception as e:
        print(f"Metric calculation failed: {e}")
        
    return scores

def run_benchmark():
    print("Loading Data...")
    X, y = load_kepler_data(limit=50) # Small limit for demo
    
    print(f"Data shape: {X.shape}, Labels: {sum(y)} planets / {len(y)} total")
    
    # 1. Main Evaluation
    scores_main = evaluate_model(X, y, "TransitDomain Features")
    
    # 2. Credibility Check: Label Shuffle
    print("\n--- Credibility: Label Shuffle Test ---")
    y_shuffled = np.random.permutation(y)
    # Metrics on shuffled labels using SAME scores (just checking if metrics collapse)
    # Wait, strict label shuffle test means we retrain? 
    # No, if we retrain on X with shuffled y, the unsupervised feature extraction (RalphBoost) shouldn't change.
    # RalphBoost is unsupervised (fits X). The *classifier* uses y.
    # Here we use heuristic score.
    # If we shuffle y, PR-AUC should drop to baseline (prevalence).
    
    prauc_shuffled = average_precision_score(y_shuffled, scores_main)
    print(f"Shuffled PR-AUC: {prauc_shuffled:.4f} (Should be close to {sum(y)/len(y):.4f})")
    
if __name__ == "__main__":
    run_benchmark()
