"""
scripts/generate_fake_models.py
Génère des modèles factices pour le smoke test CI/CD.
Les vrais modèles (.pkl, .npy) sont dans .gitignore et non disponibles sur le runner.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender import ALSRecommender

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_USERS = 50
N_ITEMS = 100
rng     = np.random.default_rng(42)

# Encodeurs
user_enc = LabelEncoder()
item_enc = LabelEncoder()
user_enc.fit([f"user_{i}" for i in range(N_USERS)])
item_enc.fit([f"item_{i}" for i in range(N_ITEMS)])

with open(MODELS_DIR / "user_encoder.pkl", "wb") as f:
    pickle.dump(user_enc, f)
with open(MODELS_DIR / "item_encoder.pkl", "wb") as f:
    pickle.dump(item_enc, f)

# SVD
R_pred = rng.random((N_USERS, N_ITEMS)).astype(np.float32)
np.save(MODELS_DIR / "svd_R_pred.npy", R_pred)

# ALS
als = ALSRecommender(n_factors=8, n_iterations=1, random_state=42)
als.user_factors = rng.random((N_USERS, 8)).astype(np.float32)
als.item_factors = rng.random((N_ITEMS, 8)).astype(np.float32)
with open(MODELS_DIR / "als_model.pkl", "wb") as f:
    pickle.dump(als, f)

# Content-based
cosine_sim  = rng.random((N_ITEMS, N_ITEMS)).astype(np.float32)
np.fill_diagonal(cosine_sim, 1.0)
item_to_idx = {f"item_{i}": i for i in range(N_ITEMS)}
idx_to_item = {i: f"item_{i}" for i in range(N_ITEMS)}
with open(MODELS_DIR / "cosine_sim.pkl", "wb") as f:
    pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)

# Popularité
popular_items = [f"item_{i}" for i in range(N_ITEMS)]
with open(MODELS_DIR / "popular_items.pkl", "wb") as f:
    pickle.dump(popular_items, f)

# Summary
summary = {
    "model":           "Hybrid (SVD + Content-Based)",
    "n_users":         N_USERS,
    "n_items":         N_ITEMS,
    "n_interactions":  500,
    "hyperparameters": {"svd_k_factors": 8, "hybrid_alpha": 1.0},
    "metrics":         {"Precision@10": 0.0, "Recall@10": 0.0, "NDCG@10": 0.0},
}
with open(MODELS_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Modèles factices générés dans {MODELS_DIR}")
print(f"   {N_USERS} users | {N_ITEMS} items")
