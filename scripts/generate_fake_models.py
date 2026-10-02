"""
scripts/generate_fake_models.py
Génère des modèles factices cohérents avec l'architecture catégories.
Utilisé par le smoke test CI/CD (les vrais modèles ne sont pas disponibles sur le runner).

Architecture :
  - user_enc encode des user_id
  - item_enc encode des NOMS DE CATÉGORIES (marketplace d'occasion)
  - products_catalog.csv avec colonnes available + product_category_name
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender import ALSRecommender

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

N_USERS = 50

# Catégories réelles de MEL Cameroun
CATEGORIES = [
    "Vêtements femmes",
    "Vêtements hommes",
    "Chaussures femmes",
    "Chaussures hommes",
    "Accessoires",
    "Téléphones",
    "Électronique",
    "Maison & Déco",
    "Ensembles femmes",
    "Ensembles hommes",
]
N_CATS = len(CATEGORIES)

# Génère un catalogue fictif de 200 articles répartis sur les catégories
# Simule une marketplace d'occasion : ~20% vendus, 80% disponibles
N_ARTICLES = 200
product_ids        = [str(i + 1) for i in range(N_ARTICLES)]
product_categories = [CATEGORIES[i % N_CATS] for i in range(N_ARTICLES)]
product_names      = [f"Article {i + 1} — {CATEGORIES[i % N_CATS]}" for i in range(N_ARTICLES)]
product_available  = [0 if (i % 5 == 0) else 1 for i in range(N_ARTICLES)]   # ~20% vendus

# ── Encodeurs ─────────────────────────────────────────────────────────────────
user_enc = LabelEncoder()
item_enc = LabelEncoder()
user_enc.fit([f"user_{i}" for i in range(N_USERS)])
item_enc.fit(CATEGORIES)   # encode les noms de catégories

with open(MODELS_DIR / "user_encoder.pkl", "wb") as f:
    pickle.dump(user_enc, f)
with open(MODELS_DIR / "item_encoder.pkl", "wb") as f:
    pickle.dump(item_enc, f)

print(f"✓ Encodeurs : {N_USERS} users, {N_CATS} catégories")

# ── SVD R_pred (user × catégorie) ─────────────────────────────────────────────
R_pred = rng.random((N_USERS, N_CATS)).astype(np.float32)
np.save(MODELS_DIR / "svd_R_pred.npy", R_pred)
print(f"✓ SVD R_pred : {R_pred.shape}")

# ── ALS (user × catégorie) ────────────────────────────────────────────────────
als = ALSRecommender(n_factors=8, n_iterations=1, random_state=42)
als.user_factors = rng.random((N_USERS, 8)).astype(np.float32)
als.item_factors = rng.random((N_CATS,  8)).astype(np.float32)
with open(MODELS_DIR / "als_model.pkl", "wb") as f:
    pickle.dump(als, f)
print(f"✓ ALS : {N_USERS} users × {N_CATS} catégories")

# ── Content-Based (similarité entre catégories) ───────────────────────────────
cosine_sim  = rng.random((N_CATS, N_CATS)).astype(np.float32)
np.fill_diagonal(cosine_sim, 1.0)
item_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
idx_to_item = {i: cat for cat, i in item_to_idx.items()}
with open(MODELS_DIR / "cosine_sim.pkl", "wb") as f:
    pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)
print(f"✓ Content-Based : matrice {cosine_sim.shape}")

# ── Popularité (catégories classées par popularité fictive) ───────────────────
popular_items = CATEGORIES[:]
with open(MODELS_DIR / "popular_items.pkl", "wb") as f:
    pickle.dump(popular_items, f)
print(f"✓ Popularité : {len(popular_items)} catégories")

# ── Catalogue produits avec disponibilité ─────────────────────────────────────
catalog = pd.DataFrame({
    "product_id":             product_ids,
    "product_name":           product_names,
    "product_category_name":  product_categories,
    "product_weight_g":       [500] * N_ARTICLES,
    "product_photos_qty":     [1]   * N_ARTICLES,
    "available":              product_available,
})
catalog.to_csv(MODELS_DIR / "products_catalog.csv", index=False)

n_available = sum(product_available)
n_sold      = N_ARTICLES - n_available
print(f"✓ Catalogue : {N_ARTICLES} articles ({n_available} disponibles, {n_sold} vendus)")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "dataset":         "MEL Cameroun (fake — CI/CD)",
    "python_version":  "3.14+",
    "model":           "Hybrid (SVD + Content-Based) — catégories",
    "n_users":         N_USERS,
    "n_items":         N_CATS,
    "n_interactions":  500,
    "hyperparameters": {
        "svd_k_factors": 8,
        "als_n_factors": 8,
        "hybrid_alpha":  0.6,
    },
    "metrics": {
        "Precision@10": 0.0,
        "Recall@10":    0.0,
        "NDCG@10":      0.0,
        "n_users_eval": 0,
    },
}
with open(MODELS_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ results_summary.json")

print()
print(f"✅  Modèles factices générés dans {MODELS_DIR}")
print(f"    {N_USERS} users | {N_CATS} catégories | {n_available}/{N_ARTICLES} articles disponibles")
