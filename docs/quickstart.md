# Guide de démarrage rapide

## Prérequis

- Python 3.13+
- pip
- ~500 MB d'espace disque (dataset + modèles)

---

## Étape 1 — Installation

```bash
# Cloner le projet
git clone https://github.com/ton-username/mel-ml.git
cd mel-ml

# Installer les dépendances
pip install -r requirements.txt
```

---

## Étape 2 — Télécharger le dataset

1. Aller sur https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Cliquer **Download** (compte Kaggle requis, gratuit)
3. Extraire et copier ces 5 fichiers dans `data/raw/` :

```
data/raw/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_products_dataset.csv
├── olist_order_reviews_dataset.csv
└── product_category_name_translation.csv
```

**Alternative avec l'API Kaggle :**
```bash
pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

---

## Étape 3 — Lancer le notebook

```bash
jupyter notebook notebooks/recommendation_system.ipynb
```

Puis **Kernel → Restart & Run All** (durée estimée : 10-20 min selon ta machine).

---

## Étape 4 — Vérifier les résultats

Après exécution complète, les fichiers suivants sont générés :

```
data/processed/
├── interactions_filtered.csv   # interactions après filtrage cold-start
├── train.csv                   # split Leave-One-Out train
├── test.csv                    # split Leave-One-Out test
├── train_random.csv            # split random 80% train
└── test_random.csv             # split random 20% test

models/
├── svd_R_pred.npy              # matrice prédite SVD
├── als_model.pkl               # modèle ALS
├── cosine_sim.pkl              # matrice similarité content-based
├── user_encoder.pkl            # encodeur users
├── item_encoder.pkl            # encodeur items
├── popular_items.pkl           # items populaires (fallback)
├── results_summary.json        # métriques + hyperparamètres
├── model_comparison.png        # graphique comparaison 4 modèles
├── tuning_comparison.png       # graphique tuning SVD & ALS
└── final_comparison.png        # graphique comparaison finale 5 modèles
```

---

## Utilisation rapide des modèles

```python
import pickle
import numpy as np

# ── Charger les artefacts ─────────────────────────────────────────────────
with open('models/user_encoder.pkl', 'rb') as f:
    user_enc = pickle.load(f)
with open('models/item_encoder.pkl', 'rb') as f:
    item_enc = pickle.load(f)
with open('models/popular_items.pkl', 'rb') as f:
    popular_items = pickle.load(f)

R_pred = np.load('models/svd_R_pred.npy')

with open('models/als_model.pkl', 'rb') as f:
    als_model = pickle.load(f)


# ── Recommandations SVD ───────────────────────────────────────────────────
def recommend_svd(user_id, n=10):
    try:
        user_idx = user_enc.transform([user_id])[0]
    except ValueError:
        return popular_items[:n]   # cold start
    scores   = R_pred[user_idx]
    top_idxs = np.argsort(scores)[::-1][:n]
    return item_enc.inverse_transform(top_idxs).tolist()


# ── Recommandations ALS ───────────────────────────────────────────────────
def recommend_als(user_id, n=10):
    try:
        user_idx = int(user_enc.transform([user_id])[0])
    except ValueError:
        return popular_items[:n]   # cold start
    top_idxs = als_model.recommend(user_idx, n=n)
    return item_enc.inverse_transform(top_idxs).tolist()


# ── Exemple ───────────────────────────────────────────────────────────────
user_id = "8d50f5eadf50201ccdcedfb9e2ac8455"  # remplacer par un vrai ID
print("SVD :", recommend_svd(user_id))
print("ALS :", recommend_als(user_id))
```

---

## Dépannage

| Erreur | Cause | Solution |
|---|---|---|
| `FileNotFoundError: olist_orders_dataset.csv` | Dataset manquant | Télécharger et placer dans `data/raw/` |
| `ValueError: y contains previously unseen labels` | User/item inconnu | Le fallback popularité est géré automatiquement |
| `MemoryError` lors du SVD | RAM insuffisante | Réduire `k_factors` (ex: 50 au lieu de 150) |
| `NotImplementedError` sur sparse | Version scipy incompatible | `pip install scipy>=1.14.0` |
