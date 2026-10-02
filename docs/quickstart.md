# Guide de démarrage rapide

## Prérequis

- Python 3.13+
- pip
- Accès à la base MySQL MEL Cameroun (pour exporter les données)
- ~500 MB d'espace disque (données + modèles)

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

## Étape 2 — Exporter les données MEL depuis MySQL

Les données doivent être exportées de la base MySQL MEL vers `data/raw/`.

### Structure attendue

```
data/raw/
├── orders.csv           # Commandes (depuis table factures)
├── order_items.csv      # Articles par commande (depuis table factures + articles)
├── products.csv         # Catalogue articles (depuis table articles)
├── reviews.csv          # Avis clients (depuis table etoiles)
└── category_names.csv   # Catégories (optionnel, depuis table categories)
```

### Requêtes SQL d'export

**orders.csv** — depuis la table `factures` :
```sql
SELECT
    f.id                AS order_id,
    f.acheteur_id       AS customer_id,
    f.status            AS order_status,
    f.created_at        AS order_purchase_timestamp
FROM factures f
INTO OUTFILE '/path/to/data/raw/orders.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

**order_items.csv** — articles par commande :
```sql
SELECT
    f.id                AS order_id,
    f.article_id        AS product_id,
    f.montant_total     AS price
FROM factures f
INTO OUTFILE '/path/to/data/raw/order_items.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

**products.csv** — catalogue articles :
```sql
SELECT
    a.id                        AS product_id,
    ac.categorie_id             AS product_category_name,
    a.nom                       AS product_name,
    a.marque                    AS product_brand,
    a.created_at
FROM articles a
LEFT JOIN article_categories ac ON ac.article_id = a.id
INTO OUTFILE '/path/to/data/raw/products.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

**reviews.csv** — avis / étoiles :
```sql
SELECT
    e.note_id       AS order_id,
    e.valeur        AS review_score
FROM etoiles e
INTO OUTFILE '/path/to/data/raw/reviews.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

**category_names.csv** (optionnel) :
```sql
SELECT
    c.id            AS product_category_name,
    c.nom           AS product_category_name_english
FROM categories c
INTO OUTFILE '/path/to/data/raw/category_names.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

> **Alternative :** utiliser phpMyAdmin (Exporter → CSV) ou un outil comme DBeaver.

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
user_id = "42"   # remplacer par un vrai ID utilisateur MEL
print("SVD :", recommend_svd(user_id))
print("ALS :", recommend_als(user_id))
```

---

## Dépannage

| Erreur | Cause | Solution |
|---|---|---|
| `FileNotFoundError: orders.csv` | Données MEL non exportées | Exporter depuis MySQL et placer dans `data/raw/` |
| `ValueError: y contains previously unseen labels` | User/item inconnu | Le fallback popularité est géré automatiquement |
| `MemoryError` lors du SVD | RAM insuffisante | Réduire `k_factors` (ex: 50 au lieu de 150) |
| `NotImplementedError` sur sparse | Version scipy incompatible | `pip install scipy>=1.14.0` |
| Trop peu d'interactions après filtrage | Volume de données insuffisant | Abaisser `min_item` à 2 dans `filter_interactions()` |
