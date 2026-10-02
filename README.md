# 🛒 Système de Recommandation E-commerce — MEL Cameroun

Projet ML de recommandation de produits pour la plateforme [melcameroun.com](https://melcameroun.com).  
Expose les recommandations via une **API REST FastAPI** dans une **pipeline MLOps complète**.

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Données](#données)
- [Architecture](#architecture)
- [Modèles implémentés](#modèles-implémentés)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API FastAPI](#api-fastapi)
- [Pipeline MLOps](#pipeline-mlops)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Structure du projet](#structure-du-projet)
- [Prochaines étapes](#prochaines-étapes)

---

## Aperçu

Ce projet implémente et compare **5 modèles de recommandation** de produits, puis les expose via une **API REST** dans une **pipeline MLOps complète** :

| Modèle | Approche |
|---|---|
| Baseline | Popularité globale |
| SVD | Collaborative Filtering (décomposition matricielle) |
| ALS | Collaborative Filtering implicite (Alternating Least Squares) |
| Content-Based | TF-IDF + similarité cosine sur les attributs produit |
| **Hybride** | SVD + Content-Based pondérés (meilleur alpha automatique) |

> ✅ **100% compatible Python 3.13+** — aucune dépendance Cython

---

## Données

Le pipeline utilise les données de la plateforme MEL Cameroun, exportées depuis la base MySQL (`factures`, `articles`, `paniers`, `categories`).

**Fichiers CSV attendus dans `data/raw/` :**

| Fichier | Description |
|---|---|
| `orders.csv` | Commandes (order_id, customer_id, order_status) |
| `order_items.csv` | Produits par commande (order_id, product_id, price) |
| `products.csv` | Catalogue produits (product_id, category, weight, photos) |
| `reviews.csv` | Avis clients (order_id, review_score 1-5) |
| `category_names.csv` | Traduction catégories (optionnel) |

**Après filtrage cold-start (validation sur données de référence) :**

| Métrique | Valeur |
|---|---|
| Utilisateurs | 3 599 |
| Produits | 882 |
| Interactions | 9 657 |
| Sparsité | ~99.7% |

---

## Architecture

```
mel-ml/
├── data/
│   ├── raw/                        # Données source (CSV exportés depuis MySQL MEL)
│   └── processed/                  # Données nettoyées, splits train/test
├── notebooks/
│   └── recommendation_system.ipynb # Notebook principal (14 sections)
├── src/
│   ├── recommender.py              # Classe ALSRecommender (partagée)
│   └── train.py                    # Pipeline d'entraînement CLI
├── api/
│   └── main.py                     # API FastAPI (5 endpoints)
├── models/                         # Modèles entraînés (.pkl, .npy)
│   ├── svd_R_pred.npy
│   ├── als_model.pkl
│   ├── cosine_sim.pkl
│   ├── user_encoder.pkl
│   ├── item_encoder.pkl
│   ├── popular_items.pkl
│   └── results_summary.json
├── tests/
│   └── test_api.py                 # 15 tests unitaires
├── scripts/
│   └── generate_fake_models.py     # Génère de faux modèles pour CI
├── docs/
│   ├── technical_documentation.md
│   └── quickstart.md
├── .github/workflows/
│   └── ci-cd.yml                   # GitHub Actions
├── Dockerfile                      # Image API
├── Dockerfile.train                # Image entraînement
├── docker-compose.yml
├── requirements.txt                # Dépendances notebook + train
└── requirements-api.txt            # Dépendances API (image légère)
```

---

## Modèles implémentés

### 1. Baseline — Popularité
Recommande les produits les plus achetés globalement.
Sert de référence minimale et de fallback cold-start pour les nouveaux utilisateurs.

### 2. SVD — Singular Value Decomposition
- Matrice user-item centrée par user (rating moyen soustrait)
- Décomposition SVD tronquée via `scipy.sparse.linalg.svds`
- Tuning du nombre de facteurs latents : **k = 150** (optimal)

### 3. ALS — Alternating Least Squares
- Implémentation from scratch dans `src/recommender.py` (numpy/scipy uniquement)
- Signal implicite : confidence `c_ui = 1 + 40 × n_achats`
- Résolution par système linéaire `(YᵀCᵘY + λI) xᵤ = YᵀCᵘpᵤ`
- Tuning : **n_factors = 32, regularization = 0.1** (optimal)

### 4. Content-Based — TF-IDF + Cosine
- Features textuelles : catégorie produit + poids + nombre de photos
- Vectorisation TF-IDF (500 features max)
- Similarité cosine entre produits
- Recommande les produits similaires aux achats passés de l'utilisateur

### 5. Hybride — SVD + Content-Based
- Normalisation [0, 1] des scores SVD et CB par user
- Score final : `α × score_SVD + (1−α) × score_CB`
- Grid search sur α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}
- **Meilleur alpha = 1.0** (SVD domine — features CB à enrichir)

---

## Résultats

Évaluation sur un échantillon de **500 utilisateurs** avec la stratégie **Leave-One-Out** (dernier achat en test).

| Modèle | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Baseline (Popularité) | 0.0015 | 0.0147 | 0.0077 |
| SVD (scipy) | 0.0032 | 0.0320 | 0.0240 |
| ALS (from scratch) | 0.0038 | 0.0380 | 0.0298 |
| Content-Based (TF-IDF) | 0.0012 | 0.0120 | 0.0054 |
| **Hybride (SVD + CB)** | **0.0038** | **0.0380** | **0.0325** |

> **Note :** Les scores absolus sont faibles car la matrice est très sparse (~99.7%). Ce qui compte : l'ALS et le modèle hybride surpassent la baseline de **+4× en NDCG**.

---

## Installation

### Prérequis
- Python 3.13+
- pip

### 1. Cloner le projet

```bash
git clone https://github.com/ton-username/mel-ml.git
cd mel-ml
```

### 2. Installer les dépendances

```bash
# Pour le notebook et l'entraînement
pip install -r requirements.txt

# Pour l'API uniquement (plus léger)
pip install -r requirements-api.txt
```

### 3. Préparer les données

Exporter les données de la base MySQL MEL dans `data/raw/` :

```sql
-- Exemple : exporter les commandes
SELECT f.id AS order_id, f.acheteur_id AS customer_id, f.status AS order_status,
       f.created_at AS order_purchase_timestamp
FROM factures f
INTO OUTFILE '/path/to/data/raw/orders.csv' FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
```

Voir `docs/quickstart.md` pour le guide complet d'export.

---

## Utilisation

### Notebook (exploration + entraînement interactif)

```bash
jupyter notebook notebooks/recommendation_system.ipynb
```

Exécuter les cellules dans l'ordre. Les modèles sont sauvegardés dans `models/`.

> ⚠️ Le notebook importe `ALSRecommender` depuis `src.recommender` — ne pas redéfinir la classe dans le notebook.

### Script d'entraînement (reproductible)

```bash
python src/train.py
```

Avec options :

```bash
python src/train.py \
  --data-dir data/raw \
  --models-dir models \
  --k-factors 150 \
  --als-factors 32 \
  --als-reg 0.1 \
  --alpha 1.0
```

---

## API FastAPI

### Démarrage

```bash
python -m uvicorn api.main:app --reload --port 8000
```

→ Documentation interactive : http://127.0.0.1:8000/docs

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut de l'API + infos modèle |
| GET | `/metrics` | Métriques NDCG/Precision/Recall |
| GET | `/popular?n=10` | Produits les plus populaires |
| GET | `/recommend/{user_id}?n=10&model=hybrid` | Recommandations personnalisées |
| GET | `/similar/{product_id}?n=10` | Produits similaires |

### Exemples

```bash
# Santé de l'API
curl http://localhost:8000/health

# Top 10 recommandations pour un user
curl "http://localhost:8000/recommend/USER_ID?n=10"

# Choisir le modèle : svd | als | hybrid
curl "http://localhost:8000/recommend/USER_ID?model=als"

# Produits similaires
curl "http://localhost:8000/similar/PRODUCT_ID?n=5"

# Nouveau client → cold-start automatique (popularité)
curl "http://localhost:8000/recommend/nouveau_client"
```

---

## Pipeline MLOps

```
data/raw/ ──► src/train.py ──► models/ ──► api/main.py ──► HTTP
                  │                              │
                  ▼                              ▼
          results_summary.json           /metrics endpoint
```

### Re-entraîner et redéployer

```bash
# 1. Entraîner
python src/train.py

# 2. Lancer l'API (recharge les modèles au démarrage)
python -m uvicorn api.main:app --reload
```

### Tests

```bash
pytest tests/ -v
# 15/15 tests passent
```

---

## Docker

### Lancer l'API avec Docker

```bash
# Build + démarrage
docker compose up --build

# API disponible sur http://localhost:8000
```

### Re-entraîner dans Docker

```bash
docker compose --profile train up trainer
```

### Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MODELS_DIR` | `models/` | Chemin vers les artefacts |

---

## CI/CD

Pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) déclenché sur chaque push vers `main` :

```
push main
    │
    ▼
[1] Tests + Linting (pytest, ruff)
    │
    ▼
[2] Build Docker image → push vers GHCR
    │
    ▼
[3] Smoke test : /health + /popular
```

---

## Structure du notebook

Le notebook `recommendation_system.ipynb` est organisé en **14 sections** :

| Section | Contenu |
|---|---|
| 0 | Imports & configuration |
| 1 | Chargement des données |
| 2 | Exploration (EDA) |
| 3 | Preprocessing & feature engineering |
| 4 | Train / Test Split (Leave-One-Out + Random 80/20) |
| 5 | Fonctions de métriques (Precision@K, Recall@K, NDCG@K) |
| 6 | Modèle 1 — Baseline popularité |
| 7 | Modèle 2 — SVD (scipy) |
| 8 | Modèle 3 — ALS (from scratch) |
| 9 | Modèle 4 — Content-Based (TF-IDF) |
| 10 | Comparaison des 4 modèles |
| 11 | Tuning des hyperparamètres |
| 12 | Modèle 5 — Hybride (SVD + CB) |
| 13 | Analyse qualitative des recommandations |
| 14 | Sauvegarde finale & résumé JSON |

---

## Prochaines étapes

- [ ] **Export MySQL → CSV** — script d'export automatique depuis la BDD MEL
- [ ] **Cold-start amélioré** — recommandations par catégorie pour les nouveaux utilisateurs
- [ ] **Features enrichies** — utiliser `nom`, `marque`, `description` des articles MEL
- [ ] **A/B Testing** — mesurer l'impact réel des recommandations sur les conversions
- [ ] **Neural CF** — embeddings appris par réseau de neurones (NCF, Two-Tower)
- [ ] **Monitoring** — tracking des métriques en production (Prometheus/Grafana)

---

## Stack technique

| Outil | Usage |
|---|---|
| Python 3.13+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scipy | SVD tronqué, matrices sparse |
| scikit-learn | Encodage, TF-IDF, métriques |
| FastAPI | API REST |
| uvicorn | Serveur ASGI |
| Docker | Conteneurisation |
| GitHub Actions | CI/CD |
| pytest | Tests unitaires |
| matplotlib / seaborn | Visualisations |
| Jupyter | Notebook interactif |

---

## Auteur

Projet développé dans le cadre de l'amélioration de la plateforme e-commerce [melcameroun.com](https://melcameroun.com).
