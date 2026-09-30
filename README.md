# 🛒 Système de Recommandation E-commerce — MEL Cameroun

Projet ML/DL de recommandation de produits basé sur le dataset **Olist Brazilian E-Commerce**.  
Conçu comme service annexe pour la plateforme [melcameroun.com](https://melcameroun.com).

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Dataset](#dataset)
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

> ✅ **100% compatible Python 3.13+** — aucune dépendance Cython (pas de scikit-surprise ni implicit)

---

## Dataset

**Olist Brazilian E-Commerce** — disponible sur [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| Fichier | Description |
|---|---|
| `olist_orders_dataset.csv` | Commandes (100k+) |
| `olist_order_items_dataset.csv` | Produits par commande |
| `olist_products_dataset.csv` | Catalogue produits |
| `olist_order_reviews_dataset.csv` | Avis clients (1-5 étoiles) |
| `product_category_name_translation.csv` | Traduction catégories PT→EN |

**Après filtrage cold-start :**

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
│   ├── raw/                        # CSV originaux Olist
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
│   └── fix_als_pickle.py           # Migration artefacts
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
Sert de référence minimale pour comparer les autres modèles.

### 2. SVD — Singular Value Decomposition
- Matrice user-item centrée par user (rating moyen soustrait)
- Décomposition SVD tronquée via `scipy.sparse.linalg.svds`
- Tuning du nombre de facteurs latents : **k = 150** (optimal)
- Évaluation RMSE/MAE sur le split aléatoire 80/20

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
- **Meilleur alpha = 1.0** (SVD domine sur ce dataset)

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

> **Note :** Les scores faibles sont normaux sur ce dataset — Olist est extrêmement sparse (~99.7%) car la plupart des clients n'achètent qu'une seule fois. L'ALS et le modèle hybride surpassent significativement la baseline (+4× en NDCG).

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

### 3. Télécharger le dataset

Télécharge le dataset Olist depuis [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) et place les fichiers CSV dans `data/raw/`.

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
curl "http://localhost:8000/recommend/8d50f5eadf50201ccdcedfb9e2ac8455?n=10"

# Choisir le modèle : svd | als | hybrid
curl "http://localhost:8000/recommend/8d50f5eadf50201ccdcedfb9e2ac8455?model=als"

# Produits similaires
curl "http://localhost:8000/similar/b623b7cb05ee3248fbe4a6ecbeed79a4?n=5"

# User inconnu → cold-start (popularité automatique)
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
| 1 | Chargement des données Olist |
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

- [ ] **Cold-start amélioré** — recommandations par catégorie pour les nouveaux utilisateurs
- [ ] **Données MEL Cameroun** — adapter le pipeline aux données réelles du site
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
