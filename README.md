# 🛒 Système de Recommandation — MEL Cameroun

Système de recommandation de produits pour la marketplace d'occasion [melcameroun.com](https://melcameroun.com).  
Expose les recommandations via une **API REST FastAPI** dans une **pipeline MLOps complète**.

> **Particularité :** chaque article est unique (vendu = retiré). Le modèle recommande des **catégories**, l'API retourne les **articles disponibles** dans ces catégories en temps réel.

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Modèles implémentés](#modèles-implémentés)
- [Données](#données)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API FastAPI](#api-fastapi)
- [Pipeline MLOps](#pipeline-mlops)
- [Docker](#docker)
- [Automatisation](#automatisation)
- [CI/CD](#cicd)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)
- [Prochaines étapes](#prochaines-étapes)
- [Stack technique](#stack-technique)

---

## Aperçu

Ce projet implémente un **système de recommandation adapté à une marketplace d'occasion** :

| Contrainte | Solution |
|---|---|
| Chaque article est unique | Collaborative filtering sur les **catégories**, pas les articles |
| Un article vendu disparaît | Colonne `available` dans le catalogue, filtre dynamique à chaque requête |
| Nouveaux utilisateurs | Cold-start automatique → articles populaires disponibles |
| Re-entraînement sans downtime | Endpoint `POST /reload` + cron hebdomadaire Docker |

**5 modèles comparés :**

| Modèle | Approche |
|---|---|
| Baseline | Popularité des catégories |
| SVD | Collaborative Filtering (décomposition matricielle user × catégorie) |
| ALS | Collaborative Filtering implicite (from scratch, numpy/scipy) |
| Content-Based | TF-IDF + similarité cosine entre catégories |
| **Hybride** | SVD + Content-Based pondérés (α = 0.6) |

> ✅ **100% compatible Python 3.14+** — aucune dépendance Cython

---

## Architecture

```
mel-ml/
├── data/
│   ├── raw/                          # CSV exportés depuis MySQL MEL
│   └── processed/                    # Splits train/test
├── notebooks/
│   └── recommendation_system.ipynb   # Notebook principal (14 sections)
├── src/
│   ├── recommender.py                # Classe ALSRecommender (partagée)
│   └── train.py                      # Pipeline d'entraînement CLI
├── api/
│   └── main.py                       # API FastAPI (6 endpoints)
├── models/                           # Artefacts entraînés
│   ├── svd_R_pred.npy                # Matrice user × catégorie
│   ├── als_model.pkl
│   ├── cosine_sim.pkl                # Similarité entre catégories
│   ├── user_encoder.pkl
│   ├── item_encoder.pkl              # Encode les noms de catégories
│   ├── popular_items.pkl
│   ├── products_catalog.csv          # Catalogue avec colonne available
│   └── results_summary.json
├── scripts/
│   ├── sql_to_csv.py                 # Parse dump MySQL → CSV
│   ├── mysql_export.py               # Export direct MySQL → CSV (production)
│   ├── retrain.sh                    # Pipeline retrain automatique
│   ├── generate_fake_models.py       # Modèles factices pour CI
│   └── docker-entrypoint-scheduler.sh
├── tests/
│   └── test_api.py                   # 32 tests unitaires
├── docs/
│   ├── technical_documentation.md
│   └── quickstart.md
├── .github/workflows/
│   └── ci-cd.yml                     # GitHub Actions
├── .env.example                      # Template de configuration
├── Dockerfile                        # Image API
├── Dockerfile.train                  # Image entraînement
├── Dockerfile.scheduler              # Image scheduler (cron)
├── docker-compose.yml
└── requirements.txt
```

---

## Modèles implémentés

### Logique commune — marketplace d'occasion

Sur MEL Cameroun, chaque article est vendu une seule fois. Recommander un article spécifique n'a pas de sens : il peut déjà être vendu à l'instant où l'utilisateur voit la recommandation.

**Solution :** le modèle apprend des préférences par **catégorie** (vêtements femmes, chaussures hommes, etc.). L'API traduit ensuite ces catégories en articles **actuellement disponibles**.

```
Modèle → [Catégorie A, Catégorie B, ...]
API    → articles disponibles dans Catégorie A + Catégorie B + ...
```

### Signaux d'interaction utilisés

| Signal | Source SQL | Score |
|---|---|---|
| Vue d'article | `shetabit_visits` | 0.2 – 1.0 |
| Ajout au panier | `paniers` | 3.0 |
| Achat (facture valide) | `factures` | 5.0 |

### 1. Baseline — Popularité
Recommande les catégories les plus interagies globalement.
Fallback cold-start pour les nouveaux utilisateurs.

### 2. SVD — Singular Value Decomposition
- Matrice user × catégorie centrée par utilisateur
- Décomposition SVD tronquée via `scipy.sparse.linalg.svds`
- Tuning du nombre de facteurs latents : **k = 150**

### 3. ALS — Alternating Least Squares
- Implémentation from scratch dans `src/recommender.py` (numpy/scipy uniquement)
- Signal implicite : confidence `c_ui = 1 + 40 × score`
- Résolution par système linéaire `(YᵀCᵘY + λI) xᵤ = YᵀCᵘpᵤ`
- Tuning : **n_factors = 32, regularization = 0.1**

### 4. Content-Based — TF-IDF + Cosine
- Features : nom de la catégorie + noms des articles disponibles dans la catégorie
- Vectorisation TF-IDF (500 features max)
- Similarité cosine entre catégories
- Recommande les catégories similaires à celles déjà achetées/visitées

### 5. Hybride — SVD + Content-Based
- Score final : `0.6 × score_SVD + 0.4 × score_CB`
- Meilleur compromis entre filtrage collaboratif et similarité de contenu

---

## Données

Le pipeline utilise les données de MEL Cameroun exportées depuis MySQL.

**Tables source :**

| Table MySQL | Rôle |
|---|---|
| `factures` | Commandes → `orders.csv` + `order_items.csv` |
| `articles` | Catalogue produits → `products.csv` (avec `available`) |
| `article_categories` + `categories` | Catégories → `products.csv` + `category_names.csv` |
| `paniers` | Ajouts au panier → `panier_interactions.csv` |
| `etoiles` | Avis clients → `reviews.csv` |
| `shetabit_visits` | Visites d'articles → `visits.csv` |

**Colonnes clés de `products.csv` :**

| Colonne | Description |
|---|---|
| `product_id` | Identifiant de l'article |
| `product_name` | Nom de l'article |
| `product_category_name` | Catégorie (en français) |
| `available` | **1 = disponible, 0 = vendu** |

---

## Installation

### Prérequis
- Python 3.14+
- pip
- Docker (pour le déploiement)

### 1. Cloner le projet

```bash
git clone https://github.com/willyshonen/systeme-de-recommandation.git
cd systeme-de-recommandation
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer l'environnement

```bash
cp .env.example .env
# Éditer .env avec les paramètres de votre environnement
```

### 4. Préparer les données

**Option A — depuis un dump SQL :**
```bash
# Placer le dump dans data/raw/mysql-mel.sql puis :
python scripts/sql_to_csv.py --sql data/raw/mysql-mel.sql --out data/raw
```

**Option B — connexion directe MySQL (production) :**
```bash
python scripts/mysql_export.py \
  --host     db.melcameroun.com \
  --database mel_cameroun \
  --user     mel_readonly \
  --password secret \
  --out      data/raw
```

---

## Utilisation

### Entraîner le modèle

```bash
python src/train.py \
  --data-dir   data/raw \
  --models-dir models \
  --k-factors  150 \
  --als-factors 32 \
  --als-reg    0.1 \
  --alpha      1.0
```

### Lancer l'API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

→ Documentation interactive : http://127.0.0.1:8000/docs

---

## API FastAPI

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut + nb users, catégories, articles disponibles |
| GET | `/metrics` | Métriques NDCG/Precision/Recall |
| GET | `/popular?n=10` | Articles disponibles les plus populaires |
| GET | `/popular?category=X` | Articles disponibles dans une catégorie |
| GET | `/recommend/{user_id}?n=10&model=hybrid` | Articles disponibles recommandés |
| GET | `/similar/{product_id}?n=10` | Articles disponibles dans des catégories similaires |
| POST | `/reload?secret=XXX` | Rechargement zero-downtime des modèles |

### Exemples

```bash
# Statut de l'API
curl http://localhost:8000/health
# {"status":"ok","model":"Hybrid","n_users":34,"n_categories":7,"n_available_products":148}

# Recommandations pour un utilisateur connu
curl "http://localhost:8000/recommend/42?n=10&model=hybrid"
# {"user_id":"42","model":"hybrid","recommended_categories":["Vêtements femmes",...],"recommendations":["12","47","83",...],"n":10}

# Utilisateur inconnu → cold-start automatique
curl "http://localhost:8000/recommend/nouveau_client"
# {"model":"popularity (cold-start)", ...}

# Articles dans des catégories similaires
curl "http://localhost:8000/similar/47?n=5"
# {"product_id":"47","product_category":"Chaussures femmes","similar_categories":[...],"similar_products":["23","61",...],"n":5}

# Articles populaires disponibles dans une catégorie
curl "http://localhost:8000/popular?category=Vêtements+femmes&n=10"
```

---

## Pipeline MLOps

```
MySQL MEL
   │
   ▼ sql_to_csv.py ou mysql_export.py
data/raw/ (CSV avec colonne available)
   │
   ▼ src/train.py
models/ (SVD, ALS, Content-Based, products_catalog.csv)
   │
   ▼ api/main.py
API :8000
   │
   ├─ GET /recommend/{user_id}  → articles disponibles dans catégories recommandées
   ├─ GET /similar/{product_id} → articles dans catégories similaires
   └─ GET /popular              → articles les plus populaires disponibles
```

### Re-entraîner manuellement

```bash
python src/train.py
# Puis recharger l'API sans downtime :
curl -X POST "http://localhost:8000/reload?secret=votre-secret"
```

---

## Docker

### Démarrage complet

```bash
# Copier et configurer l'environnement
cp .env.example .env

# Démarrage : API + MLflow + Scheduler
docker compose up --build -d

# API        → http://localhost:8000
# MLflow UI  → http://localhost:5000
# Scheduler  → retrain automatique chaque dimanche 2h
```

### Commandes utiles

```bash
# Retrain ponctuel manuel
docker compose --profile train up trainer

# Voir les logs du scheduler
docker logs mel-scheduler -f

# Recharger l'API après retrain manuel
curl -X POST "http://localhost:8000/reload?secret=$(grep RELOAD_SECRET .env | cut -d= -f2)"

# Statut des containers
docker compose ps
```

### Variables d'environnement

Voir `.env.example` pour la liste complète. Variables clés :

| Variable | Défaut | Description |
|---|---|---|
| `RELOAD_SECRET` | _(vide)_ | Clé pour `POST /reload`. **Vide = endpoint ouvert** : à renseigner impérativement sur une machine exposée (`openssl rand -hex 32`). |
| `RETRAIN_SCHEDULE` | `0 2 * * 0` | Cron du retrain (dimanche 2h) |
| `RETRAIN_ON_START` | `0` | `1` = retrain immédiat au démarrage |
| `MYSQL_HOST` | _(vide)_ | Hôte MySQL (production) |
| `MYSQL_PASSWORD` | _(vide)_ | Mot de passe MySQL |
| `NOTIFY_WEBHOOK` | _(vide)_ | Webhook Slack/Discord |

---

## Automatisation

Le système se ré-entraîne **automatiquement** chaque semaine sans intervention humaine.

### Architecture

```
Container : scheduler
  │
  cron (RETRAIN_SCHEDULE)
  │
  scripts/retrain.sh
  │
  ├─ Étape 1 : Export données
  │    MYSQL_HOST défini → mysql_export.py (connexion directe)
  │    sinon             → sql_to_csv.py   (parse dump SQL)
  │
  ├─ Étape 2 : Entraînement
  │    src/train.py → models/
  │
  └─ Étape 3 : Reload API
       POST /reload → modèles rechargés sans redémarrage
       Notification Slack/Discord (si NOTIFY_WEBHOOK défini)
```

### Notification après chaque retrain

```
✅ MEL Recommender — Retrain success
   NDCG@10=0.0325 | 45 users | 9 catégories
   Durée : 47s | 2026-10-05 02:00
```

---

## CI/CD

Pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) sur chaque push `main` :

```
push main
  │
  ▼
[1] Tests (pytest 32/32) + Linting (ruff)
  │
  ▼
[2] Génération modèles factices + train factice
  │
  ▼
[3] Build image Docker → push GHCR
  │
  ▼
[4] Smoke test : /health + /popular
```

---

## Tests

```bash
pytest tests/ -v
# 32/32 tests passent
```

**Couverture des tests :**
- `/health` — champs, nb articles disponibles (exclus les vendus)
- `/metrics` — structure de la réponse
- `/popular` — exclusion articles vendus, filtre par catégorie
- `/recommend` — SVD / ALS / Hybrid, cold-start, articles vendus exclus
- `/similar` — exclusion de l'article source, articles vendus exclus, 404 si inconnu
- `/reload` — sans secret, avec bonne/mauvaise clé

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

- [x] **Adaptation marketplace d'occasion** — recommandation par catégorie + filtre `available`
- [x] **Export MySQL → CSV** — `mysql_export.py` (connexion directe) + `sql_to_csv.py` (dump)
- [x] **Retrain automatique** — cron hebdomadaire via container `scheduler` Docker
- [x] **Rechargement zero-downtime** — endpoint `POST /reload`
- [x] **Notifications** — webhook Slack/Discord après chaque retrain
- [ ] **Intégration Laravel** — appel `Http::get("http://api/recommend/{user_id}")` depuis melcameroun.com
- [ ] **Sécurité** — clé API sur tous les endpoints ou réseau privé Docker
- [ ] **Features enrichies** — utiliser `marque` et `description` des articles pour le Content-Based
- [ ] **A/B Testing** — mesurer l'impact des recommandations sur les conversions
- [ ] **Monitoring** — Prometheus/Grafana pour suivre les métriques en production
- [ ] **Neural CF** — embeddings par réseau de neurones (NCF, Two-Tower)

---

## Stack technique

| Outil | Usage |
|---|---|
| Python 3.14+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scipy | SVD tronqué, matrices sparse |
| scikit-learn | Encodage, TF-IDF, métriques |
| FastAPI | API REST |
| uvicorn | Serveur ASGI |
| pymysql | Export direct MySQL |
| MLflow | Tracking des expériences |
| Docker | Conteneurisation (API + Scheduler) |
| GitHub Actions | CI/CD |
| pytest | 32 tests unitaires |
| matplotlib / seaborn | Visualisations |
| Jupyter | Notebook interactif |

---

## Auteur

Projet développé dans le cadre de l'amélioration de la plateforme e-commerce [melcameroun.com](https://melcameroun.com).
