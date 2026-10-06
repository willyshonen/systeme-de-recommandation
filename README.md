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
- [Structure du notebook](#structure-du-notebook)
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
| **Hybride** | SVD + Content-Based pondérés, **α choisi par validation croisée** (LOO, 3 folds) |

> ✅ **Compatible Python 3.13 et 3.14** — aucune dépendance Cython.
> Les images Docker et le runner CI sont en **3.13** (cible de déploiement),
> le développement local se fait en 3.14 : la suite de 159 tests passe sur les deux.

> 🚦 **Porte de suffisance.** Le modèle hybride n'est servi que si la validation croisée prouve qu'il bat la baseline de popularité (`cv_lift_at_k > 0`). Sinon l'API classe par popularité — ce qui reste personnalisé (les catégories déjà consommées sont exclues) mais sans combinaison invérifiée. Ce n'est pas caché : `/health` renvoie `serving_mode`, `/metrics` expose `cv_lift_at_k`, et les réponses `/recommend` portent `model` = « `popularity (personnalisation non validée)` ». Un résumé d'entraînement ancien (sans champ `alpha_cv`) est traité comme non validé.

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
│   ├── events.py                     # Chargement des événements (source unique)
│   ├── recommender.py                # Classe ALSRecommender (partagée)
│   └── train.py                      # Pipeline d'entraînement CLI
├── api/
│   └── main.py                       # API FastAPI (8 endpoints)
├── models/                           # Artefacts entraînés
│   ├── svd_R_pred.npy                # Matrice user × catégorie
│   ├── als_model.pkl
│   ├── cosine_sim.pkl                # Similarité entre catégories
│   ├── user_encoder.pkl
│   ├── item_encoder.pkl              # Encode les noms de catégories
│   ├── popular_items.pkl
│   ├── user_history.pkl              # user_id → catégories consommées (score CB)
│   ├── product_sim.pkl               # Similarité cosine entre TITRES d'articles
│   ├── product_features.csv          # Features article (prix, note, catégorie)
│   ├── products_catalog.csv          # Catalogue avec colonne available
│   └── results_summary.json          # Métriques + verdict de la CV (alpha_cv)
├── scripts/
│   ├── sql_to_csv.py                 # Parse dump MySQL → CSV
│   ├── mysql_export.py               # Export direct MySQL → CSV (production)
│   ├── retrain.sh                    # Pipeline retrain automatique
│   ├── check_test_layout.py          # Garde-fou AST (CI) : tests imbriqués/dupliqués
│   ├── compare_metrics.py            # Régression NDCG entre deux résultats
│   ├── generate_fake_data.py         # Dataset synthétique (CI)
│   ├── generate_fake_models.py       # Modèles factices (CI)
│   ├── md_to_docx.py                 # Conversion de la documentation
│   └── docker-entrypoint-scheduler.sh
├── tests/
│   ├── test_api.py                   # 101 cas (endpoints, gate, modèles)
│   ├── test_events.py                # 26 tests (chargement, statuts de commande)
│   ├── test_train.py                 # 14 tests (CV, évaluation)
│   └── test_compare_metrics.py       # 18 tests
├── docs/
│   ├── technical_documentation.md
│   ├── DEPLOIEMENT.md
│   └── quickstart.md
├── .github/workflows/
│   └── ci-cd.yml                     # GitHub Actions (4 jobs)
├── .env.example                      # Template de configuration
├── .dockerignore
├── Dockerfile                        # Image API (modèles montés en volume)
├── Dockerfile.ci                     # Image API sans modèles (smoke test CI)
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

`src/events.py::load_events()` est la **source unique** : tous les signaux sont montés dans une seule table (user, catégorie, score), et le score retenu est le **maximum** par paire.

| Signal | Source | Score |
|---|---|---|
| Vue d'article | `shetabit_visits` / `POST /events` | 0.2 – 1.0 (selon le nombre de vues) |
| Ajout au panier | `paniers` | 3.0 |
| Avis client | `etoiles` | 4.0 |
| Achat (facture valide) | `factures` | 5.0 |

**Statuts de commande :** seules les factures livrées/valides comptent (`_STATUS_OK`). Les annulations, remboursements et commandes en attente sont écartées (`_STATUS_KO`). Un statut **inconnu** ne bloque pas l'export : il est signalé par un warning explicite disant quel statut ajouter à `_STATUS_OK` — un `factures.csv` vide doit être explicable dans les logs, pas découvert après coup.

### 1. Baseline — Popularité
Recommande les catégories les plus interagies globalement.
Fallback cold-start pour les nouveaux utilisateurs.

### 2. SVD — Singular Value Decomposition
- Matrice user × catégorie centrée par utilisateur
- SVD **exact** (`numpy.linalg.svd`, LAPACK) puis troncature — déterministe, contrairement à `scipy.sparse.linalg.svds` (ARPACK, itératif) qui produisait des métriques non reproductibles d'un run à l'autre
- Tuning du nombre de facteurs latents : **k = 150**, plafonné au rang effectif de la matrice `min(n_users, n_items) − 1` (soit **6** avec 7 catégories) — valeur reportée dans `results_summary.json` (`svd_k_factors_effective`)

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
- Score final : `α × score_SVD + (1-α) × score_CB`
- **α déterminé par validation croisée** (grille `0.0 … 1.0`, 3 folds LOO, NDCG moyen) plutôt que fixé à la main : `--alpha` n'est qu'un forçage de debug (`python src/train.py --alpha 0.6`)
- `results_summary.json` conserve `alpha_cv` (α retenu, lift vs baseline, score par α) — c'est ce que la porte de suffisance lit au chargement
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

> ⚠️ **`visits.csv` est vide.** Aucun modèle n'a donc jamais vu de vue produit — la personnalisation fine ne pourra pas être validée, même après des mois de production, tant que le front n'envoie rien. C'est pour cela que `POST /events` existe : c'est la source d'événements que melcameroun peut alimenter dès aujourd'hui (sans identifiant utilisateur : `session_id` suffit).

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
- Python 3.13 ou 3.14 (CI et Docker : 3.13)
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
  --als-reg    0.1
```

L'α est sélectionné par validation croisée. `--alpha 0.6` force une valeur (debug uniquement) et la CV n'est alors pas produite : le résumé n'aura pas de `alpha_cv`, la porte restera fermée.

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
| GET | `/health` | Statut, **`serving_mode` (ce qui classe réellement)**, nb users/catégories/articles |
| GET | `/metrics` | Métriques NDCG/Precision/Recall + `serving_mode` + `cv_lift_at_k` |
| GET | `/popular?n=10` | Articles disponibles les plus populaires |
| GET | `/popular?category=X` | Articles disponibles dans une catégorie |
| GET | `/recommend/{user_id}?n=10&model=hybrid` | Articles disponibles recommandés |
| POST | `/recommend/session` | Recommandations pour un **visiteur anonyme** (catégories/articles de la session) |
| GET | `/similar/{product_id}?n=10` | Articles disponibles dans des catégories similaires |
| POST | `/events` | Ingestion d'un lot d'événements (vues, paniers, achats) par le front |
| POST | `/reload?secret=XXX` | Rechargement zero-downtime des modèles |

`model` accepte `hybrid` (défaut), `svd`, `als` — toute autre valeur renvoie `422`. `svd` et `als` court-circuitent volontairement la porte : ce sont des diagnostics, et la réponse le dit en clair (`"(non validé en CV — demandé explicitement)"`) plutôt que de se faire passer pour de la personnalisation. L'α du hybride est lu dans `results_summary.json` (source de vérité alignée sur l'entraînement).

### Exemples

```bash
# Statut de l'API — ce qui CLASSE, pas seulement ce qui est chargé
curl http://localhost:8000/health
# {"status":"ok","model":"Hybrid",
#  "serving_mode":"popularité (personnalisation non validée, lift = -0.0502)",
#  "n_users":34,"n_categories":7,"n_available_products":191}
# ↑ valeurs de l'artefact livré : la CV ne valide pas l'hybride, la porte est fermée

# Recommandations pour un utilisateur connu
curl "http://localhost:8000/recommend/42?n=10&model=hybrid"
# {"user_id":"42","model":"hybrid","recommended_categories":["Vêtements femmes",...],"recommendations":["12","47","83",...],"n":10}

# Utilisateur inconnu → cold-start automatique
curl "http://localhost:8000/recommend/nouveau_client"
# {"model":"popularity (cold-start)", ...}

# Visiteur anonyme : personnalisation depuis le contexte de session
curl -X POST http://localhost:8000/recommend/session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"abc123","product_ids":["12","47"],"n":5}'
# {"session_id":"abc123","model":"...","granularity":"produit", ...}

# Lot d'événements renvoyé par le front après une navigation
# Champ canonique : event_type (type est accepté comme alias).
# Clé inconnue → 422, pas un faux "view" silencieux.
curl -X POST http://localhost:8000/events \
  -H "Content-Type: application/json" \
  -d '{"session_id":"abc123","user_id":null,
       "events":[{"event_type":"view","product_id":12},
                 {"event_type":"cart","product_id":47}]}'
# {"accepted":2,"rejected_unknown_product":0,"rejected_invalid_type":0,
#  "session_id":"abc123","user_id":null,"details":[]}

# Articles dans des catégories similaires
curl "http://localhost:8000/similar/47?n=5"
# {"product_id":"47","product_category":"Chaussures femmes","similar_categories":[...],"similar_products":["23","61",...],"n":5}

# Articles populaires disponibles dans une catégorie
curl "http://localhost:8000/popular?category=Vêtements+femmes&n=10"
```

---

## Pipeline MLOps

```
MySQL MEL                                   melcameroun.com (front)
   │                                              │
   ▼ sql_to_csv.py / mysql_export.py              ▼ POST /events
data/raw/ (CSV avec colonne available)      événements (session_id, type, produit)
   │                                              │
   └──────────────┬───────────────────────────────┘
                  ▼ src/events.py  · load_events() = source unique
              interactions (user, catégorie, score max)
                  │
                  ▼ src/train.py  · CV alpha → results_summary.json (alpha_cv)
              models/ (SVD, ALS, Content-Based, products_catalog.csv)
                  │
                  ▼ api/main.py — porte de suffisance lue au chargement
              API :8000
                  │
                  ├─ GET  /recommend/{user_id}    → articles dispo dans les catégories
                  ├─ POST /recommend/session      → idem, visiteur anonyme
                  ├─ GET  /similar/{product_id}   → catégories/articles similaires
                  ├─ GET  /popular                → articles populaires disponibles
                  ├─ GET  /health                 → serving_mode (ce qui classe)
                  └─ POST /reload                 → rechargement sans downtime
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
   NDCG@7=0.9736 | 34 users | 7 catégories
   Durée : 47s | 2026-10-05 02:00
```

---

## CI/CD

Pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) — déclencheurs : `push` sur `main` et `develop`, `pull_request` sur `main`. 4 jobs :

```
push main
  │
  ▼
[1] Tests & Linting
    • ruff (src/ api/ tests/ scripts/…)
    • scripts/check_test_layout.py → refuse les tests imbriqués,
      dupliqués ou masqués par une classe écrasée (AST)
    • pytest — 159 tests
  │
  ▼
[2] Train & Track (MLflow)  · needs: [test], main uniquement
    • dataset synthétique (data/raw/ est gitignoré)
    • entraînement réel avec la commande du README
    • REPRODUCTIBILITÉ : 2 entraînements successifs →
      artefacts identiques au SHA-256 + résumé identique
      hors training_time_sec
    • INTÉGRITÉ : échoue si n_users_eval=0, si
      user_history.pkl ou products_catalog.csv manquent,
      ou si aucun results_summary.json n'a été produit
    • lift négatif → ::warning (voir ci-dessous)
  │
  ▼
[3] Build Docker Image  · needs: [train]
    • buildx → ghcr.io (Dockerfile.ci, sans modèles)
    • exporte image_tag pour l'étape suivante
    • ne consomme aucun artefact du job [2] : les modèles de production
      vivent dans le volume `./models` du VPS et sont mis à jour par le
      scheduler (`scripts/retrain.sh`) — les artefacts uploadés ici
      servent à la traçabilité, pas au déploiement
  │
  ▼
[4] Smoke Test API  · needs: [build]
    • lance L'IMAGE poussée (pas une réinstallation locale),
      modèles montés en volume — ce qui teste le conteneur
      réellement livré
    • /health (serving_mode) + /popular + /events
      (contrat du front) + /recommend (lit les modèles montés)
```

**Sur une pull request, seul le job [1] tourne** : les jobs [2] à [4] portent `if: github.ref == 'refs/heads/main'`. Un PR donne donc un retour rapide (lint + tests) sans brûler de minutes d'entraînement ni pousser d'image.

**Un lift négatif ne casse pas la CI, et c'est voulu.** Sur un petit jeu de données, un hybride qui perd contre la popularité est un résultat honnête — l'échec serait au contraire de masquer. La conséquence est prise au **service**, pas en build : la porte de suffisance lisant `alpha_cv.lift_at_k` bascule `/recommend` sur la popularité, et `/health` l'annonce via `serving_mode`.

> **Note de plateforme.** Les bits exécutables sont vérifiés par Ruff (`EXE001`/`EXE002`) **uniquement sous POSIX** : sous Windows la règle ne se déclenche jamais. Un lint local vert ne prouve donc rien pour le runner — d'où `scripts/check_test_layout.py` et la vérification de reproductibilité, pensés pour être rejouables dans `python:3.13-slim` avec le tar `git archive`.

---

## Tests

```bash
pytest tests/ -v
# 159 passed

# Garde-fou structure (également exécuté en CI)
python scripts/check_test_layout.py tests/
```

**Couverture des tests :**

| Fichier | Tests | Couverture |
|---|---|---|
| `test_api.py` | 101 | endpoints, conversion catégories → articles, exclusions |
| `test_events.py` | 26 | `load_events()`, statuts de commande, sources manquantes |
| `test_train.py` | 14 | CV α, évaluation, fallback popularité |
| `test_compare_metrics.py` | 18 | comparaison de régression NDCG |

- `/health` — champs, nb articles disponibles (exclus les vendus), **`serving_mode`**
- `/metrics` — structure de la réponse, **`cv_lift_at_k`**
- `/popular` — exclusion articles vendus, filtre par catégorie
- `/recommend` — SVD / ALS / Hybrid, cold-start, articles vendus exclus,
  **porte de suffisance** (lift CV ≤ 0 → popularité + libellé explicite),
  `svd`/`als` demandés explicitement → libellé « non validé en CV »
- `/recommend/session` — granularité produit vs catégorie, catégories hors espace
- `/events` — types d'événement valides, produit inconnu, session/user absents
- `/similar` — exclusion de l'article source, articles vendus exclus, 404 si inconnu
- `/reload` — sans secret, avec bonne/mauvaise clé
- statuts de commande — `valide`/`livré` acceptés, `annulé`/`remboursé` écartés,
  statut **inconnu** → warning explicite (pas de rejet silencieux)

`scripts/check_test_layout.py` existe parce que ce genre d'erreur est **silencieux** : pytest ne voit que ce qu'il collecte, donc un test jamais collecté ne rate jamais. Il parse l'AST de `tests/` et sort en erreur sur trois cas :

1. un `def test_…` écrit **à l'intérieur** d'un autre `def test_…` (une réindentation accidentelle le transforme en closure) ;
2. un test dans une **classe redéclarée** plus bas dans le même fichier (la seconde définition écrase la première) ;
3. un test défini **deux fois**.

C'est littéralement ce qui s'était produit : 5 tests imbriqués dans `test_recommend_no_sold_articles` et une classe `TestMetrics` dupliquée n'étaient jamais exécutés, sans la moindre alerte — la suite était au vert en étant moins complète.

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
- [x] **Collecte d'événements** — `POST /events` (lot du front) + `POST /recommend/session` (visiteur anonyme), `load_events()` comme source unique
- [x] **CI réellement vérifiée** — 4 jobs, 159 tests, entraînement reproductible, smoke test sur l'image poussée
- [ ] **Rotation de `RELOAD_SECRET`** — l'ancien secret est **public dans l'historique git** : régénérer (`openssl rand -hex 32`) avant mise en production, ou purger l'historique
- [ ] **Envoyer les événements depuis melcameroun.com** — `visits.csv` est vide : c'est le signal qui manque pour valider toute personnalisation fine
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
| Python 3.13 / 3.14 | Langage principal (cible Docker/CI : 3.13) |
| pandas / numpy | Manipulation des données, SVD exact (LAPACK) |
| scipy | Matrices sparse (CSR), statistiques |
| scikit-learn | Encodage, TF-IDF, métriques |
| FastAPI | API REST |
| uvicorn | Serveur ASGI |
| pymysql | Export direct MySQL |
| MLflow | Tracking des expériences |
| Docker | Conteneurisation (API + Scheduler) |
| GitHub Actions | CI/CD |
| pytest | 159 tests unitaires |
| matplotlib / seaborn | Visualisations |
| Jupyter | Notebook interactif |

---

## Auteur

Projet développé dans le cadre de l'amélioration de la plateforme e-commerce [melcameroun.com](https://melcameroun.com).
