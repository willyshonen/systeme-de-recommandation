"""
api/main.py
API FastAPI — Système de recommandation MEL Cameroun (marketplace d'occasion).

Logique marketplace d'occasion :
  1. Chaque article est UNIQUE → il disparaît dès qu'il est vendu
  2. Le modèle recommande des CATÉGORIES (collaborative filtering user×catégorie)
  3. L'API traduit ces catégories en articles DISPONIBLES (available=1)
  4. Les articles vendus sont automatiquement exclus de toutes les réponses

Endpoints :
    GET /health                         → statut de l'API
    GET /metrics                        → métriques du modèle
    GET /recommend/{user_id}?n=10       → articles disponibles dans les catégories recommandées
    GET /similar/{product_id}?n=10      → articles disponibles dans des catégories similaires
    GET /popular?n=10                   → articles disponibles les plus populaires
"""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Import nécessaire pour que pickle résolve ALSRecommender au chargement
from src.recommender import ALSRecommender  # noqa: F401

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts
# ─────────────────────────────────────────────────────────────────────────────

class ModelStore:
    """
    Conteneur pour tous les artefacts du modèle.

    Après entraînement le modèle travaille au niveau CATÉGORIE :
      - user_enc / item_enc encodent les IDs utilisateurs et les noms de catégories
      - products_catalog : table complète des articles avec colonnes available et product_category_name
      - cat_to_products  : dict {catégorie → [product_id disponibles triés par popularité]}
    """

    def __init__(self):
        self.user_enc        = None
        self.item_enc        = None   # encode les catégories
        self.R_pred          = None
        self.R_pred_norm     = None
        self.als_model       = None
        self.cosine_sim      = None
        self.item_to_idx     = None   # catégorie → index CB
        self.idx_to_item     = None   # index CB → catégorie
        self.popular_items   = None   # liste de catégories populaires
        self.summary         = None
        self.products_catalog: pd.DataFrame = pd.DataFrame()
        self.cat_to_products: dict[str, list] = {}   # catégorie → [article_id disponibles]
        self.popular_products: list = []             # articles disponibles les plus populaires
        self.ready           = False
        self._models_dir: Path = MODELS_DIR          # mémorisé au premier load()

    def load(self, models_dir: Path = MODELS_DIR):
        # Mémorise le chemin pour que /reload puisse le réutiliser
        self._models_dir = models_dir
        log.info("Chargement des modèles depuis %s", models_dir)
        try:
            with open(models_dir / "user_encoder.pkl", "rb") as f:
                self.user_enc = pickle.load(f)
            with open(models_dir / "item_encoder.pkl", "rb") as f:
                self.item_enc = pickle.load(f)  # encode les catégories

            self.R_pred = np.load(models_dir / "svd_R_pred.npy")
            r_min = self.R_pred.min(axis=1, keepdims=True)
            r_max = self.R_pred.max(axis=1, keepdims=True)
            denom = r_max - r_min
            denom[denom == 0] = 1
            self.R_pred_norm = (self.R_pred - r_min) / denom

            with open(models_dir / "als_model.pkl", "rb") as f:
                self.als_model = pickle.load(f)

            with open(models_dir / "cosine_sim.pkl", "rb") as f:
                self.cosine_sim, self.item_to_idx, self.idx_to_item = pickle.load(f)

            with open(models_dir / "popular_items.pkl", "rb") as f:
                self.popular_items = pickle.load(f)  # liste de catégories populaires

            with open(models_dir / "results_summary.json") as f:
                self.summary = json.load(f)

            # Catalogue produits avec disponibilité
            catalog_path = models_dir / "products_catalog.csv"
            if catalog_path.exists():
                self.products_catalog = pd.read_csv(catalog_path)
                self._build_availability_index()
            else:
                # Fallback : lit depuis data/raw/products.csv
                products_path = models_dir.parent / "data" / "raw" / "products.csv"
                if products_path.exists():
                    self.products_catalog = pd.read_csv(products_path)
                    self._build_availability_index()
                else:
                    log.warning("Catalogue produits introuvable — filtre disponibilité désactivé")

            self.ready = True
            log.info(
                "Modèles chargés — %d users | %d catégories | %d articles disponibles",
                len(self.user_enc.classes_),
                len(self.item_enc.classes_),
                len(self.popular_products),
            )
        except Exception as e:
            log.error("Erreur au chargement des modèles : %s", e)
            raise

    def _build_availability_index(self):
        """
        Construit deux index à partir du catalogue produits :
          1. cat_to_products : {catégorie → [product_id disponibles]}
          2. popular_products : articles disponibles triés par popularité de catégorie
        """
        df = self.products_catalog.copy()

        # Assurer les bons types
        df["product_id"] = df["product_id"].astype(str)
        if "product_category_name" not in df.columns:
            log.warning("Colonne 'product_category_name' absente du catalogue")
            return

        # Filtre disponibilité
        if "available" in df.columns:
            available_df = df[df["available"] == 1].copy()
        else:
            log.warning("Colonne 'available' absente — tous les articles considérés disponibles")
            available_df = df.copy()

        # Index catégorie → articles disponibles
        self.cat_to_products = (
            available_df.groupby("product_category_name")["product_id"]
            .apply(list)
            .to_dict()
        )

        n_available = len(available_df)
        n_sold      = len(df) - n_available
        log.info(
            "Index disponibilité : %d articles dispo, %d vendus, %d catégories",
            n_available, n_sold, len(self.cat_to_products),
        )

        # Articles populaires disponibles : triés par ordre des catégories populaires
        seen: set = set()
        result: list = []
        for cat in (self.popular_items or []):
            cat_str = str(cat)
            for pid in self.cat_to_products.get(cat_str, []):
                if pid not in seen:
                    seen.add(pid)
                    result.append(pid)
        self.popular_products = result


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    log.info("━" * 55)
    log.info("🚀  MEL Recommender API — prêt")
    log.info("━" * 55)
    log.info("  📡  API REST    → http://localhost:8000")
    log.info("  📖  Swagger UI  → http://localhost:8000/docs")
    log.info("  📘  ReDoc       → http://localhost:8000/redoc")
    log.info("  📊  MLflow UI   → http://localhost:5000")
    log.info("━" * 55)
    yield


# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MEL Cameroun — API Recommandation",
    description=(
        "Système de recommandation pour marketplace d'occasion. "
        "Le modèle recommande des catégories, l'API retourne les articles disponibles."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ready():
    if not store.ready:
        raise HTTPException(status_code=503, detail="Modèles non chargés")


def _categories_to_products(categories: list[str], n: int) -> list[str]:
    """
    Traduit une liste de catégories recommandées en articles disponibles.

    Pour chaque catégorie (dans l'ordre de score décroissant), on ajoute
    les articles disponibles jusqu'à atteindre n résultats.
    Résultat : articles triés par pertinence de catégorie, uniques, disponibles.
    """
    seen: set = set()
    result: list = []
    for cat in categories:
        for pid in store.cat_to_products.get(str(cat), []):
            if pid not in seen:
                seen.add(pid)
                result.append(pid)
                if len(result) >= n:
                    return result
    return result


def _get_cb_scores_cat(user_items: list) -> np.ndarray | None:
    """
    Scores content-based sur l'espace des catégories (item_enc).
    user_items = liste de catégories vues/achetées par l'utilisateur.
    """
    items_in_cb = [i for i in user_items if i in store.item_to_idx]
    if not items_in_cb:
        return None
    n_cats    = len(store.item_enc.classes_)
    scores_cb = np.zeros(len(store.cosine_sim))
    for cat in items_in_cb:
        scores_cb += store.cosine_sim[store.item_to_idx[cat]]
    full_scores = np.zeros(n_cats)
    for cb_idx, cat in store.idx_to_item.items():
        try:
            full_scores[int(store.item_enc.transform([cat])[0])] = scores_cb[cb_idx]
        except ValueError:
            pass
    s_min, s_max = full_scores.min(), full_scores.max()
    if s_max > s_min:
        full_scores = (full_scores - s_min) / (s_max - s_min)
    return full_scores


def _top_categories(user_id: str, model: str = "hybrid", n_cats: int = 20) -> list[str]:
    """
    Retourne les N catégories les mieux scorées pour un utilisateur.
    model : 'svd' | 'als' | 'hybrid'
    """
    user_idx = int(store.user_enc.transform([user_id])[0])

    if model == "als":
        top_idxs = store.als_model.recommend(user_idx, n=n_cats)
        return store.item_enc.inverse_transform(top_idxs).tolist()

    # SVD scores
    svd_scores = store.R_pred_norm[user_idx]

    if model == "svd":
        top_idxs = np.argsort(svd_scores)[::-1][:n_cats]
        return store.item_enc.inverse_transform(top_idxs).tolist()

    # Hybrid : SVD + Content-Based
    # Pour récupérer les catégories de l'user, on reconstruit depuis item_enc
    # On utilise les scores SVD élevés comme proxy des catégories connues de l'user
    known_cats = store.item_enc.inverse_transform(
        np.argsort(store.R_pred[user_idx])[::-1][:10]
    ).tolist()
    cb_scores = _get_cb_scores_cat(known_cats)

    if cb_scores is None:
        hybrid = svd_scores
    else:
        hybrid = 0.6 * svd_scores + 0.4 * cb_scores  # alpha fixé à 0.6

    top_idxs = np.argsort(hybrid)[::-1][:n_cats]
    return store.item_enc.inverse_transform(top_idxs).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Schémas de réponse
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    n_users: int
    n_categories: int
    n_available_products: int


class RecommendationResponse(BaseModel):
    user_id: str
    model: str
    recommended_categories: list[str]
    recommendations: list[str]   # articles disponibles dans ces catégories
    n: int


class SimilarResponse(BaseModel):
    product_id: str
    product_category: str
    similar_categories: list[str]
    similar_products: list[str]  # articles disponibles dans les catégories similaires
    n: int


class PopularResponse(BaseModel):
    popular_products: list[str]
    n: int


class MetricsResponse(BaseModel):
    model: str
    hyperparameters: dict
    metrics: dict
    n_users: int
    n_items: int
    n_interactions: int


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health():
    """Statut de l'API et informations sur le modèle chargé."""
    _check_ready()
    return HealthResponse(
        status="ok",
        model=store.summary.get("model", "Hybrid"),
        n_users=len(store.user_enc.classes_),
        n_categories=len(store.item_enc.classes_),
        n_available_products=len(store.popular_products),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Système"])
def metrics():
    """Métriques d'évaluation et hyperparamètres du modèle en production."""
    _check_ready()
    s = store.summary
    return MetricsResponse(
        model=s.get("model", "Hybrid"),
        hyperparameters=s.get("hyperparameters", {}),
        metrics=s.get("metrics", s.get("results", {})),
        n_users=s.get("n_users", 0),
        n_items=s.get("n_items", 0),
        n_interactions=s.get("n_interactions", 0),
    )


@app.get("/popular", response_model=PopularResponse, tags=["Recommandation"])
def popular(
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de produits")] = 10,
    category: Annotated[str | None, Query(description="Filtrer par catégorie")] = None,
):
    """
    Retourne les N articles les plus populaires encore disponibles.
    Fallback cold-start pour les nouveaux utilisateurs.
    Optionnellement filtrable par catégorie.
    """
    _check_ready()

    if category:
        # Articles disponibles dans une catégorie précise
        products = store.cat_to_products.get(category, [])
        return PopularResponse(popular_products=products[:n], n=n)

    return PopularResponse(popular_products=store.popular_products[:n], n=n)


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommandation"])
def recommend(
    user_id: str,
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de recommandations")] = 10,
    model: Annotated[str, Query(description="Modèle : svd | als | hybrid")] = "hybrid",
):
    """
    Retourne les N articles disponibles dans les catégories recommandées pour un utilisateur.

    Fonctionnement :
    1. Le modèle identifie les catégories qui correspondent au profil de l'utilisateur
    2. Pour chaque catégorie recommandée, on retourne les articles encore disponibles
    3. Les articles déjà vendus sont automatiquement exclus

    Si l'utilisateur est inconnu (cold-start), retourne les articles populaires disponibles.
    """
    _check_ready()

    # ── Cold-start : utilisateur inconnu ──────────────────────────────────────
    if user_id not in set(store.user_enc.classes_):
        log.info("Cold-start pour user_id=%s → popularité", user_id)
        return RecommendationResponse(
            user_id=user_id,
            model="popularity (cold-start)",
            recommended_categories=list(store.popular_items[:5]) if store.popular_items else [],
            recommendations=store.popular_products[:n],
            n=n,
        )

    # ── Récupère les catégories recommandées ─────────────────────────────────
    # On demande plus de catégories que nécessaire pour avoir assez d'articles dispo
    n_cats = max(n, 20)
    try:
        top_cats = _top_categories(user_id, model=model, n_cats=n_cats)
    except ValueError as e:
        log.warning("Erreur recommandation user=%s : %s → fallback popularité", user_id, e)
        return RecommendationResponse(
            user_id=user_id,
            model="popularity (fallback)",
            recommended_categories=[],
            recommendations=store.popular_products[:n],
            n=n,
        )

    # ── Traduit les catégories en articles disponibles ────────────────────────
    products = _categories_to_products(top_cats, n)

    # Fallback : si pas assez d'articles dans les catégories recommandées,
    # on complète avec les articles populaires disponibles
    if len(products) < n:
        seen = set(products)
        for pid in store.popular_products:
            if pid not in seen:
                products.append(pid)
                seen.add(pid)
                if len(products) >= n:
                    break

    return RecommendationResponse(
        user_id=user_id,
        model=model,
        recommended_categories=top_cats[:5],   # les 5 catégories prioritaires
        recommendations=products[:n],
        n=n,
    )


@app.post("/reload", tags=["Système"])
def reload_models(
    secret: Annotated[str | None, Query(description="Clé secrète de rechargement")] = None,
):
    """
    Recharge les modèles en mémoire sans redémarrer l'API (zero-downtime).

    Appelé automatiquement par retrain.sh après chaque ré-entraînement.
    Protégé par la variable d'environnement RELOAD_SECRET (optionnel).

    Returns le statut avant/après rechargement.
    """
    import os
    import time

    expected = os.environ.get("RELOAD_SECRET", "")
    if expected and secret != expected:
        raise HTTPException(status_code=401, detail="Clé de rechargement invalide")

    t0 = time.time()
    log.info("🔄 Rechargement des modèles demandé via /reload")

    old_n_users  = len(store.user_enc.classes_) if store.user_enc is not None else 0
    old_n_cats   = len(store.item_enc.classes_) if store.item_enc is not None else 0
    old_n_avail  = len(store.popular_products)

    try:
        store.load(store._models_dir)
    except (OSError, pickle.UnpicklingError, ValueError, RuntimeError) as e:
        log.error("Erreur au rechargement : %s", e)
        raise HTTPException(status_code=500, detail=f"Erreur rechargement : {e}")

    elapsed = round(time.time() - t0, 2)
    log.info("✅ Modèles rechargés en %.2fs", elapsed)

    return {
        "status":        "reloaded",
        "elapsed_sec":   elapsed,
        "before": {
            "n_users":             old_n_users,
            "n_categories":        old_n_cats,
            "n_available_products": old_n_avail,
        },
        "after": {
            "n_users":             len(store.user_enc.classes_),
            "n_categories":        len(store.item_enc.classes_),
            "n_available_products": len(store.popular_products),
        },
    }


@app.get("/similar/{product_id}", response_model=SimilarResponse, tags=["Recommandation"])
def similar(
    product_id: str,
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de produits similaires")] = 10,
):
    """
    Retourne les N articles disponibles dans des catégories similaires à un article donné.

    Basé sur la similarité cosine TF-IDF des catégories.
    Sur une marketplace d'occasion, on ne peut pas recommander l'article lui-même
    (il peut déjà être vendu), mais des articles dans des catégories proches.
    """
    _check_ready()

    # Trouver la catégorie de l'article
    product_id_str = str(product_id)
    if not store.products_catalog.empty and "product_id" in store.products_catalog.columns:
        row = store.products_catalog[
            store.products_catalog["product_id"].astype(str) == product_id_str
        ]
        if row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Produit '{product_id}' non trouvé dans le catalogue.",
            )
        product_category = str(row.iloc[0]["product_category_name"])
    elif product_id_str in store.item_to_idx:
        product_category = product_id_str  # product_id est déjà une catégorie
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Produit '{product_id}' non trouvé dans le catalogue.",
        )

    # Similarité entre catégories
    if product_category not in store.item_to_idx:
        # Catégorie inconnue du CB → fallback popularité dans la même catégorie
        products = store.cat_to_products.get(product_category, store.popular_products)
        products = [p for p in products if str(p) != product_id_str]
        return SimilarResponse(
            product_id=product_id,
            product_category=product_category,
            similar_categories=[product_category],
            similar_products=products[:n],
            n=n,
        )

    cat_idx = store.item_to_idx[product_category]
    scores  = store.cosine_sim[cat_idx].copy()
    scores[cat_idx] = -1  # exclut la catégorie elle-même

    top_cat_idxs = np.argsort(scores)[::-1]
    similar_cats = [store.idx_to_item[i] for i in top_cat_idxs]

    # Articles disponibles dans la même catégorie d'abord, puis catégories similaires
    all_cats = [product_category] + similar_cats
    products = _categories_to_products(all_cats, n + 1)
    # Exclut l'article source lui-même
    products = [p for p in products if str(p) != product_id_str][:n]

    return SimilarResponse(
        product_id=product_id,
        product_category=product_category,
        similar_categories=similar_cats[:5],
        similar_products=products,
        n=n,
    )
