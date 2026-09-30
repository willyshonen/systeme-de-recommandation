"""
api/main.py
API FastAPI — Système de recommandation MEL Cameroun.

Endpoints :
    GET /health                         → statut de l'API
    GET /metrics                        → métriques du modèle
    GET /recommend/{user_id}?n=10       → recommandations pour un user
    GET /similar/{product_id}?n=10      → produits similaires à un item
    GET /popular?n=10                   → items les plus populaires
"""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
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
    """Conteneur pour tous les artefacts du modèle."""

    def __init__(self):
        self.user_enc      = None
        self.item_enc      = None
        self.R_pred        = None
        self.R_pred_norm   = None
        self.als_model     = None
        self.cosine_sim    = None
        self.item_to_idx   = None
        self.idx_to_item   = None
        self.popular_items = None
        self.summary       = None
        self.ready         = False

    def load(self, models_dir: Path = MODELS_DIR):
        log.info("Chargement des modèles depuis %s", models_dir)
        try:
            with open(models_dir / "user_encoder.pkl", "rb") as f:
                self.user_enc = pickle.load(f)
            with open(models_dir / "item_encoder.pkl", "rb") as f:
                self.item_enc = pickle.load(f)

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
                self.popular_items = pickle.load(f)

            with open(models_dir / "results_summary.json") as f:
                self.summary = json.load(f)

            self.ready = True
            log.info(
                "Modèles chargés — %d users | %d items",
                len(self.user_enc.classes_),
                len(self.item_enc.classes_),
            )
        except Exception as e:
            log.error("Erreur au chargement des modèles : %s", e)
            raise


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MEL Cameroun — API Recommandation",
    description="Système de recommandation de produits e-commerce (modèle Hybride SVD + Content-Based)",
    version="1.0.0",
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


def _get_cb_scores(user_items: list) -> np.ndarray | None:
    """Scores content-based normalisés sur l'espace item_enc."""
    items_in_cb = [i for i in user_items if i in store.item_to_idx]
    if not items_in_cb:
        return None
    n_items   = len(store.item_enc.classes_)
    scores_cb = np.zeros(len(store.cosine_sim))
    for item_id in items_in_cb:
        scores_cb += store.cosine_sim[store.item_to_idx[item_id]]
    full_scores = np.zeros(n_items)
    for cb_idx, pid in store.idx_to_item.items():
        try:
            full_scores[int(store.item_enc.transform([pid])[0])] = scores_cb[cb_idx]
        except ValueError:
            pass
    s_min, s_max = full_scores.min(), full_scores.max()
    if s_max > s_min:
        full_scores = (full_scores - s_min) / (s_max - s_min)
    return full_scores


def _hybrid_scores(user_id: str, alpha: float = 1.0) -> np.ndarray:
    """Calcule les scores hybrides pour un user."""
    user_idx   = int(store.user_enc.transform([user_id])[0])
    svd_scores = store.R_pred_norm[user_idx]
    return svd_scores   # alpha=1.0 par défaut → pur SVD normalisé


# ─────────────────────────────────────────────────────────────────────────────
# Schémas de réponse
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    n_users: int
    n_items: int


class RecommendationResponse(BaseModel):
    user_id: str
    model: str
    recommendations: list[str]
    n: int


class SimilarResponse(BaseModel):
    product_id: str
    similar_products: list[str]
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
        n_items=len(store.item_enc.classes_),
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
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de produits")] = 10
):
    """Retourne les N produits les plus populaires (fallback cold-start)."""
    _check_ready()
    return PopularResponse(popular_products=store.popular_items[:n], n=n)


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommandation"])
def recommend(
    user_id: str,
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de recommandations")] = 10,
    model: Annotated[str, Query(description="Modèle : svd | als | hybrid")] = "hybrid",
):
    """
    Retourne les N meilleures recommandations pour un utilisateur.

    - **user_id** : identifiant de l'utilisateur
    - **n** : nombre de recommandations (1-100, défaut 10)
    - **model** : `svd` | `als` | `hybrid` (défaut : hybrid)

    Si l'utilisateur est inconnu (cold-start), retourne les produits populaires.
    """
    _check_ready()

    # ── Cold-start ────────────────────────────────────────────────────────────
    if user_id not in set(store.user_enc.classes_):
        log.info("Cold-start pour user_id=%s → popularité", user_id)
        return RecommendationResponse(
            user_id=user_id,
            model="popularity (cold-start)",
            recommendations=store.popular_items[:n],
            n=n,
        )

    user_idx = int(store.user_enc.transform([user_id])[0])

    # ── SVD ───────────────────────────────────────────────────────────────────
    if model == "svd":
        scores   = store.R_pred[user_idx]
        top_idxs = np.argsort(scores)[::-1][:n]
        reco     = store.item_enc.inverse_transform(top_idxs).tolist()
        return RecommendationResponse(user_id=user_id, model="svd", recommendations=reco, n=n)

    # ── ALS ───────────────────────────────────────────────────────────────────
    if model == "als":
        top_idxs = store.als_model.recommend(user_idx, n=n)
        reco     = store.item_enc.inverse_transform(top_idxs).tolist()
        return RecommendationResponse(user_id=user_id, model="als", recommendations=reco, n=n)

    # ── Hybrid (défaut) ───────────────────────────────────────────────────────
    scores   = _hybrid_scores(user_id, alpha=1.0)
    top_idxs = np.argsort(scores)[::-1][:n]
    reco     = store.item_enc.inverse_transform(top_idxs).tolist()
    return RecommendationResponse(user_id=user_id, model="hybrid", recommendations=reco, n=n)


@app.get("/similar/{product_id}", response_model=SimilarResponse, tags=["Recommandation"])
def similar(
    product_id: str,
    n: Annotated[int, Query(ge=1, le=100, description="Nombre de produits similaires")] = 10,
):
    """
    Retourne les N produits les plus similaires à un produit donné.

    Basé sur la similarité cosine TF-IDF des attributs produit.
    """
    _check_ready()

    if product_id not in store.item_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Produit '{product_id}' non trouvé dans le catalogue.",
        )

    item_idx = store.item_to_idx[product_id]
    scores   = store.cosine_sim[item_idx].copy()
    scores[item_idx] = -1

    top_idxs = np.argsort(scores)[::-1][:n]
    similar_products = [store.idx_to_item[i] for i in top_idxs]

    return SimilarResponse(product_id=product_id, similar_products=similar_products, n=n)
