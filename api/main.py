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
import os
import pickle
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Import nécessaire pour que pickle résolve ALSRecommender au chargement
from src.events import EVENT_TYPES, append_events
from src.recommender import ALSRecommender  # noqa: F401

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent / "data" / "raw")

# CORS — l'API est consommée par melcameroun.com (Laravel) depuis un autre origine.
# À restreindre à l'origine du front en production via CORS_ORIGINS.
_DEFAULT_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://melcameroun.com",
    "https://www.melcameroun.com",
]
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "").split(",")
    if o.strip()
] or _DEFAULT_ORIGINS


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts
# ─────────────────────────────────────────────────────────────────────────────

class ModelStore:
    """
    Conteneur pour tous les artefacts du modèle.

    Après entraînement le modèle travaille au niveau CATÉGORIE :
      - user_enc / item_enc encodent les IDs utilisateurs et les noms de catégories
      - products_catalog : table complète des articles avec colonnes available et product_category_name
      - cat_to_products  : dict {catégorie → [product_id disponibles]}
      - user_history     : dict {user_id → [catégories déjà consommées]}
    """

    # Attributs remplis par _read_artifacts() puis adoptés par _adopt()
    ATTRS = (
        "user_enc", "item_enc", "R_pred", "R_pred_norm", "als_model",
        "cosine_sim", "item_to_idx", "idx_to_item", "popular_items",
        "summary", "products_catalog", "user_history", "product_sim",
        # ATTRS liste uniquement ce que _read_artifacts() PRODUIT. Les index
        # dérivés (cat_to_products, all_product_ids, n_available_products) sont
        # construits par _build_indexes() après l'adoption : les lister ici
        # ferait planter _adopt() sur un KeyError au chargement.
    )

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
        self.user_history: dict[str, list] = {}      # user_id → catégories consommées
        self.n_available_products: int = 0
        self.all_product_ids: set[str] = set()        # tous les product_id, vendus inclus
        self.available_product_ids: set[str] = set()  # articles réellement en stock
        self.product_sim: tuple | None = None  # (matrice NxN, {product_id: index})
        self.hybrid_alpha: float = 0.6
        # Lus depuis results_summary.json : la personnalisation ne sert que si la
        # validation croisée a montré un lift positif face à la popularité.
        self.personalization_warranted: bool = False
        self.cv_lift: float = 0.0
        self.enc_cat_to_idx: dict[str, int] = {}     # catégorie → index item_enc
        self.ready           = False
        self._models_dir: Path = MODELS_DIR          # mémorisé au premier load()

    def load(self, models_dir: Path = MODELS_DIR):
        # Mémorise le chemin pour que /reload puisse le réutiliser
        self._models_dir = models_dir
        log.info("Chargement des modèles depuis %s", models_dir)
        try:
            artifacts = _read_artifacts(models_dir)
        except Exception as e:
            log.error("Erreur au chargement des modèles : %s", e)
            raise
        self._adopt(artifacts)
        self.ready = True
        log.info(
            "Modèles chargés — %d users | %d catégories | %d articles disponibles | alpha=%.2f",
            len(self.user_enc.classes_),
            len(self.item_enc.classes_),
            self.n_available_products,
            self.hybrid_alpha,
        )

    def _adopt(self, artifacts: dict):
        """Remplace les artefacts en une seule fois (rechargement atomique)."""
        for name in self.ATTRS:
            setattr(self, name, artifacts[name])

        self.enc_cat_to_idx = {c: i for i, c in enumerate(self.item_enc.classes_)}
        self.hybrid_alpha = float(
            self.summary.get("hyperparameters", {}).get("hybrid_alpha", 0.6)
        )

        # Porte de suffisance : la personnalisation n'est servie que si la
        # validation croisée a montré qu'elle bat la baseline de popularité.
        # Un résumé ancien (sans ces champs) est traité comme non validé : le
        # comportement sûr est la popularité, pas un modèle jamais évalué.
        cv = self.summary.get("hyperparameters", {}).get("alpha_cv") or {}
        self.cv_lift = float(cv.get("lift_at_k", 0.0))
        self.personalization_warranted = bool(
            cv.get("lift_at_k") is not None and self.cv_lift > 0.0
        )

        self._build_availability_index()

    def _build_availability_index(self):
        """
        Construit les index dérivés du catalogue produits :
          1. cat_to_products : {catégorie → [product_id disponibles]}
          2. popular_products : articles disponibles triés par popularité de catégorie
        """
        df = self.products_catalog.copy()

        # Assurer les bons types
        df["product_id"] = df["product_id"].astype(str)
        if "product_category_name" not in df.columns:
            log.warning("Colonne 'product_category_name' absente du catalogue")
            self.n_available_products = 0
            self.available_product_ids = set()
            return

        # Filtre disponibilité
        if "available" in df.columns:
            available_df = df[df["available"] == 1].copy()
        else:
            log.warning("Colonne 'available' absente — tous les articles considérés disponibles")
            available_df = df.copy()

        # Tous les product_id connus, vendus compris : la collecte d'événements
        # doit accepter la vue d'un article vendu (le front l'affiche encore en
        # fiche), mais ne pourra pas le recommander ensuite.
        self.all_product_ids = set(df["product_id"])

        # Articles réellement disponibles. Indispensable séparément de
        # cat_to_products : vérifier « la catégorie a-t-elle du stock ? » ne dit
        # PAS si CET article est vendu, et recommanderait un article déjà vendu
        # alors que sa catégorie a du stock.
        self.available_product_ids = set(available_df["product_id"])

        # Index catégorie → articles disponibles
        self.cat_to_products = (
            available_df.groupby("product_category_name")["product_id"]
            .apply(list)
            .to_dict()
        )

        self.n_available_products = len(available_df)
        n_sold = len(df) - self.n_available_products
        log.info(
            "Index disponibilité : %d articles dispo, %d vendus, %d catégories",
            self.n_available_products, n_sold, len(self.cat_to_products),
        )

        # Articles populaires disponibles : catégories par popularité d'abord, puis
        # les catégories restantes — ainsi /popular couvre tout le stock disponible
        # et pas seulement les catégories les plus vues.
        seen: set = set()
        result: list = []
        ordered_cats = [str(c) for c in (self.popular_items or [])]
        ordered_cats += sorted(c for c in self.cat_to_products if c not in set(ordered_cats))
        for cat in ordered_cats:
            for pid in self.cat_to_products.get(cat, []):
                if pid not in seen:
                    seen.add(pid)
                    result.append(pid)
        self.popular_products = result


def _read_artifacts(models_dir: Path) -> dict:
    """
    Lit tous les artefacts depuis le disque et renvoie un dict prêt à adopter.

    Isolé de ModelStore pour que /reload puisse charger en "staging" : si une
    lecture échoue, le ModelStore en service reste intact et cohérent.
    """
    with open(models_dir / "user_encoder.pkl", "rb") as f:
        user_enc = pickle.load(f)
    with open(models_dir / "item_encoder.pkl", "rb") as f:
        item_enc = pickle.load(f)  # encode les catégories

    R_pred = np.load(models_dir / "svd_R_pred.npy")
    r_min = R_pred.min(axis=1, keepdims=True)
    r_max = R_pred.max(axis=1, keepdims=True)
    denom = r_max - r_min
    denom[denom == 0] = 1
    R_pred_norm = (R_pred - r_min) / denom

    with open(models_dir / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)

    with open(models_dir / "cosine_sim.pkl", "rb") as f:
        cosine_sim, item_to_idx, idx_to_item = pickle.load(f)

    with open(models_dir / "popular_items.pkl", "rb") as f:
        popular_items = pickle.load(f)  # liste de catégories populaires

    with open(models_dir / "results_summary.json") as f:
        summary = json.load(f)

    # Historique user → catégories (optionnel : artefacts plus anciens)
    user_history_path = models_dir / "user_history.pkl"
    if user_history_path.exists():
        with open(user_history_path, "rb") as f:
            user_history = pickle.load(f)
    else:
        log.warning("user_history.pkl absent — scores Content-Based désactivés")
        user_history = {}

    # Catalogue produits avec disponibilité
    catalog_path = models_dir / "products_catalog.csv"
    if catalog_path.exists():
        products_catalog = pd.read_csv(catalog_path)
    else:
        # Fallback : lit depuis data/raw/products.csv
        products_path = models_dir.parent / "data" / "raw" / "products.csv"
        if products_path.exists():
            log.warning("products_catalog.csv absent — fallback sur data/raw/products.csv")
            products_catalog = pd.read_csv(products_path)
        else:
            log.warning("Catalogue produits introuvable — filtre disponibilité désactivé")
            products_catalog = pd.DataFrame()

    # Similarité entre ARTICLES (couche produit). Optionnelle : le modèle de
    # catégories fonctionne sans, mais il ne peut alors recommander que des
    # catégories. Son absence doit dégrader la fonctionnalité, pas casser l'API.
    product_sim = None
    product_sim_path = models_dir / "product_sim.pkl"
    if product_sim_path.exists():
        try:
            with open(product_sim_path, "rb") as f:
                sim_matrix, pid_to_idx = pickle.load(f)
            product_sim = (sim_matrix, pid_to_idx)
            log.info("Couche article chargée : %d articles", len(pid_to_idx))
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            log.warning("product_sim.pkl illisible (%s) — recommendations article désactivées", e)
    else:
        log.warning("product_sim.pkl absent — recommandations par article indisponibles")

    return {
        "user_enc": user_enc,
        "item_enc": item_enc,
        "R_pred": R_pred,
        "R_pred_norm": R_pred_norm,
        "als_model": als_model,
        "cosine_sim": cosine_sim,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "popular_items": popular_items,
        "summary": summary,
        "products_catalog": products_catalog,
        "user_history": user_history,
        "product_sim": product_sim,
    }


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
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ready():
    if not store.ready:
        raise HTTPException(status_code=503, detail="Modèles non chargés")


def _resolve_user(user_id: str) -> int | None:
    """
    Retourne l'index encodé d'un user_id, ou None si l'utilisateur est inconnu.

    Le path param est toujours un str, alors que user_enc.classes_ peut contenir
    des int64 (customer_id MySQL) selon la version des artefacts. Sans cette
    normalisation, la comparaison `"42" in {6, 12, 17}` est toujours fausse et
    chaque requête partait en cold-start.
    """
    if store.user_enc is None:
        return None

    candidates: list = [user_id]
    if store.user_enc.classes_.dtype.kind in "iuf":
        # Artefacts legacy : classes_ en int64/float64
        for cast in (int, lambda s: int(float(s))):
            try:
                candidates.append(cast(user_id))
            except (TypeError, ValueError):
                continue

    for cand in candidates:
        try:
            return int(store.user_enc.transform([cand])[0])
        except ValueError:
            continue
    return None


def _user_categories(user_idx: int) -> list[str]:
    """Catégories déjà consommées par l'utilisateur (user_history.pkl)."""
    if not store.user_history:
        return []
    raw = store.user_enc.classes_[user_idx]
    return list(store.user_history.get(str(raw), []))


def _exclude_idx(categories) -> list[int]:
    """Catégories (noms) → indices item_enc, pour filter_seen."""
    if not categories:
        return []
    return [store.enc_cat_to_idx[c] for c in categories if c in store.enc_cat_to_idx]


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


def _cb_scores_from_categories(categories) -> np.ndarray | None:
    """
    Score content-based sur l'espace des catégories, à partir d'une liste de
    catégories d'intérêt (historique d'un client connu, ou contenu de session).

    L'historique ne doit PAS être déduit des scores SVD : avec peu de
    catégories, `argsort(R_pred[u])[:10]` renvoyait toutes les catégories, ce
    qui sommait toutes les lignes de cosine_sim et produisait un score CB quasi
    plat (donc inutile dans le blend).
    """
    history = [str(c) for c in categories]
    items_in_cb = [c for c in history if c in store.item_to_idx]
    if not items_in_cb:
        return None

    scores_cb = np.zeros(len(store.cosine_sim))
    for cat in items_in_cb:
        scores_cb += store.cosine_sim[store.item_to_idx[cat]]

    full_scores = np.zeros(len(store.item_enc.classes_))
    for cb_idx, cat in store.idx_to_item.items():
        idx = store.enc_cat_to_idx.get(cat)
        if idx is not None:
            full_scores[idx] = scores_cb[cb_idx]

    s_min, s_max = full_scores.min(), full_scores.max()
    if s_max > s_min:
        full_scores = (full_scores - s_min) / (s_max - s_min)
    return full_scores


def _get_cb_scores_cat(user_idx: int) -> np.ndarray | None:
    """Scores CB d'un client connu, à partir de son historique persisté."""
    return _cb_scores_from_categories(_user_categories(user_idx))


def _top_categories(
    user_idx: int,
    model: str = "hybrid",
    n_cats: int = 20,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Retourne les N catégories les mieux scorées pour un utilisateur.
    model : 'svd' | 'als' | 'hybrid'
    exclude : catégories déjà consommées, à ne pas proposer.
    """
    n_model_cats = len(store.item_enc.classes_)
    if model not in ("svd", "als", "hybrid"):
        raise HTTPException(
            status_code=422,
            detail=f"Modèle inconnu : '{model}' (attendu : svd | als | hybrid)",
        )

    ex = _exclude_idx(exclude)

    # Catégories sans le moindre article disponible : les recommander brûlerait
    # un rang pour produire zéro produit. "Télévisions" est dans ce cas sur le
    # jeu réel (tout vendu) tout en ayant des interactions, donc l'encodeur la
    # connaît — c'est le service qui refuse de la proposer, pas le modèle.
    no_stock = [
        i for i, cat in enumerate(store.item_enc.classes_)
        if not store.cat_to_products.get(cat)
    ]
    ex = sorted(set(ex) | set(no_stock))

    # On ne peut pas classer plus de catégories qu'il n'en existe, et les
    # catégories déjà consommées sont mises à -inf : il faut donc plafonner sur
    # le nombre de catégories ENCORE CLASSABLES. Sans ce second plafond,
    # n_cats == n_model_cats réincluait les entrées à -inf (argsort les renvoie
    # en fin de tableau, mais le slice [:n_cats] les récupère quand même) et
    # filter_seen ne servait à rien.
    n_rankable = n_model_cats - len(ex)
    if n_rankable <= 0:
        return []
    n_cats = max(1, min(n_cats, n_rankable))

    if model == "als":
        top_idxs = store.als_model.recommend(user_idx, n=n_cats, filter_seen=ex or None)
        return store.item_enc.inverse_transform(top_idxs).tolist()

    # SVD scores (copie : l'indexation numpy renvoie une vue sur R_pred_norm)
    svd_scores = store.R_pred_norm[user_idx].copy()

    if model == "svd":
        if ex:
            svd_scores[ex] = -np.inf
        top_idxs = np.argsort(svd_scores)[::-1][:n_cats]
        return store.item_enc.inverse_transform(top_idxs).tolist()

    # Hybrid : SVD + Content-Based, alpha lu depuis results_summary.json
    # (source de vérité unique, alignée sur l'entraînement)
    cb_scores = _get_cb_scores_cat(user_idx)
    if cb_scores is None:
        hybrid = svd_scores
    else:
        alpha = store.hybrid_alpha
        hybrid = alpha * svd_scores + (1.0 - alpha) * cb_scores

    # ── Porte de suffisance ────────────────────────────────────────────────
    # Si la validation croisée a montré que le modèle ne bat PAS la baseline de
    # popularité, le servir quand même ajoute du bruit : sur 51 interactions, le
    # SVD descend à 0,59 de NDCG@3 contre 1,00 pour la popularité. On classe alors
    # par popularité, ce qui reste personnalisé (les catégories déjà consommées
    # sont exclues) mais sansugg collaboratif invérifié.
    if not store.personalization_warranted:
        log.warning(
            "Personnalisation désactivée : le modèle ne bat pas la baseline de "
            "popularité (lift CV = %+.4f) → classement par popularité",
            store.cv_lift,
        )
        pop_scores = np.full(n_model_cats, -np.inf)
        for rank, cat in enumerate(store.popular_items):
            try:
                pop_scores[int(store.item_enc.transform([cat])[0])] = -rank
            except ValueError:
                continue
        if ex:
            pop_scores[ex] = -np.inf
        top_idxs = np.argsort(pop_scores)[::-1][:n_cats]
        return store.item_enc.inverse_transform(top_idxs).tolist()

    if ex:
        hybrid[ex] = -np.inf
    top_idxs = np.argsort(hybrid)[::-1][:n_cats]
    return store.item_enc.inverse_transform(top_idxs).tolist()


def _serving_mode() -> str:
    """
    Ce qui classe RÉELLEMENT en ce moment, en une phrase.

    Utilisé par /health et /metrics : l'artefact chargé et le classement servi
    sont deux choses différentes dès que la porte de suffisance est fermée.
    """
    if not store.ready:
        return "non chargé"
    if store.personalization_warranted:
        return f"hybride (SVD + content-based, lift CV = {store.cv_lift:+.4f})"
    return (
        f"popularité (personnalisation non validée en CV, lift = {store.cv_lift:+.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schémas de réponse
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    # `model` décrit l'artefact CHARGÉ (entraîné en Hybrid). Ce n'est pas
    # forcément ce qui classe : si la validation croisée a montré que le modèle
    # ne bat pas la baseline de popularité, l'API classe par popularité. Sans ce
    # champ, /health annonçait « Hybrid » pendant qu'elle servait de la
    # popularité — l'observabilité disait donc le contraire de la réalité.
    serving_mode: str
    n_users: int
    n_categories: int
    n_available_products: int


class RecommendationResponse(BaseModel):
    user_id: str
    model: str
    recommended_categories: list[str]
    recommendations: list[str]   # articles disponibles dans ces catégories
    n: int


class SessionRecommendationRequest(BaseModel):
    """
    Contexte d'une session anonyme.

    `categories` : les catégories déjà vues/panifiées pendant la session.
        C'est la SEULE source de personnalisation disponible pour un visiteur
        non identifié — le modèle n'a aucun profil appris pour lui.
    """
    session_id: Annotated[
        str, Field(max_length=128, description="Identifiant de session (logs + reprise)")
    ] = "anonymous"
    categories: Annotated[
        list[str], Field(max_length=50, description="Catégories consultées dans la session")
    ] = []
    product_ids: Annotated[
        list[str],
        Field(
            max_length=200,
            description=(
                "Articles consultés dans la session. Plus précis que les "
                "catégories : la similarité est calculée sur les TITRES des "
                "produits, pas sur les noms de catégories."
            ),
        ),
    ] = []
    n: Annotated[int, Field(ge=1, le=100, description="Nombre d'articles")] = 10


class SessionRecommendationResponse(BaseModel):
    session_id: str
    model: str
    granularity: str                          # "produit" ou "categorie"
    used_categories: list[str]      # catégories de session reconnues du modèle
    ignored_categories: list[str]   # catégories hors espace du modèle
    used_product_ids: list[str]     # articles de session retenus pour le scoring
    ignored_product_ids: list[str]  # articles sans similarité exploitable
    recommended_categories: list[str]
    recommendations: list[str]
    n: int


class SimilarResponse(BaseModel):
    product_id: str
    product_category: str
    similar_categories: list[str]
    similar_products: list[str]  # articles disponibles dans les catégories similaires
    n: int
    strategy: str = "categorie"  # "produit (titre)" ou "categorie"


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
        serving_mode=_serving_mode(),
        n_users=len(store.user_enc.classes_),
        n_categories=len(store.item_enc.classes_),
        n_available_products=store.n_available_products,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Système"])
def metrics():
    """Métriques d'évaluation et hyperparamètres du modèle en production."""
    _check_ready()
    s = store.summary
    return MetricsResponse(
        model=s.get("model", "Hybrid"),
        hyperparameters={
            **s.get("hyperparameters", {}),
            # La CV et son verdict font partie de l'état opérationnel : sans
            # eux, impossible de savoir pourquoi le service classe par
            # popularité au lieu du modèle.
            "serving_mode": _serving_mode(),
            "cv_lift_at_k": store.cv_lift,
        },
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
    user_idx = _resolve_user(user_id)
    if user_idx is None:
        log.info("Cold-start pour user_id=%s → popularité", user_id)
        return RecommendationResponse(
            user_id=user_id,
            model="popularity (cold-start)",
            recommended_categories=list(store.popular_items[:5]) if store.popular_items else [],
            recommendations=store.popular_products[:n],
            n=n,
        )

    # On demande plus de catégories que nécessaire pour avoir assez d'articles
    # dispo, mais on ne peut pas classer plus de catégories qu'il n en existe.
    n_cats = max(n, 20)
    # Le libellé renvoyé doit dire ce qui a VRAIMENT servi le classement : si la
    # porte de suffisance a basculé sur la popularité, annoncer "hybrid" ferait
    # croire à une personnalisation que le client ne reçoit pas.
    #
    # La porte ne concerne QUE le chemin par défaut (`svd` sert de repli quand le
    # contenu du profil est absent, donc `hybrid` la couvre ; `als` est le
    # diagnostic et `svd` explicite est une demande de debug). Ces deux derniers
    #court-circuitent volontairement la porte — sinon il serait impossible de
    # comparer une régression à la sortie brute du modèle — et le disent donc
    # dans le libellé au lieu de se faire passer pour de la popularity.
    served_as = model
    if model == "hybrid" and not store.personalization_warranted:
        served_as = "popularity (personnalisation non validée)"
    elif model in ("svd", "als") and not store.personalization_warranted:
        served_as = f"{model} (non validé en CV — demandé explicitement)"
    try:
        top_cats = _top_categories(
            user_idx,
            model=model,
            n_cats=n_cats,
            exclude=_user_categories(user_idx),   # ne pas reproposer du déjà-consommé
        )
    except HTTPException:
        raise
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
        model=served_as,
        recommended_categories=top_cats[:5],   # les 5 catégories prioritaires
        recommendations=products[:n],
        n=n,
    )


@app.post(
    "/recommend/session",
    response_model=SessionRecommendationResponse,
    tags=["Recommandation"],
)
def recommend_session(payload: SessionRecommendationRequest):
    """
    Recommandations personnalisées pour une session anonyme.

    Melcameroun identifie le visiteur par session, pas par `customer_id`. Le
    modèle n'a donc AUCUN profil appris pour lui : appeler `/recommend/{user_id}`
    avec un identifiant de session tomberait en cold-start et renverrait la
    popularité — c'est-à-dire exactement les mêmes articles pour tout le monde.

    Cet endpoint personalize donc à partir du CONTENU de la session : les
    catégories déjà consultées ou panifiées. Le classement est purement
    content-based (similarité TF-IDF entre noms de catégories), ce qui est la
    seule personalize disponible sans identité ni volume d'interactions.

    Les catégories de session sont exclues du résultat : on propose des
    compléments, pas ce que le visiteur vient de regarder.

    Sans catégorie exploitable, on retombe sur la popularité — le contrat reste
    le même, seule la pertinence change.
    """
    _check_ready()

    cats = [str(c) for c in payload.categories]
    known = [c for c in cats if c in store.item_to_idx]
    ignored = [c for c in cats if c not in store.item_to_idx]

    if ignored:
        log.info(
            "Session %s : %d catégorie(s) hors espace du modèle ignorées : %s",
            payload.session_id, len(ignored), ignored,
        )

    # ── Priorité aux ARTICLES vus dans la session ────────────────────────────
    # C'est le seul chemin qui personnalise au niveau produit. Il ne dépend
    # d'aucun historique client : le collaboratif par article est hors de
    # portée (110 couples relevés contre 1000 requis).
    pids = [str(p) for p in payload.product_ids]
    if pids:
        products, rec_cats = _top_products(
            pids, n=payload.n, exclude_ids=pids,
            category_hint=None,
        )
        used_pids = [
            p for p in pids
            if store.product_sim and str(p) in store.product_sim[1]
        ]
        ignored_pids = [p for p in pids if p not in used_pids]

        if products:
            return SessionRecommendationResponse(
                session_id=payload.session_id,
                model="content-based produit (session)",
                granularity="produit",
                used_categories=known,
                ignored_categories=ignored,
                used_product_ids=used_pids,
                ignored_product_ids=ignored_pids,
                recommended_categories=rec_cats,
                recommendations=products,
                n=payload.n,
            )
        log.info(
            "Session %s : aucun article exploitable dans product_ids → repli catégories",
            payload.session_id,
        )
    else:
        used_pids, ignored_pids = [], []

    # ── Repli : personnalisation par catégorie ────────────────────────────────
    cb_scores = _cb_scores_from_categories(known)
    if cb_scores is None:
        # Session vide ou uniquement des catégories inconnues : pas de
        # personalize possible, on sert la popularité.
        log.info("Session %s : aucun contexte exploitable → cold-start", payload.session_id)
        return SessionRecommendationResponse(
            session_id=payload.session_id,
            model="popularity (session sans contexte)",
            granularity="produit",
            used_categories=[],
            ignored_categories=ignored,
            used_product_ids=[],
            ignored_product_ids=used_pids + ignored_pids,
            recommended_categories=list(store.popular_items[:5]),
            recommendations=store.popular_products[: payload.n],
            n=payload.n,
        )

    scores = cb_scores.copy()

    # Mêmes exclusions que pour un client connu : ce qu'on vient de voir, et ce
    # qui ne peut rien produire (aucun article disponible).
    ex = sorted(set(_exclude_idx(known)) | {
        i for i, cat in enumerate(store.item_enc.classes_)
        if not store.cat_to_products.get(cat)
    })

    if len(ex) < len(scores):
        scores[ex] = -np.inf
        top_cats = store.item_enc.inverse_transform(
            np.argsort(scores)[::-1][:5]
        ).tolist()
    else:
        # Tout est exclu : la session a déjà couvert l'espace connu.
        top_cats = []

    products = _categories_to_products(top_cats, payload.n)
    if len(products) < payload.n:
        seen = set(products)
        for pid in store.popular_products:
            if pid not in seen:
                products.append(pid)
                seen.add(pid)
                if len(products) >= payload.n:
                    break

    return SessionRecommendationResponse(
        session_id=payload.session_id,
        model="content-based (session)",
        granularity="categorie",
        used_categories=known,
        ignored_categories=ignored,
        used_product_ids=used_pids,
        ignored_product_ids=ignored_pids,
        recommended_categories=top_cats,
        recommendations=products[: payload.n],
        n=payload.n,
    )


class TrackedEvent(BaseModel):
    """
    Un événement de navigation collecté.

    Le nom de champ canonique est `event_type` (identique à la colonne du CSV
    `events.csv`). `type` reste accepté comme alias.

    Sans cela, un front envoyant `event_type` — le nom le plus évident, celui de
    la colonne et de la documentation — voit son champ ignoré en silence et tous
    ses événements enregistrés comme `view` : une corruption de données
    invisible, qui ne se voit qu'après coup dans les métriques.

    `extra="forbid"` ferme la même faille par une clé mal orthographiée : le
    client reçoit un 422 explicite au lieu d'un faux `view`.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    product_id: Annotated[int, Field(description="product_id MEL")]
    event_type: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("event_type", "type"),
            description="view | cart | purchase | review",
        ),
    ] = "view"
    ts: Annotated[
        str | None,
        Field(description="Horodatage ISO 8601 ; défaut = maintenant"),
    ] = None


class EventBatch(BaseModel):
    """
    Lot d'événements renvoyé par le front après une navigation.

    C'est LA source qui manque aujourd'hui : `visits.csv` est vide, donc aucun
    modèle n'a jamais vu une vue produit. Sans elle, aucune personnalisation
    par article ne pourra être validée même après des mois de production.

    `user_id` est optionnel : un visiteur non connecté produit quand même des
    événements, rattachés à `session_id`. C'est le cas principal ici, melcameroun
    identifiant le visiteur par session.
    """
    session_id: Annotated[
        str | None, Field(max_length=128, description="Identifiant de session")
    ] = None
    user_id: Annotated[
        str | None,
        Field(max_length=64, description="customer_id si le client est connecté"),
    ] = None
    events: Annotated[
        list[TrackedEvent], Field(min_length=1, max_length=200, description="Événements")
    ]


class EventBatchResponse(BaseModel):
    accepted: int
    rejected_unknown_product: int
    rejected_invalid_type: int
    session_id: str
    user_id: str | None
    details: list[str]


def _product_scores(product_ids) -> np.ndarray | None:
    """
    Score de similarité d'un ensemble d'articles vus, agrégé sur la couche article.

    Somme des lignes de la matrice cosinus produit × produit. Cette somme est
    l'équivalent article de ce que `_cb_scores_from_categories` fait au niveau
    catégories : le classement compare les textes des titres, pas les catégories.

    C'est ce qui rend la personnalisation par article possible AUJOURD'HUI, alors
    que le collaboratif par article reste hors de portée (110 couples relevés
    contre un seuil de 1000). Aucun historique client n'est nécessaire : la
    session suffit.
    """
    if not store.product_sim:
        return None
    sim, pid_to_idx = store.product_sim
    if sim is None or not len(sim) or not pid_to_idx:
        return None

    idxs = [pid_to_idx[str(p)] for p in product_ids if str(p) in pid_to_idx]
    if not idxs:
        return None

    scores = np.asarray(sim[idxs], dtype=np.float64).sum(axis=0)
    # Un article ne doit jamais être recommandé parce qu'on vient de le voir.
    for i in idxs:
        scores[i] = 0.0
    return scores


def _top_products(
    product_ids, n: int, exclude_ids=None, category_hint: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retourne (articles recommandés, catégories correspondantes) pour un contexte
    d'articles vus.

    Ne renvoie QUE des articles disponibles : la matrice de similarité porte sur
    tout le catalogue, y compris les 30 articles vendus.
    """
    scores = _product_scores(product_ids)
    if scores is None:
        return [], []

    _sim, pid_to_idx = store.product_sim
    idx_to_pid = {i: p for p, i in pid_to_idx.items()}
    cat_of: dict[str, str] = {}
    if not store.products_catalog.empty and "product_category_name" in store.products_catalog.columns:
        cat_of = dict(zip(
            store.products_catalog["product_id"].astype(str),
            store.products_catalog["product_category_name"].astype(str),
        ))

    # Un article vendu ne doit JAMAIS être recommandé : le similarity porte sur
    # tout le catalogue. On s'appuie sur la liste des articles réellement
    # disponibles, pas sur « la catégorie a-t-elle du stock ? » — ce test est
    # faux : un article vendu dans une catégorie approvisionnée passerait.
    sold = {pid for pid in store.all_product_ids if pid not in store.available_product_ids}
    exclude = {str(p) for p in (exclude_ids or [])} | sold

    order = np.argsort(scores)[::-1]
    out_products: list[str] = []
    out_categories: list[str] = []
    seen_cats: set[str] = set()

    for i in order:
        pid = idx_to_pid.get(int(i))
        if pid is None or pid in exclude:
            continue
        if scores[i] <= 0:
            break
        cat = cat_of.get(pid, "")
        if category_hint and cat not in category_hint:
            continue
        out_products.append(pid)
        if cat and cat not in seen_cats:
            out_categories.append(cat)
            seen_cats.add(cat)
        if len(out_products) >= n:
            break

    return out_products, out_categories


@app.post("/events", response_model=EventBatchResponse, tags=["Collecte"])
def collect_events(payload: EventBatch):
    """
    Collecte la navigation (vue / panier / achat) pour alimenter le modèle.

    C'est le seul moyen de faire progresser vers une personnalisation par
    article : aujourd'hui les seules données produit sont 39 lignes de commande
    et 82 ajouts panier, sans aucune vue.

    Le stockage est un CSV append dans `data/raw/events.csv`, repris par
    `src/events.py` à l'entraînement. Rien n'est écrit si le visitor n'a ni
    session ni user_id : un événement non rattachable ne peut servir à rien et
    gonflerait le jeu avec du bruit.

    Les `product_id` inconnus du catalogue sont comptés et rejetés, pas
    enregistrés : une recommandation ne peut pas porter sur un produit inconnu,
    et garder la ligne ferait gonfler les compteurs de faux positifs.
    """
    _check_ready()

    if not payload.session_id and not payload.user_id:
        raise HTTPException(
            status_code=422,
            detail="session_id ou user_id obligatoire : un événement sans "
                   "identité ne peut être rattaché à personne",
        )

    catalogue = set(store.all_product_ids) if store.all_product_ids else set()

    rows: list[dict] = []
    unknown = 0
    invalid_type = 0
    details: list[str] = []

    now = pd.Timestamp.now("UTC").isoformat()
    for ev in payload.events:
        if ev.event_type not in EVENT_TYPES:
            invalid_type += 1
            details.append(
                f"type '{ev.event_type}' ignoré pour product_id={ev.product_id}"
            )
            continue
        pid = str(ev.product_id)
        if catalogue and pid not in catalogue:
            unknown += 1
            details.append(f"product_id={pid} absent du catalogue")
            continue
        rows.append({
            "ts": ev.ts or now,
            "source": "api",
            "session_id": payload.session_id,
            "user_id": payload.user_id,
            "product_id": pid,
            "event_type": ev.event_type,
        })

    written = append_events(DATA_DIR, rows)
    log.info(
        "Collecte : session=%s user=%s | %d écrit(s), %d produit(s) inconnu(s), %d type(s) invalide(s)",
        payload.session_id, payload.user_id, written, unknown, invalid_type,
    )

    return EventBatchResponse(
        accepted=written,
        rejected_unknown_product=unknown,
        rejected_invalid_type=invalid_type,
        session_id=payload.session_id or "",
        user_id=payload.user_id,
        details=details[:20],
    )


@app.post("/reload", tags=["Système"])
def reload_models(
    secret: Annotated[str | None, Query(description="Clé secrète de rechargement")] = None,
    x_reload_secret: Annotated[
        str | None, Header(description="Clé secrète (header, préféré au query string)")
    ] = None,
):
    """
    Recharge les modèles en mémoire sans redémarrer l'API (zero-downtime).

    Appelé automatiquement par retrain.sh après chaque ré-entraînement.
    Protégé par la variable d'environnement RELOAD_SECRET (optionnel).

    Le chargement se fait en "staging" : si une lecture échoue, les modèles en
    service ne sont pas touché et l'API reste cohérente.

    La clé peut être passée par le header `X-Reload-Secret` (recommandé : les
    query strings sont écrites dans les logs d'accès) ou par `?secret=`.
    """
    expected = os.environ.get("RELOAD_SECRET", "")
    provided = x_reload_secret if x_reload_secret is not None else secret
    if expected and (provided is None or not secrets.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="Clé de rechargement invalide")

    t0 = time.time()
    log.info("🔄 Rechargement des modèles demandé via /reload")

    old_n_users = len(store.user_enc.classes_) if store.user_enc is not None else 0
    old_n_cats  = len(store.item_enc.classes_) if store.item_enc is not None else 0
    old_n_avail = store.n_available_products

    # Staging : rien n'est muté tant que la lecture n'a pas entièrement réussi
    try:
        artifacts = _read_artifacts(store._models_dir)
    except Exception as e:
        # Un artefact corrompu ne doit pas laisser sortir une AttributeError,
        # KeyError ou FileNotFoundError de l'endpoint.
        log.exception("Erreur au rechargement")
        raise HTTPException(status_code=500, detail=f"Erreur rechargement : {e}")

    store._adopt(artifacts)
    store.ready = True

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
            "n_available_products": store.n_available_products,
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

    # ── Couche ARTICLE : similarité sur les TITRES (précis) ──────────────────
    # Privilégiée dès que l'article a un titre indexé. Contrairement à la
    # similarité de catégories, elle distingue deux articles d'une même
    # catégorie et ne dépend pas des libellés de catégorie, parfois erronés
    # dans le catalogue source.
    if store.product_sim:
        products, _ = _top_products([product_id_str], n=n, exclude_ids=[product_id_str])
        if products:
            return SimilarResponse(
                product_id=product_id,
                product_category=product_category,
                similar_categories=[],
                similar_products=products,
                n=n,
                strategy="produit (titre)",
            )

    # Similarité entre catégories (repli : article sans titre exploitable)
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
            strategy="categorie (fallback popularite)",
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
        strategy="categorie",
    )
