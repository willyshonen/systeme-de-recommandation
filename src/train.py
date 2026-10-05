"""
src/train.py
Pipeline d'entraînement reproductible — Système de recommandation hybride MEL Cameroun.

Usage :
    python src/train.py
    python src/train.py --data-dir data/raw --models-dir models --k-factors 150 --als-factors 32

Le pipeline produit dans --models-dir :
    als_model.pkl, svd_R_pred.npy, cosine_sim.pkl, user_encoder.pkl,
    item_encoder.pkl, popular_items.pkl, user_history.pkl,
    products_catalog.csv, results_summary.json
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# ── Reproductibilité : BLAS sur un seul thread ────────────────────────────────
# Doit être posé AVANT l'import de numpy, sinon OpenBLAS/MKL ont déjà choisi
# leur pool de threads.
#
# Les réductions multi-threadées de BLAS ne sont pas bit-reproductibles : deux
# entraînements sur les données identiques produisaient deux `svd_R_pred.npy`
# différents, et un NDCG qui oscillait entre 0.9540 et 1.0000 selon le run.
# Conséquence concrète : l'alerte de régression que le scheduler compare avant
#/après se déclenchait sur du pur bruit, et un modèle « amélioré » pouvait n'être
# que le résultat d'un ordre d'arrondi différent.
#
# Posé ici, dans le point d'entrée, la reproductibilité ne dépend plus de la
# façon dont le script est lancé (local, cron, conteneur Docker).
#
# Sur ce jeu (34 clients, 7 catégories) le surcoût est nul. Sur un gros jeu,
# `TRAIN_ALLOW_MULTITHREAD=1` réactive BLAS multi-thread : plus rapide, mais les
# métriques redeviennent non reproductibles.
if os.environ.get("TRAIN_ALLOW_MULTITHREAD", "").lower() not in ("1", "true", "yes"):
    for _var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

# `python src/train.py` met src/ sur sys.path, pas la racine du projet.
# On l'ajoute pour pouvoir importer src.recommender (cheminabsolu du projet).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Classe partagée avec l'API. NE PAS la redéfinir ici : le pickle référence le
# module de la classe, et api/main.py ne résout que src.recommender.ALSRecommender.
from src.events import (
    build_product_similarity,
    data_sufficiency_report,
    load_events,
    product_features,
)
from src.recommender import ALSRecommender

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> dict:
    """
    Charge les fichiers CSV d'interactions MEL Cameroun.

    Format attendu dans data_dir/ :
      - orders.csv      : colonnes [order_id, customer_id, order_status]
      - order_items.csv : colonnes [order_id, product_id, price]
      - products.csv    : colonnes [product_id, product_category_name, product_weight_g, product_photos_qty]
      - reviews.csv     : colonnes [order_id, review_score]
      - category_names.csv (optionnel) : colonnes [product_category_name, product_category_name_english]

    Ces fichiers peuvent être exportés depuis la base MySQL MEL via :
        SELECT * FROM factures → orders + order_items
        SELECT * FROM articles → products
        SELECT * FROM etoiles  → reviews
    """
    log.info("Chargement des données depuis %s", data_dir)

    def _try_read(filename: str, required: bool = True) -> pd.DataFrame | None:
        path = data_dir / filename
        if path.exists():
            return pd.read_csv(path)
        if required:
            raise FileNotFoundError(
                f"Fichier requis introuvable : {path}\n"
                "Exporter les données MEL depuis MySQL vers CSV et les placer dans data/raw/"
            )
        return None

    orders      = _try_read("orders.csv")
    order_items = _try_read("order_items.csv")
    products    = _try_read("products.csv")
    reviews     = _try_read("reviews.csv")
    cat_names   = _try_read("category_names.csv", required=False)

    if cat_names is None:
        # Fallback : pas de traduction de catégorie, on réutilise le nom brut
        if "product_category_name" in products.columns:
            cat_names = products[["product_category_name"]].drop_duplicates().copy()
            cat_names["product_category_name_english"] = cat_names["product_category_name"]
        else:
            cat_names = pd.DataFrame(
                columns=["product_category_name", "product_category_name_english"]
            )

    return {
        "orders":      orders,
        "order_items": order_items,
        "products":    products,
        "reviews":     reviews,
        "cat_names":   cat_names,
        "data_dir":    str(data_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _orders_date_column(orders: pd.DataFrame) -> str | None:
    """Colonne de date de commande, selon le schéma (e-commerce ou MEL)."""
    for col in ("order_purchase_timestamp", "created_at", "updated_at"):
        if col in orders.columns:
            return col
    return None


def _coerce_ts(series: pd.Series) -> pd.Series:
    """Convertit une colonne de date en datetime, sans échouer sur les valeurssales."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def build_interactions(data: dict) -> pd.DataFrame:
    """
    Construit le dataframe d'interactions implicites par CATÉGORIE.

    Sur une marketplace d'occasion chaque article est unique et disparaît
    après vente → on ne recommande pas des articles spécifiques mais des
    catégories, puis on retourne les articles disponibles dans ces catégories.

    SOURCE UNIQUE : tout passe par `load_events()`, qui fusionne les commandes,
    les paniers, les visites, les avis ET les événements collectés par l'API
    (`POST /events`). Ce n'était pas le cas avant : les vues envoyées par le
    front n'arrivaient jamais ici, alors qu'elles sont justement le signal qui
    manque pourpredicted l'intérêt d'un visiteur. Les événements par session
    anonyme sont conservés pour `/recommend/session` mais exclus de
    l'entraînement, faute de profil client associé.

    Stratégie de scoring :
        vue        → score 0.2 à 1.0 selon le nombre de vues (intérêt faible)
        panier     → score 3  (intérêt fort)
        avis       → score 4  (confirmation d'un achat)
        achat      → score 5  (conversion)

    Le score final par (user, catégorie) est le MAX des signaux observés.

    Colonne `ts` : horodatage réel du dernier signal observé pour la paire
    (user, catégorie). C'est elle qui rend le split Leave-One-Out chronologique —
    un `order_id` factice ne matchait aucun order_id de orders.csv, ce qui
    rendait toutes les dates NaT et le tri vide.
    """
    products = data["products"]
    data_dir = Path(data.get("data_dir", "data/raw"))

    # Table de correspondance article → catégorie
    if "product_category_name" not in products.columns:
        log.error("products.csv manque la colonne 'product_category_name'")
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "ts"])

    events = load_events(data_dir, products=products)
    if events.empty:
        # On lève au lieu de renvoyer un DataFrame vide : le vide remontait
        # jusqu'à train_test_split, qui échouait sur un message incompréhensible
        # ("With n_samples=0, test_size=0.2...") alors que la vraie cause —
        # aucun événement exploitable — n'apparaissait qu'une ligne plus haut
        # dans les logs. Le message doit nommer la cause.
        raise ValueError(
            f"Aucun événement exploitable dans {data_dir} : ni achats valides, "
            f"ni paniers, ni visites, ni avis, ni événements API. "
            f"Vérifiez que les CSV attendus s'y trouvent et que les commandes "
            f"n'ont pas toutes un statut non livré. (generate_fake_data.py "
            f"génère un jeu minimal valide.)"
        )

    prod_cat = (
        products[["product_id", "product_category_name"]]
        .dropna()
        .drop_duplicates("product_id")
        .set_index("product_id")["product_category_name"]
        .to_dict()
    )

    ev = events.copy()
    ev["user_id"] = pd.to_numeric(ev["user_id"], errors="coerce")
    n_session = int(ev["user_id"].isna().sum())
    if n_session:
        log.info(
            "%d événement(s) de session anonyme exclus de l'entraînement "
            "(aucun profil client associé) — ils servent à /recommend/session",
            n_session,
        )
    ev = ev.dropna(subset=["user_id"])
    ev["category"] = ev["product_id"].map(prod_cat)
    ev = ev.dropna(subset=["category"])

    rows = []
    # Poids par type d'événement. Une vue seule vaut peu, la répétition monte
    # le score jusqu'à 1.0 : regarder 5 articles d'une catégorie est un signal
    # réel, en regarder 1 est presque du bruit.
    weights = {"view": None, "cart": 3.0, "review": 4.0, "purchase": 5.0}

    for event_type, weight in weights.items():
        sub = ev[ev["event_type"] == event_type]
        if sub.empty:
            continue
        agg = (sub.groupby(["user_id", "category"])
                  .agg(n=("product_id", "size"), ts=("ts", "max"))
                  .reset_index()
                  .rename(columns={"user_id": "user_id", "category": "item_id"}))
        agg["rating"] = (
            (agg["n"].clip(upper=5) / 5).round(2) if weight is None else weight
        )
        agg = agg[["user_id", "item_id", "rating", "ts"]]
        rows.append(agg)
        log.info("Signal %-8s (catégories) : %d interactions", event_type, len(agg))

    if not rows:
        log.error("Aucun signal d'interaction trouvé !")
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "ts"])

    # ── Fusion : garde le score max par (user, catégorie) ─────────────────────
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = _coerce_ts(df["ts"])
    df = df.groupby(["user_id", "item_id"], as_index=False).agg(
        rating=("rating", "max"), ts=("ts", "max")
    )

    n_without_ts = int(df["ts"].isna().sum())
    if n_without_ts:
        log.warning(
            "%d/%d interactions sans horodatage — traitées comme les plus anciennes",
            n_without_ts, len(df),
        )

    log.info(
        "Interactions catégories : %d (users: %d, catégories: %d)",
        len(df), df["user_id"].nunique(), df["item_id"].nunique(),
    )
    return df


def filter_interactions(
    df: pd.DataFrame,
    min_user: int = 2,
    min_item: int = 5,
    n_iter: int = 3,
) -> pd.DataFrame:
    """Filtrage itératif cold-start. Adapte les seuils si le dataset est petit."""
    n = len(df)

    # Adapte les seuils au volume de données
    if n < 100:
        min_user = 1
        min_item = 1
        log.warning("Petit dataset (%d interactions) — seuils cold-start désactivés (min_user=1, min_item=1).", n)
    elif n < 500:
        min_user = max(1, min_user - 1)
        min_item = max(1, min_item - 3)
        log.warning("Dataset limité (%d interactions) — seuils réduits (min_user=%d, min_item=%d).", n, min_user, min_item)

    for i in range(n_iter):
        before = len(df)
        valid_users = df.groupby("user_id")["item_id"].count()
        df = df[df["user_id"].isin(valid_users[valid_users >= min_user].index)]
        valid_items = df.groupby("item_id")["user_id"].count()
        df = df[df["item_id"].isin(valid_items[valid_items >= min_item].index)]
        log.info("Filtrage iter %d : %d → %d", i + 1, before, len(df))
    return df.reset_index(drop=True)


def build_item_space(
    products: pd.DataFrame, interaction_items, mode: str = "interactions",
) -> tuple[list[str], dict]:
    """
    Construit l'espace des catégories du modèle.

    mode="interactions" (défaut) : uniquement les catégories ayant au moins une
        interaction. Le modèle ne recommande que ce qu'il a observé chez les
        utilisateurs. C'est le mode retenu : les catégories sans interaction ne
        peuvent être classées que par similarité de nom, et sur ce jeu cela
        dominait le classement — 44 % des premiers résultats et 46 % des positions
        du top-5 étaient des catégories jamais observées, classées sur un simple
        mot partagé ("vêtements"). On préfère recommander moins, mais juste.

    mode="catalog" : ajoute les catégories du catalogue possédant au moins un
        article disponible. Gagne 43 articles disponibles atteignables (148 → 191
        sur le jeu réel), au prix de la qualité de classement ci-dessus. Conservé
        pour pouvoir arbitrer plus tard avec de vraies données.

    Les catégories du catalogue hors espace des items restent inatteignables :
    l'API ne peut servir que ce que l'encodeur connaît. Le rapport retourne la
    liste pour que la perte de couverture soit visible dans results_summary.json
    plutôt que silencieuse.

    Dans les deux modes, une catégorie SANS article disponible n'est pas écartée
    si elle a des interactions : elle porte un signal comportemental réel, et
    l'exclure invaliderait l'évaluation. L'API la renverra sans produits, ce que
    _categories_to_products gère déjà.

    Retourne (liste triée des catégories, rapport détaillé pour le résumé JSON).
    """
    interaction_items = set(interaction_items)

    if "product_category_name" not in products.columns:
        log.warning(
            "products.csv sans 'product_category_name' — espace des items "
            "restreint aux catégories vues dans les interactions"
        )
        return sorted(interaction_items), {
            "mode": mode,
            "n_items_total": len(interaction_items),
            "n_items_from_interactions": len(interaction_items),
            "n_items_added_from_catalog": 0,
            "n_unreachable_categories": 0,
            "unreachable_categories": [],
            "categories_without_stock": [],
        }

    prods = products[["product_category_name"]].dropna()
    if "available" in products.columns:
        prods = products.loc[
            products["available"].fillna(1) == 1, ["product_category_name"]
        ].dropna()
    else:
        log.warning("Colonne 'available' absente — toutes les catégories du catalogue sont retenues")

    with_stock = set(prods["product_category_name"])

    # Catégories du catalogue que le modèle ne pourra jamais recommander.
    unreachable = sorted(with_stock - interaction_items)

    if mode == "catalog":
        item_space = sorted(interaction_items | with_stock)
        added      = unreachable
    elif mode == "interactions":
        item_space = sorted(interaction_items)
        added      = []
    else:
        raise ValueError(
            f"mode d'espace des items inconnu : {mode!r} "
            "(attendu : 'interactions' | 'catalog')"
        )

    report = {
        "mode": mode,
        "n_items_total": len(item_space),
        "n_items_from_interactions": len(interaction_items),
        "n_items_added_from_catalog": len(added),
        "n_unreachable_categories": len(unreachable),
        "unreachable_categories": unreachable,
        "categories_without_stock": sorted(
            set(products["product_category_name"].dropna()) - with_stock
        ),
    }

    log.info(
        "Espace des items (mode=%s) : %d catégories", mode, len(item_space)
    )
    if added:
        log.info(
            "Catégories ajoutées du catalogue (signal CB uniquement) : %s", added
        )
    if unreachable:
        log.warning(
            "%d catégories du catalogue avec du stock restent INATTEIGNABLES "
            "(aucune interaction, mode=interactions) : %s",
            len(unreachable), unreachable,
        )

    return item_space, report


def encode_ids(df: pd.DataFrame, item_space=None):
    """
    Encode user_id et item_id en entiers contigus.

    `item_space` : liste explicite des catégories du modèle. Si elle est fournie,
    l'encodeur categorize toutes ces catégories, y compris celles sans aucune
    interaction (voir build_item_space) ; les lignes d'interactions restent
    inchangées.
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df = df.copy()
    # Les user_id sont des customer_id entiers en base. On les encode en chaînes
    # pour que l'API, qui reçoit toujours un str via le path, les retrouve tels
    # quels : sinon `"42" in {6, 12, 17}` est faux et tout part en cold-start.
    df["user_id"] = df["user_id"].astype(str)
    df["user_idx"] = user_enc.fit_transform(df["user_id"])

    if item_space is not None:
        item_enc.fit(sorted(set(item_space) | set(df["item_id"])))
    else:
        item_enc.fit(df["item_id"])

    df["item_idx"] = item_enc.transform(df["item_id"])
    unknown = sorted(set(df["item_id"]) - set(item_enc.classes_))
    if unknown:
        log.error(
            "%d catégories d'interactions absentes de l'espace des items : %s",
            len(unknown), unknown,
        )
    return df, user_enc, item_enc


def make_splits(df: pd.DataFrame, random_state: int = 42):
    """
    Retourne deux splits :
      - LOO (Leave-One-Out) chronologique → évaluation recommandation
      - Random 80/20 → évaluation RMSE/MAE SVD

    Le LOO utilise la colonne `ts` (horodatage réel) construite par
    build_interactions. `na_position="first"` place les interactions sans
    horodatage du côté train : on ne teste que sur ce qu'on sait être récent.
    """
    from sklearn.model_selection import train_test_split

    df_dated = df.sort_values(
        ["user_id", "ts"], na_position="first", kind="mergesort"
    )
    test_mask = df_dated.groupby("user_id").cumcount(ascending=False) == 0
    train_loo = df_dated[~test_mask].copy()
    test_loo  = df_dated[test_mask].copy()

    train_rand, test_rand = train_test_split(df, test_size=0.2, random_state=random_state)

    n_users = df["user_id"].nunique()
    n_empty = n_users - train_loo["user_id"].nunique()
    if n_empty:
        log.warning(
            "%d/%d users n'ont qu'une seule interaction : leur historique "
            "d'entraînement est vide et ils ne sont pas évaluables",
            n_empty, n_users,
        )

    log.info(
        "Split LOO   → train=%d | test=%d", len(train_loo), len(test_loo)
    )
    log.info(
        "Split random→ train=%d | test=%d", len(train_rand), len(test_rand)
    )
    return train_loo, test_loo, train_rand, test_rand


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modèles
# ─────────────────────────────────────────────────────────────────────────────

def train_svd(train_loo: pd.DataFrame, user_enc, item_enc, k_factors: int = 150):
    """
    SVD tronqué via scipy sur la matrice user × catégorie centrée par utilisateur.

    Retourne (R_pred, user_mean, R_centered, k_effective).

    `k_effective` est le k réellement utilisé : svds exige k < min(n_users, n_items),
    donc avec 7 catégories le k nominal (150) est ramené à 6. Cette valeur doit
    être celle rapportée dans le résumé JSON, pas le k demandé.
    """
    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    t = train_loo.copy()
    t["user_idx"] = user_enc.transform(t["user_id"])
    t["item_idx"] = item_enc.transform(t["item_id"])

    # ORDRE STABLE avant toute conversion en matrice creuse. csr_matrix
    # n'accumule pas les doublons (il les écrase) et ARPACK — le solveur derrière
    # svds — dépend de l'ordre d'indexation. Deux exécutions sur le MÊME jeu
    # pouvaient donc donner deux décompositions différentes : c'est ce qui
    # rendait l'alpha choisi par CV instable d'un entraînement à l'autre
    # (0.0, puis 0.8, puis 0.0). mergesort est le seul tri stable de pandas.
    t = t.sort_values(["user_idx", "item_idx"], kind="mergesort")

    R = csr_matrix(
        (t["rating"].values, (t["user_idx"].values, t["item_idx"].values)),
        shape=(n_users, n_items),
    )
    R_dense     = R.toarray().astype(np.float32)
    user_mean   = R_dense.mean(axis=1)
    R_centered  = R_dense - user_mean[:, np.newaxis]
    R_centered[R_dense == 0] = 0
    # Les lignes sans AUCUNE interaction ont une moyenne 0/0 -> NaN, ce qui
    # contamine tout le SVD. On force 0.
    user_mean = np.nan_to_num(user_mean, nan=0.0)

    # k doit être < min(n_users, n_items) — on cap pour les petits datasets
    k_safe = min(k_factors, min(R_centered.shape) - 1)
    if k_safe < k_factors:
        log.warning(
            "k_factors réduit de %d à %d (matrice %s) — le SVD est alors de rang "
            "quasi complet, la réduction de dimension est inexistante",
            k_factors, k_safe, R_centered.shape,
        )
    log.info("Entraînement SVD avec k=%d facteurs...", k_safe)

    # SVD COMPLET puis troncature, et non `svds`.
    #
    # `svds` passe par ARPACK, un solveur itératif : même avec un vecteur de
    # départ `v0` fixé, deux exécutions sur la MÊME matrice donnaient des
    # décompositions différentes (le NDCG à alpha=1.0 oscillait entre 0,51 et
    # 0,64 d'un run à l'autre), et l'alpha choisi par CV changeait avec —
    # 0.0 puis 0.8 puis 0.0. Des métriques non reproductibles font mentir le
    # scheduler, qui compare avant/après.
    #
    # Ici la matrice fait (n_clients × n_catégories) : 34 × 7 sur le jeu réel,
    # quelques milliers de cellules au maximum. Le SVD exact (LAPACK gesdd) est
    # donc plus rapide QUE l'itératif tout en étant déterministe, et il n'a
    # aucune approximation à justifier.
    _U_full, _S_full, Vt_full = np.linalg.svd(R_centered, full_matrices=False)
    k_use = min(k_safe, _S_full.shape[0])
    U     = _U_full[:, :k_use]
    sigma = _S_full[:k_use]
    Vt    = Vt_full[:k_use, :]
    R_pred = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_mean[:, np.newaxis]

    # Colonnes sans interaction dans train_loo.
    #
    # Attention, ce ne sont PAS des colonnes nulles après reconstruction. svds
    # calcule une approximation de rang k_safe ; le résidu de troncature se
    # projette dans les colonnes vides, qui se retrouvent avec un score VRAIMENT
    # variable d'un user à l'autre (écart-type mesuré ~0.22 sur le jeu réel, y
    # compris en mode 'interactions' où Accessoires et Cosmétiques perdent leur
    # signal dans train_loo). C'est du bruit de reconstruction, pas un signal
    # appris, et avec alpha=0.6 il pèserait 60 % du score hybride sur des
    # catégories qu'on ne connaît pas.
    #
    # On les remplace donc par le prior de l'utilisateur (sa moyenne) : une
    # estimation honnête "je ne sais rien de cette catégorie", identique pour
    # toutes les catégories inconnues. Leur classement relatif est alors décidé
    # par le seul Content-Based.
    empty_cols = np.flatnonzero(np.asarray(R.sum(axis=0)).ravel() == 0)
    if empty_cols.size:
        residual_std = float(R_pred[:, empty_cols].std(axis=0).max())
        log.info(
            "SVD : %d/%d colonnes sans interaction. Résidu de troncature "
            "mesuré (écart-type max %.3f) — remplacé par le prior user_mean",
            empty_cols.size, R_centered.shape[1], residual_std,
        )
        if residual_std > 0.05:
            log.warning(
                "Le résidu de troncature du SVD était important (%.3f) : sans "
                "neutralisation, le score de ces catégories aurait été du bruit",
                residual_std,
            )
        R_pred[:, empty_cols] = user_mean[:, np.newaxis]

    log.info("SVD terminé — matrice prédite : %s", R_pred.shape)
    return R_pred, user_mean, R_centered, k_safe


def train_als(train_loo: pd.DataFrame, user_enc, item_enc,
              n_factors: int = 32, regularization: float = 0.1,
              n_iterations: int = 15, alpha: int = 40, random_state: int = 42):
    """Entraîne le modèle ALS implicite."""
    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    t = train_loo.copy()
    t["user_idx"] = user_enc.transform(t["user_id"])
    t["item_idx"] = item_enc.transform(t["item_id"])

    pair_counts = t.groupby(["user_idx", "item_idx"]).size().reset_index(name="count")
    # Ordre stable : groupby ne garantit pas l'ordre des lignes, et l'ordre
    # d'indexation de csr_matrix influence les solveurs itératifs.
    pair_counts = pair_counts.sort_values(["user_idx", "item_idx"], kind="mergesort")
    user_item_matrix = csr_matrix(
        (pair_counts["count"].values,
         (pair_counts["user_idx"].values, pair_counts["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    seen_items = t.groupby("user_idx")["item_idx"].apply(list).to_dict()

    log.info(
        "Entraînement ALS : n_factors=%d, reg=%s, iter=%d",
        n_factors, regularization, n_iterations,
    )
    model = ALSRecommender(
        n_factors=n_factors,
        n_iterations=n_iterations,
        alpha=alpha,
        regularization=regularization,
        random_state=random_state,
    )
    model.fit(user_item_matrix)

    # ALSRecommender laisse les colonnes sans aucune interaction avec les
    # facteurs aléatoires de l'initialisation (le `continue` sur row.nnz == 0 ne
    # les réinitialise pas). Leur score était donc du bruit, capable de
    # remonter dans le classement au hasard. On les met explicitement à zéro :
    # ces catégories n'ont aucun signal collaboratif, seul le Content-Based peut
    # les proposer, et un score nul le dit honnêtement.
    empty_items = np.flatnonzero(np.asarray(user_item_matrix.sum(axis=0)).ravel() == 0)
    if empty_items.size:
        model.item_factors[empty_items] = 0.0
        log.info(
            "ALS : %d/%d catégories sans interaction → facteurs forcés à 0 "
            "(recommandables uniquement via Content-Based) : %s",
            empty_items.size, n_items,
            [item_enc.inverse_transform([i])[0] for i in empty_items],
        )

    return model, user_item_matrix, seen_items


def build_content_features(data: dict, valid_items):
    """
    Construit la matrice de similarité cosine TF-IDF par CATÉGORIE.

    Sur une marketplace d'occasion on travaille au niveau catégorie.
    Les features textuelles combinent :
      - le nom de la catégorie (le signal le plus discriminant)
      - les noms des articles de cette catégorie (si disponibles)

    valid_items contient les CATÉGORIES présentes dans les interactions.
    """
    products  = data["products"]
    cat_names = data["cat_names"]

    # Enrichissement avec les noms anglais si disponibles
    products_full = products.merge(cat_names, on="product_category_name", how="left")

    # Agrégation par catégorie : concatène les noms des articles pour enrichir le TF-IDF
    # valid_items = liste des catégories vues dans les interactions
    categories_cb = pd.DataFrame({"category": list(valid_items)})

    # Noms des articles disponibles par catégorie (pour enrichir le TF-IDF)
    if "product_name" in products_full.columns:
        pool = products_full
        if "available" in pool.columns:
            pool = pool[pool["available"].fillna(1) == 1]  # articles dispo seulement
        else:
            log.warning("Colonne 'available' absente de products.csv — TF-IDF sur tout le catalogue")
        cat_article_names = (
            pool.groupby("product_category_name")["product_name"]
            .apply(lambda names: " ".join(n for n in names if isinstance(n, str) and n.strip()))
            .reset_index()
        )
        cat_article_names.columns = ["category", "articles_text"]
        categories_cb = categories_cb.merge(cat_article_names, on="category", how="left")
        categories_cb["articles_text"] = categories_cb["articles_text"].fillna("")
    else:
        categories_cb["articles_text"] = ""

    # Nom anglais de la catégorie
    cat_en_map = (
        cat_names.set_index("product_category_name")["product_category_name_english"]
        .to_dict()
        if not cat_names.empty and "product_category_name" in cat_names.columns
        else {}
    )
    categories_cb["category_en"] = categories_cb["category"].map(cat_en_map).fillna(
        categories_cb["category"]
    )

    # Description finale = nom_catégorie + nom_anglais + noms_articles
    categories_cb["description"] = (
        categories_cb["category"] + " "
        + categories_cb["category_en"] + " "
        + categories_cb["articles_text"]
    ).str.strip()

    tfidf      = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_mat  = tfidf.fit_transform(categories_cb["description"])
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)

    item_to_idx = {cat: idx for idx, cat in enumerate(categories_cb["category"])}
    idx_to_item = {idx: cat for cat, idx in item_to_idx.items()}

    log.info(
        "Content-Based (catégories) — %d catégories, matrice %s",
        len(categories_cb), cosine_sim.shape,
    )
    return cosine_sim, item_to_idx, idx_to_item, products_full


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modèle Hybride
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise un vecteur de scores en [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return scores


def build_hybrid_recommender(
    R_pred, cosine_sim, item_to_idx, idx_to_item,
    user_enc, item_enc, train_df, popular_items,
    alpha: float = 1.0,
):
    """
    Retourne une fonction de recommandation hybride avec le alpha fourni.
    alpha=1.0 → pur SVD | alpha=0.0 → pur Content-Based
    """
    n_items      = len(item_enc.classes_)
    R_pred_norm  = np.apply_along_axis(normalize_scores, 1, R_pred)

    def get_cb_scores(user_id):
        user_items = train_df[train_df["user_id"] == user_id]["item_id"].tolist()
        user_items = [i for i in user_items if i in item_to_idx]
        if not user_items:
            return None
        scores_cb = np.zeros(len(cosine_sim))
        for item_id in user_items:
            scores_cb += cosine_sim[item_to_idx[item_id]]
        full_scores = np.zeros(n_items)
        for cb_idx, pid in idx_to_item.items():
            try:
                full_scores[int(item_enc.transform([pid])[0])] = scores_cb[cb_idx]
            except ValueError:
                pass
        return normalize_scores(full_scores)

    def hybrid_recommend(user_id: str, n: int = 10) -> list:
        try:
            user_idx = int(user_enc.transform([user_id])[0])
        except ValueError:
            return popular_items[:n]
        svd_scores = R_pred_norm[user_idx]
        cb_scores  = get_cb_scores(user_id)
        hybrid_scores = (
            svd_scores if cb_scores is None
            else alpha * svd_scores + (1 - alpha) * cb_scores
        )
        top_idxs = np.argsort(hybrid_scores)[::-1][:n]
        return item_enc.inverse_transform(top_idxs).tolist()

    return hybrid_recommend


# ─────────────────────────────────────────────────────────────────────────────
# 5. Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0

def tune_alpha(
    df, u_enc, i_enc, data, item_space,
    train_loo, test_loo, cosine_sim, item_to_idx, idx_to_item,
    n_items: int, random_state: int = 42, eval_sample: int = 500,
    grid: tuple[float, ...] | None = None, log_fn=None,
    svd_k_factors: int = 150,
):
    """
    Choisit alpha (poids SVD vs Content-Based) par validation croisée sur le LOO.

    Principe : pour chaque fold, on retire une interaction par client, on
    réentraîne un SVD sur ce qui reste, on évalue l'hybride à chaque alpha, puis
    on recommence. Le SVD de chaque fold ne voit donc jamais l'item évalué, et le
    score final est la moyenne sur tous les folds (et non sur un unique découpage,
    qui sur 14 clients serait entièrement.browser par le hasard).

    On maximise le NDCG au plus petit k discriminant (3). Avec 7 catégories, un
    k=7 renvoie le catalogue entier et tous les alpha obtiennent 1.0 : choisir
    alpha dessus revient à tirer une pièce.

    Retourne (alpha, dict d'audit).
    """
    grid = grid or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    k_eval = max(1, min(3, n_items))
    ndcg_key = f"NDCG@{k_eval}"

    # Folds : LOO par client, réordonné pour que chaque fold retire une
    # interaction DIFFÉRENTE (le tri initial surchargeait le fold 0).
    d = df.copy()
    d["_rid"] = range(len(d))
    ordered = d.sort_values(["user_id", "_rid"], kind="mergesort")
    max_n = int(ordered.groupby("user_id").cumcount(ascending=False).max()) + 1
    n_folds = max(1, min(max_n, 3))

    all_cats = [str(c) for c in i_enc.inverse_transform(np.arange(n_items))]
    seen_cats = set(all_cats)
    scores = {a: [] for a in grid}

    for fold in range(n_folds):
        fold_mask = (
            ordered.groupby("user_id").cumcount(ascending=False) == fold
        )
        tr = ordered[~fold_mask]
        te = ordered[fold_mask]
        if tr.empty or te.empty or tr["user_id"].nunique() == 0:
            continue

        R_fold, _, _, _ = train_svd(tr, u_enc, i_enc, k_factors=svd_k_factors)

        # Baseline propre au fold : popularity du train de CE fold seulement.
        pop_f = tr.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
        pop_f_items = [str(c) for c in pop_f.index.tolist() if str(c) in seen_cats]
        pop_f_items += [c for c in all_cats if c not in set(pop_f_items)]

        base_scores: list[float] = []
        base_m = evaluate(
            lambda _u, n, _p=pop_f_items: _p[:n], te, tr, k=k_eval,
            sample=eval_sample, seed=random_state, n_items=n_items,
        )
        base_scores.append(base_m.get(ndcg_key, 0.0))

        for a in grid:
            fn = build_hybrid_recommender(
                R_fold, cosine_sim, item_to_idx, idx_to_item,
                # Historique pour le CB = DONNÉES DU TRAIN DU FOLD. Passer `df`
                # complet ferait passer le content-based enifu : l'item de test
                # figurerait dans son propre historique, sa similarité avec lui-même
                # vaudrait 1.0 et le classerait premier systématiquement — alpha=0
                # gangnerait toujours, pour la mauvaise raison.
                u_enc, i_enc, tr, pop_f_items, alpha=a,
            )
            m = evaluate(
                fn, te, tr, k=k_eval, sample=eval_sample,
                seed=random_state, n_items=n_items,
            )
            scores[a].append(m.get(ndcg_key, 0.0))

    means = {a: float(np.mean(v)) if v else 0.0 for a, v in scores.items()}
    base_mean = float(np.mean(base_scores)) if base_scores else 0.0
    # En cas d'égalité stricte, on privilégie le modèle collaboratif (alpha élevé)
    # : il se généralise mieux que la similarité de noms de catégories, qui est
    # ici le seul signal disponible au content-based.
    best = max(sorted(means, reverse=True), key=lambda a: means[a])
    audit = {
        "k_eval": k_eval,
        "n_folds": n_folds,
        "grid": list(grid),
        "ndcg_mean_per_alpha": {str(a): round(means[a], 4) for a in grid},
        "baseline_ndcg_mean": round(base_mean, 4),
        "lift_at_k": round(means[best] - base_mean, 4),
    }
    if log_fn:
        log_fn(
            "CV alpha (%d folds, NDCG@%d moyen) : %s | baseline=%.4f → alpha=%.1f",
            n_folds, k_eval,
            " | ".join(f"{a:.1f}:{means[a]:.4f}" for a in grid),
            base_mean, best,
        )
    return float(best), audit


def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate(
    reco_fn, test_df, train_df, k: int = 10, sample: int = 500,
    seed: int = 42, n_items: int | None = None,
) -> dict:
    """
    Évalue un modèle sur le Leave-One-Out.

    Deux garde-fous indispensables sur de très petits jeux de données :

      - `k_eff` est plafonné au nombre de catégories. Avec 7 catégories et K=10, la
        recommandation renvoyée contenait *toutes* les catégories : l'item de test
        était donc presque toujours présent, d'où Recall@10 = 1.0 et NDCG@10 = 0.94
        sans que cela traduise une quelconque qualité de recommandation.

    - seuls les utilisateurs ayant un historique d'entraînement sont évalués ; les
      autres ne mesureraient que la branche de fallback (aucun historique ⇒
      content-based nul, SVD sans signal).
    """
    n_items = n_items or int(train_df["item_id"].nunique())
    k_eff = max(1, min(k, n_items))
    if k_eff < k:
        log.warning(
            "k=%d plafonné à %d : le jeu ne contient que %d catégories",
            k, k_eff, n_items,
        )

    seen_by_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    all_users = list(test_df["user_id"].unique())
    users = [u for u in all_users if seen_by_user.get(u)]
    n_skipped = len(all_users) - len(users)
    if n_skipped:
        log.warning(
            "%d/%d users ignorés : aucun historique d'entraînement",
            n_skipped, len(all_users),
        )
    if not users:
        log.error("Aucun utilisateur évaluable — métriques indisponibles")
        return {
            f"Precision@{k_eff}": 0.0, f"Recall@{k_eff}": 0.0,
            f"NDCG@{k_eff}": 0.0, "n_users_eval": 0, "eval_k": k_eff,
            "n_distinct_pairs": 0,
        }

    # Diagnostic de dégénérescence : sur ce jeu, les 14 users évaluables ne
    # représentent que 5 paires (historique → item attendu) distinctes, et le
    # voisin TF-IDF est l'item attendu dans 13 cas. Les métriques sont alors
    # proches de 1.0 par construction et ne mesurent aucune qualité de modèle.
    # On compte ces paires pour le signaler explicitement dans le résumé.
    signatures = {
        (frozenset(seen_by_user[u]),
         frozenset(test_df[test_df["user_id"] == u]["item_id"]))
        for u in users
    }
    n_pairs = len(signatures)
    if n_pairs <= max(3, len(users) // 4):
        log.warning(
            "Jeu d'évaluation dégénéré : %d users évaluables ne couvrent que %d "
            "paires (historique → item attendu) distinctes — NDCG/Recall proches de "
            "1.0 sont structurels, pas le signe d'un bon modèle",
            len(users), n_pairs,
        )

    if len(users) > sample:
        rng = np.random.default_rng(seed)
        users = [users[i] for i in rng.choice(len(users), size=sample, replace=False)]

    precisions, recalls, ndcgs = [], [], []
    for user in users:
        relevant     = set(test_df[test_df["user_id"] == user]["item_id"])
        already_seen = seen_by_user[user]
        try:
            reco = reco_fn(user, k_eff + len(already_seen))
            reco = [i for i in reco if i not in already_seen][:k_eff]
        except Exception as e:  # noqa: BLE001 — une reco cassée ne doit pas tuer l'eval
            log.warning("Utilisateur %s non évalué : %s: %s", user, type(e).__name__, e)
            continue
        precisions.append(precision_at_k(reco, relevant, k_eff))
        recalls.append(recall_at_k(reco, relevant, k_eff))
        ndcgs.append(ndcg_at_k(reco, relevant, k_eff))

    if not precisions:
        log.error("Toutes les évaluations ont échoué — métriques indisponibles")
        return {
            f"Precision@{k_eff}": 0.0, f"Recall@{k_eff}": 0.0,
            f"NDCG@{k_eff}": 0.0, "n_users_eval": 0, "eval_k": k_eff,
            "n_distinct_pairs": n_pairs,
        }

    return {
        f"Precision@{k_eff}": round(float(np.mean(precisions)), 4),
        f"Recall@{k_eff}":    round(float(np.mean(recalls)), 4),
        f"NDCG@{k_eff}":      round(float(np.mean(ndcgs)), 4),
        "n_users_eval":      len(precisions),
        "eval_k":            k_eff,
        "n_distinct_pairs":  n_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    models_dir: Path,
    R_pred, als_model, cosine_sim,
    item_to_idx, idx_to_item,
    user_enc, item_enc, popular_items,
    summary: dict,
    products_df: pd.DataFrame | None = None,
    user_history: dict | None = None,
    product_sim: tuple | None = None,
    product_features_df: pd.DataFrame | None = None,
):
    """Sauvegarde tous les artefacts nécessaires à l'API."""
    models_dir.mkdir(parents=True, exist_ok=True)

    np.save(models_dir / "svd_R_pred.npy", R_pred)
    log.info("Sauvegardé : svd_R_pred.npy")

    with open(models_dir / "als_model.pkl", "wb") as f:
        pickle.dump(als_model, f)
    log.info("Sauvegardé : als_model.pkl")

    with open(models_dir / "cosine_sim.pkl", "wb") as f:
        pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)
    log.info("Sauvegardé : cosine_sim.pkl")

    with open(models_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(models_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)
    log.info("Sauvegardé : encodeurs")

    with open(models_dir / "popular_items.pkl", "wb") as f:
        pickle.dump(popular_items, f)
    log.info("Sauvegardé : popular_items.pkl (%d catégories)", len(popular_items))

    # Historique user → catégories : indispensable au score Content-Based en
    # service. Sans cet artefact l'API ne peut que retomber sur du SVD pur.
    if user_history is not None:
        with open(models_dir / "user_history.pkl", "wb") as f:
            pickle.dump(user_history, f)
        log.info("Sauvegardé : user_history.pkl (%d users)", len(user_history))

    # Table produits complète avec colonnes available et product_name
    # Utilisée par l'API pour retrouver les articles disponibles dans une catégorie
    if products_df is not None:
        products_df.to_csv(models_dir / "products_catalog.csv", index=False)
        log.info("Sauvegardé : products_catalog.csv (%d lignes)", len(products_df))

    with open(models_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Sauvegardé : results_summary.json")

    # ── Couche article ───────────────────────────────────────────────────────
    # Similarité cosinus entre articles (TF-IDF sur les titres) + rapport de
    # suffisance des données. C'est ce qui permet à l'API de recommander des
    # articles, et non seulement des catégories.
    if product_sim is not None and len(product_sim[0]) > 0:
        product_ids, sim_matrix = product_sim
        with open(models_dir / "product_sim.pkl", "wb") as f:
            pickle.dump((sim_matrix, {str(p): i for i, p in enumerate(product_ids)}), f)
        log.info(
            "Sauvegardé : product_sim.pkl (%d articles, matrice %dx%d)",
            len(product_ids), sim_matrix.shape[0], sim_matrix.shape[1],
        )

    if product_features_df is not None:
        product_features_df.to_csv(models_dir / "product_features.csv", index=False)
        log.info("Sauvegardé : product_features.csv (%d lignes)", len(product_features_df))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train MEL recommendation models")
    parser.add_argument("--data-dir",      type=Path, default=Path("data/raw"))
    parser.add_argument("--models-dir",    type=Path, default=Path("models"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-factors",     type=int,  default=150,  help="Facteurs latents SVD")
    parser.add_argument("--als-factors",   type=int,  default=32,   help="Facteurs latents ALS")
    parser.add_argument("--als-reg",       type=float,default=0.1,  help="Régularisation ALS")
    parser.add_argument("--als-iter",      type=int,  default=15,   help="Itérations ALS")
    parser.add_argument("--alpha",         type=float,default=None,
                        help="Poids SVD dans le hybride (0=CB pur, 1=SVD pur). "
                             "Défaut : choisi automatiquement par validation croisée "
                             "sur le LOO (pluscourt k). Forcer une valeur pour un test.")
    parser.add_argument("--item-space",    choices=["interactions", "catalog"], default="interactions",
                        help="Catégories couvertes par item_enc : 'interactions' = seulement celles "
                             "observées chez les utilisateurs (défaut) ; 'catalog' = ajoute les "
                             "catégories du catalogue avec du stock, classées par similarité de nom seule")
    parser.add_argument("--eval-k",        type=int,  default=10,   help="K pour les métriques @K")
    parser.add_argument("--eval-sample",   type=int,  default=500,  help="Nb users pour l'évaluation")
    parser.add_argument("--random-state",  type=int,  default=42)
    return parser.parse_args()


def setup_mlflow() -> str:
    """
    Configure le tracking MLflow, avec repli sur une base SQLite locale.

    Un simple `urlopen(uri + "/health")` ne suffit pas pour décider : un proxy
    HTTP peut renvoyer 200 alors que le client MLflow n'atteint pas le serveur.
    C'est exactement ce qui est arrivé dans train_output.txt, où le health-check
    est passé puis `mlflow.set_experiment()` a planté après ~4,5 min de retries
    sur http://mlflow:5000. On teste donc une vraie API MLflow avant de
    s'engager sur une URI HTTP.
    """
    # Évite un blocage de plusieurs minutes quand le serveur est injoignable
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()

    if tracking_uri.startswith("http"):
        mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.search_experiments(max_results=1)  # vrai appel API
            log.info("MLflow tracking URI : %s", tracking_uri)
            return tracking_uri
        except Exception as e:  # noqa: BLE001 — tout échec réseau ⇒ repli local
            log.warning(
                "Serveur MLflow %s inutilisable (%s: %s) — repli SQLite local",
                tracking_uri, type(e).__name__, e,
            )
    elif tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        log.info("MLflow tracking URI : %s", tracking_uri)
        return tracking_uri

    # SQLite local — compatible MLflow 2.15+ (le file store est en maintenance)
    db_path = Path("mlflow-data") / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(sqlite_uri)
    log.info("MLflow tracking URI : %s (SQLite local)", sqlite_uri)
    return sqlite_uri


def main():
    args = parse_args()
    t0   = time.time()
    log.info("=== Pipeline MEL Recommandation ===")
    log.info("Config : %s", vars(args))

    setup_mlflow()

    try:
        mlflow.set_experiment("mel-recommandation")
    except Exception as e:  # noqa: BLE001 — le tracking ne doit pas tuer l'entraînement
        log.warning("MLflow set_experiment impossible (%s) — tracking désactivé", e)

    with mlflow.start_run(run_name="hybrid-svd-cb"):

        # Logger tous les hyperparamètres
        mlflow.log_params({
            "svd_k_factors": args.k_factors,
            "als_n_factors": args.als_factors,
            "als_reg":       args.als_reg,
            "als_iter":      args.als_iter,
            "hybrid_alpha":  args.alpha,
            "eval_k":        args.eval_k,
            "eval_sample":   args.eval_sample,
            "random_state":  args.random_state,
        })

        # ── Chargement ───────────────────────────────────────────────────────
        data = load_data(args.data_dir)

        # ── Preprocessing ────────────────────────────────────────────────────
        df = build_interactions(data)
        df = filter_interactions(df)

        # Espace des items : par défaut, seules les catégories observées dans les
        # interactions. Les autres restent inatteignables, ce qui est un choix —
        # elles seraient classées sur la similarité de leur nom uniquement.
        item_space, item_space_report = build_item_space(
            data["products"], df["item_id"].unique(), mode=args.item_space,
        )

        df, u_enc, i_enc = encode_ids(df, item_space=item_space)

        # Logger les stats du dataset (depuis les encodeurs : max()+1 est faux
        # dès qu'un encodage n'est pas contigu)
        n_users = len(u_enc.classes_)
        n_items = len(i_enc.classes_)
        mlflow.log_params({
            "n_users":        n_users,
            "n_items":        n_items,
            "n_interactions": len(df),
        })

        args.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.processed_dir / "interactions_filtered.csv", index=False)

        # ── Couche article : événements, features, similarité, suffisance ────
        # Indépendante du modèle par catégories : elle sert à recommander des
        # articles, ce que le modèle de catégories ne peut pas faire.
        events = load_events(args.data_dir, products=data["products"])
        prod_feat = product_features(data["products"], events)
        product_sim = build_product_similarity(prod_feat)
        sufficiency = data_sufficiency_report(events, prod_feat)
        events.to_csv(args.processed_dir / "events_canonical.csv", index=False)
        log.info(
            "Couche article : %d événements | similarité %dx%d | CF article : %s",
            len(events), len(product_sim[1]), len(product_sim[1]),
            sufficiency["collaborative_verdict"],
        )

        train_loo, test_loo, train_rand, test_rand = make_splits(
            df, random_state=args.random_state
        )
        train_loo.to_csv(args.processed_dir / "train.csv",        index=False)
        test_loo.to_csv(args.processed_dir  / "test.csv",         index=False)
        train_rand.to_csv(args.processed_dir / "train_random.csv", index=False)
        test_rand.to_csv(args.processed_dir  / "test_random.csv",  index=False)

        # ── Popularité (cold-start) ───────────────────────────────────────────
        # Statistique globale sur TOUTES les interactions : elle sert au
        # cold-start et au fallback, pas à l'évaluation (voir baseline_nondegeneree
        # plus bas, qui doit elle être calculée sur le train uniquement).
        all_cats = [str(c) for c in i_enc.inverse_transform(np.arange(n_items))]
        pop_counts = df.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
        seen_cats = set(all_cats)
        popular_items = [str(c) for c in pop_counts.index.tolist() if str(c) in seen_cats]
        popular_items += [c for c in all_cats if c not in set(popular_items)]
        log.info("Popularité (cold-start) : %d catégories", len(popular_items))

        # ── Baseline NON dégénérée pour l'évaluation ─────────────────────────
        # La baseline doit être classée avec la même information que le modèle :
        # le TRAIN uniquement. Calculée sur `df` complet, elle voyait l'interaction
        # mise de côté (l'item à prédire se retrouve souvent en tête parce qu'il
        # est le plus récent du client), ce qui faisait remonter son NDCG à 1.0 et
        # rendait TOUT modèle hybride apparemment négatif. Ici la popularity est
        # recalculée depuis train_loo, donc comparable àHybrid.
        train_pop = train_loo.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
        baseline_popular_items = [str(c) for c in train_pop.index.tolist() if str(c) in seen_cats]
        baseline_popular_items += [c for c in all_cats if c not in set(baseline_popular_items)]

        # ── Historique user → catégories (score Content-Based en service) ───
        # Les user_id sont déjà des str (encode_ids les a convertis) et les
        # valeurs doivent rester des LISTES de catégories.
        #
        # Ne surtout pas faire .astype(str) ici : ça sérialise chaque liste en
        # chaîne ("['Vêtements femmes', ...]") et l'API reçoit alors
        # list(chaîne) = une liste de caractères. Aucun caractère ne correspond
        # à un nom de catégorie, filter_seen ne filtre plus rien, et les
        # catégories déjà consommées sont ré-aménées. Les tests ne le voyaient
        # pas : leur fixture écrivait user_history.pkl à la main, avec de vraies
        # listes.
        user_history = df.groupby("user_id")["item_id"].apply(list).to_dict()
        bad = [u for u, v in user_history.items() if not isinstance(v, list)]
        if bad:
            raise TypeError(
                f"user_history mal sérialisé pour {len(bad)} users (ex: {bad[:3]})"
            )
        log.info(
            "Historique persisté : %d users, %.1f catégories/user",
            len(user_history),
            sum(len(v) for v in user_history.values()) / max(1, len(user_history)),
        )

# ── SVD ──────────────────────────────────────────────────────────────
        # ENTRAÎNÉ SUR `df` COMPLET : c'est le modèle qui est Sauvegardé puis servi
        # en production. Avant, il était entraîné sur `train_loo` (17 lignes sur 51)
        # — le split d'évaluation était ré-utilisé pour l'apprentissage, si bien
        # que 67 % des interactions n'arrivaient jamais dans le modèle servi et
        # que 20 des 34 clients se retrouvaient sans aucun profil en production.
        R_pred, _user_mean, _R_centered, svd_k_effective = train_svd(
            df, u_enc, i_enc, k_factors=args.k_factors
        )

        # ── ALS ──────────────────────────────────────────────────────────────
        # Identique : c'est un artefact servi, il doit voir toutes les données.
        als_model, _, _seen_items = train_als(
            df, u_enc, i_enc,
            n_factors=args.als_factors,
            regularization=args.als_reg,
            n_iterations=args.als_iter,
            random_state=args.random_state,
        )

        # ── Content-Based ─────────────────────────────────────────────────────
        # cosine_sim doit couvrir le MÊME espace que item_enc, sinon les
        # catégories ajoutées du catalogue n'ont ni similarité ni index CB.
        cosine_sim, item_to_idx, idx_to_item, _products_full = build_content_features(
            data, item_space
        )
        missing_cb = sorted(set(i_enc.classes_) - set(item_to_idx))
        if missing_cb:
            log.error(
                "Content-Based : %d catégories de item_enc sans similarité : %s",
                len(missing_cb), missing_cb,
            )

        # ── Choix d'alpha PAR VALIDATION CROISÉE ─────────────────────────────
        # L'alpha par défaut 0,6 n'avait jamais été justifié : on le sélectionne
        # donc sur le LOO, en réentraînant un SVD par fold (donc jamais sur les
        # données évaluées). On maximise le NDCG au plus petit k utile — avec
        # 7 catégories, k=7 renvoie tout et ne départage rien.
        # `alpha` force par l'utilisateur court-circuite tout ce bloc.
        if args.alpha is None:
            alpha, alpha_cv = tune_alpha(
                df, u_enc, i_enc, data, item_space,
                train_loo, test_loo, cosine_sim, item_to_idx, idx_to_item,
                n_items=n_items, random_state=args.random_state,
                eval_sample=args.eval_sample,
                svd_k_factors=args.k_factors,
                log_fn=log.info,
            )
            log.info(
                "alpha sélectionné par CV : %.2f (lift NDCG@%d = %+.4f vs baseline)",
                alpha, alpha_cv["k_eval"], alpha_cv["lift_at_k"],
            )
        else:
            alpha = args.alpha
            log.info("alpha forcé à %.2f (validation croisée ignorée)", alpha)
            alpha_cv = None

        # ── Modèle Hybride (ÉVALUATION : SVD réentraîné sur train_loo) ───────
        # Pour mesurer honnêtement, il faut un SVD qui n'a JAMAIS vu test_loo.
        # Le SVD servi (entraîné sur df complet) ne peut pas servir de juge :
        # il estPrecis sur les items qu'on lui demande de deviner.
        R_pred_eval, _, _, _ = train_svd(
            train_loo, u_enc, i_enc, k_factors=args.k_factors
        )
        # Historique CB pour l'évaluation = train_loo UNIQUEMENT. Avec `df`,
        # l'item de test serait dans l'historique du client, sa similarité avec
        # lui-même vaudrait 1.0 et il serait toujours classé premier : NDCG=1.0
        # sans aucune valeur prédictive.
        hybrid_fn = build_hybrid_recommender(
            R_pred_eval, cosine_sim, item_to_idx, idx_to_item,
            u_enc, i_enc, train_loo, baseline_popular_items,
            alpha=alpha,
        )

        # ── Évaluation ───────────────────────────────────────────────────────
        # ── Évaluation ───────────────────────────────────────────────────────
        log.info("Évaluation du modèle hybride (alpha=%.2f)...", alpha)
        metrics = evaluate(
            hybrid_fn, test_loo, train_loo,
            k=args.eval_k, sample=args.eval_sample,
            seed=args.random_state, n_items=n_items,
        )
        log.info("Résultats : %s", metrics)
        k_eff = metrics["eval_k"]

        # Baseline popularité sur le MÊME split, classée avec la seule information
        # du train (baseline_popular_items). Sans elle, un NDCG proche de 1.0
        # sur 7 catégories ne prouve rien : la baseline la plus triviale peut
        # obtenir le même score. Le lift est ce qui rend la métrique interprétable.
        baseline_metrics = evaluate(
            lambda _user, n: baseline_popular_items[:n],
            test_loo, train_loo,
            k=args.eval_k, sample=args.eval_sample,
            seed=args.random_state, n_items=n_items,
        )
        ndcg_key = f"NDCG@{k_eff}"
        lift_ndcg = round(metrics[ndcg_key] - baseline_metrics[ndcg_key], 4)
        log.info(
            "Baseline popularité (train only) : NDCG@%d = %.4f (lift hybride = %+.4f)",
            k_eff, baseline_metrics[ndcg_key], lift_ndcg,
        )
        if lift_ndcg <= 0:
            log.warning(
                "Le modèle hybride ne bat PAS la baseline de popularité "
                "(lift NDCG = %+.4f) sur ce jeu — alpha=%s n'est pas validé",
                lift_ndcg, alpha,
            )

        # Métriques à k=3 : avec 7 catégories, k=7 renvoie tout le catalogue et
        # ne discrimine plus rien. k=3 est le premier k réellement informatif.
        metrics_k3 = evaluate(
            hybrid_fn, test_loo, train_loo,
            k=3, sample=args.eval_sample,
            seed=args.random_state, n_items=n_items,
        )
        baseline_k3 = evaluate(
            lambda _user, n: baseline_popular_items[:n],
            test_loo, train_loo,
            k=3, sample=args.eval_sample,
            seed=args.random_state, n_items=n_items,
        )
        lift_ndcg_k3 = round(metrics_k3["NDCG@3"] - baseline_k3["NDCG@3"], 4)
        log.info(
            "Lift au k discriminant (3) : %+.4f (hybride %.4f vs baseline %.4f)",
            lift_ndcg_k3, metrics_k3["NDCG@3"], baseline_k3["NDCG@3"],
        )

        # Logger les métriques dans MLflow
        mlflow.log_metric(f"precision_at_{k_eff}", metrics[f"Precision@{k_eff}"])
        mlflow.log_metric(f"recall_at_{k_eff}",    metrics[f"Recall@{k_eff}"])
        mlflow.log_metric(f"ndcg_at_{k_eff}",      metrics[ndcg_key])
        mlflow.log_metric(f"baseline_ndcg_at_{k_eff}", baseline_metrics[ndcg_key])
        mlflow.log_metric("ndcg_lift_vs_baseline", lift_ndcg)
        mlflow.log_metric("ndcg_at_3",           metrics_k3["NDCG@3"])
        mlflow.log_metric("ndcg_lift_at_3",     lift_ndcg_k3)
        mlflow.log_metric("alpha_selected",     alpha)
        mlflow.log_metric("n_users_eval",          metrics["n_users_eval"])
        if alpha_cv:
            mlflow.log_param("alpha_cv_audit", alpha_cv["ndcg_mean_per_alpha"])

        training_time = round(time.time() - t0, 1)
        mlflow.log_metric("training_time_sec", training_time)

        # ── Résumé ────────────────────────────────────────────────────────────
        summary = {
            "dataset":         "MEL Cameroun",
            "python_version":  "3.13+",
            "n_users":         n_users,
            "n_items":         n_items,
            "n_interactions":  len(df),
            "n_available_products": int(
                (_products_full["available"] == 1).sum()
            ) if "available" in _products_full.columns else None,
            "item_space": item_space_report,
            "product_layer": sufficiency,
            "model":           "Hybrid (SVD + Content-Based)",
            "trained_on":      "df complet (toutes les interactions)",
            "evaluated_on":    "LOO (le SVD de l'évaluation est réentraîné sur train_loo)",
            "hyperparameters": {
                "svd_k_factors":            args.k_factors,
                "svd_k_factors_effective":  svd_k_effective,
                "als_n_factors":            args.als_factors,
                "als_reg":                  args.als_reg,
                "als_iter":                 args.als_iter,
                "hybrid_alpha":             alpha,
                "alpha_selection":          "validation croisée LOO" if alpha_cv else "forcé (--alpha)",
                "alpha_cv":                 alpha_cv,
            },
            "metrics": metrics,
            "metrics_baseline_popularity": baseline_metrics,
            "metrics_at_k3":              metrics_k3,
            "metrics_baseline_at_k3":       baseline_k3,
            "ndcg_lift_vs_baseline":      lift_ndcg,
            "ndcg_lift_at_k3":            lift_ndcg_k3,
            "metrics_caveat": (
                "Jeux de données minuscule et dégénéré : "
                f"{metrics['n_users_eval']} users évaluables, "
                f"{metrics.get('n_distinct_pairs', 0)} paires distinctes, "
                f"{n_items} catégories. Métriques non significatives."
            ),
            "training_time_sec": training_time,
        }

        # ── Sauvegarde ────────────────────────────────────────────────────────
        save_artifacts(
            args.models_dir,
            R_pred, als_model, cosine_sim,
            item_to_idx, idx_to_item,
            u_enc, i_enc, popular_items,
            summary,
            products_df=_products_full,
            user_history=user_history,
            product_sim=product_sim,
            product_features_df=prod_feat,
        )

        # Logger les artefacts dans MLflow
        mlflow.log_artifact(str(args.models_dir / "results_summary.json"))
        mlflow.log_artifact(str(args.models_dir / "svd_R_pred.npy"))
        # ALSRecommender est un modèle custom (non sklearn) → log via pickle artifact
        mlflow.log_artifact(str(args.models_dir / "als_model.pkl"), artifact_path="als_model")

        log.info("=== Pipeline terminée en %.1fs ===", training_time)
        log.info(
            "NDCG@%d = %.4f (baseline %.4f, lift %+.4f) | NDCG@3 = %.4f",
            k_eff, metrics[ndcg_key], baseline_metrics[ndcg_key], lift_ndcg,
            metrics_k3["NDCG@3"],
        )
        log.info("MLflow run ID : %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
