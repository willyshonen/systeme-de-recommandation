"""
src/train.py
Pipeline d'entraînement reproductible — Système de recommandation hybride MEL Cameroun.

Usage :
    python src/train.py
    python src/train.py --data-dir data/raw --models-dir models --k-factors 150 --als-factors 32
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

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

def build_interactions(data: dict) -> pd.DataFrame:
    """
    Construit le dataframe d'interactions implicites par CATÉGORIE.

    Sur une marketplace d'occasion chaque article est unique et disparaît
    après vente → on ne recommande pas des articles spécifiques mais des
    catégories, puis on retourne les articles disponibles dans ces catégories.

    Stratégie de scoring :
        vue        → score 1  (intérêt faible)
        panier     → score 3  (intérêt fort)
        achat      → score 5  (conversion)

    Le score final par (user, catégorie) est le MAX des signaux observés.
    """
    products    = data["products"]
    orders      = data["orders"]
    order_items = data["order_items"]
    data_dir    = Path(data.get("data_dir", "data/raw"))

    # Table de correspondance article → catégorie
    if "product_category_name" not in products.columns:
        log.error("products.csv manque la colonne 'product_category_name'")
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "order_id"])

    prod_cat = (
        products[["product_id", "product_category_name"]]
        .dropna()
        .drop_duplicates("product_id")
        .set_index("product_id")["product_category_name"]
        .to_dict()
    )

    rows = []

    # ── Signal 1 : vues (shetabit_visits) ────────────────────────────────────
    visits_path = data_dir / "visits.csv"
    if visits_path.exists():
        visits = pd.read_csv(visits_path)
        visits["customer_id"] = pd.to_numeric(visits["customer_id"], errors="coerce")
        visits["product_id"]  = pd.to_numeric(visits["product_id"],  errors="coerce")
        visits = visits.dropna(subset=["customer_id", "product_id"])
        visits["category"] = visits["product_id"].map(prod_cat)
        visits = visits.dropna(subset=["category"])
        v = (visits.groupby(["customer_id", "category"])
                   .size()
                   .reset_index(name="n_views"))
        v["rating"] = (v["n_views"].clip(upper=5) / 5).round(2)  # 0.2 à 1.0
        v = v[["customer_id", "category", "rating"]]
        v.columns = ["user_id", "item_id", "rating"]
        rows.append(v)
        log.info("Signal vues (catégories) : %d interactions", len(v))

    # ── Signal 2 : paniers ────────────────────────────────────────────────────
    panier_path = data_dir / "panier_interactions.csv"
    if panier_path.exists():
        paniers = pd.read_csv(panier_path)
        paniers["customer_id"] = pd.to_numeric(paniers["customer_id"], errors="coerce")
        paniers["product_id"]  = pd.to_numeric(paniers["product_id"],  errors="coerce")
        paniers = paniers.dropna(subset=["customer_id", "product_id"])
        paniers["category"] = paniers["product_id"].map(prod_cat)
        paniers = paniers.dropna(subset=["category"])
        p = paniers[["customer_id", "category"]].drop_duplicates()
        p.columns = ["user_id", "item_id"]
        p["rating"] = 3.0
        rows.append(p)
        log.info("Signal paniers (catégories) : %d interactions", len(p))

    # ── Signal 3 : achats (factures valides) ─────────────────────────────────
    valid_statuses = {"valide", "delivered", "completed"}
    orders_valid = orders[orders["order_status"].str.lower().isin(valid_statuses)]
    if len(orders_valid) == 0:
        orders_valid = orders  # fallback
    purchases = order_items.merge(
        orders_valid[["order_id", "customer_id"]], on="order_id"
    )
    if not purchases.empty:
        purchases["category"] = purchases["product_id"].map(prod_cat)
        purchases = purchases.dropna(subset=["category"])
        purchases = purchases[["customer_id", "category"]].drop_duplicates()
        purchases.columns = ["user_id", "item_id"]
        purchases["rating"] = 5.0
        rows.append(purchases)
        log.info("Signal achats (catégories) : %d interactions", len(purchases))

    if not rows:
        log.error("Aucun signal d'interaction trouvé !")
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "order_id"])

    # ── Fusion : garde le score max par (user, catégorie) ─────────────────────
    df = pd.concat(rows, ignore_index=True)
    df = df.groupby(["user_id", "item_id"])["rating"].max().reset_index()

    # order_id factice pour compatibilité avec make_splits
    df["order_id"] = range(len(df))

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


def encode_ids(df: pd.DataFrame):
    """Encode user_id et item_id en entiers contigus."""
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["item_id"])
    return df, user_enc, item_enc


def make_splits(df: pd.DataFrame, orders: pd.DataFrame, random_state: int = 42):
    """
    Retourne deux splits :
      - LOO (Leave-One-Out) chronologique → évaluation recommandation
      - Random 80/20 → évaluation RMSE/MAE SVD
    """
    from sklearn.model_selection import train_test_split

    # Supporte 'order_purchase_timestamp' (format e-commerce) et 'created_at' (MEL)
    date_col = None
    for col in ["order_purchase_timestamp", "created_at", "updated_at"]:
        if col in orders.columns:
            date_col = col
            break

    if date_col:
        orders_date = orders[["order_id", date_col]].copy()
        orders_date[date_col] = pd.to_datetime(orders_date[date_col], errors="coerce")
        df_dated = df.merge(orders_date, on="order_id", how="left")
        df_dated = df_dated.sort_values(["user_id", date_col])
    else:
        df_dated = df.copy()

    test_mask = df_dated.groupby("user_id").cumcount(ascending=False) == 0
    train_loo = df_dated[~test_mask].copy()
    test_loo  = df_dated[test_mask].copy()

    train_rand, test_rand = train_test_split(df, test_size=0.2, random_state=random_state)

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

def train_svd(train_random: pd.DataFrame, user_enc, item_enc, k_factors: int = 150):
    """SVD tronqué via scipy. Retourne la matrice R_pred et les données centrées."""
    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    t = train_random.copy()
    t["user_idx"] = user_enc.transform(t["user_id"])
    t["item_idx"] = item_enc.transform(t["item_id"])

    R = csr_matrix(
        (t["rating"].values, (t["user_idx"].values, t["item_idx"].values)),
        shape=(n_users, n_items),
    )
    R_dense     = R.toarray().astype(np.float32)
    user_mean   = R_dense.mean(axis=1)
    R_centered  = R_dense - user_mean[:, np.newaxis]
    R_centered[R_dense == 0] = 0

    log.info("Entraînement SVD avec k=%d facteurs...", k_factors)
    # k doit être < min(n_users, n_items) — on cap pour les petits datasets
    k_safe = min(k_factors, min(R_centered.shape) - 1)
    if k_safe != k_factors:
        log.warning("k_factors réduit de %d à %d (taille matrice %s)", k_factors, k_safe, R_centered.shape)
    U, sigma, Vt = svds(csr_matrix(R_centered), k=k_safe)
    R_pred = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_mean[:, np.newaxis]
    log.info("SVD terminé — matrice prédite : %s", R_pred.shape)
    return R_pred, user_mean, R_centered


class ALSRecommender:
    """ALS implicite from scratch — compatible Python 3.13."""

    def __init__(
        self,
        n_factors: int = 32,
        n_iterations: int = 15,
        alpha: int = 40,
        regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.n_factors    = n_factors
        self.n_iterations = n_iterations
        self.alpha        = alpha
        self.reg          = regularization
        self.random_state = random_state

    def fit(self, user_item_matrix):
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = user_item_matrix.shape

        C_ui = user_item_matrix.copy().tocsr().astype(np.float32)
        C_ui.data = 1.0 + self.alpha * C_ui.data

        self.user_factors = rng.standard_normal((n_users, self.n_factors)).astype(np.float32) * 0.01
        self.item_factors = rng.standard_normal((n_items, self.n_factors)).astype(np.float32) * 0.01

        reg_I = self.reg * np.eye(self.n_factors, dtype=np.float32)
        C_iu  = C_ui.T.tocsr()

        for _ in tqdm(range(self.n_iterations), desc="ALS"):
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                row = C_ui.getrow(u)
                if row.nnz == 0:
                    continue
                conf_u = row.data.astype(np.float32)
                Y_u    = self.item_factors[row.indices]
                A = YtY + Y_u.T @ (np.diag(conf_u - 1.0) @ Y_u) + reg_I
                b = Y_u.T @ conf_u
                self.user_factors[u] = np.linalg.solve(A, b)

            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                row = C_iu.getrow(i)
                if row.nnz == 0:
                    continue
                conf_i = row.data.astype(np.float32)
                X_i    = self.user_factors[row.indices]
                A = XtX + X_i.T @ (np.diag(conf_i - 1.0) @ X_i) + reg_I
                b = X_i.T @ conf_i
                self.item_factors[i] = np.linalg.solve(A, b)
        return self

    def recommend(self, user_idx: int, n: int = 10, filter_seen=None):
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if filter_seen is not None:
            scores[filter_seen] = -np.inf
        return np.argsort(scores)[::-1][:n]


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
        cat_article_names = (
            products_full[products_full["available"].fillna(1) == 1]  # articles dispo seulement
            .groupby("product_category_name")["product_name"]
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

def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate(reco_fn, test_df, train_df, k: int = 10, sample: int = 500, seed: int = 42) -> dict:
    """Évalue un modèle sur un échantillon du test set."""
    test_sample  = test_df.sample(min(sample, len(test_df)), random_state=seed)
    seen_by_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    precisions, recalls, ndcgs = [], [], []

    for user in test_sample["user_id"].unique():
        relevant     = set(test_sample[test_sample["user_id"] == user]["item_id"])
        already_seen = seen_by_user.get(user, set())
        try:
            reco = reco_fn(user, k + len(already_seen))
            reco = [i for i in reco if i not in already_seen][:k]
        except Exception:  # noqa: BLE001, S112
            continue
        precisions.append(precision_at_k(reco, relevant, k))
        recalls.append(recall_at_k(reco, relevant, k))
        ndcgs.append(ndcg_at_k(reco, relevant, k))

    return {
        f"Precision@{k}": round(float(np.mean(precisions)), 4),
        f"Recall@{k}":    round(float(np.mean(recalls)), 4),
        f"NDCG@{k}":      round(float(np.mean(ndcgs)), 4),
        "n_users_eval":   len(precisions),
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
    log.info("Sauvegardé : popular_items.pkl")

    # Table produits complète avec colonnes available et product_name
    # Utilisée par l'API pour retrouver les articles disponibles dans une catégorie
    if products_df is not None:
        products_df.to_csv(models_dir / "products_catalog.csv", index=False)
        log.info("Sauvegardé : products_catalog.csv (%d lignes)", len(products_df))

    with open(models_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Sauvegardé : results_summary.json")


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
    parser.add_argument("--alpha",         type=float,default=1.0,  help="Poids SVD dans le hybride")
    parser.add_argument("--eval-k",        type=int,  default=10,   help="K pour les métriques @K")
    parser.add_argument("--eval-sample",   type=int,  default=500,  help="Nb users pour l'évaluation")
    parser.add_argument("--random-state",  type=int,  default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    t0   = time.time()
    log.info("=== Pipeline MEL Recommandation ===")
    log.info("Config : %s", vars(args))

    # ── MLflow setup ─────────────────────────────────────────────────────────
    # L'URI est lue depuis l'environnement (docker-compose, CI/CD, ou local)
    # Valeur par défaut : tracking local dans ./mlruns
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        log.info("MLflow tracking URI : %s", tracking_uri)
    else:
        log.info("MLflow tracking URI : local (./mlruns)")

    mlflow.set_experiment("mel-recommandation")

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
        df         = build_interactions(data)
        df         = filter_interactions(df)
        df, u_enc, i_enc = encode_ids(df)

        # Logger les stats du dataset
        n_users = int(df["user_idx"].max()) + 1
        n_items = int(df["item_idx"].max()) + 1
        mlflow.log_params({
            "n_users":        n_users,
            "n_items":        n_items,
            "n_interactions": len(df),
        })

        args.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.processed_dir / "interactions_filtered.csv", index=False)

        train_loo, test_loo, train_rand, test_rand = make_splits(
            df, data["orders"], random_state=args.random_state
        )
        train_loo.to_csv(args.processed_dir / "train.csv",        index=False)
        test_loo.to_csv(args.processed_dir  / "test.csv",         index=False)
        train_rand.to_csv(args.processed_dir / "train_random.csv", index=False)
        test_rand.to_csv(args.processed_dir  / "test_random.csv",  index=False)

        # ── Popularité (baseline + cold-start) ───────────────────────────────
        pop_counts   = train_loo.groupby("item_id")["user_id"].count().sort_values(ascending=False)
        popular_items = pop_counts.index.tolist()

        # ── SVD ──────────────────────────────────────────────────────────────
        R_pred, _user_mean, _R_centered = train_svd(
            train_rand, u_enc, i_enc, k_factors=args.k_factors
        )

        # ── ALS ──────────────────────────────────────────────────────────────
        als_model, _, _seen_items = train_als(
            train_loo, u_enc, i_enc,
            n_factors=args.als_factors,
            regularization=args.als_reg,
            n_iterations=args.als_iter,
            random_state=args.random_state,
        )

        # ── Content-Based ─────────────────────────────────────────────────────
        valid_items = df["item_id"].unique()
        cosine_sim, item_to_idx, idx_to_item, _products_full = build_content_features(
            data, valid_items
        )

        # ── Modèle Hybride ────────────────────────────────────────────────────
        hybrid_fn = build_hybrid_recommender(
            R_pred, cosine_sim, item_to_idx, idx_to_item,
            u_enc, i_enc, train_loo, popular_items,
            alpha=args.alpha,
        )

        # ── Évaluation ───────────────────────────────────────────────────────
        log.info("Évaluation du modèle hybride (alpha=%.1f)...", args.alpha)
        metrics = evaluate(
            hybrid_fn, test_loo, train_loo,
            k=args.eval_k, sample=args.eval_sample, seed=args.random_state,
        )
        log.info("Résultats : %s", metrics)

        # Logger les métriques dans MLflow
        mlflow.log_metric(f"precision_at_{args.eval_k}", metrics[f"Precision@{args.eval_k}"])
        mlflow.log_metric(f"recall_at_{args.eval_k}",    metrics[f"Recall@{args.eval_k}"])
        mlflow.log_metric(f"ndcg_at_{args.eval_k}",      metrics[f"NDCG@{args.eval_k}"])
        mlflow.log_metric("n_users_eval",                 metrics["n_users_eval"])

        training_time = round(time.time() - t0, 1)
        mlflow.log_metric("training_time_sec", training_time)

        # ── Résumé ────────────────────────────────────────────────────────────
        summary = {
            "dataset":         "MEL Cameroun",
            "python_version":  "3.13+",
            "n_users":         n_users,
            "n_items":         n_items,
            "n_interactions":  len(df),
            "model":           "Hybrid (SVD + Content-Based)",
            "hyperparameters": {
                "svd_k_factors": args.k_factors,
                "als_n_factors": args.als_factors,
                "als_reg":       args.als_reg,
                "als_iter":      args.als_iter,
                "hybrid_alpha":  args.alpha,
            },
            "metrics": metrics,
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
        )

        # Logger les artefacts dans MLflow
        mlflow.log_artifact(str(args.models_dir / "results_summary.json"))
        mlflow.log_artifact(str(args.models_dir / "svd_R_pred.npy"))
        mlflow.log_artifact(str(args.models_dir / "als_model.pkl"))
        mlflow.sklearn.log_model(als_model, artifact_path="als_model")

        log.info("=== Pipeline terminée en %.1fs ===", training_time)
        log.info("NDCG@%d = %.4f", args.eval_k, metrics[f"NDCG@{args.eval_k}"])
        log.info("MLflow run ID : %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
