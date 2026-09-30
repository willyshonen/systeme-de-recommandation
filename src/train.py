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
import pickle
import time
from pathlib import Path

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
    """Charge les 5 fichiers CSV Olist."""
    log.info("Chargement des données depuis %s", data_dir)
    return {
        "orders":      pd.read_csv(data_dir / "olist_orders_dataset.csv"),
        "order_items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "products":    pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "reviews":     pd.read_csv(data_dir / "olist_order_reviews_dataset.csv"),
        "cat_names":   pd.read_csv(data_dir / "product_category_name_translation.csv"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_interactions(data: dict) -> pd.DataFrame:
    """Construit le dataframe d'interactions user-item avec ratings."""
    orders      = data["orders"]
    order_items = data["order_items"]
    reviews     = data["reviews"]

    interactions = (
        order_items
        .merge(orders[["order_id", "customer_id", "order_status"]], on="order_id")
        .query("order_status == 'delivered'")
    )
    df = interactions.merge(
        reviews[["order_id", "review_score"]].drop_duplicates("order_id"),
        on="order_id",
        how="left",
    )
    df["review_score"] = df["review_score"].fillna(3.0)
    df = df[["customer_id", "product_id", "review_score", "order_id"]].copy()
    df.columns = ["user_id", "item_id", "rating", "order_id"]
    log.info("Interactions brutes : %d", len(df))
    return df


def filter_interactions(
    df: pd.DataFrame,
    min_user: int = 2,
    min_item: int = 5,
    n_iter: int = 3,
) -> pd.DataFrame:
    """Filtrage itératif cold-start."""
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

    orders_date = orders[["order_id", "order_purchase_timestamp"]].copy()
    orders_date["order_purchase_timestamp"] = pd.to_datetime(
        orders_date["order_purchase_timestamp"]
    )
    df_dated = df.merge(orders_date, on="order_id", how="left")
    df_dated  = df_dated.sort_values(["user_id", "order_purchase_timestamp"])

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
    U, sigma, Vt = svds(csr_matrix(R_centered), k=k_factors)
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
    """Construit la matrice de similarité cosine TF-IDF."""
    products  = data["products"]
    cat_names = data["cat_names"]

    products_full = products.merge(cat_names, on="product_category_name", how="left")
    products_full["description"] = (
        products_full["product_category_name_english"].fillna("unknown") + " "
        + products_full["product_weight_g"].fillna(0).astype(str) + "g "
        + products_full["product_photos_qty"].fillna(0).astype(str) + "photos"
    )
    products_cb = (
        products_full[products_full["product_id"].isin(valid_items)]
        .drop_duplicates("product_id")
        .reset_index(drop=True)
    )
    tfidf       = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_mat   = tfidf.fit_transform(products_cb["description"])
    cosine_sim  = cosine_similarity(tfidf_mat, tfidf_mat)
    item_to_idx = {pid: idx for idx, pid in enumerate(products_cb["product_id"])}
    idx_to_item = {idx: pid for pid, idx in item_to_idx.items()}

    log.info("Content-Based — %d produits, matrice %s", len(products_cb), cosine_sim.shape)
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
        except Exception:
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

    # ── Chargement ───────────────────────────────────────────────────────────
    data = load_data(args.data_dir)

    # ── Preprocessing ────────────────────────────────────────────────────────
    df         = build_interactions(data)
    df         = filter_interactions(df)
    df, u_enc, i_enc = encode_ids(df)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.processed_dir / "interactions_filtered.csv", index=False)

    train_loo, test_loo, train_rand, test_rand = make_splits(
        df, data["orders"], random_state=args.random_state
    )
    train_loo.to_csv(args.processed_dir / "train.csv",        index=False)
    test_loo.to_csv(args.processed_dir  / "test.csv",         index=False)
    train_rand.to_csv(args.processed_dir / "train_random.csv", index=False)
    test_rand.to_csv(args.processed_dir  / "test_random.csv",  index=False)

    # ── Popularité (baseline + cold-start) ───────────────────────────────────
    pop_counts   = train_loo.groupby("item_id")["user_id"].count().sort_values(ascending=False)
    popular_items = pop_counts.index.tolist()

    # ── SVD ──────────────────────────────────────────────────────────────────
    R_pred, user_mean, R_centered = train_svd(
        train_rand, u_enc, i_enc, k_factors=args.k_factors
    )

    # ── ALS ──────────────────────────────────────────────────────────────────
    als_model, _, seen_items = train_als(
        train_loo, u_enc, i_enc,
        n_factors=args.als_factors,
        regularization=args.als_reg,
        n_iterations=args.als_iter,
        random_state=args.random_state,
    )

    # ── Content-Based ─────────────────────────────────────────────────────────
    valid_items = df["item_id"].unique()
    cosine_sim, item_to_idx, idx_to_item, products_full = build_content_features(
        data, valid_items
    )

    # ── Modèle Hybride ────────────────────────────────────────────────────────
    hybrid_fn = build_hybrid_recommender(
        R_pred, cosine_sim, item_to_idx, idx_to_item,
        u_enc, i_enc, train_loo, popular_items,
        alpha=args.alpha,
    )

    # ── Évaluation ───────────────────────────────────────────────────────────
    log.info("Évaluation du modèle hybride (alpha=%.1f)...", args.alpha)
    metrics = evaluate(
        hybrid_fn, test_loo, train_loo,
        k=args.eval_k, sample=args.eval_sample, seed=args.random_state,
    )
    log.info("Résultats : %s", metrics)

    # ── Résumé ────────────────────────────────────────────────────────────────
    summary = {
        "dataset":         "Olist Brazilian E-Commerce",
        "python_version":  "3.13+",
        "n_users":         int(df["user_idx"].max()) + 1,
        "n_items":         int(df["item_idx"].max()) + 1,
        "n_interactions":  int(len(df)),
        "model":           "Hybrid (SVD + Content-Based)",
        "hyperparameters": {
            "svd_k_factors": args.k_factors,
            "als_n_factors": args.als_factors,
            "als_reg":       args.als_reg,
            "als_iter":      args.als_iter,
            "hybrid_alpha":  args.alpha,
        },
        "metrics": metrics,
        "training_time_sec": round(time.time() - t0, 1),
    }

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    save_artifacts(
        args.models_dir,
        R_pred, als_model, cosine_sim,
        item_to_idx, idx_to_item,
        u_enc, i_enc, popular_items,
        summary,
    )

    log.info("=== Pipeline terminée en %.1fs ===", time.time() - t0)
    log.info("NDCG@%d = %.4f", args.eval_k, metrics[f"NDCG@{args.eval_k}"])


if __name__ == "__main__":
    main()
