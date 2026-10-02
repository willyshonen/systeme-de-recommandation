"""
tests/test_api.py
Tests unitaires de l'API FastAPI — MEL Recommender (architecture catégories).

Architecture testée :
  - Le modèle travaille sur user × catégorie (marketplace d'occasion)
  - /recommend retourne des articles disponibles dans les catégories recommandées
  - /similar retourne des articles dans des catégories similaires
  - /popular retourne uniquement les articles disponibles (available=1)
  - /reload recharge les modèles sans redémarrer l'API
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Constantes du jeu de test
# ─────────────────────────────────────────────────────────────────────────────

N_USERS = 10
CATEGORIES = [
    "Vêtements femmes",
    "Vêtements hommes",
    "Chaussures femmes",
    "Chaussures hommes",
    "Accessoires",
]
N_CATS = len(CATEGORIES)

# Articles fictifs par catégorie (certains vendus, d'autres disponibles)
PRODUCTS = [
    # product_id, category,              available
    ("1",  "Vêtements femmes",  0),   # vendu
    ("2",  "Vêtements femmes",  1),   # dispo
    ("3",  "Vêtements femmes",  1),   # dispo
    ("4",  "Vêtements hommes",  1),   # dispo
    ("5",  "Vêtements hommes",  0),   # vendu
    ("6",  "Chaussures femmes", 1),   # dispo
    ("7",  "Chaussures femmes", 1),   # dispo
    ("8",  "Chaussures hommes", 1),   # dispo
    ("9",  "Accessoires",       1),   # dispo
    ("10", "Accessoires",       0),   # vendu
]
AVAILABLE_IDS = {pid for pid, _, av in PRODUCTS if av == 1}
SOLD_IDS      = {pid for pid, _, av in PRODUCTS if av == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Stub ALS (picklable au niveau module)
# ─────────────────────────────────────────────────────────────────────────────

class FakeALS:
    """Stub picklable du modèle ALS travaillant sur les catégories."""
    def __init__(self, n_users=N_USERS, n_cats=N_CATS, seed=42):
        rng = np.random.default_rng(seed)
        self.user_factors = rng.random((n_users, 8)).astype(np.float32)
        self.item_factors = rng.random((n_cats,  8)).astype(np.float32)

    def recommend(self, user_idx, n=10, filter_seen=None):
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if filter_seen is not None:
            scores[list(filter_seen)] = -np.inf
        return np.argsort(scores)[::-1][:n]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture : artefacts factices cohérents avec la nouvelle architecture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fake_artifacts(tmp_path_factory):
    """
    Crée des artefacts factices cohérents avec l'architecture catégories :
      - user_enc encode des user_id entiers
      - item_enc encode des NOMS DE CATÉGORIES (pas des item_id)
      - products_catalog.csv avec colonnes available et product_category_name
    """
    models_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(42)

    # ── Encodeurs ─────────────────────────────────────────────────────────────
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    user_enc.fit([f"user_{i}" for i in range(N_USERS)])
    item_enc.fit(CATEGORIES)   # encode les catégories, pas des item_id

    with open(models_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(models_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)

    # ── SVD R_pred (user × catégorie) ─────────────────────────────────────────
    R_pred = rng.random((N_USERS, N_CATS)).astype(np.float32)
    np.save(models_dir / "svd_R_pred.npy", R_pred)

    # ── ALS ───────────────────────────────────────────────────────────────────
    als_stub = FakeALS(n_users=N_USERS, n_cats=N_CATS)
    with open(models_dir / "als_model.pkl", "wb") as f:
        pickle.dump(als_stub, f)

    # ── Content-Based (similarité entre catégories) ───────────────────────────
    cosine_sim  = rng.random((N_CATS, N_CATS)).astype(np.float32)
    np.fill_diagonal(cosine_sim, 1.0)
    item_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    idx_to_item = {i: cat for cat, i in item_to_idx.items()}
    with open(models_dir / "cosine_sim.pkl", "wb") as f:
        pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)

    # ── Popularité (catégories) ────────────────────────────────────────────────
    popular_items = CATEGORIES[:]   # toutes les catégories dans l'ordre
    with open(models_dir / "popular_items.pkl", "wb") as f:
        pickle.dump(popular_items, f)

    # ── Catalogue produits (avec disponibilité) ────────────────────────────────
    catalog = pd.DataFrame(
        PRODUCTS,
        columns=["product_id", "product_category_name", "available"]
    )
    catalog["product_name"]      = catalog["product_id"].apply(lambda i: f"Article {i}")
    catalog["product_weight_g"]  = 500
    catalog["product_photos_qty"] = 1
    catalog.to_csv(models_dir / "products_catalog.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "model":           "Hybrid (SVD + Content-Based) — catégories",
        "n_users":         N_USERS,
        "n_items":         N_CATS,
        "n_interactions":  100,
        "hyperparameters": {"svd_k_factors": 8, "hybrid_alpha": 0.6},
        "metrics": {
            "Precision@10": 0.004,
            "Recall@10":    0.04,
            "NDCG@10":      0.032,
        },
    }
    with open(models_dir / "results_summary.json", "w") as f:
        json.dump(summary, f)

    return models_dir


@pytest.fixture(scope="session")
def client(fake_artifacts):
    """Client de test FastAPI avec les artefacts factices."""
    from api.main import app, store
    store.load(models_dir=fake_artifacts)
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Tests /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_health_fields(self, client):
        """La nouvelle API expose n_categories et n_available_products."""
        data = client.get("/health").json()
        assert "n_users"             in data
        assert "n_categories"        in data
        assert "n_available_products" in data
        assert data["n_users"]      == N_USERS
        assert data["n_categories"] == N_CATS

    def test_health_available_products_excludes_sold(self, client):
        """Les articles vendus ne sont pas comptés dans n_available_products."""
        data = client.get("/health").json()
        assert data["n_available_products"] == len(AVAILABLE_IDS)

    def test_health_has_model_name(self, client):
        assert "model" in client.get("/health").json()


# ─────────────────────────────────────────────────────────────────────────────
# Tests /metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_ok(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_fields(self, client):
        data = client.get("/metrics").json()
        assert "metrics"          in data
        assert "hyperparameters"  in data
        assert "n_users"          in data
        assert data["n_users"]   == N_USERS


# ─────────────────────────────────────────────────────────────────────────────
# Tests /popular
# ─────────────────────────────────────────────────────────────────────────────

class TestPopular:
    def test_popular_default(self, client):
        r = client.get("/popular")
        assert r.status_code == 200
        data = r.json()
        assert "popular_products" in data
        assert data["n"] == 10

    def test_popular_custom_n(self, client):
        r = client.get("/popular?n=3")
        assert r.status_code == 200
        assert len(r.json()["popular_products"]) == 3

    def test_popular_excludes_sold_articles(self, client):
        """Aucun article vendu ne doit apparaître dans /popular."""
        data = client.get("/popular?n=50").json()
        returned = set(data["popular_products"])
        assert returned.isdisjoint(SOLD_IDS), (
            f"Articles vendus trouvés dans /popular : {returned & SOLD_IDS}"
        )

    def test_popular_only_available_articles(self, client):
        """Tous les articles retournés doivent être disponibles."""
        data = client.get("/popular?n=50").json()
        for pid in data["popular_products"]:
            assert pid in AVAILABLE_IDS, f"Article non disponible dans /popular : {pid}"

    def test_popular_n_too_large(self, client):
        assert client.get("/popular?n=200").status_code == 422

    def test_popular_n_zero(self, client):
        assert client.get("/popular?n=0").status_code == 422

    def test_popular_filter_by_category(self, client):
        """Filtrage par catégorie : tous les articles retournés sont dans cette catégorie."""
        cat = "Vêtements femmes"
        r = client.get(f"/popular?category={cat}&n=10")
        assert r.status_code == 200
        data = r.json()
        # Les articles retournés doivent appartenir à la catégorie ET être disponibles
        cat_available = {pid for pid, c, av in PRODUCTS if c == cat and av == 1}
        for pid in data["popular_products"]:
            assert pid in cat_available, f"Article hors catégorie dans /popular?category={cat}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests /recommend
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommend:
    def test_known_user_svd(self, client):
        r = client.get("/recommend/user_0?model=svd&n=5")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == "user_0"
        assert data["model"]   == "svd"
        assert len(data["recommendations"]) <= 5

    def test_known_user_als(self, client):
        r = client.get("/recommend/user_0?model=als&n=5")
        assert r.status_code == 200
        assert r.json()["model"] == "als"

    def test_known_user_hybrid(self, client):
        r = client.get("/recommend/user_0?model=hybrid&n=5")
        assert r.status_code == 200
        assert r.json()["model"] == "hybrid"

    def test_recommend_returns_categories(self, client):
        """La réponse inclut les catégories recommandées."""
        data = client.get("/recommend/user_0?n=5").json()
        assert "recommended_categories" in data
        assert isinstance(data["recommended_categories"], list)

    def test_recommend_no_sold_articles(self, client):
        """Les articles retournés doivent tous être disponibles."""
        data = client.get("/recommend/user_0?n=50").json()
        for pid in data["recommendations"]:
            assert pid not in SOLD_IDS, (
                f"Article vendu {pid} trouvé dans les recommandations"
            )

    def test_unknown_user_cold_start(self, client):
        """Un utilisateur inconnu reçoit les articles populaires disponibles."""
        r = client.get("/recommend/utilisateur_inconnu_xyz")
        assert r.status_code == 200
        data = r.json()
        assert "cold-start" in data["model"]
        assert len(data["recommendations"]) > 0

    def test_cold_start_no_sold_articles(self, client):
        """Le cold-start ne retourne pas d'articles vendus."""
        data = client.get("/recommend/inconnu?n=50").json()
        for pid in data["recommendations"]:
            assert pid not in SOLD_IDS

    def test_recommend_returns_strings(self, client):
        """Les IDs de produits retournés sont des chaînes."""
        data = client.get("/recommend/user_1?n=3").json()
        for pid in data["recommendations"]:
            assert isinstance(pid, str)

    def test_recommend_n_param(self, client):
        r = client.get("/recommend/user_2?n=3")
        assert r.status_code == 200
        assert r.json()["n"] == 3

    def test_recommend_n_too_large(self, client):
        assert client.get("/recommend/user_0?n=200").status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Tests /similar
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilar:
    def test_similar_known_product(self, client):
        """Un produit connu retourne des produits similaires disponibles."""
        r = client.get("/similar/2?n=5")   # article 2 = Vêtements femmes, disponible
        assert r.status_code == 200
        data = r.json()
        assert data["product_id"] == "2"
        assert "product_category"   in data
        assert "similar_categories" in data
        assert "similar_products"   in data

    def test_similar_excludes_source_product(self, client):
        """L'article source ne doit pas être dans les similarités."""
        data = client.get("/similar/2?n=10").json()
        assert "2" not in data["similar_products"]

    def test_similar_no_sold_articles(self, client):
        """Les articles vendus ne doivent pas apparaître dans /similar."""
        data = client.get("/similar/2?n=20").json()
        for pid in data["similar_products"]:
            assert pid not in SOLD_IDS, (
                f"Article vendu {pid} trouvé dans /similar"
            )

    def test_similar_unknown_product(self, client):
        r = client.get("/similar/produit_inexistant_xyz")
        assert r.status_code == 404

    def test_similar_returns_n_items(self, client):
        r = client.get("/similar/4?n=3")
        assert r.status_code == 200
        assert len(r.json()["similar_products"]) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests /reload
# ─────────────────────────────────────────────────────────────────────────────

class TestReload:
    def test_reload_without_secret(self, client):
        """Sans RELOAD_SECRET configuré, /reload est accessible librement."""
        r = client.post("/reload")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "reloaded"
        assert "elapsed_sec" in data
        assert "before"      in data
        assert "after"       in data

    def test_reload_response_structure(self, client):
        """La réponse contient les stats avant/après rechargement."""
        data = client.post("/reload").json()
        for key in ("n_users", "n_categories", "n_available_products"):
            assert key in data["before"], f"Clé '{key}' manquante dans before"
            assert key in data["after"],  f"Clé '{key}' manquante dans after"

    def test_reload_wrong_secret(self, client, monkeypatch):
        """Avec un RELOAD_SECRET défini, une mauvaise clé retourne 401."""
        monkeypatch.setenv("RELOAD_SECRET", "super-secret")
        r = client.post("/reload?secret=mauvaise-cle")
        assert r.status_code == 401

    def test_reload_correct_secret(self, client, monkeypatch):
        """Avec un RELOAD_SECRET défini, la bonne clé retourne 200."""
        monkeypatch.setenv("RELOAD_SECRET", "super-secret")
        r = client.post("/reload?secret=super-secret")
        assert r.status_code == 200
