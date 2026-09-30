"""
tests/test_api.py
Tests unitaires de l'API FastAPI avec des modèles factices.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder


# ── Stub ALS au niveau module (picklable) ─────────────────────────────────────

N_USERS = 10
N_ITEMS = 20


class FakeALS:
    """Stub picklable du modèle ALS."""
    def __init__(self, n_users=N_USERS, n_items=N_ITEMS, seed=42):
        rng = np.random.default_rng(seed)
        self.user_factors = rng.random((n_users, 8)).astype(np.float32)
        self.item_factors = rng.random((n_items, 8)).astype(np.float32)

    def recommend(self, user_idx, n=10, filter_seen=None):
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if filter_seen is not None:
            scores[list(filter_seen)] = -np.inf
        return np.argsort(scores)[::-1][:n]

@pytest.fixture(scope="session")
def fake_artifacts(tmp_path_factory):
    """Crée de faux artefacts dans un répertoire temporaire."""
    models_dir = tmp_path_factory.mktemp("models")

    rng = np.random.default_rng(42)

    # Encodeurs
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    user_enc.fit([f"user_{i}" for i in range(N_USERS)])
    item_enc.fit([f"item_{i}" for i in range(N_ITEMS)])

    with open(models_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(models_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)

    # SVD R_pred
    R_pred = rng.random((N_USERS, N_ITEMS)).astype(np.float32)
    np.save(models_dir / "svd_R_pred.npy", R_pred)

    # ALS model stub (classe au niveau module → picklable)
    als_stub = FakeALS(n_users=N_USERS, n_items=N_ITEMS)
    with open(models_dir / "als_model.pkl", "wb") as f:
        pickle.dump(als_stub, f)

    # Content-based
    cosine_sim  = rng.random((N_ITEMS, N_ITEMS)).astype(np.float32)
    np.fill_diagonal(cosine_sim, 1.0)
    item_to_idx = {f"item_{i}": i for i in range(N_ITEMS)}
    idx_to_item = {i: f"item_{i}" for i in range(N_ITEMS)}
    with open(models_dir / "cosine_sim.pkl", "wb") as f:
        pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)

    # Popularité
    popular_items = [f"item_{i}" for i in range(N_ITEMS)]
    with open(models_dir / "popular_items.pkl", "wb") as f:
        pickle.dump(popular_items, f)

    # Summary
    summary = {
        "model": "Hybrid (SVD + Content-Based)",
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "n_interactions": 100,
        "hyperparameters": {"svd_k_factors": 50, "hybrid_alpha": 1.0},
        "metrics": {"Precision@10": 0.0038, "Recall@10": 0.038, "NDCG@10": 0.0325},
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


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["n_users"] == N_USERS
        assert data["n_items"] == N_ITEMS

    def test_health_has_model_name(self, client):
        r = client.get("/health")
        assert "model" in r.json()


class TestMetrics:
    def test_metrics_ok(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data
        assert "hyperparameters" in data
        assert data["n_users"] == N_USERS


class TestPopular:
    def test_popular_default(self, client):
        r = client.get("/popular")
        assert r.status_code == 200
        data = r.json()
        assert len(data["popular_products"]) == 10
        assert data["n"] == 10

    def test_popular_custom_n(self, client):
        r = client.get("/popular?n=5")
        assert r.status_code == 200
        assert len(r.json()["popular_products"]) == 5

    def test_popular_n_too_large(self, client):
        r = client.get("/popular?n=200")
        assert r.status_code == 422   # validation error

    def test_popular_n_zero(self, client):
        r = client.get("/popular?n=0")
        assert r.status_code == 422


class TestRecommend:
    def test_known_user_svd(self, client):
        r = client.get("/recommend/user_0?model=svd&n=5")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == "user_0"
        assert data["model"] == "svd"
        assert len(data["recommendations"]) == 5

    def test_known_user_als(self, client):
        r = client.get("/recommend/user_0?model=als&n=5")
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "als"

    def test_known_user_hybrid(self, client):
        r = client.get("/recommend/user_0?model=hybrid&n=10")
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "hybrid"
        assert len(data["recommendations"]) == 10

    def test_unknown_user_cold_start(self, client):
        r = client.get("/recommend/unknown_user_xyz")
        assert r.status_code == 200
        data = r.json()
        assert "cold-start" in data["model"]
        assert len(data["recommendations"]) > 0

    def test_recommend_returns_item_ids(self, client):
        r = client.get("/recommend/user_1?n=3")
        data = r.json()
        for item_id in data["recommendations"]:
            assert isinstance(item_id, str)


class TestSimilar:
    def test_similar_known_product(self, client):
        r = client.get("/similar/item_0?n=5")
        assert r.status_code == 200
        data = r.json()
        assert data["product_id"] == "item_0"
        assert "item_0" not in data["similar_products"]   # ne se recommande pas lui-même
        assert len(data["similar_products"]) == 5

    def test_similar_unknown_product(self, client):
        r = client.get("/similar/product_does_not_exist")
        assert r.status_code == 404

    def test_similar_returns_n_items(self, client):
        r = client.get("/similar/item_5?n=8")
        assert len(r.json()["similar_products"]) == 8
