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
from typing import ClassVar

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
    "Ordinateurs",     # jamais dans l'historique, mais avec du stock
    "Télévisions",     # dans l'historique, mais plus aucun article disponible
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
    # Catégorie SANS interaction dans l'historique : c'est le cas que l'ancien
    # encodeur rendait invisible (item_enc fit sur les seules catégories vues
    # dans les interactions).
    ("11", "Ordinateurs",       1),   # dispo, mais absent des interactions
    ("12", "Ordinateurs",       0),   # vendu
    # Catégorie avec interactions mais SANS le moindre article disponible :
    # la recommander brûlerait un rang pour zéro produit (cas réel : "Télévisions").
    ("13", "Télévisions",       0),   # vendu
]
AVAILABLE_IDS = {pid for pid, _, av in PRODUCTS if av == 1}
SOLD_IDS      = {pid for pid, _, av in PRODUCTS if av == 0}
# Catégories ayant au moins un article disponible (cf. build_item_space côté
# entraînement : une catégorie sans stock ne doit jamais être recommandée).
CATEGORIES_WITH_STOCK = {
    cat for _, cat, av in PRODUCTS if av == 1
}


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


@pytest.fixture(scope="session")
def int_user_artifacts(tmp_path_factory):
    """
    Artefacts reproduisant le cas de production cassé : user_encoder contient des
    entiers (les customer_id de la base) et non des chaînes, et un
    user_history.pkl est présent.

    C'est exactement la configuration qui faisait tomber tous les utilisateurs
    connus en cold-start : l'API recevait "42" (str) et testait
    `"42" in {6, 12, 17, 19}` → faux.
    """
    models_dir = tmp_path_factory.mktemp("models_int_users")
    rng = np.random.default_rng(7)

    # Encodeurs : user_id ENTIERS, comme les anciens artefacts
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    user_enc.fit([6, 12, 17, 19, 20])
    item_enc.fit(CATEGORIES)

    with open(models_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(models_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)

    np.save(models_dir / "svd_R_pred.npy", rng.random((5, N_CATS)).astype(np.float32))

    with open(models_dir / "als_model.pkl", "wb") as f:
        pickle.dump(FakeALS(n_users=5, n_cats=N_CATS, seed=7), f)

    cosine_sim = rng.random((N_CATS, N_CATS)).astype(np.float32)
    np.fill_diagonal(cosine_sim, 1.0)
    item_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    idx_to_item = {i: cat for cat, i in item_to_idx.items()}
    with open(models_dir / "cosine_sim.pkl", "wb") as f:
        pickle.dump((cosine_sim, item_to_idx, idx_to_item), f)

    with open(models_dir / "popular_items.pkl", "wb") as f:
        pickle.dump(CATEGORIES[:], f)

    # Historique persistant : chaque user a déjà vu une catégorie
    history = {
        "6":  ["Vêtements femmes"],
        "12": ["Accessoires"],
        "17": ["Chaussures hommes"],
        "19": ["Chaussures femmes"],
        "20": ["Vêtements hommes"],
    }
    with open(models_dir / "user_history.pkl", "wb") as f:
        pickle.dump(history, f)

    catalog = pd.DataFrame(
        PRODUCTS, columns=["product_id", "product_category_name", "available"]
    )
    catalog["product_name"] = catalog["product_id"].apply(lambda i: f"Article {i}")
    catalog["product_weight_g"] = 500
    catalog["product_photos_qty"] = 1
    catalog.to_csv(models_dir / "products_catalog.csv", index=False)

    summary = {
        "model": "Hybrid (SVD + Content-Based) — catégories",
        "n_users": 5,
        "n_items": N_CATS,
        "hyperparameters": {"svd_k_factors": 8, "hybrid_alpha": 0.6},
        "metrics": {},
    }
    with open(models_dir / "results_summary.json", "w") as f:
        json.dump(summary, f)

    return models_dir


@pytest.fixture()
def int_user_client(int_user_artifacts, fake_artifacts):
    """
    Charge les artefacts à user_id entiers et restaure les artefacts par défaut
    en teardown : le store est un singleton global, sans cette restauration les
    tests suivants tourneraient sur le mauvais jeu d'artefacts.
    """
    from api.main import app, store
    store.load(models_dir=int_user_artifacts)
    try:
        yield TestClient(app)
    finally:
        store.load(models_dir=fake_artifacts)


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

    def test_health_annonce_le_verdict_de_la_cv(self, client):
        """
        /health doit dire ce qui classe RÉELLEMENT, pas seulement l'artefact.

        La fixture n'a pas de lift CV, donc la porte de suffisance est fermée :
        annoncer « Hybrid » sans le lift donnerait un état de santé rassurant
        sur un service qui sert en réalité de la popularité.
        """
        from api.main import store

        data = client.get("/health").json()
        assert "serving_mode" in data
        assert "popularité" in data["serving_mode"]
        assert "non validée" in data["serving_mode"]
        assert store.personalization_warranted is False

    def test_health_serving_mode_suit_la_porte(self, client):
        """Porte ouverte → /health annonce l'hybride, avec le lift."""
        from api.main import store

        avant = store.personalization_warranted
        try:
            store.personalization_warranted = True
            data = client.get("/health").json()
            assert "hybride" in data["serving_mode"].lower()
        finally:
            store.personalization_warranted = avant


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

    def test_metrics_expose_le_lift_cv_et_le_mode(self, client):
        """
        Sans `cv_lift_at_k` dans /metrics, on ne peut pas comprendre pourquoi le
        service classe par popularité — c'est la métrique qui décide de tout.
        """
        hp = client.get("/metrics").json()["hyperparameters"]
        assert "serving_mode" in hp
        assert "cv_lift_at_k" in hp


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
        # `model=svd` est une demande explicite de debug : elle court-circuite
        # volontairement la porte de suffisance (sinon on ne pourrait plus
        # comparer une régression à la sortie brute du SVD). Le libellé doit
        # donc dire « svd » ET rappeler que la CV ne l'a pas validé — surtout
        # pas annoncer « popularity », qui ne serait pas la vérité ici.
        assert data["model"].startswith("svd")
        assert "non validé" in data["model"]
        assert len(data["recommendations"]) <= 5

    def test_known_user_als(self, client):
        r = client.get("/recommend/user_0?model=als&n=5")
        assert r.status_code == 200
        label = r.json()["model"]
        assert label.startswith("als")
        assert "non validé" in label

    def test_known_user_hybrid(self, client):
        r = client.get("/recommend/user_0?model=hybrid&n=5")
        assert r.status_code == 200
        assert r.json()["model"] in (
            "hybrid", "popularity (personnalisation non validée)",
        )

def test_hybrid_porte_fermee_sert_de_la_popularite_pas_du_svd(client):
    """
    Le chemin par défaut doit vraiment basculer sur la popularité.

    Une porte qui ne changerait que le libellé laisserait servir du SVD non
    validé en croyant avoir écarté le modèle — le pire des deux mondes, car le
    front afficherait « popularity » pour un classement collaboratif.
    """
    import numpy as np

    from api.main import _top_categories, _user_categories, store

    avant = store.personalization_warranted
    try:
        store.personalization_warranted = False
        servi = _top_categories(0, model="hybrid", n_cats=3)

        # Référence reconstruite : le même ordre que la branche popularity.
        no_stock = [
            i for i, cat in enumerate(store.item_enc.classes_)
            if not store.cat_to_products.get(cat)
        ]
        ex = sorted(set(_user_categories(0)) | set(no_stock))
        pop_scores = np.full(len(store.item_enc.classes_), -np.inf)
        for rang, cat in enumerate(store.popular_items):
            try:
                pop_scores[int(store.item_enc.transform([cat])[0])] = -rang
            except ValueError:
                continue
        pop_scores[ex] = -np.inf
        attendu = store.item_enc.inverse_transform(
            np.argsort(pop_scores)[::-1][:3]
        ).tolist()

        assert servi == attendu, (
            f"porte fermée : le classement servi {servi} n'est pas celui de la "
            f"popularité {attendu}"
        )
        assert "non validée" in client.get(
            "/recommend/user_0?model=hybrid&n=5"
        ).json()["model"]
    finally:
        store.personalization_warranted = avant


def test_porte_de_sufficance_ferme_sans_lift(client):
    """
    Sans lift positif en validation croisée, la personnalisation est coupée.

    C'est le comportement observé sur les données réelles : à 51 interactions,
    le SVD descend à 0,59 de NDCG@3 contre 1,00 pour la popularité. Servir le
    modèle quand même reviendrait à ajouter du bruit.
    """
    from api.main import store

    assert store.personalization_warranted is False
    data = client.get("/recommend/user_0?model=hybrid&n=5").json()
    assert "non validée" in data["model"]


def test_porte_de_sufficance_ouverte_avec_lift_positif(client):
    """Un lift CV positif rouvre la porte et l'annonce honnêtement."""
    from api.main import store

    # `store` est un singleton chargé au démarrage de l'API : le muter sans
    # le restaurer fuite dans tous les tests exécutés après (et l'ordre de
    # pytest n'est pas garanti). D'où le try/finally.
    avant = store.personalization_warranted
    try:
        store.personalization_warranted = True
        data = client.get("/recommend/user_0?model=hybrid&n=5").json()
        assert data["model"] == "hybrid"
    finally:
        store.personalization_warranted = avant


class TestRecommendCategories:
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
# Tests de régression — résolution des user_id
#
# Régression du bug P0 : avec un user_encoder dont les classes sont des entiers,
# l'API testait l'appartenance du user_id reçu (une chaîne, issu du path) à un
# set d'entiers → toujours faux → 100 % des utilisateurs connus renvoyaient la
# branche cold-start, identique pour tout le monde.
# ─────────────────────────────────────────────────────────────────────────────

class TestUserResolutionRegression:
    def test_int_encoded_user_is_not_cold_start(self, int_user_client):
        """Un user_id connu mais encodé en entier ne doit PAS tomber en cold-start."""
        data = int_user_client.get("/recommend/6").json()
        assert data["model"] != "cold-start", (
            f"L'utilisateur 6 est connu du modèle mais est servi en cold-start : {data}"
        )
        assert data["user_id"] == "6"

    @pytest.mark.parametrize("uid", [6, 12, 17, 19, 20])
    def test_every_int_encoded_user_is_known(self, int_user_client, uid):
        data = int_user_client.get(f"/recommend/{uid}").json()
        assert data["model"] != "cold-start", (
            f"L'utilisateur {uid} est connu mais servi en cold-start"
        )

    def test_unknown_user_still_cold_start(self, int_user_client):
        """La résolution entière ne doit pas faire inventer des utilisateurs."""
        data = int_user_client.get("/recommend/424242").json()
        assert "cold-start" in data["model"]

    def test_float_like_user_id_resolves(self, int_user_client):
        """Un client qui envoie 6.0 (float JSON) doit être résolu comme 6."""
        data = int_user_client.get("/recommend/6.0").json()
        assert data["model"] != "cold-start", (
            f"Le user_id 6.0 n'a pas été résolu en 6 : {data}"
        )

    def test_known_users_get_distinct_recommendations(self, int_user_client):
        """La personnalisation doit produire des résultats distincts par user."""
        recos = {
            uid: tuple(int_user_client.get(f"/recommend/{uid}?n_cats=5")
                       .json()["recommended_categories"])
            for uid in (6, 12, 17, 19)
        }
        assert len(set(recos.values())) > 1, (
            "Tous les utilisateurs connus reçoivent exactement la même "
            f"recommandation : {recos}"
        )


class TestUserHistoryFilterSeen:
    """L'historique persistant doit alimenter le Content-Based et le filtre."""

    SEEN_BY_USER: ClassVar[dict] = {
        "6":  "Vêtements femmes",
        "12": "Accessoires",
        "17": "Chaussures hommes",
        "19": "Chaussures femmes",
        "20": "Vêtements hommes",
    }

    @pytest.mark.parametrize("uid,seen_cat", sorted(SEEN_BY_USER.items()))
    def test_seen_category_excluded(self, int_user_client, uid, seen_cat):
        """La catégorie déjà vue ne doit pas être recommandée."""
        data = int_user_client.get(f"/recommend/{uid}?n_cats={N_CATS}").json()
        assert seen_cat not in data["recommended_categories"], (
            f"La catégorie déjà vue '{seen_cat}' est ré-aménée à l'utilisateur {uid} "
            f"(filter_seen inopérant) : {data['recommended_categories']}"
        )

    def test_history_is_loaded(self, int_user_client):
        from api.main import store
        assert store.user_history, "user_history.pkl n'a pas été chargé"
        assert len(store.user_history) == 5

    def test_filter_seen_still_returns_products(self, int_user_client):
        """Exclure les catégories vues ne doit pas vider la réponse."""
        data = int_user_client.get("/recommend/6?n_cats=5").json()
        assert data["recommendations"], "Aucune recommandation après filtrage"
        assert set(data["recommendations"]).issubset(AVAILABLE_IDS)


class TestSessionRecommend:
    """
    Melcameroun identifie le visiteur par SESSION, pas par customer_id. Sans cet
    endpoint, /recommend/{user_id} avec un identifiant de session tombe en
    cold-start et renvoie la popularité : mêmes articles pour tout le monde.
    La personnalisation passe donc par le contenu de la session.
    """

    def test_empty_session_falls_back_to_popularity(self, int_user_client):
        from api.main import store
        data = int_user_client.post(
            "/recommend/session", json={"session_id": "s1", "categories": []}
        ).json()
        assert data["used_categories"] == []
        assert "popularity" in data["model"]
        assert data["recommendations"] == store.popular_products[:10]

    def test_known_category_personalizes(self, int_user_client):
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s2", "categories": ["Vêtements femmes"]},
        ).json()
        assert data["used_categories"] == ["Vêtements femmes"]
        assert data["model"] == "content-based (session)"
        assert data["recommended_categories"], "Aucune catégorie recommandée"

    def test_session_categories_not_recommended_back(self, int_user_client):
        """On propose des compléments, pas ce que le visiteur vient de voir."""
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s3", "categories": ["Vêtements femmes"]},
        ).json()
        assert "Vêtements femmes" not in data["recommended_categories"]

    def test_unknown_category_is_ignored_not_fatal(self, int_user_client):
        """'Sacs' est une vraie catégorie MEL mais hors de l'espace du modèle."""
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s4", "categories": ["Sacs"]},
        ).json()
        assert data["ignored_categories"] == ["Sacs"]
        assert data["used_categories"] == []
        assert data["recommendations"], "Doit retomber sur la popularité, pas renvoyer vide"

    def test_only_unknown_categories_falls_back(self, int_user_client):
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s5", "categories": ["Sacs", "Mobilier"]},
        ).json()
        assert set(data["ignored_categories"]) == {"Sacs", "Mobilier"}
        assert "popularity" in data["model"]

    def test_different_sessions_rank_differently(self, int_user_client):
        """
        Le cœur de la fonctionnalité : deux sessions distinctes ne doivent pas
        recevoir le même classement. Sinon la personnalisation est décorative.
        """
        a = int_user_client.post(
            "/recommend/session",
            json={"session_id": "a", "categories": ["Accessoires"]},
        ).json()["recommended_categories"]
        b = int_user_client.post(
            "/recommend/session",
            json={"session_id": "b", "categories": ["Chaussures hommes"]},
        ).json()["recommended_categories"]
        assert a != b, f"Même classement pour deux sessions différentes : {a}"

    def test_repeated_call_is_stable(self, int_user_client):
        """Même session => même réponse (pas de non-déterminisme)."""
        body = {"session_id": "stable", "categories": ["Accessoires"]}
        first = int_user_client.post("/recommend/session", json=body).json()
        second = int_user_client.post("/recommend/session", json=body).json()
        assert first == second

    def test_no_sold_articles(self, int_user_client):
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s6", "categories": ["Vêtements femmes"], "n": 10},
        ).json()
        assert set(data["recommendations"]).issubset(AVAILABLE_IDS)

    def test_recommended_categories_have_stock(self, int_user_client):
        """'Télévisions' n'a plus aucun article : inutile de la proposer."""
        data = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s7", "categories": ["Accessoires"]},
        ).json()
        assert not (set(data["recommended_categories"]) - CATEGORIES_WITH_STOCK), (
            f"Catégorie sans stock recommandée : "
            f"{set(data['recommended_categories']) - CATEGORIES_WITH_STOCK}"
        )

    def test_n_is_respected(self, int_user_client):
        for n in (1, 3, 7):
            data = int_user_client.post(
                "/recommend/session",
                json={"session_id": "s8", "categories": ["Accessoires"], "n": n},
            ).json()
            assert len(data["recommendations"]) == n

    def test_n_zero_is_rejected(self, int_user_client):
        r = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s9", "categories": ["Accessoires"], "n": 0},
        )
        assert r.status_code == 422

    def test_too_many_categories_rejected(self, int_user_client):
        r = int_user_client.post(
            "/recommend/session",
            json={"session_id": "s10", "categories": [f"Cat{i}" for i in range(60)]},
        )
        assert r.status_code == 422


class TestUserHistoryArtifactFormat:
    """
    Le fixture écrit user_history.pkl à la main, il ne peut donc pas détecter
    un pipeline qui sérialise les listes en chaînes. Régression réelle : un
    `.astype(str)` dans train.py produisait {user: "['Cat A', 'Cat B']"},
    l'API faisait list(chaine) et filtrait une liste de caractères — donc
    filter_seen ne retirait rien et les catégories déjà consommées étaient
    ré-aménées. Ces tests vérifient le format tel que produit par l'entraînement.
    """

    def test_history_values_are_lists(self, int_user_client):
        from api.main import store
        assert store.user_history, "user_history vide"
        for uid, cats in store.user_history.items():
            assert isinstance(cats, list), (
                f"user_history[{uid}] est un {type(cats).__name__}, pas une liste : "
                "les catégories ont été sérialisées en chaîne"
            )
            assert all(isinstance(c, str) for c in cats), (
                f"user_history[{uid}] contient des non-str : {cats}"
            )

    def test_history_keys_are_strings(self, int_user_client):
        from api.main import store
        for uid in store.user_history:
            assert isinstance(uid, str), f"clé non-str dans user_history : {uid!r}"

    def test_history_categories_resolve_to_indices(self, int_user_client):
        """Chaque catégorie de l'historique doit être convertible en index."""
        from api.main import _exclude_idx, _user_categories, store
        resolved = 0
        for user_idx in range(len(store.user_enc.classes_)):
            cats = _user_categories(user_idx)
            assert isinstance(cats, list)
            resolved += len(_exclude_idx(cats))
        assert resolved > 0, (
            "Aucune catégorie d'historique ne se convertit en index : "
            "filter_seen ne peut rien exclure"
        )


class TestItemSpaceCoverage:
    """
    L'encodeur doit couvrir toutes les catégories que l'API peut servir.
    Régression : item_enc ne contenait que les catégories vues dans les
    interactions, donc les autres étaient impossibles à recommander.
    """

    def test_encoder_covers_all_categories_with_stock(self, int_user_client):
        from api.main import store
        encoder_cats = set(store.item_enc.classes_)
        cats_with_stock = {
            cat for cat, ids in store.cat_to_products.items() if ids
        }
        missing = cats_with_stock - encoder_cats
        assert not missing, (
            f"Catégories du catalogue avec du stock mais absentes de item_enc "
            f"→ articles jamais recommandables : {sorted(missing)}"
        )

    def test_every_recommended_category_has_stock(self, int_user_client):
        """Une catégorie sans article disponible ne doit pas être recommandée."""
        r = int_user_client.get("/recommend/6?n_cats=5").json()
        for cat in r["recommended_categories"]:
            assert cat in CATEGORIES_WITH_STOCK, (
                f"Catégorie sans aucun article disponible recommandée : {cat}"
            )


class TestCors:
    def test_cors_header_on_allowed_origin(self, client):
        """Une origine autorisée doit recevoir l'en-tête CORS."""
        r = client.get("/health", headers={"Origin": "https://melcameroun.com"})
        assert r.headers.get("access-control-allow-origin") == "https://melcameroun.com", (
            f"Pas d'en-tête CORS pour une origine autorisée : {dict(r.headers)}"
        )

    def test_cors_header_on_preflight(self, client):
        """Le preflight OPTIONS doit être traité par le middleware."""
        r = client.options(
            "/recommend/6",
            headers={
                "Origin": "https://melcameroun.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://melcameroun.com"

    def test_cors_blocks_unknown_origin(self, client):
        """Une origine non autorisée ne doit pas recevoir l'accès."""
        r = client.get("/health", headers={"Origin": "https://piloteur-typosquat.xyz"})
        assert "access-control-allow-origin" not in {
            k.lower() for k in r.headers
        }, f"Origine inconnue autorisée : {dict(r.headers)}"


class TestReloadSecretViaHeader:
    def test_reload_header_secret_accepted(self, int_user_client, monkeypatch):
        """La clé peut être passée par en-tête X-Reload-Secret."""
        monkeypatch.setenv("RELOAD_SECRET", "jeton-ultra-secret")
        r = int_user_client.post(
            "/reload", headers={"X-Reload-Secret": "jeton-ultra-secret"}
        )
        assert r.status_code == 200

    def test_reload_header_secret_rejected(self, int_user_client, monkeypatch):
        """Un mauvais en-tête est refusé même si la query est correcte."""
        monkeypatch.setenv("RELOAD_SECRET", "jeton-ultra-secret")
        r = int_user_client.post(
            "/reload?secret=jeton-ultra-secret",
            headers={"X-Reload-Secret": "mauvais"},
        )
        assert r.status_code == 401

    def test_reload_open_when_no_secret_configured(self, int_user_client, monkeypatch):
        """Sans RELOAD_SECRET, /reload reste ouvert (usage local)."""
        monkeypatch.delenv("RELOAD_SECRET", raising=False)
        assert int_user_client.post("/reload").status_code == 200


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


# �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?��
# Tests POST /events (collecte) et session par article
# �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?��
# Ces tests n'écrivent JAMAIS dans data/raw/events.csv : DATA_DIR est redirigé
# vers tmp_path. Sans cela, la suite de tests polluerait les données de prod.


def _patch_data_dir(monkeypatch, tmp_path):
    from api import main
    target = tmp_path / "raw"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "DATA_DIR", target)
    return target


class TestCollectEvents:
    def test_lot_valide_accepté_et_écrit(self, client, monkeypatch, tmp_path):
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        now = pd.Timestamp.utcnow().isoformat()
        r = client.post("/events", json={
            "session_id": "sess-test-1",
            "events": [
                {"event_type": "view", "product_id": 1, "ts": now},
                {"event_type": "cart", "product_id": 2, "ts": now},
            ],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["accepted"] == 2
        assert body["rejected_unknown_product"] == 0
        assert body["rejected_invalid_type"] == 0
        assert body["session_id"] == "sess-test-1"
        assert (data_dir / "events.csv").exists()
        written = pd.read_csv(data_dir / "events.csv")
        # Les deux types doivent être conservés : c'est le bug de contrat que
        # ce test couvre (un `event_type` ignoré registrait tout en "view").
        assert set(written["event_type"]) == {"view", "cart"}

    def test_event_type_canonique(self, client, monkeypatch, tmp_path):
        """`event_type` — le nom de la colonne CSV — doit être honoré."""
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s", "events": [{"product_id": 2, "event_type": "purchase"}],
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 1
        written = pd.read_csv(data_dir / "events.csv")
        assert written.iloc[0]["event_type"] == "purchase"

    def test_alias_type_accepté(self, client, monkeypatch, tmp_path):
        """`type` reste accepté comme alias de `event_type`."""
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s", "events": [{"product_id": 2, "type": "cart"}],
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 1
        written = pd.read_csv(data_dir / "events.csv")
        assert written.iloc[0]["event_type"] == "cart"

    def test_champ_mal_orthographié_refusé(self, client, monkeypatch, tmp_path):
        """
        Une clé mal orthographiée doit être refusée, pas ignorée en silence.

        Sinon le client croit avoir collecté des vues alors que l'API a
        enregistré autre chose : la faute n'apparaît qu'en production, dans les
        métriques, des semaines plus tard.
        """
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s", "events": [{"product_id": 1, "evnt_type": "cart"}],
        })
        assert r.status_code == 422
        assert not (data_dir / "events.csv").exists()

    def test_produit_inconnu_rejeté_sans_écrire(self, client, monkeypatch, tmp_path):
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        now = pd.Timestamp.utcnow().isoformat()
        r = client.post("/events", json={
            "session_id": "s",
            "events": [
                {"event_type": "view", "product_id": 1, "ts": now},
                {"event_type": "view", "product_id": 999999, "ts": now},
            ],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["accepted"] == 1
        assert body["rejected_unknown_product"] == 1
        assert any("999999" in d for d in body["details"])
        written = pd.read_csv(data_dir / "events.csv")
        assert list(written["product_id"]) == [1]

    def test_type_evenement_invalide_rejeté(self, client, monkeypatch, tmp_path):
        _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s",
            "events": [{"event_type": "teleportation", "product_id": 1}],
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 0
        assert r.json()["rejected_invalid_type"] == 1

    def test_lot_vide_rejeté(self, client, monkeypatch, tmp_path):
        _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={"session_id": "s", "events": []})
        assert r.status_code == 422

    def test_produit_acheté_collecté_même_vendu(self, client, monkeypatch, tmp_path):
        """Un article vendu reste affichable en fiche : sa vue doit être gardée."""
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s", "events": [{"event_type": "view", "product_id": 1}],
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 1
        assert (data_dir / "events.csv").exists()

    def test_sans_session_ni_user_rejeté(self, client, monkeypatch, tmp_path):
        _patch_data_dir(monkeypatch, tmp_path)
        now = pd.Timestamp.utcnow().isoformat()
        r = client.post("/events", json={
            "events": [{"event_type": "view", "product_id": 1, "ts": now}],
        })
        # Sans identité, l'événement n'est rattachable à personne : refusé.
        assert r.status_code == 422

    def test_identifié_par_user_id_seul(self, client, monkeypatch, tmp_path):
        """Un client connecté peut être identifié sans session."""
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "user_id": "user_7",
            "events": [{"event_type": "purchase", "product_id": 1}],
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 1
        written = pd.read_csv(data_dir / "events.csv")
        assert written.iloc[0]["user_id"] == "user_7"

    def test_product_id_obligatoire(self, client, monkeypatch, tmp_path):
        """Un événement sans product_id est refusé par la validation."""
        _patch_data_dir(monkeypatch, tmp_path)
        r = client.post("/events", json={
            "session_id": "s", "events": [{"event_type": "view"}],
        })
        assert r.status_code == 422

    def test_evenement_rejoué_non_dupliqué(self, client, monkeypatch, tmp_path):
        """
        Un rejeu du même lot ne doit pas gonfler les compteurs.

        La déduplication porte sur (user, session, article, type, ts) : elle ne
        peut donc jouer que si le front renvoie son `ts` d'origine. C'est le
        comportement attendu d'un retry — le client rejoue le lot tel quel.
        """
        data_dir = _patch_data_dir(monkeypatch, tmp_path)
        payload = {
            "session_id": "s",
            "events": [{
                "event_type": "view", "product_id": 2,
                "ts": "2026-03-05T12:00:00",
            }],
        }
        assert client.post("/events", json=payload).json()["accepted"] == 1
        assert client.post("/events", json=payload).json()["accepted"] == 0
        written = pd.read_csv(data_dir / "events.csv")
        assert len(written) == 1

    def test_sans_ts_rejoué_non_dédupliquable(self, client, monkeypatch, tmp_path):
        """
        Sans `ts` client, deux envois sont deux événements distincts.

        C'est correct : sans horodatage, on ne peut pas distinguer un rejeu d'une
        nouvelle visite. C'est pourquoi la déduplication s'appuie sur le ts.
        """
        _patch_data_dir(monkeypatch, tmp_path)
        payload = {"session_id": "s", "events": [{"event_type": "view", "product_id": 2}]}
        assert client.post("/events", json=payload).json()["accepted"] == 1
        assert client.post("/events", json=payload).json()["accepted"] == 1


# �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?��
# Tests de la personnalisation par ARTICLE (product_sim.pkl)
# �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?��

@pytest.fixture()
def product_client(fake_artifacts, monkeypatch):
    """Charge une couche article (product_sim.pkl) au-dessus des artefacts factices."""
    from api import main

    sim = np.zeros((len(PRODUCTS), len(PRODUCTS)), dtype=np.float32)
    # Articles 1 et 3 se ressemblent, 2 est distinct.
    sim[0, 2] = sim[2, 0] = 0.9
    sim[0, 1] = sim[1, 0] = 0.1
    sim[1, 2] = sim[2, 1] = 0.05
    pid_to_idx = {str(p[0]): i for i, p in enumerate(PRODUCTS)}

    path = fake_artifacts / "product_sim.pkl"
    with open(path, "wb") as f:
        pickle.dump((sim, pid_to_idx), f)
    try:
        main.store.load(models_dir=fake_artifacts)
        yield TestClient(main.app)
    finally:
        path.unlink(missing_ok=True)
        main.store.product_sim = None
        main.store.load(models_dir=fake_artifacts)


class TestSessionParArticle:
    def test_product_ids_donne_une_granularité_produit(self, product_client):
        r = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["1"], "n": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["granularity"] == "produit"
        assert data["used_product_ids"] == ["1"]
        # L'article le plus proche de "1" est "3" (similarité 0.9).
        assert data["recommendations"][0] == "3"

    def test_article_vu_jamais_recommandé(self, product_client):
        data = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["1"], "n": 10,
        }).json()
        assert "1" not in data["recommendations"]

    def test_plusieurs_articles_vus(self, product_client):
        data = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["1", "3"], "n": 10,
        }).json()
        # 1 et 3 sont mutuellement proches mais tous deux vus : on ne doit
        # proposer que les articles restants.
        assert "1" not in data["recommendations"]
        assert "3" not in data["recommendations"]

    def test_article_vendu_exclu(self, product_client):
        """Un article vendu ne doit pas être proposé, même s'il est très proche."""
        data = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["3"], "n": 10,
        }).json()
        # "1" est le plus proche de "3" (0.9) mais il est vendu.
        assert "1" not in data["recommendations"]

    def test_article_hors_couche_article_ignoré(self, product_client):
        data = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["1", "999999"], "n": 5,
        }).json()
        assert data["used_product_ids"] == ["1"]
        assert "999999" in data["ignored_product_ids"]

    def test_product_ids_inconnus_repli_catégories(self, product_client):
        """Sans article exploitable, on retombe sur la personnalisation catégorie."""
        data = product_client.post("/recommend/session", json={
            "session_id": "s",
            "product_ids": ["999999"],
            "categories": ["Vêtements femmes"],
            "n": 5,
        }).json()
        assert data["granularity"] == "categorie"
        assert data["ignored_product_ids"] == ["999999"]
        assert data["recommendations"]

    def test_absence_de_product_sim_repli_catégories(self, client):
        """Sans couche article, l'API reste fonctionnelle au niveau catégorie."""
        data = client.post("/recommend/session", json={
            "session_id": "s",
            "product_ids": ["1"],
            "categories": ["Vêtements femmes"],
            "n": 5,
        }).json()
        assert data["granularity"] == "categorie"
        assert data["recommendations"]

    def test_aucun_contexte_retombe_sur_la_popularité(self, product_client):
        """Ni article ni catégorie : on renvoie de la popularité, pas une erreur."""
        data = product_client.post("/recommend/session", json={
            "session_id": "s", "product_ids": ["999999"], "n": 5,
        }).json()
        assert data["granularity"] == "produit"
        assert "popularity" in data["model"]
        assert data["recommendations"]

    def test_categories_et_articles_combines(self, product_client):
        """Les articles de session priment, les catégories servent de repli."""
        data = product_client.post("/recommend/session", json={
            "session_id": "s",
            "product_ids": ["1"],
            "categories": ["Chaussures femmes"],
            "n": 10,
        }).json()
        # Le contexte article l'emporte : on ne retombe pas sur les chaussures.
        assert data["granularity"] == "produit"
        assert data["used_categories"] == ["Chaussures femmes"]

    def test_similar_texte_prioritaire(self, product_client):
        r = product_client.get("/similar/1?n=5")
        assert r.status_code == 200
        data = r.json()
        assert data["strategy"] == "produit (titre)"
        assert "1" not in data["similar_products"]
        assert data["similar_products"][0] == "3"

    def test_similar_repli_catégorie_sans_couche_article(self, client):
        data = client.get("/similar/2?n=5").json()
        assert data["strategy"] == "categorie"
        assert "2" not in data["similar_products"]