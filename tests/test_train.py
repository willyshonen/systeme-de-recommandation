"""
Garde-fous du pipeline d'entraînement.

Ces tests ne réentraînent pas le modèle (trop lent) : ils verrouillent les
propriétés dont dépend la reproductibilité et la lecture des métriques.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


# ── Reproductibilité ───────────────────────────────────────────────────────

def test_blas_figé_à_un_seul_thread():
    """
    L'entraînement doit poser les variables de thread AVANT l'import de numpy.

    Les réductions multi-threadées de BLAS ne sont pas bit-reproductibles : deux
    entraînements identiques donnaient deux modèles différents et un NDCG qui
    oscillait entre 0.9540 et 1.0000. L'alerte de régression du scheduler se
    déclenchait alors sur du bruit.
    """
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import src.train;"
        "import os;"
        "print(os.environ.get('OMP_NUM_THREADS'),"
        "os.environ.get('OPENBLAS_NUM_THREADS'),"
        "os.environ.get('MKL_NUM_THREADS'))"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT, env=env,
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "1 1 1"


def test_opt_out_multithread_honoré():
    """La variable d opting-out doit être respectée quand elle vaut 1."""
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import src.train;"
        "import os; print(os.environ.get('OMP_NUM_THREADS'))"
    )
    env = dict(os.environ, TRAIN_ALLOW_MULTITHREAD="1")
    env.pop("OMP_NUM_THREADS", None)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT, env=env,
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert out.returncode == 0, out.stderr[-500:]
    # Pas de valeur forcée : le choix revient à l'environnement (None si absent).
    assert out.stdout.strip().splitlines()[-1] in ("", "None")


def test_svd_est_déterministe():
    """
    Le SVD ne doit plus dépendre d'un tirage aléatoire.

    `svds` passait par ARPACK, un solveur itératif : même avec un `v0` fixé, il
    donnait des décompositions différentes d'une exécution à l'autre. Le NDCG à
    alpha=1.0 oscillait entre 0,51 et 0,64, et l'alpha choisi par validation
    croisée changeait (0.0, puis 0.8, puis 0.0). Le scheduler compare les
    métriques avant/après : il ne peut pas se fier à ça.

    On exige maintenant un SVD COMPLET (LAPACK gesdd), déterministe et — sur une
    matrice clients × catégories — plus rapide que l'itératif.
    """
    import numpy as np
    import pandas as pd

    from src.train import train_svd

    class Enc:
        def __init__(self, classes):
            self.classes_ = np.array(classes, dtype=object)

        def transform(self, ids):
            return np.array([list(self.classes_).index(i) for i in ids])

    rng = np.random.default_rng(0)
    n_u, n_i = 34, 7
    rows = []
    for u in range(n_u):
        for i in rng.choice(n_i, size=rng.integers(1, 4), replace=False):
            rows.append({"user_id": f"u{u}", "item_id": f"c{i}", "rating": 5.0})
    df = pd.DataFrame(rows)

    u_enc, i_enc = Enc([f"u{i}" for i in range(n_u)]), Enc([f"c{i}" for i in range(n_i)])

    r1, _, _, _ = train_svd(df, u_enc, i_enc, k_factors=150)
    r2, _, _, _ = train_svd(df, u_enc, i_enc, k_factors=150)
    assert np.allclose(r1, r2), "train_svd n'est pas reproductible"

    # L'ordre des lignes d'entrée ne doit pas changer le résultat.
    r3, _, _, _ = train_svd(df.sample(frac=1.0, random_state=1), u_enc, i_enc, k_factors=150)
    assert np.allclose(r1, r3), "train_svd dépend de l'ordre des lignes"

    src = (_ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert "svds(" not in src, "svds (ARPACK, non déterministe) est encore utilisé"


# ── Espace des catégories ──────────────────────────────────────────────────

def test_espace_des_items_trié():
    """
    L'espace des catégories doit être trié avant d'encoder.

    Un ordre issu d'un set varie d'un processus à l'autre et change l'ordre des
    colonnes de la matrice, donc le modèle.
    """
    from src.train import build_item_space

    products = pytest.importorskip("pandas").DataFrame({
        "product_category_name": ["B", "A", "C"],
        "available": [1, 1, 1],
    })
    space, _ = build_item_space(products, {"B", "C"}, mode="interactions")
    assert space == sorted(space)
    assert space == ["B", "C"]


def test_catégorie_interactions_sans_stock_conservée():
    """
    Une catégorie vue dans les interactions reste dans l'espace du modèle, même
    sans article en stock — décision explicite : elle est signalée, pas supprimée.

    Supprimer une catégorie observée ferait disparaître le item_encoder des
    profils existants et casserait le Warm Start des Visitors.
    """
    import pandas as pd

    from src.train import build_item_space

    products = pd.DataFrame({"product_category_name": ["A"], "available": [1]})
    space, _ = build_item_space(products, {"A", "FANTOME"}, mode="interactions")
    assert space == ["A", "FANTOME"]


def test_mode_catalog_ajoute_les_catégories_avec_stock():
    """
    En mode catalogue, une catégorie avec du stock mais sans interaction est
    ajoutée — et signalée comme inatteignable, car aucun profil ne la porte.
    """
    import pandas as pd

    from src.train import build_item_space

    products = pd.DataFrame({
        "product_category_name": ["A", "B"], "available": [1, 1],
    })
    space, report = build_item_space(products, {"A"}, mode="catalog")
    assert space == ["A", "B"]
    assert report["n_items_added_from_catalog"] == 1
    # Avec du stock mais aucune interaction : le modèle ne peut pas y arriver.
    assert report["unreachable_categories"] == ["B"]


def test_mode_interactions_ignore_les_catégories_non_observées():
    """Le mode par défaut n'invente pas de catégories : c'est le choix utilisateur."""
    import pandas as pd

    from src.train import build_item_space

    products = pd.DataFrame({
        "product_category_name": ["A", "B", "C"], "available": [1, 1, 0],
    })
    space, report = build_item_space(products, {"A"}, mode="interactions")
    assert space == ["A"]
    assert report["n_items_added_from_catalog"] == 0
    assert report["categories_without_stock"] == ["C"]


def test_mode_inconnu_rejeté():
    """Un mode mal orthographié doit échouer, pas retomber silencieusement."""
    import pandas as pd

    from src.train import build_item_space

    products = pd.DataFrame({"product_category_name": ["A"], "available": [1]})
    with pytest.raises(ValueError, match="interactions"):
        build_item_space(products, {"A"}, mode="interractions")


# ── Vues et achats nourrissent la prédiction de catégories ─────────────────

def _jeu_de_donnees(tmp_path, events_csv: str | None = None):
    """Écrit un jeu minimal : 2 clients, achats et vues optionnels."""
    import pandas as pd

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "product_name": ["Chemise", "Robe", "Canape", "TV"],
        "product_category_name": ["Vetements", "Vetements", "Mobilier", "TV"],
        "available": [1, 1, 1, 1],
    }).to_csv(raw / "products.csv", index=False)
    pd.DataFrame({
        "order_id": [1], "customer_id": [10], "order_status": ["Valide"],
        "created_at": ["2026-03-01"],
    }).to_csv(raw / "orders.csv", index=False)
    pd.DataFrame({
        "order_id": [1], "product_id": [1], "price": [10.0],
    }).to_csv(raw / "order_items.csv", index=False)
    if events_csv:
        (raw / "events.csv").write_text(events_csv, encoding="utf-8")
    return {"data_dir": str(raw), "products": pd.read_csv(raw / "products.csv")}


def test_achat_entre_dans_les_interactions(tmp_path):
    """Le contract central : un achat doit produire une interaction de catégorie."""
    from src.train import build_interactions

    df = build_interactions(_jeu_de_donnees(tmp_path))
    assert len(df)
    assert set(df["item_id"]) == {"Vetements"}
    assert df.iloc[0]["rating"] == 5.0


def test_vues_collectées_entrent_dans_les_interactions(tmp_path):
    """
    Les vues envoyées par POST /events doivent.predict l'intérêt du client.

    C'est le cœur de la demande : prédire les catégories à partir de ce que le
    client achète OU regarde. Avant, build_interactions lisait visits.csv en
    dur et ignorait events.csv : les vues du front n'arrivaient jamais ici.
    """
    from src.train import build_interactions

    events = (
        "ts,source,session_id,user_id,product_id,event_type\n"
        "2026-03-05T12:00:00,api,,77,3,view\n"
        "2026-03-05T12:00:05,api,,77,3,view\n"
        "2026-03-05T12:00:10,api,,77,4,view\n"
    )
    df = build_interactions(_jeu_de_donnees(tmp_path, events))
    assert set(df["item_id"]) == {"Vetements", "Mobilier", "TV"}
    assert set(df["user_id"]) == {10, 77}


def test_vues_seules_donnent_un_profil(tmp_path):
    """Un client qui n'a jamais acheté mais seulement regardé doit avoir un profil."""
    from src.train import build_interactions

    events = (
        "ts,source,session_id,user_id,product_id,event_type\n"
        "2026-03-05T12:00:00,api,,77,3,view\n"
        "2026-03-05T12:00:10,api,,77,4,view\n"
    )
    df = build_interactions(_jeu_de_donnees(tmp_path, events))
    assert 77 in set(df["user_id"])


def test_vues_anonymes_exclues_de_l_entraînement(tmp_path):
    """
    Un événement de session anonyme ne peut pas créer de profil.

    Il reste disponible pour /recommend/session, mais l'entraînement n'a pas de
    client auquel rattacher le signal.
    """
    from src.train import build_interactions

    events = (
        "ts,source,session_id,user_id,product_id,event_type\n"
        "2026-03-05T12:00:00,web,sess-1,,3,view\n"
    )
    df = build_interactions(_jeu_de_donnees(tmp_path, events))
    assert df["user_id"].isna().sum() == 0


def test_répétition_de_vues_augmente_le_poids(tmp_path):
    """Regarder plusieurs articles d'une catégorie pèse plus qu'un seul."""
    from src.train import build_interactions

    events = (
        "ts,source,session_id,user_id,product_id,event_type\n"
        "2026-03-05T12:00:00,api,,77,3,view\n"
        "2026-03-05T12:00:05,api,,77,3,view\n"
        "2026-03-05T12:00:10,api,,77,3,view\n"
        "2026-03-05T12:00:15,api,,77,3,view\n"
        "2026-03-05T12:00:20,api,,77,3,view\n"
        "2026-03-05T12:01:00,api,,77,4,view\n"
    )
    df = build_interactions(_jeu_de_donnees(tmp_path, events))
    poids = dict(zip(df["item_id"], df["rating"]))
    assert poids["Mobilier"] == 1.0     # 5 vues
    assert poids["TV"] < 1.0            # 1 vue


def test_achat_prime_sur_la_vue(tmp_path):
    """Un achat (5.0) doit l'emporter sur une vue (≤1.0) pour la même catégorie."""
    from src.train import build_interactions

    events = (
        "ts,source,session_id,user_id,product_id,event_type\n"
        "2026-03-05T12:00:00,api,,10,1,view\n"
        "2026-03-05T12:00:05,api,,10,2,view\n"
    )
    df = build_interactions(_jeu_de_donnees(tmp_path, events))
    assert df.iloc[0]["rating"] == 5.0