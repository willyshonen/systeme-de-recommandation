"""
Tests du flux d'événements produit (src/events.py).

Hermétiques : tous les tests construisent leurs propres CSV dans tmp_path.
Aucune donnée réelle n'est lue ni écrite — c'est ce qui manquait jusqu'ici et
la raison pour laquelle les défauts de timestamp et de rattachement des avis
n'avaient pas été vus.
"""

import pandas as pd

from src import events as ev


def _products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [1, 2, 3],
        "product_name": [
            "Chemise Homme a Manches Longues",
            "Robe bebe en maille creme",
            "Pantalon de sport Asics",
        ],
        "product_category_name": ["Vetements hommes", "Vetements enfants", "Vetements hommes"],
    })


def _write(tmp_path, name, frame):
    frame.to_csv(tmp_path / name, index=False)


# ── Achats ────────────────────────────────────────────────────────────────

def test_achat_récupère_le_client_et_le_timestamp(tmp_path):
    """Le timestamp d'un achat est celui de la COMMANDE, pas celui du produit."""
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1], "customer_id": [100], "order_status": ["Valide"],
        "created_at": ["2026-03-01 10:30:00"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1], "product_id": [1], "price": [5000.0],
    }))

    out = ev.load_events(tmp_path)
    assert len(out) == 1
    assert out.iloc[0]["user_id"] == 100
    # created_at est PERDU après _normalise s'il n'est pas lu avant : ce test
    # échouait avec ts = NaT.
    assert pd.notna(out.iloc[0]["ts"]), "ts d'achat perdu"
    assert out.iloc[0]["ts"] == pd.Timestamp("2026-03-01 10:30:00")


def test_commande_annulée_compte_comme_achat(tmp_path):
    """Une commande annulée ne doit jamais produire d'événement d'achat."""
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [100, 101, 102],
        "order_status": ["Valide", "Annulee", "Valide"],
        "created_at": ["2026-03-01", "2026-03-02", "2026-03-03"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1, 2, 3], "product_id": [1, 2, 3], "price": [1.0, 2.0, 3.0],
    }))

    out = ev.load_events(tmp_path)
    assert len(out) == 2
    assert set(out["user_id"]) == {100, 102}
    assert 2 not in set(out["product_id"]), "achat annulé conservé"


def test_statut_insensible_à_la_casse(tmp_path):
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1, 2, 3], "customer_id": [1, 2, 3],
        "order_status": ["valide", "VALIDE", " Valide "],
        "created_at": ["2026-03-01"] * 3,
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1, 2, 3], "product_id": [1, 2, 3], "price": [1.0, 2.0, 3.0],
    }))
    assert len(ev.load_events(tmp_path)) == 3


# ── Avis ──────────────────────────────────────────────────────────────────

def test_avis_rattaché_aux_articles_de_la_commande(tmp_path):
    """Sans jointure order_items, l'avis n'a aucun product_id et était jeté."""
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1], "customer_id": [100], "order_status": ["Valide"],
        "created_at": ["2026-03-01"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1, 1], "product_id": [1, 2], "price": [10.0, 20.0],
    }))
    _write(tmp_path, "reviews.csv", pd.DataFrame({
        "order_id": [1], "review_score": [5.0],
    }))

    out = ev.load_events(tmp_path)
    avis = out[out["event_type"] == "review"]
    # Un avis sur une commande de 2 articles = 1 événement par article.
    assert set(avis["product_id"]) == {1, 2}
    assert set(avis["user_id"]) == {100}


def test_statuts_de_livraison_magento_acceptés(tmp_path):
    """
    Un export Magento ne dit pas « valide » mais « delivered » / « shipped ».

    Le filtre ne portait que sur « valide » : sur un tel export TOUS les achats
    étaient écartés, et le modèle s'entraînait sur un jeu vide — silencieusement,
    puisque le seul symptôme était un message d'erreur tardif et trompeur.
    """
    statuses = ["delivered", "shipped", "processing", "complete",
                "Valide", " delivered ", "DELIVERED"]
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": list(range(1, len(statuses) + 1)),
        "customer_id": [100] * len(statuses),
        "order_status": statuses,
        "created_at": ["2026-03-01"] * len(statuses),
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": list(range(1, len(statuses) + 1)),
        "product_id": [1] * len(statuses),
        "price": [10.0] * len(statuses),
    }))
    assert len(ev.load_events(tmp_path)) == len(statuses)


def test_statuts_non_livrés_rejetés(tmp_path):
    """Annulation, attente, remboursement : ce ne sont pas des achats."""
    bad = ["canceled", "cancelled", "annule", "annulée", "pending",
           "failed", "refunded", "en attente", "fraud"]
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": list(range(1, len(bad) + 1)),
        "customer_id": [100] * len(bad),
        "order_status": bad,
        "created_at": ["2026-03-01"] * len(bad),
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": list(range(1, len(bad) + 1)),
        "product_id": [1] * len(bad),
        "price": [10.0] * len(bad),
    }))
    assert len(ev.load_events(tmp_path)) == 0


def test_statut_inconnu_signalé_sans_crash(tmp_path, caplog):
    """
    Un statut inconnu écarte les lignes mais doit être visible dans les logs.

    Sinon des achats disparaissent du modèle sans que personne ne s'en aperçoive.
    """
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1, 2], "customer_id": [100, 101],
        "order_status": ["delivered", "paiement_virement"],
        "created_at": ["2026-03-01", "2026-03-02"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1, 2], "product_id": [1, 1], "price": [10.0, 10.0],
    }))
    with caplog.at_level("WARNING", logger="mel.events"):
        out = ev.load_events(tmp_path)
    assert len(out) == 1                      # seul 'delivered' est retenu
    assert "paiement_virement" in caplog.text  # et le cas est signalé


def test_avis_de_commande_annulée_ignoré(tmp_path):
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1], "customer_id": [100], "order_status": ["Annulee"],
        "created_at": ["2026-03-01"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1], "product_id": [1], "price": [10.0],
    }))
    _write(tmp_path, "reviews.csv", pd.DataFrame({"order_id": [1], "review_score": [1.0]}))
    assert len(ev.load_events(tmp_path)) == 0


# ── Collecte live ─────────────────────────────────────────────────────────

def test_événement_live_conserve_son_event_type(tmp_path):
    """events.csv porte son propre event_type : il ne doit pas être écrasé."""
    pd.DataFrame([{
        "ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
        "user_id": None, "product_id": 1, "event_type": "view",
    }]).to_csv(tmp_path / "events.csv", index=False)

    out = ev.load_events(tmp_path)
    assert len(out) == 1
    assert out.iloc[0]["event_type"] == "view"
    assert out.iloc[0]["session_id"] == "s1"


def test_append_events_écrit_les_lignes(tmp_path):
    n = ev.append_events(tmp_path, [
        {"ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
         "user_id": None, "product_id": 1, "event_type": "view"},
    ])
    assert n == 1
    assert (tmp_path / "events.csv").exists()
    assert len(ev.load_events(tmp_path)) == 1


def test_append_events_déduplique_les_rejouements(tmp_path):
    """Un retry ou un double-clic ne doit pas gonfler les compteurs."""
    row = {"ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
           "user_id": None, "product_id": 1, "event_type": "view"}
    assert ev.append_events(tmp_path, [row]) == 1
    assert ev.append_events(tmp_path, [row]) == 0, "doublon non filtré"
    assert len(ev.load_events(tmp_path)) == 1


def test_append_events_conserve_les_événements_différents(tmp_path):
    base = {"ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
            "user_id": None, "product_id": 1, "event_type": "view"}
    assert ev.append_events(tmp_path, [base]) == 1
    # Autre article -> pas un doublon
    assert ev.append_events(tmp_path, [dict(base, product_id=2)]) == 1
    # Autre type -> pas un doublon
    assert ev.append_events(tmp_path, [dict(base, event_type="cart")]) == 1
    # Autre session -> pas un doublon
    assert ev.append_events(tmp_path, [dict(base, session_id="s2")]) == 1
    assert len(ev.load_events(tmp_path)) == 4


def test_append_events_conserve_les_doublons_du_même_lot(tmp_path):
    """
    Deux lignes identiques dans le même lot sont deux événements, pas un rejeu.

    Écraser reviendrait à perdre des vues réelles : un front qui envoie deux
    fois la même vue parce qu'elle a réellement eu lieu verrait son signal
    amputé en silence.
    """
    row = {"ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
           "user_id": None, "product_id": 1, "event_type": "view"}
    assert ev.append_events(tmp_path, [dict(row), dict(row), dict(row)]) == 3
    assert len(ev.load_events(tmp_path)) == 3


def test_rejeu_du_meme_lot_ecfiltre(tmp_path):
    """La protection reste efficace contre le rejeu d'un lot déjà écrit."""
    lot = [{"ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
            "user_id": None, "product_id": 1, "event_type": "view"}]
    assert ev.append_events(tmp_path, lot) == 1
    assert ev.append_events(tmp_path, lot) == 0
    assert len(ev.load_events(tmp_path)) == 1


def test_append_events_ts_invalide_écarté(tmp_path):
    assert ev.append_events(tmp_path, [{"ts": "pas-une-date", "product_id": 1}]) == 0


# ── Filtrage ──────────────────────────────────────────────────────────────

def test_produit_hors_catalogue_ignoré(tmp_path):
    pd.DataFrame([{
        "ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
        "user_id": None, "product_id": 999, "event_type": "view",
    }]).to_csv(tmp_path / "events.csv", index=False)
    # Le filtrage catalogue n'a lieu que si le catalogue est fourni.
    assert len(ev.load_events(tmp_path, _products())) == 0


def test_produit_du_catalogue_conservé(tmp_path):
    pd.DataFrame([{
        "ts": "2026-03-05T12:00:00", "source": "web", "session_id": "s1",
        "user_id": None, "product_id": 1, "event_type": "view",
    }]).to_csv(tmp_path / "events.csv", index=False)
    assert len(ev.load_events(tmp_path, _products())) == 1


def test_événement_sans_identité_ni_session_ignoré(tmp_path):
    pd.DataFrame([{
        "ts": "2026-03-05T12:00:00", "source": "web", "session_id": "",
        "user_id": None, "product_id": 1, "event_type": "view",
    }]).to_csv(tmp_path / "events.csv", index=False)
    assert len(ev.load_events(tmp_path)) == 0


# ── Features & similarité ─────────────────────────────────────────────────

def test_prix_médian_ignore_les_commandes_annulées(tmp_path):
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1, 2, 3], "customer_id": [1, 2, 3],
        "order_status": ["Valide", "Annulee", "Valide"],
        "created_at": ["2026-03-01", "2026-03-02", "2026-03-03"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1, 2, 3], "product_id": [1, 1, 1], "price": [100.0, 9999.0, 200.0],
    }))
    out = ev.load_events(tmp_path)
    feat = ev.product_features(_products(), out)
    # Médiane de {100, 200} = 150. Si l'annulée (9999) entrait, on aurait ~5100.
    assert feat.loc[feat["product_id"] == 1, "price"].iloc[0] == 150.0


def test_produit_jamais_vendu_sans_prix_pas_zéro(tmp_path):
    """Un prix manquant n'est pas un article gratuit."""
    _write(tmp_path, "orders.csv", pd.DataFrame({
        "order_id": [1], "customer_id": [1], "order_status": ["Valide"],
        "created_at": ["2026-03-01"],
    }))
    _write(tmp_path, "order_items.csv", pd.DataFrame({
        "order_id": [1], "product_id": [1], "price": [100.0],
    }))
    feat = ev.product_features(_products(), ev.load_events(tmp_path))
    assert pd.isna(feat.loc[feat["product_id"] == 2, "price"].iloc[0])


def test_similarité_trouve_un_parent_proche():
    feat = ev.product_features(_products(), ev._empty_events())
    pids, sim = ev.build_product_similarity(feat)
    assert len(pids) == 3
    assert sim.shape == (3, 3)
    # Diagonale nulle : un article ne doit jamais se recommander lui-même.
    assert (sim.diagonal() == 0).all()


def test_similarité_refuse_un_catalogue_vide():
    pids, sim = ev.build_product_similarity(pd.DataFrame({"product_id": [], "name": []}))
    assert pids == [] and sim.shape == (0, 0)


def test_similarité_refuse_un_catalogue_sans_titre():
    pids, sim = ev.build_product_similarity(
        pd.DataFrame({"product_id": [1, 2], "name": ["", None]})
    )
    assert pids == [] and sim.shape == (0, 0)


# ── Suffisance ─────────────────────────────────────────────────────────────

def test_suffisance_refuse_les_sessions_seules():
    """Des sessions anonymes ne doivent pas faire monter le volume CF."""
    n = 300
    out = pd.DataFrame({
        "ts": pd.date_range("2026-03-01", periods=n, freq="h"),
        "source": "live", "session_id": [f"s{i}" for i in range(n)],
        "user_id": [None] * n,
        "product_id": [1 + (i % 3) for i in range(n)],
        "event_type": ["view"] * n,
    })
    r = ev.data_sufficiency_report(out)
    assert r["n_events"] == 300
    assert r["n_events_identified"] == 0
    assert r["n_events_session_only"] == 300
    assert r["n_users_identified"] == 0
    assert not r["collaborative_ready"], "sessions seules ne doivent pas valider le CF"
    assert r["content_based_ready"]


def test_suffisance_valide_avec_des_données_suffisantes():
    rows = []
    for u in range(120):
        for p in range(10):
            rows.append({"ts": "2026-03-01", "source": "live", "session_id": None,
                         "user_id": u, "product_id": p, "event_type": "view"})
    r = ev.data_sufficiency_report(pd.DataFrame(rows))
    assert r["collaborative_ready"]
    assert "activable" in r["collaborative_verdict"]


def test_suffisance_signale_ce_qui_manque():
    out = pd.DataFrame({
        "ts": pd.to_datetime(["2026-03-01"] * 5),
        "source": ["live"] * 5, "session_id": [None] * 5,
        "user_id": [1, 1, 1, 2, 2], "product_id": [1, 1, 1, 2, 2],
        "event_type": ["view"] * 5,
    })
    r = ev.data_sufficiency_report(out)
    assert not r["collaborative_ready"]
    verdict = r["collaborative_verdict"]
    assert "événements identifiés" in verdict
    assert "clients identifiés" in verdict
    assert "couples" in verdict