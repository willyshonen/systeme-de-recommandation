"""
Flux d'événements canonique au niveau ARTICLE.

Pourquoi ce module existe
-------------------------
L'historique d'entraînement (`build_interactions` dans train.py) agrège tout au
niveau CATÉGORIE : un clic sur « Robe Africaine 8 Mars » et un achat de « T-shirt
Léopard » ne produisent qu'un signal par catégorie. C'est suffisant pour classer
des catégories, pas pour recommander un article à une personne.

Ce module reconstruit la granularité article en unifiant les cinq tables
dispersées dans `data/raw/` :

| Source                  | Événement  | Identité user            |
|-------------------------|------------|--------------------------|
| `orders.csv`            | —          | `order_id → customer_id` |
| `order_items.csv`       | `purchase` | via le join sur orders   |
| `panier_interactions.csv` | `cart`   | `customer_id` direct     |
| `visits.csv`            | `view`     | `customer_id` direct     |
| `events.csv`            | tous       | `session_id` et/ou user  |
| `reviews.csv`           | `review`   | via le join sur orders   |

Deux pièges de données traités ici, tous deux vérifiés sur le jeu réel :

1. **`order_items.csv` n'a pas de `customer_id`**, seulement `order_id`. Sans le
   join sur `orders.csv`, les 39 achats étaient des événements sans propriétaire
   — inexploitables pour toute personnalisation.

2. **9 commandes sur 39 sont `annule`**. Les compter comme des achats apprend au
   modèle qu'un client a acheté ce qu'il a en réalité annulé. Le filtre est sur
   `order_status`, pas sur la présence de la ligne.

État réel du jeu au moment de l'écriture : 30 achats valides (7 acheteurs), 82
ajouts panier (32 clients), 0 vue, 1 avis. Suffisant pour du content-based,
insuffisant pour du collaboratif — voir `data_sufficiency_report`.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

# Schéma canonique. Toute source est ramenée à ces colonnes.
EVENT_COLUMNS = [
    "user_id",
    "session_id",
    "product_id",
    "event_type",
    "event_value",
    "ts",
    "price",
    "source",
]

EVENT_TYPES = ("view", "cart", "purchase", "review")

# Poids d'un événement en "intérêt équivalent achat".
# Un achat vaut 1.0, un ajout panier 0.6 (intention forte, pas de conversion),
# une vue 0.2. Ces poids sont un choix produit, pas une mesure.
EVENT_WEIGHTS = {"purchase": 1.0, "cart": 0.6, "view": 0.2, "review": 0.8}

# `events.csv` est écrit par l'API pendant que l'entraînement le lit : on sérialise
# les lectures/écritures pour ne pas lire un CSV tronqué.
_IO_LOCK = threading.Lock()

_EVENTS_FILE = "events.csv"
_EVENTS_COLUMNS_FILE = ["ts", "source", "session_id", "user_id", "product_id", "event_type"]


def _coerce_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return df if not df.empty else None


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in EVENT_COLUMNS})


def _normalise(df: pd.DataFrame, event_type: str | None, source: str) -> pd.DataFrame:
    """
    Ramène une table source au schéma canonique.

    `event_type` : valeur par défaut. Si la table porte déjà une colonne
    `event_type` (cas de `events.csv`, alimenté par l'API avec view/cart/purchase),
    elle fait foi — sinon tous les événements collectés seraient enregistrés en
    `view` et le poids d'un achat serait perdu.
    """
    out = pd.DataFrame()
    out["product_id"] = pd.to_numeric(df.get("product_id"), errors="coerce")
    # Les tables MEL utilisent `customer_id`, `events.csv` (API) utilise `user_id`.
    user_col = next(
        (c for c in ("customer_id", "user_id") if c in df.columns), None
    )
    out["user_id"] = (
        pd.to_numeric(df[user_col], errors="coerce") if user_col else np.nan
    )
    out["session_id"] = df["session_id"] if "session_id" in df.columns else None

    if "event_type" in df.columns:
        out["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
        out.loc[~out["event_type"].isin(EVENT_TYPES), "event_type"] = event_type or "view"
    else:
        out["event_type"] = event_type
    out["source"] = source

    ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
    out["ts"] = _coerce_ts(df[ts_col]) if ts_col else pd.NaT

    if "price" in df.columns:
        out["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        out["price"] = np.nan

    if "review_score" in df.columns:
        out["event_value"] = pd.to_numeric(df["review_score"], errors="coerce")
    else:
        out["event_value"] = np.nan

    return out[EVENT_COLUMNS]


def load_events(data_dir: Path | str, products: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construit le flux d'évérations canonique à partir de toutes les sources.

    Retourne un DataFrame aux colonnes `EVENT_COLUMNS`, filtré :
      - `product_id` connu du catalogue (sinon aucun produit à recommander) ;
      - au moins une identité (user_id ou session_id) — un événement anonyme
        sans session ne sert à rien.
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []

    orders = _read(data_dir / "orders.csv")
    items = _read(data_dir / "order_items.csv")
    panier = _read(data_dir / "panier_interactions.csv")
    visits = _read(data_dir / "visits.csv")
    reviews = _read(data_dir / "reviews.csv")
    live = _read(data_dir / _EVENTS_FILE)

    if live is not None:
        # `events.csv` porte son propre event_type : c'est lui qui fait foi.
        frames.append(_normalise(live, None, "live"))

    # ── Achats : order_items ⋈ orders ───────────────────────────────────────────
    # C'est le seul endroit où l'identité d'un achat doit être reconstruite.
    if items is not None and orders is not None and "order_id" in items.columns:
        merged = items.merge(
            orders[["order_id", "customer_id", "order_status", "created_at"]],
            on="order_id",
            how="left",
        )
        status = merged.get("order_status")
        if status is not None:
            # Comparaison insensible à la casse et aux espaces : 'valide',
            # 'Valide', 'VALID'.
            #
            # Le mot « valide » ne doit pas être le SEUL statut accepté. Sur les
            # données réelles le statut s'appelle « valide », mais un export
            # Magento (orders.csv en vient) écrit 'delivered', 'shipped',
            # 'processing' : le filtre les écartait TOUS, et le modèle s'entraînait
            # sur le jeu synthétique de la CI tout en ignorant silencieusement
            # toutes les ventes réelles. On accepte donc les statuts qui
            # signifient « la commande a bien eu lieu », et on rejette les autres
            # — surtout les annulations et les paniers abandonnés.
            _STATUS_OK = {
                "valide", "valid", "delivered", "shipped", "processing",
                "complete", "completed", "delivered_shipped",
                "livree", "livré", "livraison", "expedie", "expédié",
            }
            # États terminaux sans transaction : la commande n'a pas abouti.
            _STATUS_KO = {
                "canceled", "cancelled", "annule", "annulée", "annul",
                "pending", "en attente", "attente", "failed", "echec",
                "échec", "refunded", "rembourse", "remboursé", "fraud",
            }
            norm = status.astype(str).str.strip().str.lower()
            keep = norm.isin(_STATUS_OK) & ~norm.isin(_STATUS_KO)

            unknown = sorted(set(norm.unique()) - _STATUS_OK - _STATUS_KO)
            if unknown:
                # Un statut inconnu ne doit pas être rejeté en silence : c'est
                # soit une nouvelle valeur de Magento, soit une coquille, et dans
                # les deux cas des achats disparaissent du modèle.
                log.warning(
                    "Achats : %d statut(s) de commande non reconnu(s) — ces "
                    "lignes sont ÉCARTÉES. Valeurs : %s. Si ces commandes sont "
                    "bien livrées, ajouter ces statuts à _STATUS_OK dans "
                    "src/events.py.",
                    len(unknown), unknown[:8],
                )

            dropped = int((~keep).sum())
            merged = merged[keep]
            if dropped:
                log.info(
                    "Achats : %d ligne(s) ignorée(s) — commande annulée, non "
                    "livrée ou statut inconnu (un achat annulé n'est pas un achat)",
                    dropped,
                )
        # `created_at` est l'horodatage de la commande, pas celui de la ligne
        # produit. Il faut le lire AVANT `_normalise`, qui ne conserve que les
        # colonnes EVENT_COLUMNS : lu après, il a disparu et tous les achats
        # se retrouvaient avec un ts NaT — donc triés en dernier et ignorés par
        # la validation temporelle.
        order_ts = (
            _coerce_ts(merged["created_at"]) if "created_at" in merged.columns
            else pd.Series(pd.NaT, index=merged.index)
        )
        merged = _normalise(merged, "purchase", "order_items")
        if len(merged):
            merged["ts"] = order_ts.reindex(merged.index)
            n_na = int(merged["ts"].isna().sum())
            if n_na:
                log.warning(
                    "Achats : %d ligne(s) sans created_at exploitable", n_na,
                )
            log.info("Achats valides : %d lignes", len(merged))
            frames.append(merged)

    if panier is not None:
        f = _normalise(panier, "cart", "panier")
        if len(f):
            frames.append(f)

    if visits is not None:
        f = _normalise(visits, "view", "visits")
        if len(f):
            frames.append(f)

    # ── Avis : review_score attachée à la commande ET à ses articles ─────────────
    # `reviews.csv` ne porte qu'un `order_id` : sans la jointure aux lignes de
    # commande, l'avis n'a aucun product_id et était systématiquement jeté
    # (le seul avis du jeu de données actuel). Un avis portant sur une commande
    # de plusieurs articles devient un événement par article — c'est la seule
    # lecture honnête, on ne sait pas quel article précis a été noté.
    if reviews is not None and orders is not None and "order_id" in reviews.columns:
        r = reviews.merge(
            orders[["order_id", "customer_id", "order_status"]], on="order_id", how="left",
        )
        status = r.get("order_status")
        if status is not None:
            r = r[status.astype(str).str.strip().str.lower() == "valide"]
        if items is not None and "order_id" in items.columns and "product_id" in items.columns:
            bought = items[["order_id", "product_id"]].drop_duplicates()
            r = r.merge(bought, on="order_id", how="left")
            log.info(
                "Avis : %d ligne(s) rattachée(s) à %d article(s) de commande",
                len(r), r["product_id"].nunique(),
            )
        else:
            log.warning("Avis : order_items.csv absent — avis non rattachable à un article")
        f = _normalise(r, "review", "reviews")
        if len(f):
            frames.append(f)

    if not frames:
        log.warning("Aucun événement produit exploitable dans %s", data_dir)
        return _empty_events()

    events = pd.concat(frames, ignore_index=True, sort=False)
    events["product_id"] = pd.to_numeric(events["product_id"], errors="coerce")
    events = events.dropna(subset=["product_id"])
    events["product_id"] = events["product_id"].astype("int64")

    if products is not None:
        known = set(pd.to_numeric(products["product_id"], errors="coerce").dropna().astype("int64"))
        unknown = ~events["product_id"].isin(known)
        if unknown.any():
            log.warning(
                "Événements : %d ligne(s) ignorée(s), product_id absent du catalogue",
                int(unknown.sum()),
            )
            events = events[~unknown]

    # Un événement sans user_id ET sans session_id n'est rattachable à personne.
    has_identity = events["user_id"].notna() | events["session_id"].astype(str).str.len().gt(0)
    n_orphan = int((~has_identity).sum())
    if n_orphan:
        log.warning("Événements : %d ligne(s) sans identité ni session, ignorée(s)", n_orphan)
    events = events[has_identity]

    events = events.sort_values("ts", na_position="last").reset_index(drop=True)
    log.info(
        "Flux d'événements : %d lignes | %d produits | %d users | %d sessions",
        len(events), events["product_id"].nunique(),
        events["user_id"].nunique(), events["session_id"].nunique(),
    )
    return events


def _dedup_keys(df: pd.DataFrame) -> pd.Series:
    """Clé d'identité d'un événement, utilisée par append_events()."""
    def norm(col):
        return df[col].fillna("").astype(str).str.strip()

    return (
        norm("user_id") + "|" + norm("session_id") + "|" + norm("product_id")
        + "|" + norm("event_type") + "|" + df["ts"].astype(str)
    )


def append_events(data_dir: Path | str, rows: list[dict]) -> int:
    """
    Ajoute des événements collectés par l'API au flux canonique.

    Écriture append + verrou : l'API reçoit des requêtes concurrent pendant que
    l'entraînement lit le fichier. Retourne le nombre de lignes écrites.

    DÉDUPLICATION : un front qui rejoue un lot déjà enregistré (retry,
    double-clic, React StrictMode) ne doit pas gonfler les compteurs ni les
    métriques. La comparaison se fait donc contre le flux DÉJÀ ENREGISTRÉ, et
    non entre les événements du lot lui-même : un lot qui contient deux vues
    légitimes du même article à la même seconde dit deux choses arrivées, et les
    écraser serait perdre de la donnée. La clé est
    (user_id, session_id, product_id, event_type, ts).

    Un rejeu du même lot reste protégé : le premier envoi écrit, le second trouve
    ses lignes dans le fichier et n'écrit rien.
    """
    if not rows:
        return 0
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / _EVENTS_FILE

    frame = pd.DataFrame(rows)
    for col in _EVENTS_COLUMNS_FILE:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[_EVENTS_COLUMNS_FILE]
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", format="mixed")
    frame = frame.dropna(subset=["ts"])

    with _IO_LOCK:
        # Relire l'existant est nécessaire : la déduplication ne peut pas se
        # faire en mémoire, l'API est un processus à part de l'entraînement.
        existing = _read(path)
        if existing is not None and len(existing):
            # Uniquement contre le flux déjà enregistré : deux lignes identiques
            # dans le MÊME lot sont deux événements, pas un rejeu.
            seen = set(_dedup_keys(existing))
            duplicated = _dedup_keys(frame).isin(seen)
            n_dup = int(duplicated.sum())
            if n_dup:
                log.info("Collecte : %d événement(s) déjà présents, ignoré(s)", n_dup)
                frame = frame[~duplicated]
            if not len(frame):
                return 0

        header = not path.exists() or path.stat().st_size == 0
        frame.to_csv(path, mode="a", header=header, index=False)
    return len(frame)


def product_features(products: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la table de features article.

    Colonnes : product_id, name, category, available, price, n_views, n_carts,
    n_purchases, rating, signal.

    Le prix vient des lignes de commande VALIDES (médiane par article, les prix
    variant légèrement d'une commande à l'autre). Les produits jamais vendus
    n'ont pas de prix : on laisse NaN, pas 0 — un prix manquant n'est pas un
    article gratuit.
    """
    prod = products.copy()
    prod["product_id"] = pd.to_numeric(prod["product_id"], errors="coerce")
    prod = prod.dropna(subset=["product_id"])
    prod["product_id"] = prod["product_id"].astype("int64")
    prod = prod.drop_duplicates("product_id")

    name_col = "product_name" if "product_name" in prod.columns else None
    feat = pd.DataFrame({
        "product_id": prod["product_id"].values,
        "name": (prod[name_col].fillna("").astype(str).str.strip()
                 if name_col else ""),
        "category": (prod["product_category_name"].fillna("").astype(str)
                     if "product_category_name" in prod.columns else ""),
        "available": (pd.to_numeric(prod["available"], errors="coerce").fillna(1).astype(int)
                      if "available" in prod.columns else 1),
    })

    if len(events):
        ev = events.copy()
        ev["user_id"] = pd.to_numeric(ev["user_id"], errors="coerce")
        ev = ev.dropna(subset=["user_id"])  # panier/achat : il faut un client

        for etype in ("view", "cart", "purchase"):
            counts = (ev[ev["event_type"] == etype]
                      .groupby("product_id").size()
                      .rename(f"n_{etype}s"))
            feat = feat.merge(counts, left_on="product_id", right_index=True, how="left")

        prices = (ev[ev["event_type"] == "purchase"].dropna(subset=["price"])
                  .groupby("product_id")["price"].median())
        feat = feat.merge(prices.rename("price"), left_on="product_id",
                          right_index=True, how="left")

        ratings = (ev[ev["event_type"] == "review"].dropna(subset=["event_value"])
                   .groupby("product_id")["event_value"].mean())
        feat = feat.merge(ratings.rename("rating"), left_on="product_id",
                          right_index=True, how="left")
    else:
        for col in ("n_views", "n_carts", "n_purchases", "price", "rating"):
            feat[col] = np.nan

    for col in ("n_views", "n_carts", "n_purchases"):
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = feat[col].fillna(0.0).astype(float)

    feat["signal"] = (
        feat["n_views"] * EVENT_WEIGHTS["view"]
        + feat["n_carts"] * EVENT_WEIGHTS["cart"]
        + feat["n_purchases"] * EVENT_WEIGHTS["purchase"]
    )
    feat["category"] = feat["category"].str.strip()
    feat["name"] = feat["name"].str.strip()
    return feat.reset_index(drop=True)


_WS = re.compile(r"\s+")
_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")


def _normalise_name(text: str) -> str:
    """Minuscules, sans accents, espaces compactés : pour la TF-IDF."""
    text = str(text).lower().translate(_ACCENTS)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _WS.sub(" ", text).strip()


def build_product_similarity(feat: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    """
    Similarité cosinus entre articles, sur le NOM du produit.

    Les noms MEL sont des titres descriptifs réels (« Robe Africaine Spéciale 8
    Mars », « Chemise Manches Longues à Motif Chevrons ») : la TF-IDF y capte un
    vrai sens, là où elle ne captait qu'un mot générique sur les noms de
    catégories. C'est ce qui rend le content-based par article possible avec les
    données actuelles.

    Le prix est deliberately exclu de la similarité : deux articles de 500 et
    5000 FCFA ne sont pas « similaires » sur cette base, et l'inclure pushed
    les robes chères à proximité des robes bon marché pour une simple raison de
    prix — un biais de catalogue, pas une affinité de goût.

    Retourne (liste ordonnée des product_id, matrice cosinus NxN).
    """
    if feat is None or not len(feat) or "name" not in feat.columns:
        log.warning("Aucun titre produit exploitable : pas de similarité article")
        return [], np.zeros((0, 0))

    # astype(str) avant .str.len() : sur un DataFrame vide la colonne est
    # typée float/object et l'accès .str lève une AttributeError.
    names = feat["name"].fillna("").astype(str).str.strip()
    usable = feat[names.str.len() > 0].reset_index(drop=True)
    if len(usable) < 2:
        log.warning("Titres produits insuffisants (%d) : pas de similarité article",
                    len(usable))
        return [], np.zeros((0, 0))

    texts = [_normalise_name(t) for t in usable["name"]]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return usable["product_id"].tolist(), sim


def data_sufficiency_report(events: pd.DataFrame, feat: pd.DataFrame | None = None) -> dict:
    """
    Mesure ce que les données permettent réellement, et se prononce sur ce qu'elles
    ne permettent pas.

    Le collaboratif par article a besoin d'un volume par couple (client, article).
    En dessous, ALS/SVD produisent un modèle qui a l'airRotationnel mais classe
    au hasard — et dont les métriques flatteuses sur 14 clients donnent une fausse
    confiance. On affiche donc un verdict explicite plutôt que de laisser croire
    que le modèle fonctionne.
    """
    n_events = len(events)
    ev = events.copy()
    ev["user_id"] = pd.to_numeric(ev.get("user_id"), errors="coerce")
    identified = ev.dropna(subset=["user_id"])

    n_users = int(identified["user_id"].nunique())
    n_products = int(events["product_id"].nunique()) if n_events else 0
    n_pairs = len(identified[["user_id", "product_id"]].drop_duplicates())

    # Le collaboratif par article ne se construit que sur des clients identifiés.
    # Compter les sessions anonymes dans le volume donnerait un «Volume » flatteur
    # alors que ni n_users ni n_pairs ne progressent : le seuil pourrait être
    # atteint sur le papier et le modèle rester non activable.
    n_events_identified = len(identified)
    n_events_session_only = n_events - n_events_identified

    # Seuils prudents pour un ALS item-item : en dessous, le bruit domine.
    CF_MIN_EVENTS = 500
    CF_MIN_USERS = 100
    CF_MIN_PAIRS = 1000

    cf_ready = (
        n_events_identified >= CF_MIN_EVENTS
        and n_users >= CF_MIN_USERS
        and n_pairs >= CF_MIN_PAIRS
    )

    # Le content-based n'a pas ce problème : il n'a besoin d'aucun historique.
    cb_ready = n_products > 0

    if cf_ready:
        cf_verdict = "collaboratif par article activable"
    else:
        manque = []
        if n_events_identified < CF_MIN_EVENTS:
            manque.append(
                f"{n_events_identified}/{CF_MIN_EVENTS} événements identifiés"
            )
        if n_users < CF_MIN_USERS:
            manque.append(f"{n_users}/{CF_MIN_USERS} clients identifiés")
        if n_pairs < CF_MIN_PAIRS:
            manque.append(f"{n_pairs}/{CF_MIN_PAIRS} couples (client, article)")
        cf_verdict = "insuffisant — " + ", ".join(manque)

    by_type = (
        events["event_type"].value_counts().to_dict() if n_events else {}
    )

    report = {
        "n_events": n_events,
        "n_events_identified": n_events_identified,
        "n_events_session_only": n_events_session_only,
        "n_users_identified": n_users,
        "n_products": n_products,
        "n_user_product_pairs": n_pairs,
        "events_by_type": {str(k): int(v) for k, v in by_type.items()},
        "cf_thresholds": {
            "min_events": CF_MIN_EVENTS,
            "min_users": CF_MIN_USERS,
            "min_pairs": CF_MIN_PAIRS,
        },
        "collaborative_ready": cf_ready,
        "collaborative_verdict": cf_verdict,
        "content_based_ready": cb_ready,
    }

    if feat is not None and len(feat):
        report["n_products_with_price"] = int(feat["price"].notna().sum())
        report["n_products_with_rating"] = int(feat["rating"].notna().sum())
        report["catalogue_coverage_pct"] = round(100.0 * n_products / len(feat), 1)

    log.info(
        "Suffisance : %d événements (%d identifiés, %d sessions seules), "
        "%d clients, %d produits, %d couples | CF article : %s",
        n_events, n_events_identified, n_events_session_only,
        n_users, n_products, n_pairs, cf_verdict,
    )
    return report