"""
scripts/generate_fake_data.py
Génère un jeu de données synthétique MEL dans data/raw/ (schéma du loader).

Pourquoi ce script existe : en CI, data/raw/ est gitignoré (données client), donc
`python src/train.py` ne pouvait pas s'exécuter. Le step de training en CI était
donc wrappé dans `continue-on-error: true`, ce qui masquait toute régression du
pipeline. Avec ce générateur, la CI entraîne réellement le pipeline de bout en
bout et peut échouer bruyamment.

Aucune donnée réelle n'est utilisée : tout est tiré d'un RNG à graine fixe, donc
le run est déterministe.

Usage :
    python scripts/generate_fake_data.py [--data-dir data/raw] [--n-users 200]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    ("Vêtements femmes", "women_clothing"),
    ("Vêtements hommes", "men_clothing"),
    ("Chaussures femmes", "women_shoes"),
    ("Chaussures hommes", "men_shoes"),
    ("Accessoires", "accessories"),
    ("Ensembles femmes", "women_suits"),
    ("Cosmétiques", "cosmetics"),
    ("Télévisions", "television"),
    ("Ordinateurs", "computers"),
    ("Ustensiles cuisine", "kitchen_utensils"),
]
# Chaque catégorie a une "affinité" : les utilisateurs achètent surtout 2-3
# catégories cohérentes, ce qui donne un signal apprenable (et un jeu de test
# qui ressemble un peu à la réalité).
AFFINITY = {
    "Vêtements femmes":    ["Chaussures femmes", "Accessoires", "Ensembles femmes"],
    "Vêtements hommes":    ["Chaussures hommes", "Accessoires"],
    "Chaussures femmes":   ["Vêtements femmes", "Accessoires"],
    "Chaussures hommes":   ["Vêtements hommes", "Accessoires"],
    "Accessoires":         ["Vêtements femmes", "Vêtements hommes", "Cosmétiques"],
    "Ensembles femmes":    ["Vêtements femmes", "Chaussures femmes"],
    "Cosmétiques":         ["Accessoires"],
    "Télévisions":          ["Ordinateurs"],
    "Ordinateurs":         ["Télévisions"],
    "Ustensiles cuisine":  ["Télévisions"],
}


def generate(data_dir: Path, n_users: int = 200, n_products: int = 400,
             n_orders: int = 900, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    cat_names = [c for c, _ in CATEGORIES]
    en_map = dict(CATEGORIES)

    # ── products.csv ─────────────────────────────────────────────────────────
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "product_category_name": rng.choice(cat_names, size=n_products),
        "product_name": [f"Article {i:04d}" for i in range(1, n_products + 1)],
        "product_weight_g": rng.integers(50, 5000, size=n_products),
        "product_photos_qty": rng.integers(1, 6, size=n_products),
        # ~15 % de rupture, comme sur le vrai catalogue
        "available": rng.choice([0, 1], size=n_products, p=[0.15, 0.85]),
    })

    # ── orders.csv (statut + horodatage réel) ────────────────────────────────
    order_status = rng.choice(
        ["delivered", "shipped", "processing", "canceled"],
        size=n_orders, p=[0.55, 0.2, 0.15, 0.10],
    )
    base = pd.Timestamp("2024-01-01")
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": rng.integers(1, n_users + 1, size=n_orders),
        "order_status": order_status,
        # created_at est la colonne de date lue par build_interactions
        "created_at": base + pd.to_timedelta(
            np.sort(rng.integers(0, 400, size=n_orders)), unit="D"
        ),
    })

    # ── order_items.csv : chaque commande porte 1 à 3 articles ───────────────
    # Pas de colonne customer_id ici : build_interactions fusionne order_items
    # avec orders sur order_id, et une colonne homonyme serait suffixes en
    # customer_id_x / customer_id_y → KeyError au groupby.
    rows = []
    for oid in orders["order_id"]:
        # 2 catégories de base + affinités, puis tirage pondéré
        base_cats = rng.choice(cat_names, size=2, replace=False)
        pool = list(base_cats)
        for c in base_cats:
            pool.extend(AFFINITY.get(c, []))
        weights = np.array([3.0 if c in base_cats else 1.0 for c in pool])
        weights /= weights.sum()
        chosen = rng.choice(pool, size=rng.integers(1, 4), replace=False, p=weights)
        for cat in np.atleast_1d(chosen):
            prod_ids = products.loc[
                products["product_category_name"] == cat, "product_id"
            ].to_numpy()
            if prod_ids.size == 0:
                continue
            rows.append({
                "order_id": int(oid),
                "product_id": int(rng.choice(prod_ids)),
                "price": float(rng.integers(1000, 80000)),
            })
    order_items = pd.DataFrame(rows)

    # ── reviews.csv ──────────────────────────────────────────────────────────
    n_reviews = int(len(order_items) * 0.4)
    reviews = pd.DataFrame({
        "order_id": rng.choice(order_items["order_id"].to_numpy(), size=n_reviews),
        "review_score": rng.choice([1, 2, 3, 4, 5], size=n_reviews, p=[.05, .05, .1, .3, .5]),
    })

    # ── category_names.csv ───────────────────────────────────────────────────
    categories = pd.DataFrame({
        "product_category_name": cat_names,
        "product_category_name_english": [en_map[c] for c in cat_names],
    })

    written = {}
    for name, df in [
        ("products.csv", products),
        ("orders.csv", orders),
        ("order_items.csv", order_items),
        ("reviews.csv", reviews),
        ("category_names.csv", categories),
    ]:
        path = data_dir / name
        df.to_csv(path, index=False)
        written[name] = len(df)

    return written


def main():
    parser = argparse.ArgumentParser(description="Génère un dataset MEL synthétique")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--n-products", type=int, default=400)
    parser.add_argument("--n-orders", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    written = generate(
        args.data_dir, n_users=args.n_users,
        n_products=args.n_products, n_orders=args.n_orders, seed=args.seed,
    )
    print("Dataset synthétique écrit dans", args.data_dir)
    for name, n in written.items():
        print(f"  {name:22s} {n:>7d} lignes")


if __name__ == "__main__":
    main()