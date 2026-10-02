"""
scripts/mysql_export.py
Export direct depuis la base MySQL MEL Cameroun → fichiers CSV pour train.py.

Contrairement à sql_to_csv.py (qui parse un dump .sql statique), ce script
se connecte en direct à la BDD de production et exporte les données en temps réel.
À utiliser en production quand MYSQL_HOST est défini.

Usage :
    python scripts/mysql_export.py \\
        --host     localhost \\
        --port     3306 \\
        --database mel_cameroun \\
        --user     mel_user \\
        --password secret \\
        --out      data/raw

Variables d'environnement (alternative aux arguments) :
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD, DATA_DIR

Dépendances :
    pip install pymysql  (ou mysqlclient)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Connexion
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(host: str, port: int, database: str, user: str, password: str):
    """Retourne une connexion MySQL. Essaie pymysql puis mysqlclient."""
    try:
        import pymysql
        conn = pymysql.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
        )
        log.info("Connexion MySQL via pymysql — %s:%d/%s", host, port, database)
        return conn
    except ImportError:
        pass

    try:
        import MySQLdb
        conn = MySQLdb.connect(
            host=host,
            port=port,
            db=database,
            user=user,
            passwd=password,
            charset="utf8mb4",
            connect_timeout=10,
        )
        log.info("Connexion MySQL via mysqlclient — %s:%d/%s", host, port, database)
        return conn
    except ImportError:
        pass

    log.error(
        "Aucun driver MySQL trouvé. Installer avec : pip install pymysql"
    )
    sys.exit(1)


def query_to_df(conn, sql: str, params=None):
    """Exécute une requête SQL et retourne un DataFrame pandas."""
    import pandas as pd
    with conn.cursor() as cursor:
        cursor.execute(sql, params or ())
        rows = cursor.fetchall()
    # pymysql retourne des dicts, mysqlclient retourne des tuples
    if rows and isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    # Pour mysqlclient : récupère les noms de colonnes
    with conn.cursor() as cursor:
        cursor.execute(sql, params or ())
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction JSON multilang
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_fr(value, fallback: str = "inconnu") -> str:
    """Extrait la valeur française d'un champ JSON multilang MEL."""
    if value is None:
        return fallback
    try:
        s = value if isinstance(value, str) else str(value)
        s = s.replace('\\"', '"')
        d = json.loads(s)
        if isinstance(d, dict):
            return d.get("fr", d.get("en", next(iter(d.values()), fallback)))
        return str(d)
    except Exception:
        return str(value).strip('"').strip("'") or fallback


# ─────────────────────────────────────────────────────────────────────────────
# Exports par table
# ─────────────────────────────────────────────────────────────────────────────

def export_orders(conn, out_dir: Path):
    """
    Export factures → orders.csv + order_items.csv
    Chaque facture correspond à UN article (marketplace d'occasion).
    """
    import pandas as pd

    log.info("Export factures...")
    df = query_to_df(conn, """
        SELECT
            f.id            AS order_id,
            f.acheteur_id   AS customer_id,
            f.article_id    AS product_id,
            f.montant_total AS price,
            f.status        AS order_status,
            f.created_at    AS created_at
        FROM factures f
        WHERE f.acheteur_id IS NOT NULL
          AND f.article_id  IS NOT NULL
        ORDER BY f.created_at ASC
    """)

    if df.empty:
        log.warning("Aucune facture trouvée")
        return

    # orders.csv
    orders = df[["order_id", "customer_id", "order_status", "created_at"]].copy()
    orders.to_csv(out_dir / "orders.csv", index=False)
    log.info("orders.csv : %d lignes", len(orders))

    # order_items.csv
    items = df[["order_id", "product_id", "price"]].copy()
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
    items.to_csv(out_dir / "order_items.csv", index=False)
    log.info("order_items.csv : %d lignes", len(items))


def export_products(conn, out_dir: Path):
    """
    Export articles + categories + article_categories → products.csv
    Inclut product_name, product_category_name, available.
    """
    import pandas as pd

    log.info("Export articles...")

    # Articles avec leur première catégorie et statut de vente
    df = query_to_df(conn, """
        SELECT
            a.id                    AS product_id,
            a.nom                   AS product_name_raw,
            a.prix                  AS product_weight_g,
            a.image                 AS image_json,
            c.nom                   AS category_name_raw,
            -- article vendu = facture valide existe
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM factures f
                    WHERE f.article_id = a.id
                      AND LOWER(f.status) IN ('valide','delivered','completed')
                ) THEN 0
                ELSE 1
            END                     AS available
        FROM articles a
        LEFT JOIN article_categories ac ON ac.article_id = a.id
        LEFT JOIN categories c          ON c.id = ac.categorie_id
        GROUP BY a.id
        ORDER BY a.id ASC
    """)

    if df.empty:
        log.warning("Aucun article trouvé")
        return

    # Décode les champs JSON
    df["product_name"]          = df["product_name_raw"].apply(
        lambda v: extract_json_fr(v, "Article inconnu")
    )
    df["product_category_name"] = df["category_name_raw"].apply(
        lambda v: extract_json_fr(v, "inconnu")
    )

    # Nombre de photos
    def count_photos(img_json):
        if not img_json:
            return 0
        try:
            imgs = json.loads(img_json) if isinstance(img_json, str) else img_json
            if isinstance(imgs, list):
                return len(imgs)
            if isinstance(imgs, dict):
                return len(imgs)
            return 1
        except Exception:
            return 1

    df["product_photos_qty"] = df["image_json"].apply(count_photos)

    result = df[[
        "product_id", "product_name", "product_category_name",
        "product_weight_g", "product_photos_qty", "available"
    ]].copy()

    n_available = int(result["available"].sum())
    n_sold      = len(result) - n_available
    log.info(
        "products.csv : %d articles (%d disponibles, %d vendus)",
        len(result), n_available, n_sold
    )
    result.to_csv(out_dir / "products.csv", index=False)


def export_reviews(conn, out_dir: Path):
    """Export etoiles → reviews.csv"""
    log.info("Export avis...")
    df = query_to_df(conn, """
        SELECT
            e.id     AS order_id,
            e.valeur AS review_score
        FROM etoiles e
        WHERE e.valeur IS NOT NULL
    """)
    if df.empty:
        log.warning("Aucun avis trouvé")
        return
    df.to_csv(out_dir / "reviews.csv", index=False)
    log.info("reviews.csv : %d lignes", len(df))


def export_category_names(conn, out_dir: Path):
    """Export categories → category_names.csv"""
    log.info("Export catégories...")
    df = query_to_df(conn, "SELECT id, nom FROM categories ORDER BY id")
    if df.empty:
        log.warning("Aucune catégorie trouvée")
        return
    df["product_category_name"]         = df["nom"].apply(extract_json_fr)
    df["product_category_name_english"] = df["product_category_name"]
    result = df[["product_category_name", "product_category_name_english"]].drop_duplicates()
    result.to_csv(out_dir / "category_names.csv", index=False)
    log.info("category_names.csv : %d lignes", len(result))


def export_panier_interactions(conn, out_dir: Path):
    """Export paniers → panier_interactions.csv"""
    log.info("Export paniers...")
    df = query_to_df(conn, """
        SELECT
            p.user_id    AS customer_id,
            p.article_id AS product_id,
            p.created_at AS timestamp
        FROM paniers p
        WHERE p.user_id    IS NOT NULL
          AND p.article_id IS NOT NULL
        ORDER BY p.created_at ASC
    """)
    if df.empty:
        log.warning("Aucune interaction panier trouvée")
        return
    df.to_csv(out_dir / "panier_interactions.csv", index=False)
    log.info("panier_interactions.csv : %d lignes", len(df))


def export_visits(conn, out_dir: Path):
    """Export shetabit_visits → visits.csv (visites d'articles par users connectés)"""
    log.info("Export visites...")
    df = query_to_df(conn, """
        SELECT
            v.visitor_id  AS customer_id,
            v.visitable_id AS product_id,
            v.created_at  AS timestamp
        FROM shetabit_visits v
        WHERE v.visitable_type LIKE '%Article%'
          AND v.visitor_id  IS NOT NULL
          AND v.visitable_id IS NOT NULL
        ORDER BY v.created_at ASC
    """)
    if df.empty:
        log.warning("Aucune visite d'article trouvée")
        return
    import pandas as pd
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["product_id"]  = pd.to_numeric(df["product_id"],  errors="coerce")
    df = df.dropna(subset=["customer_id", "product_id"])
    df.to_csv(out_dir / "visits.csv", index=False)
    log.info("visits.csv : %d lignes", len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export direct MySQL MEL → CSV pour train.py"
    )
    parser.add_argument("--host",     default=os.environ.get("MYSQL_HOST", "localhost"))
    parser.add_argument("--port",     type=int, default=int(os.environ.get("MYSQL_PORT", "3306")))
    parser.add_argument("--database", default=os.environ.get("MYSQL_DB", "mel_cameroun"))
    parser.add_argument("--user",     default=os.environ.get("MYSQL_USER", "mel_user"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD", ""))
    parser.add_argument("--out",      default=os.environ.get("DATA_DIR", "data/raw"))
    return parser.parse_args()


def main():
    args   = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Export MySQL → %s", out_dir)
    log.info("Connexion : %s@%s:%d/%s", args.user, args.host, args.port, args.database)

    conn = get_connection(args.host, args.port, args.database, args.user, args.password)

    try:
        export_orders(conn, out_dir)
        export_products(conn, out_dir)
        export_reviews(conn, out_dir)
        export_category_names(conn, out_dir)
        export_panier_interactions(conn, out_dir)
        export_visits(conn, out_dir)
    finally:
        conn.close()

    log.info("✅ Export terminé → %s", out_dir)
    for f in sorted(out_dir.glob("*.csv")):
        size = f.stat().st_size
        log.info("  %-35s %8d octets", f.name, size)


if __name__ == "__main__":
    main()
