"""
scripts/sql_to_csv.py
Convertit le dump MySQL MEL Cameroun en fichiers CSV compatibles avec src/train.py.

Usage :
    python scripts/sql_to_csv.py
    python scripts/sql_to_csv.py --sql data/raw/mysql-mel.sql --out data/raw
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parser SQL — approche ligne par ligne
# ─────────────────────────────────────────────────────────────────────────────

def parse_table(lines: list[str], table_name: str) -> pd.DataFrame:
    """
    Extrait les données d'une table depuis les lignes du dump SQL.
    Chaque ligne de données est de la forme :
        (val1,val2,...),
    """

    # 1. Colonnes depuis CREATE TABLE
    columns = _extract_columns(lines, table_name)
    if not columns:
        log.warning("Table '%s' : impossible d'extraire les colonnes.", table_name)
        return pd.DataFrame()

    # 2. Lignes de données — on cherche les blocs INSERT INTO `table` VALUES
    rows = []
    in_insert = False

    for line in lines:
        line = line.rstrip()

        if re.match(rf"^INSERT INTO `{table_name}` VALUES", line):
            in_insert = True
            # La première ligne peut contenir des données après VALUES
            rest = re.sub(rf"^INSERT INTO `{table_name}` VALUES\s*", "", line)
            if rest:
                _extract_rows_from_line(rest, columns, rows)
            continue

        if in_insert:
            # Fin du bloc INSERT
            if line.startswith("UNLOCK TABLES") or line.startswith("/*!") or \
               line.startswith("--") or (line == "" and not rows):
                in_insert = False
                continue
            _extract_rows_from_line(line, columns, rows)
            # Un ; seul sur une ligne termine l'INSERT
            if line.strip() == ";":
                in_insert = False

    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    log.info("Table '%s' : %d lignes extraites.", table_name, len(df))
    return df


def _extract_columns(lines: list[str], table_name: str) -> list[str]:
    """Extrait les noms de colonnes depuis CREATE TABLE."""
    in_create = False
    columns = []

    for line in lines:
        if re.match(rf"^CREATE TABLE `{table_name}`", line):
            in_create = True
            continue
        if in_create:
            # Colonne : `nom` type ...
            m = re.match(r"^\s+`(\w+)`\s+", line)
            if m:
                col = m.group(1)
                columns.append(col)
            # Fin du CREATE TABLE
            if line.startswith(") ENGINE"):
                break

    return columns


def _extract_rows_from_line(line: str, columns: list, rows: list):
    """
    Parse les tuples d'une ligne du type :
        (v1,v2,...),(v1,v2,...),
    Gère les chaînes avec guillemets simples, JSON imbriqués, NULL, etc.
    """
    # On itère caractère par caractère pour trouver les tuples de haut niveau
    i = 0
    n = len(line)

    while i < n:
        # Cherche le début d'un tuple
        if line[i] == "(":
            # Extrait le contenu du tuple (gère parenthèses imbriquées dans strings)
            content, end = _extract_tuple_content(line, i + 1)
            if content is not None:
                values = _parse_values(content)
                if len(values) == len(columns):
                    rows.append(values)
                elif len(values) > 0:
                    # Padding si légèrement différent (colonnes optionnelles)
                    if len(values) < len(columns):
                        values += [None] * (len(columns) - len(values))
                    rows.append(values[:len(columns)])
            i = end + 1
        else:
            i += 1


def _extract_tuple_content(s: str, start: int) -> tuple[str | None, int]:
    """
    À partir de start (après le '('), extrait le contenu jusqu'au ')' fermant
    de haut niveau, en respectant les chaînes SQL entre guillemets simples.
    Retourne (contenu, index_du_paren_fermant).
    """
    depth = 1  # on est déjà à l'intérieur du premier (
    i = start
    n = len(s)
    in_string = False
    escape_next = False

    while i < n:
        c = s[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if c == "'" and not in_string:
            in_string = True
        elif c == "'" and in_string:
            # Guillemet doublé '' = guillemet échappé
            if i + 1 < n and s[i + 1] == "'":
                i += 2
                continue
            in_string = False
        elif not in_string:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return s[start:i], i

        i += 1

    return None, n


def _parse_values(content: str) -> list:
    """
    Parse les valeurs séparées par des virgules d'un tuple SQL.
    Respecte les guillemets simples et les virgules dans les chaînes.
    """
    values = []
    current = []
    in_string = False
    escape_next = False
    i = 0
    n = len(content)

    while i < n:
        c = content[i]

        if escape_next:
            current.append(c)
            escape_next = False
            i += 1
            continue

        if c == "\\" and in_string:
            current.append(c)
            escape_next = True
            i += 1
            continue

        if c == "'" and not in_string:
            in_string = True
            current.append(c)
        elif c == "'" and in_string:
            if i + 1 < n and content[i + 1] == "'":
                current.append("'")
                i += 2
                continue
            in_string = False
            current.append(c)
        elif c == "," and not in_string:
            values.append(_clean_value("".join(current).strip()))
            current = []
        else:
            current.append(c)

        i += 1

    # Dernière valeur
    if current or not values:
        values.append(_clean_value("".join(current).strip()))

    return values


def _clean_value(val: str):
    """Nettoie une valeur SQL brute."""
    if val.upper() == "NULL":
        return None
    if val.startswith("'") and val.endswith("'"):
        inner = val[1:-1]
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        inner = inner.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        return inner
    # Nombre entier
    try:
        return int(val)
    except ValueError:
        pass
    # Nombre flottant
    try:
        return float(val)
    except ValueError:
        pass
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Construction des CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_orders(factures: pd.DataFrame, out_dir: Path):
    orders = factures[["id", "acheteur_id", "status", "created_at"]].copy()
    orders.columns = ["order_id", "customer_id", "order_status", "created_at"]
    orders.to_csv(out_dir / "orders.csv", index=False)
    log.info("orders.csv : %d lignes", len(orders))

    order_items = factures[["id", "article_id", "montant_total"]].copy()
    order_items.columns = ["order_id", "product_id", "price"]
    order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce").fillna(0)
    order_items.to_csv(out_dir / "order_items.csv", index=False)
    log.info("order_items.csv : %d lignes", len(order_items))


def build_products(articles: pd.DataFrame, article_categories: pd.DataFrame,
                   categories: pd.DataFrame, factures: pd.DataFrame, out_dir: Path):
    # Première catégorie par article
    art_cat = article_categories.copy()
    art_cat["article_id"] = pd.to_numeric(art_cat["article_id"], errors="coerce")
    art_cat["categorie_id"] = pd.to_numeric(art_cat["categorie_id"], errors="coerce")
    art_cat = art_cat.groupby("article_id")["categorie_id"].first().reset_index()
    art_cat.columns = ["product_id", "categorie_id"]

    # Nom de catégorie (JSON multilang)
    def extract_cat_name(nom_json):
        if nom_json is None:
            return "inconnu"
        try:
            s = nom_json if isinstance(nom_json, str) else str(nom_json)
            # Le parser SQL peut laisser des \" au lieu de "
            s = s.replace('\\"', '"')
            d = json.loads(s)
            if isinstance(d, dict):
                return d.get("fr", d.get("en", next(iter(d.values()), "inconnu")))
            return str(d)
        except Exception:
            # Tente de lire directement si c'est déjà une chaîne propre
            return str(nom_json).strip('"').strip("'")

    categories = categories.copy()
    categories["id"] = pd.to_numeric(categories["id"], errors="coerce")
    categories["category_name"] = categories["nom"].apply(extract_cat_name)
    cat_map = categories.set_index("id")["category_name"].to_dict()
    art_cat["product_category_name"] = art_cat["categorie_id"].map(cat_map).fillna("inconnu")

    articles = articles.copy()
    articles["id"] = pd.to_numeric(articles["id"], errors="coerce")

    # Nombre de photos
    def count_photos(img_json):
        if img_json is None:
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

    # Nom de l'article (JSON multilang ou texte brut)
    def extract_nom(nom_json):
        if nom_json is None:
            return ""
        try:
            s = nom_json if isinstance(nom_json, str) else str(nom_json)
            s = s.replace('\\"', '"')
            d = json.loads(s)
            if isinstance(d, dict):
                return d.get("fr", d.get("en", next(iter(d.values()), "")))
            return str(d)
        except Exception:
            return str(nom_json).strip('"').strip("'")

    # Déterminer les articles vendus (facture avec status 'valide')
    # Sur une marketplace d'occasion, un article vendu n'est plus disponible
    sold_ids: set = set()
    if not factures.empty and "article_id" in factures.columns and "status" in factures.columns:
        factures_cp = factures.copy()
        factures_cp["article_id"] = pd.to_numeric(factures_cp["article_id"], errors="coerce")
        valid_statuses = {"valide", "delivered", "completed"}
        sold_mask = factures_cp["status"].str.lower().isin(valid_statuses)
        sold_ids = set(factures_cp.loc[sold_mask, "article_id"].dropna().astype(int))
        log.info("Articles vendus (non disponibles) : %d", len(sold_ids))

    products = articles[["id", "prix"]].copy()
    products.columns = ["product_id", "product_weight_g"]
    products["product_photos_qty"] = articles["image"].apply(count_photos)

    # Nom du produit — champ 'nom' de la table articles
    if "nom" in articles.columns:
        products["product_name"] = articles["nom"].apply(extract_nom)
    else:
        products["product_name"] = ""

    products = products.merge(art_cat[["product_id", "product_category_name"]],
                              on="product_id", how="left")
    products["product_category_name"] = products["product_category_name"].fillna("inconnu")

    # Disponibilité : 1 = disponible (pas encore vendu), 0 = vendu
    products["available"] = products["product_id"].apply(
        lambda pid: 0 if int(pid) in sold_ids else 1
    )

    n_available = products["available"].sum()
    log.info("products.csv : %d articles total, %d disponibles, %d vendus",
             len(products), n_available, len(products) - n_available)

    products = products[["product_id", "product_name", "product_category_name",
                          "product_weight_g", "product_photos_qty", "available"]]
    products.to_csv(out_dir / "products.csv", index=False)
    log.info("products.csv : %d lignes", len(products))


def build_reviews(etoiles: pd.DataFrame, out_dir: Path):
    reviews = etoiles[["id", "valeur"]].copy()
    reviews.columns = ["order_id", "review_score"]
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    reviews.to_csv(out_dir / "reviews.csv", index=False)
    log.info("reviews.csv : %d lignes", len(reviews))


def build_category_names(categories: pd.DataFrame, out_dir: Path):
    def extract_name(nom_json):
        if nom_json is None:
            return "inconnu"
        try:
            s = nom_json if isinstance(nom_json, str) else str(nom_json)
            s = s.replace('\\"', '"')
            d = json.loads(s)
            if isinstance(d, dict):
                return d.get("fr", d.get("en", next(iter(d.values()), "inconnu")))
            return str(d)
        except Exception:
            return str(nom_json).strip('"').strip("'")

    cat_names = categories[["nom"]].copy()
    cat_names["product_category_name"] = cat_names["nom"].apply(extract_name)
    cat_names["product_category_name_english"] = cat_names["product_category_name"]
    cat_names = cat_names[["product_category_name", "product_category_name_english"]]
    cat_names.to_csv(out_dir / "category_names.csv", index=False)
    log.info("category_names.csv : %d lignes", len(cat_names))


def build_panier_interactions(paniers: pd.DataFrame, out_dir: Path):
    paniers_out = paniers[["user_id", "article_id", "created_at"]].copy()
    paniers_out.columns = ["customer_id", "product_id", "timestamp"]
    paniers_out.to_csv(out_dir / "panier_interactions.csv", index=False)
    log.info("panier_interactions.csv : %d lignes", len(paniers_out))


def build_visits(visits: pd.DataFrame, out_dir: Path):
    """
    shetabit_visits → visits.csv
    Filtre uniquement les visites sur des articles (visitable_type = App\\Models\\Article)
    par des utilisateurs connectés (visitor_id non null).
    """
    if visits.empty:
        log.warning("Pas de données dans 'shetabit_visits' → visits.csv non généré")
        return

    v = visits.copy()
    # Garde uniquement les visites d'articles par des users connectés
    mask = (
        v["visitable_type"].astype(str).str.contains("Article", na=False) &
        v["visitor_id"].notna() &
        v["visitable_id"].notna()
    )
    v = v[mask][["visitor_id", "visitable_id", "created_at"]].copy()
    v.columns = ["customer_id", "product_id", "timestamp"]
    v["customer_id"] = pd.to_numeric(v["customer_id"], errors="coerce")
    v["product_id"]  = pd.to_numeric(v["product_id"],  errors="coerce")
    v = v.dropna(subset=["customer_id", "product_id"])
    v.to_csv(out_dir / "visits.csv", index=False)
    log.info("visits.csv : %d lignes (sur %d visites totales)", len(v), len(visits))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convertit mysql-mel.sql en CSVs pour train.py")
    parser.add_argument("--sql", default="data/raw/mysql-mel.sql")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    sql_path = Path(args.sql)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sql_path.exists():
        raise FileNotFoundError(f"Dump SQL introuvable : {sql_path}")

    log.info("Lecture de %s (%.1f Mo)…", sql_path, sql_path.stat().st_size / 1e6)
    lines = sql_path.read_text(encoding="utf-8", errors="replace").splitlines()
    log.info("%d lignes lues.", len(lines))

    log.info("Extraction des tables…")
    factures           = parse_table(lines, "factures")
    articles           = parse_table(lines, "articles")
    article_categories = parse_table(lines, "article_categories")
    paniers            = parse_table(lines, "paniers")
    etoiles            = parse_table(lines, "etoiles")
    categories         = parse_table(lines, "categories")
    visits             = parse_table(lines, "shetabit_visits")

    log.info("Génération des CSVs…")

    if not factures.empty:
        build_orders(factures, out_dir)
    else:
        log.warning("Pas de données dans 'factures' → orders.csv non généré")

    if not articles.empty:
        build_products(articles, article_categories, categories, factures, out_dir)
    else:
        log.warning("Pas de données dans 'articles' → products.csv non généré")

    if not etoiles.empty:
        build_reviews(etoiles, out_dir)
    else:
        log.warning("Pas de données dans 'etoiles' → reviews.csv non généré")

    if not categories.empty:
        build_category_names(categories, out_dir)

    if not paniers.empty:
        build_panier_interactions(paniers, out_dir)

    build_visits(visits, out_dir)

    log.info("✅ Conversion terminée → %s", out_dir)
    for f in sorted(out_dir.glob("*.csv")):
        log.info("  %-35s %8d octets", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
