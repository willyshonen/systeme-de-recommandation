"""
Script pour injecter les cellules MEL dans le notebook recommendation_system.ipynb.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\Users\wilfried\Documents\mel-ml\notebooks\recommendation_system.ipynb")

# ── Helpers ──────────────────────────────────────────────────────────────────

_cell_counter = 0

def mk_md(lines):
    global _cell_counter
    _cell_counter += 1
    return {
        "cell_type": "markdown",
        "id": f"mel_md_{_cell_counter:04d}",
        "metadata": {},
        "source": lines,
    }


def mk_code(lines):
    global _cell_counter
    _cell_counter += 1
    return {
        "cell_type": "code",
        "id": f"mel_code_{_cell_counter:04d}",
        "metadata": {},
        "outputs": [],
        "source": lines,
        "execution_count": None,
    }


# ── Section 0.5 – Chargement SQL ─────────────────────────────────────────────

sec05_md = mk_md([
    "## 0.5 — Chargement du dataset MEL Cameroun (mysql-mel.sql)\n",
    "\n",
    "On parse le dump SQL directement en Python sans avoir besoin de MySQL installé.\n",
    "\n",
    "**Source :** `data/raw/mysql-mel.sql` — export de la base de production melcameroun.com  \n",
    "**Tables exploitées :**\n",
    "- `articles` — catalogue produits (~228 articles)\n",
    "- `factures` — historique d'achats (39 total, 25 valides)\n",
    "- `categories` + `article_categories` — catégorisation des articles\n",
    "- `paniers` — articles ajoutés au panier (~148 entrées, signal implicite)\n",
])

sec05_code = mk_code([
    "import re\n",
    "import json as _json\n",
    "\n",
    "SQL_PATH = DATA_DIR / 'mysql-mel.sql'\n",
    "\n",
    "\n",
    "def extract_table_to_df(sql_text, table_name, columns):\n",
    "    \"\"\"\n",
    "    Extrait une table du dump SQL MariaDB et retourne un DataFrame pandas.\n",
    "    Stratégie : regex pour trouver le bloc VALUES, puis csv pour parser les lignes.\n",
    "    \"\"\"\n",
    "    import csv, io\n",
    "\n",
    "    pattern = rf'INSERT INTO `{table_name}` VALUES\\n(.*?)\\n/\\*!40000'\n",
    "    match = re.search(pattern, sql_text, re.DOTALL)\n",
    "    if not match:\n",
    "        print(f'Table {table_name} non trouvée dans le SQL')\n",
    "        return pd.DataFrame(columns=columns)\n",
    "\n",
    "    values_text = match.group(1)\n",
    "\n",
    "    # Chaque ligne VALUES est séparée par ),\\n(\n",
    "    clean = values_text.strip()\n",
    "    clean = clean.lstrip('(')\n",
    "    # Supprimer le );  final\n",
    "    if clean.endswith(';'):\n",
    "        clean = clean[:-1]\n",
    "    if clean.endswith(')'):\n",
    "        clean = clean[:-1]\n",
    "\n",
    "    parts = re.split(r'\\),\\n\\(', clean)\n",
    "\n",
    "    rows = []\n",
    "    for part in parts:\n",
    "        part = part.strip().lstrip('(').rstrip(');')\n",
    "        try:\n",
    "            reader = csv.reader(io.StringIO(part), quotechar=\"'\")\n",
    "            for row in reader:\n",
    "                row = [None if str(v).strip() == 'NULL' else v for v in row]\n",
    "                rows.append(row)\n",
    "                break\n",
    "        except Exception:\n",
    "            continue\n",
    "\n",
    "    df = pd.DataFrame(rows)\n",
    "    if len(df.columns) >= len(columns):\n",
    "        df = df.iloc[:, :len(columns)]\n",
    "        df.columns = columns\n",
    "    else:\n",
    "        # Si mismatch, on tronque les colonnes\n",
    "        df.columns = columns[:len(df.columns)]\n",
    "\n",
    "    return df\n",
    "\n",
    "\n",
    "# Lire le fichier SQL complet en mémoire\n",
    "print('Chargement du fichier SQL MEL...')\n",
    "with open(SQL_PATH, encoding='utf-8') as f:\n",
    "    sql_text = f.read()\n",
    "\n",
    "print(f'Taille du fichier : {len(sql_text):,} caractères')\n",
    "print('OK — fichier SQL chargé en mémoire')\n",
])


# ── Section 1-MEL – DataFrames ────────────────────────────────────────────────

sec1mel_md = mk_md([
    "## 1-MEL — Construction des DataFrames MEL Cameroun\n",
    "\n",
    "On extrait les 4 tables clés du dump SQL et on construit des DataFrames pandas propres.\n",
])

sec1mel_articles = mk_code([
    "# ── 1.1 Articles ─────────────────────────────────────────────────────────────\n",
    "ARTICLES_COLS = [\n",
    "    'id', 'nom', 'marque', 'slug', 'couleur', 'taille',\n",
    "    'couleur_haut', 'taille_haut', 'couleur_pantalon', 'taille_pantalon',\n",
    "    'taille_porteur', 'garantie', 'ville', 'stockage',\n",
    "    'pixel_camera_avant', 'pixel_camera_arriere', 'ram', 'empreinte',\n",
    "    'cpu', 'model_cpu', 'systeme_exploitation',\n",
    "    'carte_graphique_dedie', 'model_carte_graphique_dedie',\n",
    "    'etat', 'taille_pouce', 'quantite_litre', 'type_cosmetique',\n",
    "    'quartier', 'essence', 'superficie', 'nombre_piece', 'nombre_chambre',\n",
    "    'nombre_salon', 'nombre_douche', 'nombre_cuisine',\n",
    "    'prix_minimal_enchere', 'prix', 'numero_direct', 'numero_whatsapp',\n",
    "    'description', 'image', 'user_id', 'select_article_enchere',\n",
    "    'date_ajout', 'status_enchere', 'status',\n",
    "    'price_verification_status', 'price_restriction_active',\n",
    "    'price_max_allowed', 'price_protection_until',\n",
    "    'created_at', 'updated_at', 'enchere_id', 'video',\n",
    "    'video_processing', 'video_processing_token'\n",
    "]\n",
    "\n",
    "df_articles = extract_table_to_df(sql_text, 'articles', ARTICLES_COLS)\n",
    "df_articles['id']      = pd.to_numeric(df_articles['id'],      errors='coerce')\n",
    "df_articles['user_id'] = pd.to_numeric(df_articles['user_id'], errors='coerce')\n",
    "df_articles['prix']    = pd.to_numeric(df_articles['prix'],    errors='coerce')\n",
    "\n",
    "print(f'Articles chargés : {len(df_articles)}')\n",
    "print(df_articles['status'].value_counts())\n",
    "display(df_articles[['id', 'nom', 'marque', 'prix', 'etat', 'status', 'user_id']].head())\n",
])

sec1mel_categories = mk_code([
    "# ── 1.2 Catégories ────────────────────────────────────────────────────────────\n",
    "CATS_COLS = ['id', 'nom', 'input', 'autorisation', 'status', 'position', 'garantie',\n",
    "             'created_at', 'updated_at']\n",
    "df_categories_raw = extract_table_to_df(sql_text, 'categories', CATS_COLS)\n",
    "df_categories_raw['id'] = pd.to_numeric(df_categories_raw['id'], errors='coerce')\n",
    "\n",
    "def parse_nom_fr(nom_json):\n",
    "    \"\"\"Extrait le nom français depuis le JSON {\"fr\":\"...\",\"en\":\"...\"}.\"\"\"\n",
    "    if nom_json is None:\n",
    "        return None\n",
    "    try:\n",
    "        return _json.loads(str(nom_json)).get('fr', nom_json)\n",
    "    except Exception:\n",
    "        return str(nom_json)\n",
    "\n",
    "df_categories_raw['nom_fr'] = df_categories_raw['nom'].apply(parse_nom_fr)\n",
    "df_categories = df_categories_raw[['id', 'nom_fr']].copy()\n",
    "df_categories.columns = ['categorie_id', 'categorie_nom']\n",
    "\n",
    "# ── 1.3 article_categories (table de jonction) ────────────────────────────────\n",
    "AC_COLS = ['id', 'categorie_id', 'article_id', 'created_at', 'updated_at']\n",
    "df_article_cats = extract_table_to_df(sql_text, 'article_categories', AC_COLS)\n",
    "df_article_cats['categorie_id'] = pd.to_numeric(df_article_cats['categorie_id'], errors='coerce')\n",
    "df_article_cats['article_id']   = pd.to_numeric(df_article_cats['article_id'],   errors='coerce')\n",
    "df_article_cats = df_article_cats.merge(df_categories, on='categorie_id', how='left')\n",
    "\n",
    "print(f'Catégories : {len(df_categories)}')\n",
    "display(df_categories)\n",
    "print(f'Associations article_categories : {len(df_article_cats)}')\n",
])

sec1mel_factures = mk_code([
    "# ── 1.4 Factures (historique achats) ──────────────────────────────────────────\n",
    "FACTURES_COLS = [\n",
    "    'id', 'matricule', 'quantite', 'montant_verse', 'montant_total',\n",
    "    'cni_acheteur', 'cni_vendeur', 'chemin', 'numero_acheteur',\n",
    "    'relai_id', 'acheteur_id', 'vendeur_id', 'article_id',\n",
    "    'status', 'date_delivery', 'created_at', 'updated_at',\n",
    "    'alerte_expiration_envoyee_at', 'alerte_livraison_vendeur_envoyee_at'\n",
    "]\n",
    "df_factures = extract_table_to_df(sql_text, 'factures', FACTURES_COLS)\n",
    "for col in ['id', 'acheteur_id', 'vendeur_id', 'article_id', 'montant_total']:\n",
    "    df_factures[col] = pd.to_numeric(df_factures[col], errors='coerce')\n",
    "\n",
    "print(f'Factures : {len(df_factures)}')\n",
    "print(df_factures['status'].value_counts())\n",
    "display(df_factures[['id', 'acheteur_id', 'article_id', 'status', 'montant_total', 'created_at']].head())\n",
])

sec1mel_paniers = mk_code([
    "# ── 1.5 Paniers (signaux implicites) ─────────────────────────────────────────\n",
    "PANIERS_COLS = ['id', 'user_id', 'article_id', 'created_at', 'updated_at']\n",
    "df_paniers = extract_table_to_df(sql_text, 'paniers', PANIERS_COLS)\n",
    "for col in ['id', 'user_id', 'article_id']:\n",
    "    df_paniers[col] = pd.to_numeric(df_paniers[col], errors='coerce')\n",
    "\n",
    "print(f'Paniers : {len(df_paniers)}')\n",
    "print(f'Utilisateurs uniques : {df_paniers[\"user_id\"].nunique()}')\n",
    "print(f'Articles uniques : {df_paniers[\"article_id\"].nunique()}')\n",
    "display(df_paniers.head())\n",
])

sec1mel_summary = mk_code([
    "# ── 1.6 Résumé global ────────────────────────────────────────────────────────\n",
    "print('=' * 55)\n",
    "print('DATASET MEL CAMEROUN — RÉSUMÉ')\n",
    "print('=' * 55)\n",
    "print(f'Articles total          : {len(df_articles)}')\n",
    "print(f'  - en vente            : {(df_articles[\"status\"] == \"en vente\").sum()}')\n",
    "print(f'  - vendus              : {(df_articles[\"status\"] == \"vendu\").sum()}')\n",
    "print(f'Catégories              : {len(df_categories)}')\n",
    "print(f'Factures total          : {len(df_factures)}')\n",
    "print(f'  - valides             : {(df_factures[\"status\"] == \"valide\").sum()}')\n",
    "print(f'  - annulées            : {(df_factures[\"status\"] == \"annule\").sum()}')\n",
    "print(f'  - en cours            : {(df_factures[\"status\"] == \"en cour\").sum()}')\n",
    "print(f'Paniers (signaux impl.) : {len(df_paniers)}')\n",
    "print(f'Utilisateurs acheteurs  : {df_factures[df_factures[\"status\"] == \"valide\"][\"acheteur_id\"].nunique()}')\n",
    "print(f'Utilisateurs vendeurs   : {df_articles[\"user_id\"].nunique()}')\n",
    "print('=' * 55)\n",
])


# ── Section 2-MEL – EDA ───────────────────────────────────────────────────────

sec2mel_md = mk_md([
    "## 2-MEL — Exploration des données MEL (EDA)\n",
    "\n",
    "Analyse des distributions de prix, catégories et comportements d'achat.\n",
])

sec2mel_eda = mk_code([
    "# ── 2.1 Distributions ────────────────────────────────────────────────────────\n",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
    "\n",
    "# Distribution des prix\n",
    "prix_valid = df_articles['prix'].dropna()\n",
    "prix_clipped = prix_valid.clip(0, prix_valid.quantile(0.95))\n",
    "axes[0].hist(prix_clipped, bins=30, color='steelblue', edgecolor='white')\n",
    "axes[0].set_title('Distribution des prix (articles MEL)')\n",
    "axes[0].set_xlabel('Prix (XAF)')\n",
    "axes[0].set_ylabel('Fréquence')\n",
    "\n",
    "# Top catégories\n",
    "cat_counts = df_article_cats.groupby('categorie_nom')['article_id'].count().sort_values(ascending=False)\n",
    "cat_counts.head(10).plot(kind='bar', ax=axes[1], color='coral', edgecolor='white')\n",
    "axes[1].set_title('Top 10 catégories (nb articles)')\n",
    "axes[1].tick_params(axis='x', rotation=45)\n",
    "\n",
    "# Status des articles\n",
    "status_counts = df_articles['status'].value_counts()\n",
    "axes[2].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',\n",
    "            colors=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])\n",
    "axes[2].set_title('Status des articles MEL')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
])

sec2mel_interactions = mk_code([
    "# ── 2.2 Interactions — achats & paniers ──────────────────────────────────────\n",
    "df_achats = df_factures[df_factures['status'] == 'valide'].copy()\n",
    "\n",
    "print('Achats par utilisateur (factures valides):')\n",
    "print(df_achats.groupby('acheteur_id')['article_id'].count().describe())\n",
    "\n",
    "# Top articles achetés\n",
    "top_achetes = (\n",
    "    df_achats.groupby('article_id')['acheteur_id'].count()\n",
    "    .sort_values(ascending=False)\n",
    "    .reset_index()\n",
    ")\n",
    "top_achetes.columns = ['article_id', 'nb_achats']\n",
    "top_achetes = top_achetes.merge(\n",
    "    df_articles[['id', 'nom', 'prix']].rename(columns={'id': 'article_id'}),\n",
    "    on='article_id', how='left'\n",
    ").head(10)\n",
    "print('\\nTop 10 articles achetés :')\n",
    "display(top_achetes)\n",
    "\n",
    "# Sparsité\n",
    "n_users_total = max(df_achats['acheteur_id'].nunique(), df_paniers['user_id'].nunique())\n",
    "n_items_total = df_articles['id'].nunique()\n",
    "n_interactions = len(df_achats) + len(df_paniers.drop_duplicates(['user_id', 'article_id']))\n",
    "sparsity = 1 - n_interactions / (n_users_total * n_items_total)\n",
    "print(f'\\nSparsité MEL : {sparsity:.2%}')\n",
    "print(f'Utilisateurs : {n_users_total}  |  Articles : {n_items_total}  |  Interactions : {n_interactions}')\n",
])


# ── Section 3-MEL – Preprocessing ────────────────────────────────────────────

sec3mel_md = mk_md([
    "## 3-MEL — Preprocessing — Matrice user-item MEL\n",
    "\n",
    "Signaux implicites mixtes :\n",
    "\n",
    "| Signal | Confiance |\n",
    "|---|---|\n",
    "| Achat validé (facture `valide`) | **5** |\n",
    "| Mise en panier | **2** |\n",
    "\n",
    "> Avec ~25 achats et ~148 paniers, le dataset MEL est très sparse.\n",
    "> On prend la confiance **max** si un article est à la fois acheté et en panier.\n",
])

sec3mel_matrix = mk_code([
    "# ── 3.1 Signaux implicites ───────────────────────────────────────────────────\n",
    "# Signal achat (confiance 5)\n",
    "df_achats_mel = (\n",
    "    df_factures[df_factures['status'] == 'valide'][['acheteur_id', 'article_id']]\n",
    "    .copy()\n",
    "    .rename(columns={'acheteur_id': 'user_id', 'article_id': 'item_id'})\n",
    ")\n",
    "df_achats_mel['confidence'] = 5\n",
    "df_achats_mel['source'] = 'achat'\n",
    "\n",
    "# Signal panier (confiance 2)\n",
    "df_paniers_mel = (\n",
    "    df_paniers[['user_id', 'article_id']]\n",
    "    .copy()\n",
    "    .rename(columns={'article_id': 'item_id'})\n",
    ")\n",
    "df_paniers_mel['confidence'] = 2\n",
    "df_paniers_mel['source'] = 'panier'\n",
    "\n",
    "# Fusionner — garder confiance max par paire (user, item)\n",
    "df_interactions_mel = pd.concat([df_achats_mel, df_paniers_mel], ignore_index=True)\n",
    "df_interactions_mel = (\n",
    "    df_interactions_mel\n",
    "    .groupby(['user_id', 'item_id'], as_index=False)\n",
    "    .agg(confidence=('confidence', 'max'), source=('source', 'first'))\n",
    ")\n",
    "\n",
    "# Garder seulement les items présents dans df_articles\n",
    "valid_items = df_articles['id'].dropna().astype(int).values\n",
    "df_interactions_mel = df_interactions_mel[\n",
    "    df_interactions_mel['item_id'].isin(valid_items)\n",
    "]\n",
    "\n",
    "print(f'Interactions MEL : {len(df_interactions_mel)}')\n",
    "print(f'  - achats    : {(df_interactions_mel[\"source\"] == \"achat\").sum()}')\n",
    "print(f'  - paniers   : {(df_interactions_mel[\"source\"] == \"panier\").sum()}')\n",
    "print(f'Users : {df_interactions_mel[\"user_id\"].nunique()}  |  Items : {df_interactions_mel[\"item_id\"].nunique()}')\n",
    "display(df_interactions_mel.head(10))\n",
])

sec3mel_encode = mk_code([
    "# ── 3.2 Encodage des IDs MEL ─────────────────────────────────────────────────\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "mel_user_enc = LabelEncoder()\n",
    "mel_item_enc = LabelEncoder()\n",
    "\n",
    "df_interactions_mel['user_idx'] = mel_user_enc.fit_transform(df_interactions_mel['user_id'])\n",
    "df_interactions_mel['item_idx'] = mel_item_enc.fit_transform(df_interactions_mel['item_id'])\n",
    "\n",
    "N_USERS_MEL = int(df_interactions_mel['user_idx'].nunique())\n",
    "N_ITEMS_MEL = int(df_interactions_mel['item_idx'].nunique())\n",
    "\n",
    "print(f'Users MEL : {N_USERS_MEL}  |  Items MEL : {N_ITEMS_MEL}')\n",
    "print(f'Sparsité  : {1 - len(df_interactions_mel) / (N_USERS_MEL * N_ITEMS_MEL):.2%}')\n",
    "\n",
    "# DataFrame articles enrichi avec catégorie\n",
    "df_articles_mel = df_articles[['id', 'nom', 'marque', 'prix', 'etat', 'status']].copy()\n",
    "df_articles_mel['id'] = df_articles_mel['id'].astype(float).astype('Int64')\n",
    "\n",
    "cat_principale = (\n",
    "    df_article_cats.groupby('article_id')['categorie_nom']\n",
    "    .first()\n",
    "    .reset_index()\n",
    "    .rename(columns={'article_id': 'id', 'categorie_nom': 'categorie'})\n",
    ")\n",
    "cat_principale['id'] = cat_principale['id'].astype(float).astype('Int64')\n",
    "df_articles_mel = df_articles_mel.merge(cat_principale, on='id', how='left')\n",
    "\n",
    "print('\\nArticles enrichis (extrait) :')\n",
    "display(df_articles_mel.head(8))\n",
])

sec3mel_split = mk_code([
    "# ── 3.3 Train / Test Split MEL (Leave-One-Out) ───────────────────────────────\n",
    "df_loo = df_interactions_mel.sort_values(['user_id', 'item_idx']).copy()\n",
    "\n",
    "user_counts = df_loo.groupby('user_id').size()\n",
    "users_with_ge2 = user_counts[user_counts >= 2].index\n",
    "\n",
    "test_mel_indices = df_loo.groupby('user_id').tail(1).index\n",
    "test_mel_indices = test_mel_indices[\n",
    "    df_loo.loc[test_mel_indices, 'user_id'].isin(users_with_ge2)\n",
    "]\n",
    "\n",
    "train_mel_df = df_loo.drop(test_mel_indices).copy()\n",
    "test_mel_df  = df_loo.loc[test_mel_indices].copy()\n",
    "\n",
    "print(f'Train MEL : {len(train_mel_df)} interactions  |  Test MEL : {len(test_mel_df)}')\n",
    "\n",
    "# Split aléatoire 80/20 (optionnel)\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_mel_random, test_mel_random = train_test_split(\n",
    "    df_interactions_mel, test_size=0.2, random_state=42\n",
    ")\n",
    "print(f'Train Random : {len(train_mel_random)}  |  Test Random : {len(test_mel_random)}')\n",
    "\n",
    "# Matrice sparse pour ALS\n",
    "from scipy.sparse import csr_matrix as csr\n",
    "R_mel = csr(\n",
    "    (train_mel_df['confidence'].values,\n",
    "     (train_mel_df['user_idx'].values, train_mel_df['item_idx'].values)),\n",
    "    shape=(N_USERS_MEL, N_ITEMS_MEL)\n",
    ")\n",
    "print(f'\\nMatrice user-item MEL : {R_mel.shape}, {R_mel.nnz} éléments non nuls')\n",
])


# ── Section 6-MEL – Baseline ──────────────────────────────────────────────────

sec6mel_md = mk_md([
    "## 6-MEL — Baseline MEL — Popularité\n",
    "\n",
    "Recommande les articles les plus fréquemment achetés / ajoutés au panier.\n",
])

sec6mel_code = mk_code([
    "# ── Popularité MEL ──────────────────────────────────────────────────────────\n",
    "mel_item_popularity = (\n",
    "    train_mel_df.groupby('item_id')['confidence']\n",
    "    .sum()\n",
    "    .sort_values(ascending=False)\n",
    ")\n",
    "K_MEL = min(10, N_ITEMS_MEL)\n",
    "mel_popular_items = mel_item_popularity.head(K_MEL).index.tolist()\n",
    "\n",
    "pop_df = (\n",
    "    pd.DataFrame({'item_id': mel_popular_items})\n",
    "    .merge(df_articles_mel.rename(columns={'id': 'item_id'}), on='item_id', how='left')\n",
    ")\n",
    "print(f'Top {K_MEL} articles populaires MEL :')\n",
    "display(pop_df[['item_id', 'nom', 'categorie', 'prix']])\n",
    "\n",
    "\n",
    "def mel_baseline_recommender(user_id, n=10):\n",
    "    user_history = train_mel_df[train_mel_df['user_id'] == user_id]['item_id'].values\n",
    "    recs = [i for i in mel_popular_items if i not in user_history]\n",
    "    return recs[:n]\n",
    "\n",
    "\n",
    "mel_p_base, mel_r_base, mel_n_base = [], [], []\n",
    "for _, row in test_mel_df.iterrows():\n",
    "    uid, relevant = row['user_id'], [row['item_id']]\n",
    "    rec = mel_baseline_recommender(uid)\n",
    "    mel_p_base.append(precision_at_k(rec, relevant, K_MEL))\n",
    "    mel_r_base.append(recall_at_k(rec, relevant, K_MEL))\n",
    "    mel_n_base.append(ndcg_at_k(rec, relevant, K_MEL))\n",
    "\n",
    "mel_results_baseline = {\n",
    "    f'Precision@{K_MEL}': np.mean(mel_p_base),\n",
    "    f'Recall@{K_MEL}':    np.mean(mel_r_base),\n",
    "    f'NDCG@{K_MEL}':      np.mean(mel_n_base),\n",
    "}\n",
    "print('\\nBaseline MEL :', mel_results_baseline)\n",
])


# ── Section 8-MEL – ALS ───────────────────────────────────────────────────────

secals_mel_md = mk_md([
    "## 8-MEL — ALS implicite sur les données MEL\n",
    "\n",
    "On applique l'ALS from-scratch sur la matrice de confiance MEL.\n",
    "Le signal mixte (achat=5, panier=2) est utilisé directement.\n",
])

secals_mel_code = mk_code([
    "# ── ALS MEL ──────────────────────────────────────────────────────────────────\n",
    "import sys; sys.path.insert(0, str(Path.cwd().parent))\n",
    "from src.recommender import ALSRecommender\n",
    "\n",
    "als_mel = ALSRecommender(\n",
    "    n_factors=min(10, N_USERS_MEL - 1, N_ITEMS_MEL - 1),\n",
    "    regularization=0.1,\n",
    "    n_iterations=20,\n",
    "    alpha=40,\n",
    ")\n",
    "print(f'Entraînement ALS MEL (matrice {R_mel.shape})...')\n",
    "als_mel.fit(R_mel)\n",
    "print('OK')\n",
    "\n",
    "\n",
    "def mel_als_recommender(user_id, n=10):\n",
    "    if user_id not in mel_user_enc.classes_:\n",
    "        return mel_popular_items[:n]\n",
    "    user_idx = int(mel_user_enc.transform([user_id])[0])\n",
    "    scores = als_mel.user_factors[user_idx] @ als_mel.item_factors.T\n",
    "    seen = train_mel_df[train_mel_df['user_id'] == user_id]['item_idx'].values\n",
    "    scores[seen] = -np.inf\n",
    "    top_idxs = np.argsort(scores)[::-1][:n]\n",
    "    return mel_item_enc.inverse_transform(top_idxs).tolist()\n",
    "\n",
    "\n",
    "mel_p_als, mel_r_als, mel_n_als = [], [], []\n",
    "for _, row in test_mel_df.iterrows():\n",
    "    uid, relevant = row['user_id'], [row['item_id']]\n",
    "    rec = mel_als_recommender(uid)\n",
    "    mel_p_als.append(precision_at_k(rec, relevant, K_MEL))\n",
    "    mel_r_als.append(recall_at_k(rec, relevant, K_MEL))\n",
    "    mel_n_als.append(ndcg_at_k(rec, relevant, K_MEL))\n",
    "\n",
    "mel_results_als = {\n",
    "    f'Precision@{K_MEL}': np.mean(mel_p_als) if mel_p_als else 0,\n",
    "    f'Recall@{K_MEL}':    np.mean(mel_r_als) if mel_r_als else 0,\n",
    "    f'NDCG@{K_MEL}':      np.mean(mel_n_als) if mel_n_als else 0,\n",
    "}\n",
    "print('ALS MEL :', mel_results_als)\n",
])


# ── Section 9-MEL – Content-Based ────────────────────────────────────────────

seccb_mel_md = mk_md([
    "## 9-MEL — Content-Based MEL (TF-IDF)\n",
    "\n",
    "Feature textuelle = catégorie + nom + marque + état de l'article.\n",
])

seccb_mel_code = mk_code([
    "# ── Content-Based MEL ────────────────────────────────────────────────────────\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity as cos_sim\n",
    "\n",
    "encoded_item_ids = mel_item_enc.classes_\n",
    "df_cb_mel = (\n",
    "    df_articles_mel[df_articles_mel['id'].isin(encoded_item_ids)]\n",
    "    .copy()\n",
    ")\n",
    "df_cb_mel['item_idx'] = mel_item_enc.transform(df_cb_mel['id'].values)\n",
    "df_cb_mel = df_cb_mel.sort_values('item_idx').reset_index(drop=True)\n",
    "\n",
    "\n",
    "def build_mel_feature(row):\n",
    "    parts = []\n",
    "    for field in ['categorie', 'nom', 'marque', 'etat']:\n",
    "        val = row.get(field)\n",
    "        if val and pd.notna(val) and str(val).strip():\n",
    "            parts.append(str(val).strip())\n",
    "    return ' '.join(parts)\n",
    "\n",
    "\n",
    "df_cb_mel['features'] = df_cb_mel.apply(build_mel_feature, axis=1)\n",
    "\n",
    "mel_tfidf        = TfidfVectorizer(max_features=200)\n",
    "mel_tfidf_matrix = mel_tfidf.fit_transform(df_cb_mel['features'])\n",
    "mel_cosine_sim   = cos_sim(mel_tfidf_matrix, mel_tfidf_matrix)\n",
    "\n",
    "print(f'Matrice TF-IDF MEL : {mel_tfidf_matrix.shape}')\n",
    "print(f'Cosine sim matrix  : {mel_cosine_sim.shape}')\n",
    "\n",
    "mel_item_to_cbidx = {row['id']: i for i, row in df_cb_mel.iterrows()}\n",
    "\n",
    "\n",
    "def mel_content_based_recommender(user_id, n=10):\n",
    "    user_items = train_mel_df[train_mel_df['user_id'] == user_id]['item_id'].values\n",
    "    if len(user_items) == 0:\n",
    "        return mel_popular_items[:n]\n",
    "    scores = np.zeros(len(df_cb_mel))\n",
    "    for iid in user_items:\n",
    "        if iid in mel_item_to_cbidx:\n",
    "            scores += mel_cosine_sim[mel_item_to_cbidx[iid]]\n",
    "    for iid in user_items:\n",
    "        if iid in mel_item_to_cbidx:\n",
    "            scores[mel_item_to_cbidx[iid]] = -np.inf\n",
    "    top_idxs = np.argsort(scores)[::-1][:n]\n",
    "    return df_cb_mel.iloc[top_idxs]['id'].tolist()\n",
    "\n",
    "\n",
    "mel_p_cb, mel_r_cb, mel_n_cb = [], [], []\n",
    "for _, row in test_mel_df.iterrows():\n",
    "    uid, relevant = row['user_id'], [row['item_id']]\n",
    "    rec = mel_content_based_recommender(uid)\n",
    "    mel_p_cb.append(precision_at_k(rec, relevant, K_MEL))\n",
    "    mel_r_cb.append(recall_at_k(rec, relevant, K_MEL))\n",
    "    mel_n_cb.append(ndcg_at_k(rec, relevant, K_MEL))\n",
    "\n",
    "mel_results_cb = {\n",
    "    f'Precision@{K_MEL}': np.mean(mel_p_cb) if mel_p_cb else 0,\n",
    "    f'Recall@{K_MEL}':    np.mean(mel_r_cb) if mel_r_cb else 0,\n",
    "    f'NDCG@{K_MEL}':      np.mean(mel_n_cb) if mel_n_cb else 0,\n",
    "}\n",
    "print('Content-Based MEL :', mel_results_cb)\n",
])


# ── Section 10-MEL – Comparaison ─────────────────────────────────────────────

seccmp_mel_md = mk_md([
    "## 10-MEL — Comparaison des modèles MEL\n",
])

seccmp_mel_code = mk_code([
    "# ── Comparaison finale MEL ────────────────────────────────────────────────────\n",
    "mel_results_all = {\n",
    "    'Baseline (Popularité)': mel_results_baseline,\n",
    "    'ALS implicite':         mel_results_als,\n",
    "    'Content-Based':         mel_results_cb,\n",
    "}\n",
    "\n",
    "mel_results_df = pd.DataFrame(mel_results_all).T.round(4)\n",
    "print('=' * 60)\n",
    "print('COMPARAISON DES MODÈLES — DATASET MEL CAMEROUN')\n",
    "print('=' * 60)\n",
    "display(mel_results_df)\n",
    "\n",
    "# Visualisation\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "fig.suptitle('Métriques — Dataset MEL Cameroun', fontsize=14)\n",
    "colors = ['#2ecc71', '#e74c3c', '#3498db']\n",
    "for ax, metric in zip(axes, list(mel_results_baseline.keys())):\n",
    "    vals = [mel_results_all[m][metric] for m in mel_results_all]\n",
    "    ax.bar(list(mel_results_all.keys()), vals, color=colors)\n",
    "    ax.set_title(metric)\n",
    "    ax.tick_params(axis='x', rotation=30)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print('''\n",
    "Note : Avec ~25 achats valides et ~148 paniers sur 228 articles,\n",
    "le dataset MEL est extrêmement sparse. L ALS implicite est\n",
    "la stratégie optimale pour ce type de données.\n",
    "''')\n",
])


# ── Section 13-MEL – Analyse qualitative ─────────────────────────────────────

secqual_mel_md = mk_md([
    "## 13-MEL — Analyse qualitative des recommandations MEL\n",
])

secqual_mel_code = mk_code([
    "# ── Analyse qualitative MEL ──────────────────────────────────────────────────\n",
    "def show_mel_recommendations(user_id, recommender_fn, model_name, n=10):\n",
    "    print(f'\\n=== {model_name} — user {user_id} ===')\n",
    "    history = train_mel_df[train_mel_df['user_id'] == user_id]\n",
    "    if len(history) > 0:\n",
    "        hist_info = history.merge(\n",
    "            df_articles_mel.rename(columns={'id': 'item_id'}), on='item_id', how='left'\n",
    "        )[['item_id', 'nom', 'categorie', 'prix', 'confidence', 'source']]\n",
    "        print(f'Historique ({len(history)}) :')\n",
    "        display(hist_info)\n",
    "    else:\n",
    "        print('Cold start (aucun historique)')\n",
    "    recs = recommender_fn(user_id, n=n)\n",
    "    recs_df = (\n",
    "        pd.DataFrame({'item_id': recs})\n",
    "        .merge(df_articles_mel.rename(columns={'id': 'item_id'}), on='item_id', how='left')\n",
    "    )[['item_id', 'nom', 'categorie', 'prix', 'etat']]\n",
    "    print(f'Recommandations ({len(recs)}) :')\n",
    "    display(recs_df)\n",
    "\n",
    "\n",
    "top_user_mel = (\n",
    "    train_mel_df.groupby('user_id').size()\n",
    "    .sort_values(ascending=False)\n",
    "    .index[0]\n",
    ")\n",
    "print(f'Utilisateur le plus actif : {top_user_mel}')\n",
    "show_mel_recommendations(top_user_mel, mel_baseline_recommender, 'Baseline')\n",
    "show_mel_recommendations(top_user_mel, mel_content_based_recommender, 'Content-Based')\n",
    "if top_user_mel in mel_user_enc.classes_:\n",
    "    show_mel_recommendations(top_user_mel, mel_als_recommender, 'ALS')\n",
])


# ── Section 14-MEL – Sauvegarde ───────────────────────────────────────────────

secsave_mel_md = mk_md(["## 14-MEL — Sauvegarde des modèles MEL\n"])

secsave_mel_code = mk_code([
    "# ── Sauvegarde MEL ───────────────────────────────────────────────────────────\n",
    "import pickle, json as _json_save\n",
    "\n",
    "for name, obj in [\n",
    "    ('mel_user_encoder.pkl',  mel_user_enc),\n",
    "    ('mel_item_encoder.pkl',  mel_item_enc),\n",
    "    ('mel_als_model.pkl',     als_mel),\n",
    "    ('mel_cosine_sim.pkl',    mel_cosine_sim),\n",
    "    ('mel_popular_items.pkl', mel_popular_items),\n",
    "]:\n",
    "    with open(MODELS_DIR / name, 'wb') as f:\n",
    "        pickle.dump(obj, f)\n",
    "\n",
    "mel_summary = {\n",
    "    'dataset':        'MEL Cameroun',\n",
    "    'n_users':        N_USERS_MEL,\n",
    "    'n_items':        N_ITEMS_MEL,\n",
    "    'n_interactions': len(df_interactions_mel),\n",
    "    'sparsity':       round(1 - len(df_interactions_mel) / (N_USERS_MEL * N_ITEMS_MEL), 4),\n",
    "    'results': {\n",
    "        k: {m: round(float(v), 4) for m, v in r.items()}\n",
    "        for k, r in mel_results_all.items()\n",
    "    },\n",
    "}\n",
    "with open(MODELS_DIR / 'mel_results_summary.json', 'w') as f:\n",
    "    _json_save.dump(mel_summary, f, indent=2, ensure_ascii=False)\n",
    "\n",
    "print('Modèles MEL sauvegardés dans', MODELS_DIR)\n",
    "print(_json_save.dumps(mel_summary, indent=2, ensure_ascii=False))\n",
])


# ── Assemblage et injection ───────────────────────────────────────────────────

new_cells = [
    sec05_md, sec05_code,
    sec1mel_md, sec1mel_articles, sec1mel_categories, sec1mel_factures,
    sec1mel_paniers, sec1mel_summary,
    sec2mel_md, sec2mel_eda, sec2mel_interactions,
    sec3mel_md, sec3mel_matrix, sec3mel_encode, sec3mel_split,
    sec6mel_md, sec6mel_code,
    secals_mel_md, secals_mel_code,
    seccb_mel_md, seccb_mel_code,
    seccmp_mel_md, seccmp_mel_code,
    secqual_mel_md, secqual_mel_code,
    secsave_mel_md, secsave_mel_code,
]

print(f"Ouverture du notebook : {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, encoding="utf-8") as f:
    nb = json.load(f)

original_cells = nb["cells"]
insert_pos = 3  # après imports (cell 0=titre, 1=Section0 md, 2=code imports), avant Section 1

nb["cells"] = original_cells[:insert_pos] + new_cells + original_cells[insert_pos:]

print(f"Cellules originales : {len(original_cells)}")
print(f"Nouvelles cellules  : {len(new_cells)}")
print(f"Total après         : {len(nb['cells'])}")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook sauvegardé avec succès !")
