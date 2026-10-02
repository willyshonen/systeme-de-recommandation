# Documentation Technique — Système de Recommandation MEL Cameroun

## 1. Pipeline de données

### 1.1 Source
Données e-commerce MEL Cameroun. Les interactions utilisateur-produit sont construites à partir des tables MySQL de la plateforme (`factures`, `paniers`, `articles`) et exportées en CSV dans `data/raw/`.

**Fichiers attendus dans `data/raw/` :**

| Fichier | Colonnes clés |
|---|---|
| `orders.csv` | `order_id`, `customer_id`, `order_status` |
| `order_items.csv` | `order_id`, `product_id`, `price` |
| `products.csv` | `product_id`, `product_category_name`, `product_weight_g`, `product_photos_qty` |
| `reviews.csv` | `order_id`, `review_score` |
| `category_names.csv` (optionnel) | `product_category_name`, `product_category_name_english` |

### 1.2 Construction des interactions

```
orders (order_id, customer_id, order_status)
    ↓ jointure order_id
order_items (order_id, product_id, price)
    ↓ filtre order_status == 'delivered'
    ↓ jointure order_id
reviews (order_id, review_score)
    → interactions (user_id, item_id, rating, order_id)
```

- Si pas d'avis → `rating = 3.0` (signal neutre)

### 1.3 Filtrage cold-start (itératif, 3 passes)

| Seuil | Valeur |
|---|---|
| Interactions minimales par user | 2 |
| Interactions minimales par item | 5 |

### 1.4 Splits

| Split | Stratégie | Train | Test |
|---|---|---|---|
| Leave-One-Out | Dernier achat chronologique en test | ~63% | ~37% |
| Random 80/20 | Shuffle aléatoire | 80% | 20% |

Le split LOO est utilisé pour évaluer la capacité à prédire le prochain achat.
Le split random est utilisé pour mesurer le RMSE/MAE du SVD.

---

## 2. Modèles

### 2.1 Baseline — Popularité

**Principe :** recommande les N items ayant le plus grand nombre d'achats en train.

```python
item_popularity = train_df.groupby('item_id')['user_id'].count().sort_values(ascending=False)
popular_items   = item_popularity.index.tolist()
```

**Complexité :** O(I log I) une seule fois à l'entraînement, O(1) à l'inférence.
**Usage :** fallback cold-start pour tout nouveau user.

---

### 2.2 SVD — Singular Value Decomposition

**Bibliothèque :** `scipy.sparse.linalg.svds`

**Étapes :**
1. Construire la matrice R (users × items) avec les ratings
2. Centrer par user : `R_centered[u] = R[u] - mean(R[u])`
3. SVD tronquée : `R_centered ≈ U · Σ · Vᵀ` avec k facteurs latents
4. Reconstruction : `R_pred = U · Σ · Vᵀ + user_mean`

**Hyperparamètre optimal :**

| Paramètre | Valeur | Sélection |
|---|---|---|
| `k_factors` | 150 | Grid search [20, 50, 100, 150] → max NDCG@10 |

**Inférence :**
```python
scores   = R_pred[user_idx]          # vecteur de scores pour tous les items
top_idxs = np.argsort(scores)[::-1]  # tri décroissant
```

---

### 2.3 ALS — Alternating Least Squares

**Implémentation :** from scratch, numpy/scipy uniquement.

**Modèle implicite (Hu, Koren, Volinsky 2008) :**
- Préférence binaire : `p_ui = 1` si l'user u a acheté l'item i, sinon `0`
- Confidence : `c_ui = 1 + α × r_ui` avec α = 40
- Minimise : `Σ c_ui (p_ui - xᵤᵀ yᵢ)² + λ (||xᵤ||² + ||yᵢ||²)`

**Mise à jour alternée :**
```
# User factors (pour chaque user u)
A_u = YᵀCᵘY + λI
b_u = YᵀCᵘp_u
x_u = A_u⁻¹ b_u

# Item factors (pour chaque item i)
A_i = XᵀCⁱX + λI
b_i = XᵀCⁱp_i
y_i = A_i⁻¹ b_i
```

**Hyperparamètres optimaux :**

| Paramètre | Valeur | Sélection |
|---|---|---|
| `n_factors` | 32 | Grid search 5 configs → max NDCG@10 |
| `regularization` | 0.1 | Grid search 5 configs → max NDCG@10 |
| `alpha` | 40 | Valeur standard de la littérature |
| `n_iterations` | 15 | Convergence empirique |

**Complexité :** O(I · k² + U · k²) par itération.

---

### 2.4 Content-Based — TF-IDF + Cosine Similarity

**Features produit :**
```python
description = category_name + product_weight_g + "g " + product_photos_qty + "photos"
```

**Pipeline :**
1. TF-IDF vectorisation (max 500 features, stop words anglais)
2. Matrice de similarité cosine (items × items)
3. Pour un user : moyenne des vecteurs de similarité de ses achats passés

```python
scores = np.zeros(n_items)
for item_id in user_history:
    scores += cosine_sim[item_to_idx[item_id]]
```

**Note :** Performance dépendante de la richesse des métadonnées produit disponibles. Des descriptions textuelles complètes (champ `nom`, `marque`, `description` de la table `articles`) amélioreraient significativement ce modèle.

---

### 2.5 Hybride — SVD + Content-Based

**Score final :**
```
score_hybrid(u, i) = α × score_SVD_norm(u, i) + (1 − α) × score_CB_norm(u, i)
```

**Normalisation [0, 1] par user :**
```python
score_norm = (score - score.min()) / (score.max() - score.min())
```

**Sélection de alpha :** Grid search sur α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} → max NDCG@10.

---

## 3. Évaluation

### 3.1 Stratégie
**Leave-One-Out** : pour chaque user, le dernier achat (chronologique) est le ground truth.
Les items déjà vus en train sont exclus des recommandations.

Évaluation sur un échantillon de **500 users** (pour limiter le temps de calcul).

### 3.2 Métriques

**Precision@K**
```
Precision@K = |recommandés[:K] ∩ pertinents| / K
```

**Recall@K**
```
Recall@K = |recommandés[:K] ∩ pertinents| / |pertinents|
```

**NDCG@K** (Normalized Discounted Cumulative Gain)
```
DCG@K  = Σ(i=1 à K) rel_i / log2(i+1)
IDCG@K = DCG@K optimal (si tous les pertinents en premier)
NDCG@K = DCG@K / IDCG@K
```

### 3.3 Résultats (dataset de validation)

| Modèle | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Baseline | 0.0015 | 0.0147 | 0.0077 |
| SVD | 0.0032 | 0.0320 | 0.0240 |
| ALS | 0.0038 | 0.0380 | 0.0298 |
| Content-Based | 0.0012 | 0.0120 | 0.0054 |
| **Hybride** | **0.0038** | **0.0380** | **0.0325** |

**Note sur les scores absolus :** les métriques absolues semblent faibles car la matrice user-item est extrêmement sparse (~99.7%). La plupart des utilisateurs n'achètent qu'une seule fois (one-shot buyers), ce qui est un problème inhérent aux données e-commerce. Ce qui compte : les **gains relatifs** — l'ALS et le modèle hybride surpassent la baseline de **+4× en NDCG**.

---

## 4. Artefacts sauvegardés

| Fichier | Contenu | Chargement |
|---|---|---|
| `models/svd_R_pred.npy` | Matrice prédite SVD (N_users × N_items) | `np.load()` |
| `models/als_model.pkl` | Instance ALSRecommender avec user/item factors | `pickle.load()` |
| `models/cosine_sim.pkl` | Tuple (cosine_sim, item_to_idx, idx_to_item) | `pickle.load()` |
| `models/user_encoder.pkl` | LabelEncoder des user_ids | `pickle.load()` |
| `models/item_encoder.pkl` | LabelEncoder des item_ids | `pickle.load()` |
| `models/popular_items.pkl` | Liste d'item_ids triés par popularité | `pickle.load()` |
| `models/results_summary.json` | Métriques + hyperparamètres optimaux | `json.load()` |

---

## 5. Pistes d'amélioration

### Court terme
- **Features produit plus riches** : utiliser les champs `nom`, `marque`, `description` de la table `articles` MEL → améliorerait significativement le Content-Based
- **Temporal weighting** : pondérer les achats récents plus fortement dans l'ALS
- **BPR (Bayesian Personalized Ranking)** : optimise directement le ranking au lieu du RMSE

### Moyen terme
- **Neural Collaborative Filtering (NCF)** : embeddings user/item appris par un réseau de neurones
- **Session-based recommendations** : LSTM/Transformer sur la séquence d'achats (paniers MEL)
- **Two-tower model** : encodeur user + encodeur item, recherche ANN (FAISS)

### Adaptation MEL Cameroun
- Exporter les tables `factures`, `paniers`, `articles` de MySQL vers CSV pour alimenter `data/raw/`
- Enrichir les features produit avec les descriptions et images du catalogue MEL
- Implémenter un cold-start par catégorie : nouveaux users → popularité dans leur catégorie préférée
