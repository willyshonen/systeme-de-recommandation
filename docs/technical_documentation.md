# Documentation Technique — Modèles de Recommandation

## 1. Pipeline de données

### 1.1 Source
Dataset Olist Brazilian E-Commerce (Kaggle). 5 fichiers CSV joints par `order_id` et `product_id`.

### 1.2 Construction des interactions
```
orders (order_id, customer_id, order_status)
    ↓ join order_id
order_items (order_id, product_id, price)
    ↓ filtre order_status == 'delivered'
    ↓ join order_id
reviews (order_id, review_score)
    → interactions (user_id, item_id, rating, order_id)
```
- Si pas d'avis → `rating = 3.0` (signal neutre)
- Résultat brut : **110 197 interactions**

### 1.3 Filtrage cold-start (itératif, 3 passes)
| Seuil | Valeur |
|---|---|
| Interactions minimales par user | 2 |
| Interactions minimales par item | 5 |

Résultat après filtrage : **9 657 interactions**, 3 599 users, 882 items.

### 1.4 Splits
| Split | Stratégie | Train | Test |
|---|---|---|---|
| Leave-One-Out | Dernier achat chronologique en test | 6 058 | 3 599 |
| Random 80/20 | Shuffle aléatoire | 7 725 | 1 932 |

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

**Métriques sur test_random :**
- RMSE : calculé sur les paires (user, item) du test split
- MAE : calculé sur les paires (user, item) du test split

**Inférence :**
```python
scores   = R_pred[user_idx]          # vecteur de scores pour tous les items
top_idxs = np.argsort(scores)[::-1]  # tri décroissant
```
Complexité : O(I) par user.

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
description = category_name_english + product_weight_g + "g " + product_photos_qty + "photos"
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

**Limites :** dépend uniquement des métadonnées produit disponibles (catégorie, poids, photos). Peu de signal discriminant sur ce dataset.

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

**Sélection de alpha :**
Grid search sur α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} → max NDCG@10.

**Résultat :** `alpha = 1.0` — le SVD domine entièrement. Le Content-Based n'apporte pas de signal supplémentaire sur ce dataset (features produits trop pauvres).

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
Avantage : pénalise les recommandations pertinentes mais mal classées.

### 3.3 Résultats

| Modèle | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Baseline | 0.0015 | 0.0147 | 0.0077 |
| SVD | 0.0032 | 0.0320 | 0.0240 |
| ALS | 0.0038 | 0.0380 | 0.0298 |
| Content-Based | 0.0012 | 0.0120 | 0.0054 |
| **Hybride** | **0.0038** | **0.0380** | **0.0325** |

**Pourquoi les scores sont faibles ?**  
La matrice user-item est sparse à **99.7%**. Sur Olist, 97% des clients n'achètent qu'une seule fois (one-shot buyers). Avec un seul achat par user, le Leave-One-Out met cet unique achat en test → le train est vide pour ces users → impossible de personnaliser. C'est un problème inhérent au dataset, pas au modèle.

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
- **Features produit plus riches** : description textuelle, images → améliorerait le Content-Based
- **Temporal weighting** : pondérer les achats récents plus fortement dans l'ALS
- **BPR (Bayesian Personalized Ranking)** : optimise directement le ranking au lieu du RMSE

### Moyen terme
- **Neural Collaborative Filtering (NCF)** : embeddings user/item appris par un réseau de neurones
- **Session-based recommendations** : LSTM/Transformer sur la séquence d'achats
- **Two-tower model** : encodeur user + encodeur item, recherche ANN (FAISS)

### Adaptation MEL Cameroun
- Remplacer les `customer_id` Olist par les IDs utilisateurs réels du site
- Enrichir les features produit avec les descriptions et images du catalogue
- Implémenter un cold-start par catégorie : nouveaux users → popularité dans leur catégorie préférée
