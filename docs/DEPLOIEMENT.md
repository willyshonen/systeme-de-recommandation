# Déploiement VPS et contrat d'intégration melcameroun

Cible : VPS avec Docker + Compose v2.
MLflow est **optionnel** — `src/train.py` bascule sur une base SQLite locale si le
serveur est injoignable. L'API ne lit que des fichiers `.pkl`/`.npy`, elle ne
dépend donc pas de MLflow.

---

## 1. Ce que melcameroun doit appeler

### 1.1 Visiteur non identifié → `/recommend/session`

**C'est le point important.** Melcameroun identifie le visiteur par session, pas
par `customer_id`. Le modèle n'a aucun profil appris pour un visiteur anonyme :
appeler `GET /recommend/{user_id}` avec un identifiant de session **tomberait en
cold-start** et renverrait la popularité, c'est-à-dire les mêmes articles pour
tout le monde.

La personnalisation passe donc par le **contenu de la session**. Melcameroun
n'envoie pas des catégories mais les **`product_id` réellement consultés** :
c'est ce qui permet de recommander un article précis, et non « un article
quelconque de la même catégorie ».

```http
POST /recommend/session
Content-Type: application/json

{
  "session_id": "sess_7f3a91",
  "product_ids": ["20", "21"],
  "categories": ["Vêtements femmes", "Chaussures"],
  "n": 10
}
```

```json
{
  "session_id": "sess_7f3a91",
  "model": "content-based produit (session)",
  "granularity": "produit",
  "used_categories": ["Vêtements femmes"],
  "ignored_categories": ["Chaussures"],
  "used_product_ids": ["20", "21"],
  "ignored_product_ids": [],
  "recommended_categories": ["Vêtements hommes"],
  "recommendations": ["148", "18", "156", "19", "139"],
  "n": 10
}
```

À lire côté intégration :

| Champ | Signification |
|---|---|
| `granularity` | `produit` = classement par similarité d'article, `categorie` = repli |
| `used_product_ids` | articles de session ayant un titre exploitable |
| `ignored_product_ids` | envoyés mais absents du catalogue ou sans titre |
| `used_categories` | catégories de session réellement reconnues du modèle |
| `ignored_categories` | envoyées mais hors espace du modèle (voir §4) |
| `recommended_categories` | **compléments**, jamais les catégories déjà vues |
| `recommendations` | `product_id` disponibles, dans l'ordre de priorité |

`recommendations` ne contient **jamais** de produit vendu, **jamais** un article
déjà présent dans `product_ids`, et `recommended_categories` ne contient
**jamais** de catégorie sans stock.

Ordre de priorité appliqué par l'API :

1. **`product_ids`** — similarité TF-IDF sur les TITRES. C'est le chemin
   recommandé : il distingue deux articles d'une même catégorie et ne dépend pas
   des libellés de catégorie, qui comportent au moins une erreur dans le
   catalogue source.
2. **`categories`** — repli si aucun article n'est exploitable. Moins précis :
   deux articles d'une même catégorie se valent.
3. **Popularité** — si la session n'a aucun contexte. Même forme de réponse,
   seule la pertinence change.

Le champ `categories` reste donc utile mais devient secondaire : si melcameroun
peut transmettre les `product_id`, il n'a plus besoin de maintenir une liste de
catégories.

### 1.2 Client connecté → `GET /recommend/{user_id}`

À réserver aux vrais `customer_id` MEL (les 34 users du jeu d'entraînement).
Le classement combine alors SVD + content-based, et l'historique persistant est
utilisé pour ne pas reproposer du déjà-consommé.

### 1.3 Autres endpoints

| Endpoint | Usage |
|---|---|
| `GET /health` | sonde de santé (liveness/readiness) |
| `GET /metrics` | métriques du dernier entraînement |
| `GET /popular?n=10` | articles populaires disponibles |
| `GET /similar/{product_id}` | « vous aimerez aussi » |
| `POST /reload` | rechargement des modèles (appelé par le scheduler) |

---

### 1.3 Collecte d'événements → `POST /events`

Sans collecte, le modèle ne s'améliore jamais. C'est **la** source manquante :
`visits.csv` est vide, donc aucun modèle n'a jamais vu une vue produit.

```http
POST /events
Content-Type: application/json

{
  "session_id": "sess_7f3a91",
  "events": [
    {"event_type": "view",  "product_id": 20, "ts": "2026-03-05T12:00:00"},
    {"event_type": "cart",  "product_id": 21, "ts": "2026-03-05T12:00:05"}
  ]
}
```

```json
{
  "accepted": 2,
  "rejected_unknown_product": 0,
  "rejected_invalid_type": 0,
  "session_id": "sess_7f3a91",
  "user_id": null,
  "details": []
}
```

| Champ | Règle |
|---|---|
| `event_type` | `view`, `cart`, `purchase` ou `review`. `type` est accepté comme alias |
| `product_id` | obligatoire, doit exister au catalogue |
| `ts` | ISO 8601, **à envoyer** — voir ci-dessous |
| `session_id` | obligatoire sauf si `user_id` est fourni |
| `events` | 1 à 200 éléments par lot |

Trois points qui ont caused des incidents de conception, à lire avant
d'intégrer :

1. **`event_type` est le nom canonique**, aligné sur la colonne du CSV
   `events.csv`. Une clé mal orthographiée (`evnt_type`) est **rejetée en 422**,
   pas ignorée. C'est volontaire : un champ ignoré en silence transforme tous les
   événements en `view`, et la faute n'apparaît qu'en production, des semaines
   plus tard, dans les métriques.

2. **`ts` doit être envoyé par le front.** La déduplication repose sur
   (client, session, article, type, `ts`). Un retry qui rejoue le même lot avec
   le même `ts` n'est pas compté deux fois ; sans `ts`, l'API horodate au
   reception et deux envois sont deux événements distincts. C'est le comportement
   correct : sans horodatage, on ne peut pas distinguer un rejeu d'une nouvelle
   visite.

3. **Un article vendu reste collectable.** Une vue sur un article vendu est
   enregistrée (le front l'affiche encore en fiche) mais ne sera jamais
   recommandé ensuite. `POST /events` et `POST /recommend/session` ont donc des
   règles de catalogue différentes, volontairement.

À prévoir côté melcameroun : `session_id` stable pour toute la session, envoi
d'un lot au changement de page plutôt qu'à chaque clic, et un `ts` ISO 8601 en
UTC. La rétention des sessions n'est pas gérée par l'API — c'est une décision de
conformité à prendre côté MEL.

## 2. Mise en place

```bash
cp .env.example .env
```

À remplir impérativement dans `.env` :

```bash
# Générer une vraie clé, jamais la valeur par défaut
RELOAD_SECRET=$(openssl rand -hex 32)

# Base MEL (le ré-entraînement lit la base, en lecture seule)
MYSQL_HOST=db.melcameroun.com
MYSQL_USER=mel_readonly
MYSQL_PASSWORD=...

# Webhook Slack/Discord : sans lui, personne ne voit les régressions NDCG
NOTIFY_WEBHOOK=https://hooks.slack.com/services/...

CORS_ORIGINS=https://www.melcameroun.com,https://melcameroun.com
```

Puis :

```bash
docker compose up -d --build
docker compose ps                       # api + scheduler doivent être 'healthy'
curl -s localhost:8000/health | jq
```

Premier entraînement (nécessaire pour peupler `models/`) :

```bash
docker compose run --rm trainer
docker compose restart api
```

---

## 3. Terminer le TLS

L'API est volontairement **boundée sur `127.0.0.1`** : melcameroun doit passer
par un reverse proxy qui termine le TLS, pas consommer un port HTTP nu.

Caddy, en une ligne :

```
mel-recommend.melcameroun.com {
    reverse_proxy 127.0.0.1:8000
}
```

Si le front et l'API sont sur des hôtes distincts, mettre un Nginx entre les
deux et laisser `API_BIND=127.0.0.1`.

---

## 4. Ré-entraînement automatique

Le scheduler lance `scripts/retrain.sh` selon `RETRAIN_SCHEDULE`
(défaut : dimanche 2h) :

1. export MySQL → CSV (ou parse du dump SQL local si `MYSQL_HOST` est vide) ;
2. `python src/train.py` ;
3. comparaison des métriques avec le modèle précédent ;
4. `POST /reload` — sans downtime, l'API adopts les nouveaux modèles à chaud.

### Politique de déploiement : toujours déployer

Si le NDCG régresse, **le modèle neuf est quand même mis en production** (il a
appris sur des données plus récentes), et une notification `⚠️ warning` part sur
`NOTIFY_WEBHOOK` avec l'amplitude de la baisse.

C'est un arbitrage : on privilégie la fraîcheur des données, et on délègue la
surveillance à un humain. Si vous préférez bloquer, la bascule se fait dans
`scripts/retrain.sh` (variable `DEGRADED` déjà calculée) — il suffit de
conditionner le `reload_api`.

Sans `NOTIFY_WEBHOOK` configuré, cette régression passe **inaperçue**. C'est la
première chose à vérifier après le déploiement.

### Espace des catégories

`ITEM_SPACE=interactions` (défaut) : le modèle ne couvre que les catégories déjà
observées chez les utilisateurs. `ITEM_SPACE=catalog` élargit aux catégories du
catalogue ayant du stock — +43 articles disponibles joignables, mais classés par
similarité de nom seule, ce qui poussait 44 % des premiers résultats sur des
catégories jamais vues. Ne changez pas cette valeur sans le savoir.

---

## 5. Limites connues — à lire avant de promettre la personnalisation

**Ce n'est vrai que du classement par profil client.** Le content-based **par
article** fonctionne dès maintenant : la similarité TF-IDF porte sur les
TITRES des produits, qui sont suffisamment descriptifs dans ce catalogue
(« Robe Bébé En Maille Crème Avec Broderie Cœurs Et Poussins », « Chemise Homme
À Carreaux — Style Chic Et Polyvalent »). Aucune donnée client n'est nécessaire :
un visiteur qui regarde une chemise reçoit d'autres chemises, pas « un article
de Vêtements hommes » choisi au hasard dans le catalogue.

Ce qui reste hors de portée, c'est le **collaboratif par article** — « les gens
qui ont acheté ceci ont aussi acheté cela ». Il exige des interactions par
couple (client, article), et le jeu actuel en compte 110 pour un seuil fixé à
1000. `collaborative_ready: false` dans `results_summary.json` le dit
explicitement. Ce seuil n'est pas un caprice : sous 1000 couples, ALS/SVD
produisent un modèle qui classe au hasard tout en affichant d'excellentes
métriques.

| | ce qui marche aujourd'hui | ce qui attend des données |
|---|---|---|
| Visiteur anonyme | similarité sur les titres, par session | — |
| Client connecté | SVD + ALS par catégorie | classement par article |
| « Vous aimerez aussi » | similarité sur les titres | co-achat |

`GET /similar/{product_id}` utilise la couche article quand elle est disponible
(voir `strategy` dans la réponse) et se replie sur la similarité de catégories
sinon.

**Les métriques ne valident rien.** Sur le jeu réel : 14 clients évaluables,
5 paires distinctes, NDCG@7 = 1.000 pour le modèle **et** pour la baseline de
popularité (lift 0.0). Ces données sont trop pauvres pour distinguer les deux.
Sur le jeu synthétique, plus large, **aucune valeur d'`ALPHA` ne bat la
popularité**. `ALPHA=0.6` est un défaut non validé.

Pour réellement personnaliser par article, il faut collecter :
prix, marque, note moyenne, et les interactions produit par utilisateur
(clics, panier, achat). Tant que ces données manquent, ne présentez pas le
service comme un moteur de personnalisation par article.