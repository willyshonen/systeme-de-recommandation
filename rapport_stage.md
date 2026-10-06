# RAPPORT DE STAGE

**Système de recommandation de produits — MEL Cameroun (melcameroun.com)**

| Étudiant | *(Nom Prénom)* | Filière | *(M2 Informatique / Data Science)* |
|---|---|---|---|
| Établissement | *(École / Université)* | Période | *(jj/mm – jj/mm)* |
| Tuteur entreprise | *(Nom)* | Tuteur académique | *(Nom)* |

> 📸 **IMAGE 1 — À PLACER ICI, juste sous le tableau d'en-tête.**
> Capture de la **page d'accueil de melcameroun.com** (catalogue + catégories visibles), sans retraitement. Elle sert de contexte visuel.

---

## 1. Contexte
MEL Cameroun est une **marketplace d'occasion** : chaque article est unique et disparaît dès qu'il est vendu. Les visiteurs voient aujourd'hui tous le même catalogue, sans aucune personnalisation.

## 2. Objectifs
Concevoir un moteur de recommandation adapté à l'occasion, l'exposer via une **API REST** consommable par le front Laravel, et l'industrialiser (entraînement reproductible, déploiement automatisé, supervision).

## 3. Données
Sources : base MySQL MEL (factures, paniers, articles, avis) + événements de navigation collectés par `POST /events`. Jeu actuel : **34 clients, 7 catégories, 51 interactions, 191 articles disponibles**.

> 📸 **IMAGE 2 — À PLACER ICI, entre « Données » et « Méthodes ».**
> **Schéma du pipeline de données** (MySQL → CSV → entraînement → API). À dessiner avec draw.io ou Excalidraw : 4 blocs plus des flèches, rien de plus.

## 4. Méthodes
**Cinq modèles comparés** : popularité (baseline), SVD, ALS implicite (numpy, écrit à la main), content-based TF-IDF, et hybride SVD+CB. Le modèle classe des **catégories**, l'API les traduit en articles **encore disponibles**.

## 5. Architecture et MLOps
Chaîne complète : export MySQL → `src/train.py` → artefacts dans `models/` → API **FastAPI (8 endpoints)** → rechargement à chaud (`POST /reload`) → scheduler Docker hebdomadaire → **MLflow**. CI/CD **GitHub Actions** (4 jobs) : lint, **159 tests**, entraînement **reproductible** (hash SHA-256), build de l'image et smoke test sur l'image poussée.

> 📸 **IMAGE 3 — À PLACER ICI, sous le paragraphe MLOps.**
> Capture du terminal après `pytest tests/ -v` montrant la ligne **« 159 passed »**. Possibilité d'ajouter à côté le workflow GitHub Actions au vert.

## 6. Résultats
Sur le Leave-One-Out, NDCG@7 = **0,97** contre **0,89** pour la baseline. Mais la **validation croisée invalide la personnalisation** (lift **−0,05**) : par la « porte de suffisance », l'API sert alors la **popularité** et l'annonce via `serving_mode`. Le collaboratif reste bloqué par le volume (110 couples relevés / 1000 requis).

> 📸 **IMAGE 4 — À PLACER ICI, après le paragraphe « Résultats ».**
> Capture de **http://localhost:8000/docs** (Swagger) répondant à `GET /recommend/{user_id}`, ou du JSON `GET /health` montrant le champ `serving_mode`. Elle prouve ce que sert réellement l'API.

## 7. Difficultés rencontrées
Extrême sparsité et cold-start ; contraintes **Python 3.13** (pas de bibliothèque Cython → ALS écrit à la main) ; **reproductibilité des métriques** (SVD LAPACK déterministe, BLAS forcé sur un seul thread).

## 8. Conclusion et perspectives
Un système **complet et honnête** est livré : il échoue proprement vers la popularité tant que les données ne valident pas la personnalisation. Perspectives : collecter les événements côté front, **intégration Laravel**, sécurisation des endpoints, **A/B testing** et monitoring.

> 📸 **IMAGE 5 — À PLACER ICI, en toute fin de rapport.**
> Capture de l'**arborescence du projet** dans VS Code (`src/`, `api/`, `tests/`, `scripts/`, `models/`, `Dockerfile`, `.github/`). Elle donne une vue d'ensemble concrète du livrable.
