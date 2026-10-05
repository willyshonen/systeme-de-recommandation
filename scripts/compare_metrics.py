#!/usr/bin/env python
"""
Compare deux results_summary.json et signale une régression de NDCG.

Utilisé par scripts/retrain.sh entre l'entraînement et le rechargement de l'API.
Politique de déploiement : « toujours déployer ». Ce script ne bloque donc rien,
il produit un message d'alerte si le modèle neuf est moins bon que le précédent.

Sortie :
  - chaîne vide (et code 0) si pas de régression, ou si la comparaison est
    impossible (premier entraînement, fichier illisible, k différent...)
  - message descriptif (et code 0) si régression
  - code 1 uniquement en cas d'erreur d'appel

Pourquoi la clé est dynamique : `eval_k` est plafonné au nombre de catégories
(7 sur le jeu réel, pas 10), donc 'NDCG@10' n'existe pas dans le résumé. Une
clé en dur afficherait '?' et ne détecterait jamais rien.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tolérance : une baisse millésimale due au bruit numérique n'est pas une
# régression. Au-delà, on alerte.
TOLERANCE = 1e-6


def extract_ndcg(summary: dict) -> tuple[float | None, int | None]:
    """Retourne (NDCG@eval_k, eval_k) ou (None, None) si absent."""
    metrics = summary.get("metrics") or {}
    k = metrics.get("eval_k")
    if k is None:
        return None, None
    value = metrics.get(f"NDCG@{k}")
    if not isinstance(value, (int, float)):
        return None, k
    return float(value), int(k)


def compare(old: dict, new: dict) -> str:
    """Retourne un message si le NDCG a baissé, sinon une chaîne vide."""
    old_ndcg, old_k = extract_ndcg(old)
    new_ndcg, new_k = extract_ndcg(new)

    if old_ndcg is None or new_ndcg is None:
        # Première exécution, ou résumé sans métriques exploitables : on ne peut
        # pas conclure, donc on n'alerte pas (faux positif pire que silence).
        return ""
    if old_k != new_k:
        return (
            f"NDCG non comparable : eval_k a changé ({old_k} -> {new_k}), "
            f"{old_ndcg:.4f} -> {new_ndcg:.4f}"
        )
    if new_ndcg < old_ndcg - TOLERANCE:
        drop = old_ndcg - new_ndcg
        return (
            f"NDCG@{new_k} dégradé : {old_ndcg:.4f} -> {new_ndcg:.4f} (-{drop:.4f}) | "
            f"users {old.get('n_users', '?')} -> {new.get('n_users', '?')} | "
            f"interactions {old.get('n_interactions', '?')} -> "
            f"{new.get('n_interactions', '?')}"
        )
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="results_summary.json AVANT entraînement")
    parser.add_argument("new", help="results_summary.json APRÈS entraînement")
    args = parser.parse_args(argv)

    try:
        old = json.loads(Path(args.old).read_text(encoding="utf-8"))
        new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"comparaison impossible : {e}", file=sys.stderr)
        return 1

    message = compare(old, new)
    if message:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())