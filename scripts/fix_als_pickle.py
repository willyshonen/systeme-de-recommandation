"""
scripts/fix_als_pickle.py
Ré-écrit models/als_model.pkl avec la bonne référence de classe.

Pourquoi : un modèle picklé depuis un notebook (ou via `python -c`) stocke la
classe sous le module `__main__`. L'API n'importe que `src.recommender`, donc
`store.load()` lève :

    AttributeError: module '__main__' has no attribute 'ALSRecommender'

Ce script migre les anciens pickles vers `src.recommender.ALSRecommender`.
Depuis la déduplication de la classe dans src/train.py, les nouveaux pickles sont
corrects : ce script n'est plus nécessaire que pour les artefacts existants.

Usage :
    python scripts/fix_als_pickle.py [--models-dir models]
"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import ALSRecommender


class LegacyPickleError(Exception):
    """Le pickle référence une classe définie dans __main__ (notebook / python -c)."""


class StrictUnpickler(pickle.Unpickler):
    """
    Refuse les pickles qui stockent la classe sous `__main__`.

    Ne PAS utiliser un simple `pickle.load` pour détecter le problème : ce script
    fait lui-même `from src.recommender import ALSRecommender`, ce qui place la
    classe dans son propre `__main__` et ferait passer un pickle cassé pour valide.
    """

    def find_class(self, module, name):
        if module == "__main__":
            raise LegacyPickleError(f"référence __main__.{name}")
        return super().find_class(module, name)


class FixedUnpickler(pickle.Unpickler):
    """Redirige __main__.ALSRecommender → src.recommender.ALSRecommender."""

    def find_class(self, module, name):
        if name == "ALSRecommender":
            return ALSRecommender
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    path = args.models_dir / "als_model.pkl"
    if not path.exists():
        print(f"❌ {path} introuvable — rien à réparer")
        return 1

    print(f"Chargement de {path}...")
    try:
        with open(path, "rb") as f:
            StrictUnpickler(f).load()
        print("  → déjà valide, rien à faire")
        return 0
    except LegacyPickleError as e:
        print(f"  → pickle cassé : {e}")

    with open(path, "rb") as f:
        model = FixedUnpickler(f).load()

    print(f"  user_factors : {model.user_factors.shape}")
    print(f"  item_factors : {model.item_factors.shape}")

    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(model, f)
    tmp.replace(path)
    print(f"✅ {path} réécrit avec src.recommender.ALSRecommender")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
