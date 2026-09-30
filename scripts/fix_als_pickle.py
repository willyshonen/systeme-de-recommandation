"""
scripts/fix_als_pickle.py
Resauvegarde le modèle ALS avec la bonne référence de classe.
"""

import pickle
import sys
import io
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender import ALSRecommender

MODELS_DIR = ROOT / "models"


class FixedUnpickler(pickle.Unpickler):
    """Redirige __main__.ALSRecommender → src.recommender.ALSRecommender."""

    def find_class(self, module, name):
        if name == "ALSRecommender":
            return ALSRecommender
        return super().find_class(module, name)


print("Chargement de l'ancien pickle...")
with open(MODELS_DIR / "als_model.pkl", "rb") as f:
    old_model = FixedUnpickler(f).load()

print(f"  type : {type(old_model).__module__}.{type(old_model).__name__}")
print(f"  user_factors : {old_model.user_factors.shape}")
print(f"  item_factors : {old_model.item_factors.shape}")

# Resauvegarder avec la bonne classe
print("Resauvegarde...")
with open(MODELS_DIR / "als_model.pkl", "wb") as f:
    pickle.dump(old_model, f)

print("✅ als_model.pkl corrigé — lance maintenant :")
print("   python -m uvicorn api.main:app --reload")
