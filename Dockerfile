FROM python:3.13-slim

# Métadonnées
LABEL maintainer="MEL Cameroun"
LABEL description="API de recommandation de produits — modèle Hybride"

# Dossier de travail
WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (couche cachée séparément)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source de l'API
COPY api/ ./api/
COPY src/__init__.py ./src/__init__.py
COPY src/recommender.py ./src/recommender.py
COPY src/train.py ./src/train.py

# Le dossier models/ est monté comme volume en production
# En dev, on copie les modèles pré-entraînés
COPY models/ ./models/

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Port exposé
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
