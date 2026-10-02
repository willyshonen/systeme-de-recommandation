#!/bin/sh
# scripts/retrain.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de ré-entraînement automatique — MEL Recommender
#
# Étapes :
#   1. Export MySQL → CSV  (si MYSQL_HOST défini, sinon parse mysql-mel.sql)
#   2. Entraînement du modèle
#   3. Rechargement de l'API sans downtime (POST /reload)
#   4. Notification (webhook Slack/Discord optionnel)
#
# Variables d'environnement :
#   DATA_DIR             chemin données brutes         (défaut: data/raw)
#   MODELS_DIR           chemin modèles                (défaut: models)
#   MLFLOW_TRACKING_URI  URL MLflow                    (défaut: http://mlflow:5000)
#   API_URL              URL de l'API à recharger      (défaut: http://api:8000)
#   RELOAD_SECRET        clé secrète endpoint /reload  (défaut: vide)
#   NOTIFY_WEBHOOK       URL webhook Slack/Discord      (défaut: vide = pas de notif)
#   MYSQL_HOST           hôte MySQL pour export direct  (défaut: vide = parse dump SQL)
#   MYSQL_PORT           port MySQL                     (défaut: 3306)
#   MYSQL_DB             nom de la BDD MEL              (défaut: mel_cameroun)
#   MYSQL_USER           utilisateur MySQL              (défaut: mel_user)
#   MYSQL_PASSWORD       mot de passe MySQL             (défaut: vide)
#   LOG_FILE             fichier de log                 (défaut: /tmp/retrain.log)
# ─────────────────────────────────────────────────────────────────────────────

set -e

DATA_DIR="${DATA_DIR:-data/raw}"
MODELS_DIR="${MODELS_DIR:-models}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
API_URL="${API_URL:-http://api:8000}"
RELOAD_SECRET="${RELOAD_SECRET:-}"
NOTIFY_WEBHOOK="${NOTIFY_WEBHOOK:-}"
LOG_FILE="${LOG_FILE:-/tmp/retrain.log}"
MYSQL_HOST="${MYSQL_HOST:-}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_DB="${MYSQL_DB:-mel_cameroun}"
MYSQL_USER="${MYSQL_USER:-mel_user}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-}"

START_TIME=$(date +%s)

# ── Fonctions utilitaires ──────────────────────────────────────────────────────

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

# Notifie via webhook Slack ou Discord si NOTIFY_WEBHOOK est défini
notify() {
    STATUS="$1"   # success | failure
    MESSAGE="$2"
    ELAPSED="$3"

    if [ -z "$NOTIFY_WEBHOOK" ]; then
        return 0
    fi

    if [ "$STATUS" = "success" ]; then
        EMOJI="✅"
        COLOR="good"
    else
        EMOJI="❌"
        COLOR="danger"
    fi

    PAYLOAD=$(cat <<EOF
{
  "text": "$EMOJI MEL Recommender — Retrain $STATUS",
  "attachments": [{
    "color": "$COLOR",
    "fields": [
      {"title": "Message",  "value": "$MESSAGE",             "short": false},
      {"title": "Durée",    "value": "${ELAPSED}s",          "short": true},
      {"title": "Heure",    "value": "$(date '+%Y-%m-%d %H:%M')", "short": true}
    ]
  }]
}
EOF
)
    curl -s -X POST -H 'Content-type: application/json' \
        --data "$PAYLOAD" "$NOTIFY_WEBHOOK" > /dev/null 2>&1 || true
}

# Recharge l'API sans downtime via POST /reload
reload_api() {
    log "🔄  Rechargement de l'API..."

    RELOAD_URL="${API_URL}/reload"
    if [ -n "$RELOAD_SECRET" ]; then
        RELOAD_URL="${RELOAD_URL}?secret=${RELOAD_SECRET}"
    fi

    RESPONSE=$(curl -s -o /tmp/reload_response.json -w "%{http_code}" \
        -X POST "$RELOAD_URL" 2>/dev/null || echo "000")

    if [ "$RESPONSE" = "200" ]; then
        log "✅  API rechargée avec succès"
        # Affiche les stats avant/après si disponibles
        if command -v python3 > /dev/null 2>&1; then
            python3 -c "
import json, sys
try:
    with open('/tmp/reload_response.json') as f:
        d = json.load(f)
    before = d.get('before', {})
    after  = d.get('after', {})
    print(f'   Avant  : {before.get(\"n_users\",\"?\")} users, {before.get(\"n_categories\",\"?\")} catégories, {before.get(\"n_available_products\",\"?\")} articles dispo')
    print(f'   Après  : {after.get(\"n_users\",\"?\")} users, {after.get(\"n_categories\",\"?\")} catégories, {after.get(\"n_available_products\",\"?\")} articles dispo')
    print(f'   Temps  : {d.get(\"elapsed_sec\",\"?\")}s')
except Exception as e:
    print(f'   (lecture réponse impossible: {e})')
" | tee -a "$LOG_FILE" || true
        fi
        return 0
    else
        log "⚠️   API non accessible (HTTP $RESPONSE) — les nouveaux modèles seront chargés au prochain démarrage"
        return 0  # Ne pas faire échouer le retrain si l'API est temporairement indispo
    fi
}

# ── Début du pipeline ──────────────────────────────────────────────────────────

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "🚀  Démarrage du ré-entraînement MEL Recommender"
log "    DATA_DIR   = $DATA_DIR"
log "    MODELS_DIR = $MODELS_DIR"
log "    MLFLOW     = $MLFLOW_TRACKING_URI"
log "    API_URL    = $API_URL"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Étape 1 : Export des données ───────────────────────────────────────────────
log "📦  Étape 1/3 : Export des données..."

if [ -n "$MYSQL_HOST" ]; then
    # Export direct depuis MySQL (production)
    log "    Mode : export direct MySQL → $MYSQL_HOST:$MYSQL_PORT/$MYSQL_DB"
    python scripts/mysql_export.py \
        --host     "$MYSQL_HOST" \
        --port     "$MYSQL_PORT" \
        --database "$MYSQL_DB" \
        --user     "$MYSQL_USER" \
        --password "$MYSQL_PASSWORD" \
        --out      "$DATA_DIR"
    if [ $? -ne 0 ]; then
        log "❌  Erreur lors de l'export MySQL"
        END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
        notify "failure" "Erreur export MySQL" "$ELAPSED"
        exit 1
    fi
    log "✅  Export MySQL terminé"
else
    # Parse le dump SQL local (développement / CI)
    SQL_FILE="$DATA_DIR/mysql-mel.sql"
    if [ ! -f "$SQL_FILE" ]; then
        log "⚠️   Dump SQL introuvable ($SQL_FILE) — skip export"
    else
        log "    Mode : parsing dump SQL ($SQL_FILE)"
        python scripts/sql_to_csv.py \
            --sql "$SQL_FILE" \
            --out "$DATA_DIR"
        if [ $? -ne 0 ]; then
            log "❌  Erreur lors du parsing SQL"
            END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
            notify "failure" "Erreur parsing SQL" "$ELAPSED"
            exit 1
        fi
        log "✅  Parsing SQL terminé"
    fi
fi

# ── Étape 2 : Entraînement ─────────────────────────────────────────────────────
log "🤖  Étape 2/3 : Entraînement du modèle..."

python src/train.py \
    --data-dir    "$DATA_DIR" \
    --models-dir  "$MODELS_DIR" \
    --k-factors   150 \
    --als-factors 32 \
    --als-reg     0.1 \
    --alpha       1.0

if [ $? -ne 0 ]; then
    log "❌  Erreur lors de l'entraînement"
    END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
    notify "failure" "Erreur entraînement" "$ELAPSED"
    exit 1
fi
log "✅  Entraînement terminé"

# Affiche les métriques
if [ -f "$MODELS_DIR/results_summary.json" ] && command -v python3 > /dev/null 2>&1; then
    python3 -c "
import json
try:
    with open('$MODELS_DIR/results_summary.json') as f:
        d = json.load(f)
    m = d.get('metrics', {})
    print(f'   Precision@10 = {m.get(\"Precision@10\", \"?\")}')
    print(f'   Recall@10    = {m.get(\"Recall@10\", \"?\")}')
    print(f'   NDCG@10      = {m.get(\"NDCG@10\", \"?\")}')
    print(f'   Users        = {d.get(\"n_users\", \"?\")}')
    print(f'   Catégories   = {d.get(\"n_items\", \"?\")}')
except Exception as e:
    print(f'   (lecture métriques impossible: {e})')
" | tee -a "$LOG_FILE" || true
fi

# ── Étape 3 : Rechargement de l'API ───────────────────────────────────────────
log "🔄  Étape 3/3 : Rechargement de l'API..."
reload_api

# ── Fin ────────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✅  Pipeline terminé en ${ELAPSED}s"
log "    Modèles disponibles dans : $MODELS_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Lecture métriques pour la notification
METRICS_MSG="Pipeline OK"
if [ -f "$MODELS_DIR/results_summary.json" ] && command -v python3 > /dev/null 2>&1; then
    METRICS_MSG=$(python3 -c "
import json
try:
    with open('$MODELS_DIR/results_summary.json') as f:
        d = json.load(f)
    m = d.get('metrics', {})
    print(f'NDCG@10={m.get(\"NDCG@10\",\"?\")} | {d.get(\"n_users\",\"?\")} users | {d.get(\"n_items\",\"?\")} catégories')
except Exception:
    print('Pipeline OK')
" 2>/dev/null || echo "Pipeline OK")
fi

notify "success" "$METRICS_MSG" "$ELAPSED"

exit 0
