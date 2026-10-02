#!/bin/sh
# scripts/docker-entrypoint-scheduler.sh
# Point d'entrée du container scheduler.
# Installe le cron job avec la planification configurée puis démarre le daemon.

set -e

# ── Planification ──────────────────────────────────────────────────────────────
# RETRAIN_SCHEDULE : expression cron (défaut: dimanche à 2h du matin)
# Exemples :
#   "0 2 * * 0"   → chaque dimanche à 2h
#   "0 3 * * *"   → chaque jour à 3h
#   "0 2 * * 1"   → chaque lundi à 2h
SCHEDULE="${RETRAIN_SCHEDULE:-0 2 * * 0}"

LOGFILE="/var/log/retrain.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🕐  MEL Scheduler — démarrage"
echo "    Planification : $SCHEDULE"
echo "    Prochain retrain : $(date -d 'next sunday 02:00' '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'voir cron')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Exporte toutes les variables d'environnement vers /etc/environment ─────────
# (nécessaire pour que cron hérite des variables Docker)
printenv | grep -v "no_proxy" > /etc/environment

# ── Génère le fichier crontab ──────────────────────────────────────────────────
cat > /etc/cron.d/mel-retrain <<EOF
# MEL Recommender — ré-entraînement automatique
# Planification : $SCHEDULE
SHELL=/bin/sh
PATH=/usr/local/bin:/usr/bin:/bin

$SCHEDULE root cd /app && sh scripts/retrain.sh >> $LOGFILE 2>&1
EOF

chmod 0644 /etc/cron.d/mel-retrain

echo "✅  Cron job installé :"
cat /etc/cron.d/mel-retrain

# ── Premier retrain immédiat si RETRAIN_ON_START=1 ────────────────────────────
if [ "${RETRAIN_ON_START:-0}" = "1" ]; then
    echo "🚀  RETRAIN_ON_START=1 — lancement immédiat du retrain..."
    sh /app/scripts/retrain.sh >> "$LOGFILE" 2>&1 &
fi

# ── Crée le fichier de log et le lie sur stdout ────────────────────────────────
touch "$LOGFILE"

echo "✅  Scheduler démarré. Logs dans $LOGFILE"
echo "    Pour voir les logs : docker logs mel-scheduler"
echo ""

# Démarre cron en foreground (pour que le container reste actif)
# et tail le log en parallèle pour que docker logs fonctionne
cron && tail -f "$LOGFILE"
