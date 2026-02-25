#!/bin/bash
# ═══════════════════════════════════════════
#  Cleanup Script — Delete old session data
#  Run via cron: 0 */6 * * * /app/scripts/cleanup.sh
#  Deletes uploads older than 1 hour, outputs older than 24 hours
# ═══════════════════════════════════════════

echo "[$(date)] Running cleanup..."

# Delete uploaded files older than 1 hour
find /app/storage/uploads -mindepth 1 -maxdepth 1 -mmin +60 -exec rm -rf {} \; 2>/dev/null
UPLOADS_DELETED=$?

# Delete output files older than 24 hours
find /app/storage/outputs -mindepth 1 -maxdepth 1 -mmin +1440 -exec rm -rf {} \; 2>/dev/null
OUTPUTS_DELETED=$?

echo "[$(date)] Cleanup complete."
