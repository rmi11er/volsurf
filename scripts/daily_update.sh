#!/bin/bash
# Daily update script for volsurf
#
# This script performs the daily data ingestion, surface fitting, and analytics
# calculation pipeline. Schedule with cron to run after market close.
#
# Example crontab entry (runs at 6 PM ET Monday-Friday):
# 0 18 * * 1-5 /path/to/vol-modeling/scripts/daily_update.sh
#
# Prerequisites:
# - Theta Terminal must be running (or auto-started by this script)
# - .env file must be configured with credentials

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/daily_update_$(date +%Y%m%d).log"
SYMBOL="${1:-SPY}"
DATE="${2:-today}"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Helper function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Change to project directory
cd "$PROJECT_DIR"

log "=========================================="
log "Starting daily update for $SYMBOL"
log "=========================================="

# Resolve date
if [ "$DATE" == "today" ]; then
    DATE=$(date +%Y-%m-%d)
fi

log "Target date: $DATE"

# Check if Theta Terminal is needed and start it
if [ -f "$PROJECT_DIR/vendor/ThetaTerminal.jar" ]; then
    # Check if terminal is running
    if ! pgrep -f "ThetaTerminal" > /dev/null; then
        log "Starting Theta Terminal..."
        uv run volsurf terminal start
        sleep 10  # Wait for terminal to initialize
    else
        log "Theta Terminal already running"
    fi
fi

# Step 1: Ingest daily data
log ""
log "Step 1: Ingesting options data..."
if uv run volsurf ingest daily "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"; then
    log "Data ingestion completed"
else
    log "WARNING: Data ingestion failed or no new data"
fi

# Step 2: Fit volatility surfaces
log ""
log "Step 2: Fitting volatility surfaces..."
if uv run volsurf fit surfaces "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"; then
    log "Surface fitting completed"
else
    log "ERROR: Surface fitting failed"
    exit 1
fi

# Step 3: Fit term structure
log ""
log "Step 3: Fitting term structure..."
if uv run volsurf fit term-structure "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"; then
    log "Term structure fitting completed"
else
    log "WARNING: Term structure fitting failed"
fi

# Step 4: Calculate realized volatility
log ""
log "Step 4: Calculating realized volatility..."
if uv run volsurf analyze realized-vol "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"; then
    log "Realized vol calculation completed"
else
    log "WARNING: Realized vol calculation failed (may need more historical data)"
fi

# Step 5: Calculate VRP metrics
log ""
log "Step 5: Calculating VRP metrics..."
if uv run volsurf analyze vrp "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"; then
    log "VRP calculation completed"
else
    log "WARNING: VRP calculation failed (may need more data)"
fi

# Step 6: Generate summary
log ""
log "Step 6: Generating summary..."
uv run volsurf analyze surface-summary "$SYMBOL" --date "$DATE" 2>&1 | tee -a "$LOG_FILE"

log ""
log "=========================================="
log "Daily update completed successfully!"
log "=========================================="

# Cleanup old logs (keep 30 days)
find "$LOG_DIR" -name "daily_update_*.log" -mtime +30 -delete 2>/dev/null || true
