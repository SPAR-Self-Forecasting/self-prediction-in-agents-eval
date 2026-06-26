#!/usr/bin/env bash
# Periodically kill orphan SWE-bench containers older than the threshold.
# mini-extra's container cleanup uses backgrounded `docker stop &` which
# often races on Windows; cells finish & cache files land but containers
# linger as `sleep 2h` until killed. They eat Docker daemon registry.
LOG="experiments/2026-05-16_self_prediction/logs/container_janitor.log"
echo "=== janitor start $(date -Iseconds) ===" >> "$LOG"
THRESHOLD_MIN=20
while true; do
    # List "ID|Status" lines with status >= THRESHOLD_MIN minutes ago
    killed=0
    while IFS='|' read -r cid status; do
        # "Up X minutes" / "Up X hours" / "Up X seconds"
        if [[ "$status" =~ Up[[:space:]]+([0-9]+)[[:space:]]+(second|minute|hour) ]]; then
            n=${BASH_REMATCH[1]}; unit=${BASH_REMATCH[2]}
            mins=0
            case "$unit" in
                second*) mins=0 ;;
                minute*) mins=$n ;;
                hour*)   mins=$((n*60)) ;;
            esac
            if (( mins >= THRESHOLD_MIN )); then
                docker kill "$cid" >/dev/null 2>&1 && ((killed++))
            fi
        fi
    done < <(docker ps --format '{{.ID}}|{{.Status}}' 2>/dev/null)
    if (( killed > 0 )); then
        echo "[$(date -Iseconds)] killed $killed orphan containers; remaining=$(docker ps -q | wc -l)" >> "$LOG"
    fi
    # Also reap exited containers
    pruned=$(docker container prune -f --filter "until=10m" 2>/dev/null | grep -oE 'Total reclaimed space: [^ ]+ [^ ]+' || true)
    if [ -n "$pruned" ]; then
        echo "[$(date -Iseconds)] pruned exited: $pruned" >> "$LOG"
    fi
    sleep 300
done
