#!/usr/bin/env bash
set -u
LIST="experiments/2026-05-16_self_prediction/logs/image_list.txt"
LOG="experiments/2026-05-16_self_prediction/logs/pull_images.log"
echo "=== Starting pull at $(date -Iseconds) ===" > "$LOG"

pull_one() {
    local img="$1"
    local start=$(date +%s)
    echo "[START] $img" >> "$LOG"
    if docker pull "$img" >/dev/null 2>"$LOG.err"; then
        local end=$(date +%s)
        echo "[DONE]  $img in $((end-start))s" >> "$LOG"
    else
        local end=$(date +%s)
        echo "[FAIL]  $img in $((end-start))s" >> "$LOG"
        echo "---- error output ----" >> "$LOG"
        cat "$LOG.err" >> "$LOG"
        echo "---------------------" >> "$LOG"
    fi
}
export -f pull_one
export LOG

cat "$LIST" | xargs -I{} -P 4 bash -c 'pull_one "$@"' _ {}

echo "=== Finished at $(date -Iseconds) ===" >> "$LOG"
