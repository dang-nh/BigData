#!/usr/bin/env bash
# Start Neo4j via Compose, wait for healthy, run run_pipeline.py --stage all.
# Logs: data/results/pipeline_full.log, summary: data/results/agent_pipeline_summary.txt
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$PROJ/data/results"
LOG="$RESULTS/pipeline_full.log"
SUMMARY="$RESULTS/agent_pipeline_summary.txt"
mkdir -p "$RESULTS"
: > "$LOG"

cd "$PROJ" || exit 1

NEO4J_OK="no"
NEO4J_FINAL_STATUS="unknown"

{
  echo "=== $(date -Iseconds) docker compose up neo4j ==="
  docker compose -f docker/docker-compose.yml up -d neo4j
} 2>&1 | tee -a "$LOG"

NEO4J_CID=$(docker compose -f docker/docker-compose.yml ps -q neo4j)
if [ -n "$NEO4J_CID" ]; then
  for i in $(seq 1 90); do
    NEO4J_FINAL_STATUS=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$NEO4J_CID" 2>/dev/null || echo "inspect_error")
    echo "$(date -Iseconds) poll $i neo4j Health.Status=$NEO4J_FINAL_STATUS" | tee -a "$LOG"
    if [ "$NEO4J_FINAL_STATUS" = "healthy" ]; then
      NEO4J_OK="yes"
      break
    fi
    sleep 2
  done
else
  NEO4J_FINAL_STATUS="no_container_id"
  echo "$(date -Iseconds) ERROR: empty neo4j container id" | tee -a "$LOG"
fi

set +e
echo "=== $(date -Iseconds) run_pipeline.py --stage all ===" | tee -a "$LOG"
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ] && [ -x "$PROJ/.venv/bin/python" ]; then PYTHON="$PROJ/.venv/bin/python"; fi
if [ -z "$PYTHON" ]; then PYTHON="python3"; fi
"$PYTHON" run_pipeline.py --stage all 2>&1 | tee -a "$LOG"
PIPE_EXIT="${PIPESTATUS[0]}"
set -e

{
  echo "=== agent_pipeline_summary $(date -Iseconds) ==="
  echo "Pipeline exit code: $PIPE_EXIT"
  echo "Neo4j started OK: $NEO4J_OK (final Health.Status='$NEO4J_FINAL_STATUS')"
  echo "--- Last 50 lines of $LOG ---"
  tail -n 50 "$LOG"
} > "$SUMMARY"

printf '%s' "$PIPE_EXIT" > "$RESULTS/pipeline_exit_code.txt"

exit "$PIPE_EXIT"
