#!/usr/bin/env bash
# Dispatcher for the COS760 Docker image.
# Usage: docker compose run --rm cos760 [rq1|rq2|viz1|viz2|all|shell]
set -euo pipefail

case "${1:-help}" in
  rq1)
    shift
    exec python run_rq1.py "$@"
    ;;
  rq2)
    shift
    exec python run_rq2.py "$@"
    ;;
  viz1)
    exec python visualize_rq1.py
    ;;
  viz2)
    exec python visualize_rq2.py
    ;;
  all)
    echo "[entrypoint] Running full pipeline: rq1 → viz1 → rq2 → viz2"
    python run_rq1.py
    python visualize_rq1.py
    python run_rq2.py
    python visualize_rq2.py
    echo "[entrypoint] Done. Results written to /app/results/ and /app/outputs/."
    ;;
  shell)
    exec bash
    ;;
  help | *)
    cat <<'EOF'
COS760 Cross-Lingual Embedding Project — Docker image

Usage:
  docker compose run --rm cos760 <command> [args...]

Commands:
  rq1   [args]  Run RQ1 alignment pipeline (FastText + CCA/KCCA/VecMap)
  rq2   [args]  Run RQ2 data-efficiency pipeline (zero-shot NER learning curves)
  viz1          Regenerate RQ1 visualisations from results/rq1_results.csv
  viz2          Regenerate RQ2 visualisations from results/rq2_results.csv
  all           Run the full pipeline end-to-end (rq1 → viz1 → rq2 → viz2)
  shell         Open an interactive bash shell inside the container

RQ2 options (passed directly to run_rq2.py):
  --langs zul nso tsn
  --fractions 1.0 0.75 0.5 0.25 0.1 0.05
  --methods CCA KCCA VecMap
  --split test|dev
  --force          (retrain and overwrite cached artifacts)

Examples:
  docker compose run --rm cos760 rq1
  docker compose run --rm cos760 rq2 --langs tsn --fractions 0.25 0.05
  docker compose run --rm cos760 all
EOF
    ;;
esac
