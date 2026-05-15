#!/usr/bin/env bash
# Reliability of Tabular ML Under Distribution Shift — end-to-end reproduction.
#
# Runs the five experimental blocks in order, then regenerates the three paper
# figures. All experiments are CPU-only. Total wall time on a 16-core laptop
# with warm dataset caches: approximately 15 minutes.
#
# Usage:
#   ./reproduce.sh           # full reproduction
#   ./reproduce.sh --smoke   # smoke test: figure regeneration only (~30 s)
#
# Outputs land back in results/ and paper-figures/.

set -euo pipefail

cd "$(dirname "$0")"

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then
    SMOKE=1
fi

PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

mkdir -p results paper-figures

if [ "$SMOKE" = "1" ]; then
    echo "==> SMOKE TEST: figure regeneration only"
    python3 src/figures/make_figures.py
    echo "==> Smoke test complete. Outputs: paper-figures/fig{1,2,3}_*.pdf"
    exit 0
fi

cd "$(dirname "$0")"
echo "==> [1/5] Block 1 v2  main 10-task grid (~6 min)"
( cd "$PWD" && python3 -u src/block1_full_v2.py )

echo "==> [2/5] Block 3 v1  original-7 budget axis (~7 min)"
( cd "$PWD" && python3 -u src/block3_budget_axis.py )

echo "==> [3/5] Block 3 v2  budget axis on 3 new tasks (~3 min)"
( cd "$PWD" && python3 -u src/block3_budget_axis_v2.py )

echo "==> [4/5] Block 4.5  deploy-time decision tree (~1 min)"
( cd "$PWD" && python3 -u src/block4_5_v2_deploy_time.py )

echo "==> [5/5] Block 5  ESS diagnostics + Online News no-kw_* (~3 min)"
( cd "$PWD" && python3 -u src/block5_robustness.py )

echo "==> Regenerating paper figures (~30 s)"
python3 -u src/figures/make_figures.py

echo
echo "==> DONE. Outputs:"
echo "    results/block1_full_v2.parquet"
echo "    results/block3_budget_axis.parquet"
echo "    results/block3_budget_axis_v2.parquet"
echo "    results/block4_5_v2_deploy_time.{parquet,json}"
echo "    results/block5_weight_diagnostics.parquet"
echo "    results/block5_onlinenews_no_kw.parquet"
echo "    paper-figures/fig{1,2,3}_*.{pdf,png}"
