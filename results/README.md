# results/

Selected small result artefacts shipped with the code release so that the paper's tables and figures can be reproduced without re-running the full experimental grid.

| Block | Files | Purpose |
|-------|-------|---------|
| Block 1 v2 (10-task main grid) | `block1_full_v2.parquet` (198 KB, 4,140 rows × 16 cols), `block1_full_analysis.md` | Per-cell coverage / set size / ECE / Brier / accuracy for every (task, learner, calibrator, CP variant, seed, split). Source of Table 1 and §4.1. |
| Block 3 v1 + v2 (budget axis) | `block3_budget_axis.parquet`, `block3_budget_axis_v2.parquet`, `block3_budget_axis_analysis.md` | Per-cell coverage / set size for the 3 budget-axis strategies (id_postcp, recalibrate, retrain) across budgets B ∈ {0, 50, 500, ≈all}. Source of Tables 4 and 5 and Figure 3. |
| Block 4.5 v2 (deploy-time tree) | `block4_5_v2_deploy_time.{parquet,json}`, `block4_5_deploy_time.md` | Per-task deploy-time feature vectors (ESS / n_cal, log-density-ratio statistics, model-output KL) plus the LOO-CV results. Source of §4.7 and Aid 1 in §5. |
| Block 5 (ESS + robustness) | `block5_weight_diagnostics.parquet`, `block5_onlinenews_no_kw.parquet`, `block5_robustness_analysis.md` | Per-task ESS and max-weight diagnostics (Figure 2 + Table 3). Online News-without-`kw_*` robustness check (Appendix B). |
| Diagnostics & dataset notes | `dataset_quality_notes.md`, `block0_sanity_v3.json`, `block_v2_analysis.md`, `block4_recommender_analysis.md`, `block4_recommender.json`, `block4_5_deploy_time.json` | Dataset-quality decisions (esp. HELOC label-leakage exclusion); v1 → v2 task-expansion analysis; earlier Block 4 label-derived recommender results (kept for transparency on the 7→10-task regression). |

## Regenerating these files from scratch

Run `bash ../reproduce.sh` from the repo root. The script overwrites every parquet/JSON in this directory and re-renders the figures in `../paper-figures/`.

End-to-end runtime on a 16-core CPU laptop with warm dataset caches: ≈ 15 minutes.

## What is NOT in this directory

Intermediate logs (`*.log`) and stdout transcripts from the live experiment runs are NOT shipped. They are recoverable by re-running `reproduce.sh`. We ship only the structured parquet/JSON outputs that the paper actually cites.
