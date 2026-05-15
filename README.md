# Reliability of Tabular ML Under Distribution Shift — Code Release

Companion code and reproducibility package for the paper
**"Reliability of Tabular ML Under Distribution Shift: An Empirical Failure-Modes Map for Conformal Prediction and Post-hoc Calibration."**

## What this repository contains

| Path | Contents |
|------|----------|
| `src/` | Python source for all experimental blocks (split / weighted / Mondrian CP with APS score; 4 post-hoc calibrators; budget-axis baseline; deploy-time decision-tree analysis; ESS-diagnostic robustness checks). |
| `src/figures/make_figures.py` | Renders the three publication figures from result parquets. |
| `results/` | Selected small result artefacts needed to reproduce paper tables and figures without re-running the full grid: per-cell parquet logs, JSON summaries, and per-block analyses (Markdown). See `results/README.md`. |
| `paper-figures/` | The three rendered figures (`fig1_coverage_drop.pdf`, `fig2_ess_vs_coverage.pdf`, `fig3_budget_axis.pdf`) included in the paper. |
| `reproduce.sh` | One-command end-to-end reproduction script (Blocks 1 v2 → 3 → 4.5 → 5 → figures). |
| `requirements.txt` | Pinned dependency versions used in the paper. |
| `DATA.md` | Per-dataset notes: source, license, download mode (automatic vs manual), known caveats. |
| `CITATION.cff` | Machine-readable citation for this repository. |
| `LICENSE` | MIT (code) — see file for the precise terms. |
| `PUSH_TO_GITHUB.md` | One-time commands for pushing this directory as a separate GitHub repo. |

## What the paper reproduces from this code

The paper reports five experimental blocks:

- **Block 1 v2** — main 10-task grid: 3 CP variants × 4 calibrators × 3 GBM learners × 5 seeds → 4,140 result rows (`results/block1_full_v2.parquet`). Source: `src/block1_full.py`, `src/block1_full_v2.py`, `src/extra_tasks.py`, `src/cp_variants.py`, `src/calibrators.py`.
- **Block 3** — budget-axis baseline (post-hoc CP vs retrain at B ∈ {0, 50, 500, ~all}): combined `results/block3_budget_axis.parquet` + `block3_budget_axis_v2.parquet` → 360 rows. Source: `src/block3_budget_axis.py`, `src/block3_budget_axis_v2.py`.
- **Block 4.5** — deploy-time decision-tree analysis: 10-task feature vectors → LOO-CV (`results/block4_5_v2_deploy_time.{parquet,json}`). Source: `src/block4_5_v2_deploy_time.py`.
- **Block 5** — ESS-diagnostic robustness checks: per-task ESS / max-weight (`results/block5_weight_diagnostics.parquet`) and Online News `kw_*`-removal (`results/block5_onlinenews_no_kw.parquet`). Source: `src/block5_robustness.py`.
- **Figures 1–3** — rendered from the above by `src/figures/make_figures.py`.

## Setup

```bash
# Tested with Python 3.12 on macOS 14 (16-core, CPU only).
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No GPU required. All experiments run on CPU only.

## Reproduction

### Quick mode — just regenerate figures from cached results

```bash
python3 src/figures/make_figures.py
# Writes fig{1,2,3}_*.{pdf,png} into ./paper-figures/
```

Runtime: <30 s. Uses the parquet files already in `results/`.

### Full mode — rerun all experiments end-to-end

```bash
bash reproduce.sh
```

This runs, in order:

1. **Block 1 v2** main grid (~6 min on 16-core CPU).
2. **Block 3 v2** budget axis (3 new tasks; combine with Block 3 v1 for full table) (~3 min).
3. **Block 4.5 v2** deploy-time decision-tree analysis (~1 min).
4. **Block 5** ESS diagnostics + Online News no-`kw_*` robustness (~3 min).
5. **Figure regeneration** (~30 s).

Total wall time: **≈ 15 minutes** on a 16-core laptop. Outputs land back in `results/` (overwriting the cached versions) and `paper-figures/`.

If you also want to run **Block 3 v1** (the original 7-task budget axis on which Block 3 v2 builds), run `python3 src/block3_budget_axis.py` separately (~7 min).

## Where outputs appear

- Block 1 v2: `results/block1_full_v2.parquet` (4,140 rows × 16 columns).
- Block 3 v1: `results/block3_budget_axis.parquet` (252 rows; run separately).
- Block 3 v2: `results/block3_budget_axis_v2.parquet` (108 new rows).
- Block 4.5: `results/block4_5_v2_deploy_time.{parquet,json}`.
- Block 5: `results/block5_weight_diagnostics.parquet`, `block5_onlinenews_no_kw.parquet`.
- Figures: `paper-figures/fig{1,2,3}_*.pdf` and `.png`.
- Per-block analyses: `results/block{1,3,4_5,5}_*.md` (paper-aligned summary tables).

## Data sources and licenses

Public datasets only. See `DATA.md` for the per-dataset breakdown including download mode, license, and any preprocessing. Briefly:

| Dataset | Source | Download |
|---------|--------|----------|
| ACSIncome (4 OOD tasks: state, temporal, sex, race) | Hugging Face mirror `birkhoffg/folktables-acs-income` (≈ 200 MB cached) | Automatic via `datasets` package |
| UCI Adult | OpenML id=2 (`adult` v2) | Automatic via `sklearn.datasets.fetch_openml` |
| Bank Marketing | OpenML id=1461 (`bank-marketing` v1) | Automatic |
| Taiwan Credit Default | OpenML id=42477 | Automatic |
| Diabetes 130-US | OpenML id=4541 | Automatic |
| SpeedDating | OpenML id=40536 | Automatic |
| OnlineNews Popularity | OpenML id=4545 | Automatic |

First-time fetches will cache under `~/.cache/huggingface/` and `~/scikit_learn_data/`. Cumulative cache size: < 1 GB.

## Known limitations

- **HGB-only Block 3.** The budget-axis comparison uses `HistGradientBoostingClassifier` only; cross-learner generalisation in this block is left as future work.
- **10-task panel size.** The ESS / coverage diagnostic patterns reported in `paper-figures/fig2_ess_vs_coverage.pdf` are empirical separations from a 10-task panel; treat them as diagnostic findings rather than universal thresholds.
- **HELOC dropped.** OpenML id=43890 exhibited Brier = 0.000 across all CP / calibrator combinations on our pipeline (a clear label-leakage signature). It is excluded from the final 10-task study; see `results/dataset_quality_notes.md`.
- **`folktables` package** itself currently 403s against `www2.census.gov` PUMS endpoints on many networks (Cloudflare). We work around this by using the Hugging Face mirror `birkhoffg/folktables-acs-income`. If the original Census mirror is restored, switching loaders is a single-import change in `src/block1_full.py`.

## Citation

```
@misc{tabular_cp_failure_modes_2025,
  title  = {Reliability of Tabular ML Under Distribution Shift: An Empirical
            Failure-Modes Map for Conformal Prediction and Post-hoc Calibration},
  year   = {2025},
  note   = {Preprint},
  url    = {<paper URL to be inserted>},
}
```

(See `CITATION.cff` for machine-readable form.)

## Contact

Issues / questions: open an issue in this repository.
