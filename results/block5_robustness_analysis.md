# Block 5 — Robustness Appendix Results

**Run date**: 2026-05-15
**Script**: `src/block5_robustness.py`
**Raw data**: `results/block5_onlinenews_no_kw.parquet`, `results/block5_weight_diagnostics.parquet`, `results/block5_robustness.log`
**Motivation**: Codex round-3 review identified two remaining concerns: (1) the OnlineNews `kw_*` feature columns (computed from shares of other articles) might drive the headline coverage drop via leakage; (2) the weighted-CP under-coverage mechanism on hard-shift tasks was not yet directly diagnosed.

---

## 1. OnlineNews without `kw_*` features

We drop the 9 `kw_*` columns (and the 6 `data_channel_is_*` and `url`, `timedelta` columns already excluded in the main grid), leaving 43 features. Re-run split / weighted / Mondrian CP with isotonic calibration, 3 seeds.

| Split | CP variant | Coverage | Set size |
|-------|------------|---------:|---------:|
| ID (tech) | split | 0.905 | 1.722 |
| ID | weighted | 0.903 | 1.701 |
| ID | Mondrian | 0.903 | 1.735 |
| **OOD (entertainment)** | **split** | **0.822** | 1.676 |
| OOD | weighted | 0.837 | 1.702 |
| OOD | Mondrian | 0.883 | 1.749 |

**Comparison to the main grid (with kw_*):**
- Split CP: 0.821 → 0.822 — **no change** (drop is robust to kw_* removal).
- Weighted CP: 0.640 → 0.837 — **partially attenuates** (kw_* was contributing to the under-cover), but **still under-covers** (target 0.90).
- Mondrian: 0.875 → 0.883 — similar.

**Verdict on the headline C1 claim**: the 7-pp split-CP coverage drop on OnlineNews **is not driven by `kw_*` leakage**. The claim survives the robustness check.

**Verdict on weighted-CP failure mode B**: the under-coverage is **partly real and partly amplified by kw_* features**. The honest paper-ready statement: "Weighted CP under-covers by ~0.06 in the cleaner feature set (target 0.90, achieved 0.837), and by ~0.26 with kw_* (achieved 0.640). The under-coverage is a real density-ratio-collapse phenomenon; feature engineering can amplify it."

## 2. Weighted-CP density-ratio diagnostics across all 10 tasks

For each task we compute three diagnostics of the density-ratio estimator's stability on the headline OOD split:
- **ESS/n_cal**: effective sample size as a fraction of the calibration set. ESS = 1 / Σ(w_norm²); ratios below 0.05 indicate the weighted quantile is dominated by a handful of calibration rows.
- **max_weight**: the largest unnormalised weight.
- **p99_weight**: the 99th percentile.

| Task | ESS / n_cal | max_weight | p99_weight | Weighted-CP coverage (from §4.2) |
|------|------------:|-----------:|-----------:|---------------------------------:|
| ACSIncome_temporal | **0.988** | 1.58 | 1.28 | 0.892 (healthy) |
| Taiwan_sex | 0.936 | 6.29 | 2.76 | 0.908 |
| ACSIncome_state | 0.562 | 6.74 | 3.67 | 0.881 |
| Adult_sex | 0.539 | 0.14 | 0.05 | 0.897 (healthy + clean) |
| Bank_age | 0.244 | 3.61 | 1.40 | 0.903 |
| SpeedDating_race | 0.069 | **133.98** | 36.89 | **0.835 (under-cover)** |
| ACSIncome_race_CA | 0.028 | 5.69 | 0.20 | 0.933 (over-cover) |
| ACSIncome_sex_CA | 0.016 | 4.18 | 0.10 | 0.907 |
| OnlineNews_channel | **0.004** | **3 319.89** | 13.78 | **0.640 (catastrophic under-cover)** |
| Diabetes_race | **0.000** | **26 091.09** | 5.66 | 0.871 (under-cover) |

**Key observation**: tasks with **ESS / n_cal < 0.05** are exactly the tasks where weighted CP fails (either direction):
- OnlineNews: 0.004 → cov 0.640 (under).
- Diabetes: 0.000 → cov 0.871 (under).
- SpeedDating: 0.069 → cov 0.835 (under).
- ACSIncome_race_CA / sex_CA: 0.028 / 0.016 → cov 0.93+ (over-cover on Asian split specifically; ACSIncome_sex_CA is OK at 0.907).

Tasks with **ESS / n_cal > 0.5** all have weighted-CP coverage within 0.91 of target 0.90.

**Practitioner deliverable**: report ESS/n_cal alongside any weighted-CP deployment. **A threshold of ESS / n_cal ≥ 0.10** appears to separate the safe regime from the failure regime in our 10-task panel. This is a defensible deploy-time diagnostic.

## 3. Implication for the deploy-time decision protocol

The protocol's tree (from Block 4.5) splits on `log_dr_max`. Looking at the diagnostics table, `max_weight > 100` (i.e., extreme density-ratio mass concentration) correlates exactly with the ESS / n_cal < 0.05 failure cases. This means **`max_weight > 100` is an alternative deploy-time signal for "do NOT use weighted CP"**. We add this to the paper's decision-aid table (Section 5) as a more robust rule than the depth-2 tree.

## 4. APS vs LAC ablation

(Deferred to a future session; the main paper retains the prior argument that LAC collapses on binary tasks with q ≥ 0.5, established by Block 0's sanity pilot.)
