# Block v2 — Expansion to 10-Task Pool: Analysis

**Run date**: 2026-05-15
**Inputs**: `results/block1_full_v2.parquet` (4,140 rows, 10 tasks), `results/block3_budget_axis_v2.parquet` (108 rows, 3 new tasks) — combined with Block 3 v1 = 360 budget-axis rows across 10 tasks; `results/block4_5_v2_deploy_time.parquet` (10 task feature vectors); `results/block4_5_v2_deploy_time.json`.

The expansion adds **Diabetes_race**, **SpeedDating_race**, and **OnlineNews_channel** to the original 7-task panel (`paper/CODEX_REVIEW_1.md` action item #2). HELOC remains excluded.

---

## 1. Per-task split-CP marginal coverage drops (10 tasks)

| Task | ID coverage | Worst-OOD coverage | Δ (pp) |
|------|------------:|-------------------:|-------:|
| **OnlineNews_channel** | 0.901 | 0.821 | **+7.94** |
| ACSIncome_state (MS) | 0.901 | 0.868 | +3.30 |
| ACSIncome_race_CA (Black) | 0.900 | 0.886 | +1.53 |
| Adult_sex (Female) | 0.896 | 0.883 | +1.53 |
| ACSIncome_temporal | 0.899 | 0.892 | +0.74 |
| SpeedDating_race (Asian) | 0.899 | 0.893 | +0.55 |
| Diabetes_race (African) | 0.903 | 0.902 | +0.16 |
| ACSIncome_sex_CA | 0.900 | 0.900 | +0.15 |
| Bank_age (≥35) | 0.898 | 0.901 | −0.30 |
| Taiwan_sex (Female) | 0.899 | 0.908 | −1.03 |

**Claim C1 (restored to original)**: split-CP marginal coverage drops by ≥ 5 pp on at least one task family. **PASS**: OnlineNews_channel hits 7.94 pp (tech → entertainment).

## 2. Claim C2 — within-1-pp restoration

For each task, the CP variant closest to nominal 0.90 on the worst OOD split:

| Task | Best CP | Coverage | Within 1 pp? |
|------|---------|---------:|:------------:|
| ACSIncome_state | split | 0.881 | ✗ |
| **OnlineNews_channel** | **Mondrian** | **0.875** | ✗ |
| ACSIncome_temporal | Mondrian | 0.894 | ✓ |
| ACSIncome_sex_CA | Mondrian | 0.900 | ✓ |
| ACSIncome_race_CA | split | 0.892 | ✓ (within 1.5 pp) |
| Adult_sex | weighted | 0.896 | ✓ |
| Bank_age | split | 0.899 | ✓ |
| Taiwan_sex | Mondrian | 0.902 | ✓ |
| Diabetes_race | split | 0.902 | ✓ |
| SpeedDating_race | Mondrian | 0.894 | ✓ |

**Claim C2**: 8 / 10 tasks have a CP variant within 1 pp of nominal (80 % — at the threshold). The 2 failures (ACSIncome_state, OnlineNews_channel) are the same hard-shift cases where no CP wrapper closes the gap. **PASS at the stated 80 % threshold (exactly).**

## 3. Mondrian-vs-split set-size growth (10 tasks)

| Task | \|P(y=1\|ID) − P(y=1\|OOD)\| | Mondrian Δ set size (vs split) |
|------|-----------------------------:|--------------------------------:|
| **Diabetes_race** | ≈ 0.00 (both 0.112) | **+53.8 %** |
| Taiwan_sex | 0.111 | +21.5 % |
| SpeedDating_race | 0.032 | +20.6 % |
| Bank_age | 0.052 | +14.8 % |
| Adult_sex | 0.196 | +4.7 % |
| OnlineNews_channel | ≈ small | +4.1 % |
| ACSIncome_temporal | 0.012 | +0.8 % |
| ACSIncome_state | 0.034 | +0.2 % |
| ACSIncome_sex_CA | 0.083 | 0.0 % |
| ACSIncome_race_CA | small | −0.1 % |

**Claim G — updated**: Mondrian's set-size penalty is **NOT explained by label-prior shift alone**. Diabetes_race shows 53.8 % growth despite identical class rates between ID (Caucasian) and OOD (AfricanAmerican). The mechanism likely combines class-conditional miscalibration AND minority-class quantile sensitivity. The original "monotone in |ΔP(y=1)|" framing in v1 is too narrow — Mondrian-failure is **multi-factor and task-specific**. We weaken claim G to *"Mondrian inflates set size on a subset of OOD-shifted tasks; the predictors of this failure include but are not limited to class-rate differences."*

## 4. Weighted CP failure modes — two distinct patterns

**Pattern 1 — Over-cover on small / noisy OOD populations** (already documented in v1):
- ACSIncome_race_CA / OOD_Asian: 0.969 (target 0.90).
- HELOC (dropped from main study): 0.977.

**Pattern 2 — NEW — Under-cover on hard-shift tasks with high-dim or text-style features**:
- **OnlineNews_channel**: 0.640 — catastrophic.
- **SpeedDating_race**: 0.835.
- **Diabetes_race**: 0.871.

The under-cover mode appears when the OOD population is *different enough* that the logistic-regression density-ratio estimator produces extreme weights on a few calibration rows, collapsing the weighted quantile. This is the **opposite** of the noise-driven over-cover mode. **Weighted CP is therefore not a uniformly safer default than split CP** — it is a high-variance tool that can fail in either direction.

## 5. Budget-axis (Block 3 v1 + v2) — combined verdict on C5

Combined 10-task budget axis (HGB, 3 seeds, B ∈ {0, 50, 500, 5000 or full}):

| Budget B | Post-hoc CP wins/ties retrain | % | Threshold |
|---------:|-------------------------------:|--:|:---------:|
| 0 | 5 / 10 | **50 %** | 60 % — **FAIL** |
| 50 | 6 / 10 | 60 % | — |
| 500 | 3 / 10 | **30 %** | 30 % — PASS (exactly) |

**Claim C5 — revised**: in the 7-task pool, post-hoc CP won 71 % at B=0 (PASS). With 3 new tasks added — Diabetes, SpeedDating, OnlineNews, all of which show weighted CP failure modes — **C5 narrowly fails its 60% B=0 threshold (50%) but still PASSES the B=500 threshold (30%)**. The honest takeaway: **post-hoc CP is competitive with retraining at B=0 but does not dominate.** Retraining is the safer default when target labels are available, even at low budgets.

### Recalibrate anti-pattern severity (10 tasks)

Largest mean distance from nominal 0.90 across (task, B) cells for the `recalibrate` method:

| Task | B | distance from 0.90 |
|------|--:|------------------:|
| **OnlineNews_channel** | **50** | **0.191** |
| Bank_age | 500 | 0.135 |
| ACSIncome_race_CA | 50 | 0.116 |
| SpeedDating_race | 0 | 0.104 |
| OnlineNews_channel | 500 | 0.098 |

Up from the v1 maximum of 0.13 (Bank_age B=500). The OnlineNews B=50 case (coverage 0.71) is a **paper-ready cautionary example**: appending 50 entertainment-channel labels to a calibration set fitted on tech-channel articles makes the weighted-CP density ratios swing wildly.

## 6. Block 4.5 v2 — deploy-time decision protocol on 10 tasks

LOO-CV top-1 accuracy on the 10-task pool with deploy-time features only:

| Tree depth | LOO-CV accuracy | Random baseline |
|-----------:|----------------:|-----------------:|
| 2 | 2 / 10 = **20 %** | 33 % |
| 3 | 2 / 10 = **20 %** | 33 % |

The v1 result of 5/7 = 71 % was **overfitted to the 7-task panel**. On the 10-task pool, the deploy-time tree is **below the random baseline**.

**Honest verdict on C6′**: the deploy-time decision tree **does not generalise** at our task-pool size. We can no longer claim a recommender — we present the **descriptive depth-2 rule fit on the full 10-task data**

```
if log_dr_max ≤ 1.74:
    if mean_pred_kl_id_to_ood ≤ 0.10: use split CP
    else:                              use weighted CP
else (log_dr_max > 1.74):              use Mondrian CP
```

as a sensible *starting heuristic*, while explicitly stating that it does **not** survive leave-one-task-out cross-validation. The protocol is a v0 starting point; productionising it requires a benchmark expansion to 30+ tasks.

This is the largest paper-level negative finding from v2 and **must be the headline limitation** in the discussion.

## 7. Updated paper claim register

| Claim | v1 verdict | **v2 verdict** | Notes |
|-------|------------|----------------|-------|
| **C1** | FAIL @ 5 pp; relaxed to ≥3 pp | **PASS @ original 5 pp** | OnlineNews 7.94 pp |
| **C2** | PASS 100% (7/7) | **PASS 80% (8/10)** at exact threshold | Hard-shift cases (ACSIncome_state, OnlineNews) miss |
| **C3** | NULL (narrow) | **NULL (narrow)** — unchanged | Calibrator rerank does not appear in 4-cal × 3-GBM setting |
| **C5** | PASS @ B=0 (71%) | **FAIL @ B=0 (50%)**, PASS @ B=500 (30%) | Hard-shift tasks pull post-hoc CP below 60% threshold |
| **C6′** | MARGINAL PASS (71%, n=7) | **FAIL** (20%, n=10; below random) | Overfitting on small pool; demote to descriptive rule |
| **G** | PASS as label-prior story | **PASS (existence); mechanism BROADER** | Diabetes shows 53.8% growth with equal class rates |

## 8. What the v2 expansion reveals about the paper's actual contribution

The paper is **now strongest as an empirical map of failure modes**:

- **Three documented weighted-CP failure modes**: over-cover (small OOD), under-cover (hard shift / text features), and noise (Diabetes intermittent crashes at certain budgets).
- **A class of hard-shift tasks** (OnlineNews) where no CP wrapper closes the gap and retraining is required.
- **A practitioner anti-pattern** (recalibrate at small B) with a vivid example (OnlineNews B=50 → coverage 0.71).
- **A descriptive starting heuristic** that explicitly does not survive LOO-CV.
- **A reliable case study** (Adult_sex) where weighted CP demonstrably works.

The paper should be reframed as a **failure-modes-and-case-studies** paper, not a "decision protocol" paper.
