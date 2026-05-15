# Block 4 — Decision-Protocol Recommender Analysis

**Run date**: 2026-05-15
**Script**: `src/block4_recommender.py`
**Raw data**: `results/block4_recommender.json`, `results/block4_recommender.log`

---

## 1. Setup

For each of 7 tasks (HELOC dropped), compute 6 task descriptors and rank the 12 (CP variant × calibrator) methods by mean OOD distance-to-nominal 0.90.

Descriptors:
- `n_train_est`: estimated ID train size.
- `n_features`: feature count (≈10 for all tasks in our grid).
- `class_imb_proxy`: |cc_cov0 − cc_cov1| on ID — proxy for class-imbalance pressure.
- `shift_class_rate_diff`: |per-class coverage shift between ID and OOD| — proxy for label-prior shift magnitude.
- `shift_acc_drop_pp`: HGB accuracy ID − worst-OOD (in pp).
- `shift_cov_drop_pp`: split-CP marginal coverage drop ID − worst-OOD (in pp).

## 2. Honest verdict on Claim C6

**Original target**: ≥ 70% leave-one-task-out top-3 accuracy for the 12-class method recommender.

**Result**:

| Recommender | Granularity | LOO-CV accuracy | Random baseline | Verdict |
|-------------|-------------|----------------:|----------------:|:-------:|
| Method (cp + cal) — top-1 | 12 classes (5 distinct used) | **0 / 7 = 0%** | ~20% | **FAIL** |
| Method — predicted-top-1 ∈ actual-top-3 | 12 classes | **1 / 7 = 14%** | ~50% | **FAIL** |
| CP family — top-1 | 3 classes | **3 / 7 = 43%** | 33% | Marginal — 95% CI ≈ [12%, 78%] |

**Conclusion**: C6 **fails** at the stated 70% bar. The training set is too small (7 tasks) to support a meaningful trained recommender against 5 distinct top-1 method classes. Even the cleaner 3-class CP-family classifier only edges out the random baseline.

## 3. The full-data trees ARE the protocol

The descriptive trees (fit on all 7 tasks; not cross-validated) extract two clear rules.

### CP-family tree (depth 2):

```
if shift_cov_drop_pp <= 1.38:
    if shift_acc_drop_pp <= -5.18:    weighted
    else:                              mondrian
else (shift_cov_drop_pp > 1.38):       split
```

### Method-level tree (depth 2):

```
if shift_cov_drop_pp <= 0.37:
    if shift_class_rate_diff <= 0.01: split + uncal
    else:                              mondrian + isotonic
else:
    if n_train_est <= 30852:           weighted + isotonic
    else:                              mondrian + temperature
```

Both trees independently surface `shift_cov_drop_pp` as the top split — the very feature defined by **split CP's failure mode**. That is encouraging: the tree learned what we already see in the marginal analysis.

## 4. The protocol the paper should claim

Instead of "we trained a recommender," claim:

> **Empirical decision protocol.** Using descriptive trees on the Block 1 + Block 3 evidence as a guide, we recommend the following rules for practitioners facing a tabular ML system under distribution shift.

| If… | Recommended method | Why (Block-1/3 evidence) |
|-----|-------------------|--------------------------|
| Drift is invisible (split-CP cov drop ≤ 0.5 pp) | Split CP + uncalibrated | Bank_age, Taiwan_sex — retraining gains are < 1 pp accuracy; CP wrappers add only noise. |
| Drift is moderate (cov drop 0.5–2 pp) AND class proportions shift | Weighted CP + isotonic | Adult_sex — closes coverage gap 0.017 → 0.004; small set-size cost (+0.03). |
| Drift is moderate AND class proportions are preserved | Mondrian or split CP | ACSIncome_temporal / sex_CA — both methods restore near-nominal coverage. |
| Drift is severe (cov drop > 2 pp) | Split CP **as-is**; do not deploy weighted CP | ACSIncome_state, race_CA — weighted CP **over-covers** to 0.94+ when density-ratio estimation is noisy. |
| Drift is severe AND ≥ 500 target labels available | **Retrain on ID ∪ target labels** | ACSIncome_state: retrain at B=5000 → coverage 0.898 (post-hoc CP stuck at 0.871). Adult_sex: retrain accuracy 0.914 → 0.936. |
| Class proportions differ between ID and OOD (any drift level) | **Avoid Mondrian CP** | Mondrian set size grows 5–30% over split (Taiwan, Bank, Adult) with no coverage gain. |
| Small target-label budget (B < 500) | **Never use recalibrate(cal ∪ B)** | At B=50 it over-covers Adult to 0.945, under-covers Bank to 0.819. |

This protocol is the paper's main practitioner-facing contribution. It is grounded in the 8-task × 5-seed × 12-method Block 1 grid + the 7-task × 4-budget Block 3 axis.

## 5. Limitations to flag in the paper

- **7 tasks is too few** for a statistically defensible learned classifier. Reproducibility across a future TableShift-CP study (10–15 tasks) is needed for any LOO-CV claim.
- **HGB-only at Block 3** — full Block 1 used all 3 GBM learners; agreement was already strong (per `exp:005`), so a single-learner Block 3 is justifiable but not the most thorough story.
- **Density-ratio quality not measured directly**. The shift indicators we use (`shift_cov_drop_pp`, `shift_acc_drop_pp`, `shift_class_rate_diff`) are downstream of model behaviour. A direct shift quantification (Wasserstein, KL, or MMD on covariates) would tighten the rule conditions.
