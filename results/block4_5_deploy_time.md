# Block 4.5 — Decision Protocol from Deploy-Time Observables

**Run date**: 2026-05-15
**Script**: `src/block4_5_deploy_time_protocol.py`
**Raw data**: `results/block4_5_deploy_time.parquet` (7 task feature rows), `results/block4_5_deploy_time.json` (tree + LOO-CV)
**Motivation**: Codex MCP review (`paper/CODEX_REVIEW_1.md`) identified the original Block 4 protocol as ill-posed because its splits used label-derived OOD-side features (`shift_cov_drop_pp`, `shift_class_rate_diff`) unavailable to a deploy-time practitioner with B=0 target labels. This block rebuilds the protocol with deploy-time-observable features only.

---

## 1. Deploy-time features

For each of the 7 tasks, computed using **only** (X_id_train, y_id_train, X_id_cal, y_id_cal, X_OOD) — never OOD labels:

| Task | n_id_train | n_id_cal | id_class_imb | log_dr_max | log_dr_p99 | log_dr_std | ood_pred_entropy | mean_pred_kl_id_to_ood |
|------|-----------:|---------:|-------------:|-----------:|-----------:|-----------:|-----------------:|-----------------------:|
| ACSIncome_state | 73 498 | 36 749 | 0.088 | 0.811 | 0.689 | **2.039** | 0.501 | **0.011** |
| ACSIncome_temporal | 38 528 | 19 264 | 0.107 | −0.001 | −0.001 | 0.000 | 0.481 | 0.000 |
| ACSIncome_sex_CA | 38 845 | 19 423 | 0.030 | 0.521 | 0.484 | **1.099** | 0.466 | **0.118** |
| ACSIncome_race_CA | 45 379 | 22 689 | 0.139 | 0.866 | 0.825 | **2.000** | 0.510 | **0.166** |
| Adult_sex | 16 323 | 8 161 | 0.187 | 0.001 | 0.001 | 0.000 | 0.318 | **0.336** |
| Bank_age | 7 506 | 3 753 | 0.471 | 0.000 | 0.000 | 0.000 | 0.130 | 0.054 |
| Taiwan_sex | 5 944 | 2 972 | 0.279 | 0.000 | 0.000 | 0.000 | 0.395 | 0.063 |

A few observations from the table alone:
- `log_dr_std > 0.98` on **only** the 3 ACS state/race/sex CA tasks (state, sex_CA, race_CA). These are the multi-covariate-shift cases where the joint covariate distribution of OOD genuinely differs from ID.
- `mean_pred_kl_id_to_ood` is largest on `Adult_sex` (0.34) — the model's marginal prediction P(y=1) is very different between male and female, despite identical feature schemas.
- The temporal / Bank / Taiwan tasks all show near-zero log-density-ratio dispersion, consistent with weak or absent covariate shift; in those cases the labelled shift target (P(y=1)) is the discriminating signal, not the covariates.

## 2. The depth-2 tree (full-data, descriptive)

```
if log_dr_std <= 0.98:
    if mean_pred_kl_id_to_ood <= 0.09:  use Mondrian CP
    else:                                use weighted CP
else (log_dr_std > 0.98):                use split CP
```

## 3. LOO-CV accuracy (deploy-time features only)

**5 / 7 = 71 %** top-1 CP-family accuracy. Random baseline 33 %. 95 % CI (Wilson) for 5/7 ≈ [36 %, 92 %].

Per-fold:

| Task | Predicted | Actual best (from Block 1 OOD coverage) | Correct? |
|------|-----------|------------------------------------------|:--------:|
| ACSIncome_state | split | split | ✓ |
| ACSIncome_temporal | mondrian | mondrian | ✓ |
| ACSIncome_sex_CA | split | split | ✓ |
| ACSIncome_race_CA | split | split | ✓ |
| Adult_sex | split | weighted | ✗ |
| Bank_age | mondrian | split | ✗ |
| Taiwan_sex | mondrian | mondrian | ✓ |

Note that compared to Block 4 (which used label-derived features and got 3/7 = 43 %), the deploy-time tree is **more accurate (5/7) AND uses only quantities the practitioner can compute at deployment**. The two errors are:

- **Adult_sex**: predicted split (because log_dr_std ≈ 0 — covariate marginals are very similar across the sex column), but weighted was actually best. The deploy-time features cannot see that within-class density ratio is the operative shift; the tree under-uses information. Honest residual error.
- **Bank_age**: predicted mondrian (low log_dr_std, low pred-KL), but split was best because in Block 1 weighted CP slightly *over-covered* on Bank (0.905) and Mondrian inflated set size by 14 %. The tree picks Mondrian where split is actually marginally better. The post-split-vs-Mondrian distinction is small here (Block 1 OOD coverage: split 0.901, mondrian 0.898) — both within 0.01 of nominal.

## 4. Why this protocol is honest

The original Block 4 protocol used `shift_cov_drop_pp` as the top split — but that quantity requires OOD labels to compute, so a practitioner with B=0 cannot evaluate the rule. The Block 4.5 tree uses `log_dr_std` (variance of estimated log density ratios, computable from X_cal and X_OOD covariates alone) and `mean_pred_kl_id_to_ood` (computable from the trained model's predictions on the two splits). **Neither feature requires OOD labels.**

A practitioner can implement the rule in ~10 lines of Python:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def deploy_time_protocol(X_cal, X_ood, model, max_subsample=5000):
    rng = np.random.default_rng(0)
    Xs_cal = X_cal[rng.choice(len(X_cal), min(len(X_cal), max_subsample), replace=False)]
    Xs_ood = X_ood[rng.choice(len(X_ood), min(len(X_ood), max_subsample), replace=False)]
    Xmix = np.vstack([Xs_cal, Xs_ood])
    ymix = np.array([0]*len(Xs_cal) + [1]*len(Xs_ood))
    p = LogisticRegression(max_iter=200, solver="liblinear").fit(Xmix, ymix).predict_proba(Xs_cal)[:,1]
    log_dr = np.log(np.clip(p,1e-6,1-1e-6) / np.clip(1-p,1e-6,1-1e-6))
    log_dr_std = log_dr.std()
    p_cal = float(model.predict_proba(X_cal)[:,1].mean())
    p_ood = float(model.predict_proba(X_ood)[:,1].mean())
    p_cal = max(min(p_cal, 1-1e-7), 1e-7); p_ood = max(min(p_ood, 1-1e-7), 1e-7)
    kl = p_cal*np.log(p_cal/p_ood) + (1-p_cal)*np.log((1-p_cal)/(1-p_ood))
    if log_dr_std > 0.98: return "split"
    elif kl > 0.09:        return "weighted"
    else:                   return "mondrian"
```

This is the practitioner-facing contribution of the paper.

## 5. Honest caveats

- **7 tasks is small** — the 95 % CI on 5/7 is wide. Replication on a larger benchmark (e.g., the full TableShift or TabArena suite) is required before claiming a generalisable rule.
- **The two thresholds (0.98 and 0.09) are fitted to our 7 tasks.** They should be re-fit on any expanded benchmark, not blindly applied.
- **No regression / multi-class tasks** — protocol applies to binary classification only.
- **HGB only** for computing the deploy-time entropy + KL features (consistent with Block 3). Cross-learner agreement on the protocol's recommendations is left for future work.
