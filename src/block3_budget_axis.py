"""
Block 3 — budget-axis baseline.

Answers the reviewer objection: 'If you have ANY target labels, retraining is free.
Why bother with conformal-prediction wrappers?'

Per task: vary the number of OOD-distribution labels available (B in {0, 50, 500, all})
and compare three strategies for delivering 90% coverage on the unseen OOD test:
  A. id_postcp        : train on ID only; calibrate on ID; apply weighted CP + isotonic
  B. recalibrate      : train on ID only; calibrate on (ID_cal U B target labels); apply weighted CP + isotonic
  C. retrain          : train on (ID_train U B target labels); calibrate on ID_cal; apply split CP + isotonic

Metrics on the held-out OOD test (full OOD minus the B sampled budget points):
  coverage (target 0.90), set_size, accuracy, brier.

7 tasks (HELOC dropped due to label leakage). HGB only. 3 seeds. CPU.

Acceptance for claim:C5 — at B=0, post-hoc CP wins (closest to 0.90 coverage) on >=60%
of tasks; at B=500, >=30%. 'Wins' is defined as having the smallest |coverage - 0.90|
AND not under-shooting set size growth that would make retrain a free win.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from cp_variants import (
    split_cp, weighted_cp_from_weights, estimate_likelihood_ratio, summarize,
)
from calibrators import ece_uniform_width, ece_uniform_mass_jackknife, brier
from block1_full import (
    load_acs_full, task_acs_state, task_acs_temporal, task_acs_sex_ca,
    task_acs_race_ca, task_adult_sex, task_bank_age, task_taiwan_sex,
    _cache,  # for OpenML lazy-load reuse
)

ALPHA = 0.10
SEEDS = [0, 1, 2]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pick the "headline" OOD split per task — usually the worst one. Each task gets a single
# OOD population for the budget axis (so the budget points are sampled from a coherent
# distribution).
HEADLINE_OOD = {
    "ACSIncome_state":    "OOD_MS",
    "ACSIncome_temporal": "OOD_2018",
    "ACSIncome_sex_CA":   "OOD_F",
    "ACSIncome_race_CA":  "OOD_Black",
    "Adult_sex":          "OOD_F",
    "Bank_age":           "OOD_age_ge_35",
    "Taiwan_sex":         "OOD_F",
}

TASK_LOADERS = [
    ("ACSIncome_state",    task_acs_state),
    ("ACSIncome_temporal", task_acs_temporal),
    ("ACSIncome_sex_CA",   task_acs_sex_ca),
    ("ACSIncome_race_CA",  task_acs_race_ca),
    ("Adult_sex",          task_adult_sex),
    ("Bank_age",           task_bank_age),
    ("Taiwan_sex",         task_taiwan_sex),
]


def isotonic_calibrate(p_cal, y_cal, p_target):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    return iso.transform(p_target)


def to_pcal(p_pos):
    p_pos = np.clip(p_pos, 1e-7, 1 - 1e-7)
    return np.stack([1 - p_pos, p_pos], axis=1)


def get_hgb(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)


def evaluate_pset(model, X_eval, y_eval, X_cal, y_cal, alpha, rng, use_weighted=False, X_te_for_ratio=None):
    """Train (or trained) model -> probs -> isotonic on cal -> CP set on eval."""
    p_eval_raw = model.predict_proba(X_eval)[:, 1]
    p_cal_raw  = model.predict_proba(X_cal)[:, 1]
    p_eval_cal = isotonic_calibrate(p_cal_raw, y_cal, p_eval_raw)
    p_cal_cal  = isotonic_calibrate(p_cal_raw, y_cal, p_cal_raw)
    p_eval_full = to_pcal(p_eval_cal); p_cal_full = to_pcal(p_cal_cal)
    if use_weighted:
        w_cal = estimate_likelihood_ratio(X_cal, X_te_for_ratio, seed=int(rng.integers(0, 10**6)))
        inc, q = weighted_cp_from_weights(p_cal_full, y_cal, p_eval_full, w_cal, alpha, rng)
    else:
        inc, q = split_cp(p_cal_full, y_cal, p_eval_full, alpha, rng)
    s = summarize(inc, y_eval)
    acc = float((np.argmax(p_eval_full, axis=1) == y_eval).mean())
    br = brier(p_eval_full[:, 1], y_eval)
    return {
        "coverage": s["coverage"],
        "set_size": s["set_size"],
        "cc_cov0": s["cc_coverage"].get(0, float("nan")),
        "cc_cov1": s["cc_coverage"].get(1, float("nan")),
        "accuracy": acc,
        "brier": br,
        "ece_biased": ece_uniform_width(p_eval_full[:, 1], y_eval, n_bins=10),
        "ece_debiased": ece_uniform_mass_jackknife(p_eval_full[:, 1], y_eval, n_bins=15),
    }


def main():
    t0 = time.time()
    acs = load_acs_full()
    rows = []

    for task_name, loader in TASK_LOADERS:
        for seed in SEEDS:
            X_tr, y_tr, X_cal, y_cal, splits = loader(acs, seed)
            ood_name = HEADLINE_OOD[task_name]
            if ood_name not in splits:
                print(f"[skip] {task_name}: no headline OOD {ood_name}", flush=True)
                continue
            X_ood, y_ood = splits[ood_name]
            n_ood = len(X_ood)
            print(f"[task] {task_name:22s} seed={seed} ID_tr={len(X_tr):>6d} cal={len(X_cal):>6d} "
                  f"headline OOD={ood_name} (n={n_ood:>6d})", flush=True)

            # determine budgets in a task-aware way
            budgets = [0, 50, 500]
            if n_ood > 500:
                budgets.append(min(n_ood - 100, 5000))  # 'all' bounded
            for B in budgets:
                # sample B target points; eval on the rest
                rng = default_rng(seed * 17 + B)
                idx = rng.permutation(n_ood)
                Xtgt = X_ood[idx[:B]]; ytgt = y_ood[idx[:B]]
                Xev  = X_ood[idx[B:]]; yev  = y_ood[idx[B:]]
                if len(yev) < 50:
                    continue

                # method A: id_postcp (B=0 only meaningful at B=0; but eval shrinks with B)
                # train on ID, calibrate on ID, weighted CP wrt OOD eval covariates
                model_A = get_hgb(seed); model_A.fit(X_tr, y_tr)
                rA = evaluate_pset(model_A, Xev, yev, X_cal, y_cal, ALPHA,
                                   default_rng(seed * 991 + B),
                                   use_weighted=True, X_te_for_ratio=Xev)
                rA.update({"task": task_name, "seed": seed, "B": B,
                           "method": "id_postcp", "n_eval": int(len(yev))})
                rows.append(rA)

                # method B: recalibrate on cal U B target labels
                if B > 0:
                    Xcal_aug = np.vstack([X_cal, Xtgt])
                    ycal_aug = np.concatenate([y_cal, ytgt])
                else:
                    Xcal_aug, ycal_aug = X_cal, y_cal
                rB = evaluate_pset(model_A, Xev, yev, Xcal_aug, ycal_aug, ALPHA,
                                   default_rng(seed * 991 + B + 1),
                                   use_weighted=True, X_te_for_ratio=Xev)
                rB.update({"task": task_name, "seed": seed, "B": B,
                           "method": "recalibrate", "n_eval": int(len(yev))})
                rows.append(rB)

                # method C: retrain on ID train U B target labels
                if B > 0:
                    Xtr_aug = np.vstack([X_tr, Xtgt])
                    ytr_aug = np.concatenate([y_tr, ytgt])
                else:
                    Xtr_aug, ytr_aug = X_tr, y_tr
                model_C = get_hgb(seed); model_C.fit(Xtr_aug, ytr_aug)
                rC = evaluate_pset(model_C, Xev, yev, X_cal, y_cal, ALPHA,
                                   default_rng(seed * 991 + B + 2),
                                   use_weighted=False)
                rC.update({"task": task_name, "seed": seed, "B": B,
                           "method": "retrain", "n_eval": int(len(yev))})
                rows.append(rC)

                print(f"   B={B:>5d}  Apc:cov={rA['coverage']:.3f}/sz={rA['set_size']:.2f}  "
                      f"Brc:cov={rB['coverage']:.3f}/sz={rB['set_size']:.2f}  "
                      f"Crt:cov={rC['coverage']:.3f}/sz={rC['set_size']:.2f}  acc_C={rC['accuracy']:.3f}",
                      flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "block3_budget_axis.parquet")
    df.to_parquet(out)
    print(f"\nSaved: {out} ({len(df)} rows)")
    print(f"Wall time: {time.time()-t0:.1f}s")

    # Quick summary
    print("\n=== Mean coverage by (task, method, B) ===")
    print(df.groupby(["task", "method", "B"])["coverage"].mean().unstack("method").round(3))
    print("\n=== Mean set size ===")
    print(df.groupby(["task", "method", "B"])["set_size"].mean().unstack("method").round(3))


if __name__ == "__main__":
    main()
