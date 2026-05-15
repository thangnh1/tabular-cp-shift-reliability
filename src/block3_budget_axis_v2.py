"""
Block 3 v2 — budget-axis baseline, extended to the 3 new tasks.

Same protocol as `src/block3_budget_axis.py` but only runs on the 3 new tasks
to avoid re-doing the original 7. Concatenate at analysis time with the
existing `results/block3_budget_axis.parquet`.
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
from extra_tasks import task_diabetes_race, task_speeddating_race, task_onlinenews_channel

ALPHA = 0.10
SEEDS = [0, 1, 2]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HEADLINE_OOD = {
    "Diabetes_race":       "OOD_African",
    "SpeedDating_race":    "OOD_Asian",
    "OnlineNews_channel":  "OOD_entertainment",
}

TASK_LOADERS = [
    ("Diabetes_race",       task_diabetes_race),
    ("SpeedDating_race",    task_speeddating_race),
    ("OnlineNews_channel",  task_onlinenews_channel),
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
    rows = []
    for task_name, loader in TASK_LOADERS:
        for seed in SEEDS:
            X_tr, y_tr, X_cal, y_cal, splits = loader(None, seed)
            ood_name = HEADLINE_OOD[task_name]
            if ood_name not in splits:
                print(f"[skip] {task_name}: no headline OOD {ood_name}", flush=True)
                continue
            X_ood, y_ood = splits[ood_name]
            n_ood = len(X_ood)
            print(f"[task] {task_name:22s} seed={seed} ID_tr={len(X_tr):>6d} cal={len(X_cal):>6d} "
                  f"headline OOD={ood_name} (n={n_ood:>6d})", flush=True)
            budgets = [0, 50, 500]
            if n_ood > 1000:
                budgets.append(min(n_ood - 100, 5000))
            for B in budgets:
                rng = default_rng(seed * 17 + B)
                idx = rng.permutation(n_ood)
                Xtgt = X_ood[idx[:B]]; ytgt = y_ood[idx[:B]]
                Xev  = X_ood[idx[B:]]; yev  = y_ood[idx[B:]]
                if len(yev) < 50:
                    continue

                model_A = get_hgb(seed); model_A.fit(X_tr, y_tr)
                rA = evaluate_pset(model_A, Xev, yev, X_cal, y_cal, ALPHA,
                                   default_rng(seed * 991 + B),
                                   use_weighted=True, X_te_for_ratio=Xev)
                rA.update({"task": task_name, "seed": seed, "B": B,
                           "method": "id_postcp", "n_eval": int(len(yev))})
                rows.append(rA)

                if B > 0:
                    Xcal_aug = np.vstack([X_cal, Xtgt]); ycal_aug = np.concatenate([y_cal, ytgt])
                else:
                    Xcal_aug, ycal_aug = X_cal, y_cal
                rB = evaluate_pset(model_A, Xev, yev, Xcal_aug, ycal_aug, ALPHA,
                                   default_rng(seed * 991 + B + 1),
                                   use_weighted=True, X_te_for_ratio=Xev)
                rB.update({"task": task_name, "seed": seed, "B": B,
                           "method": "recalibrate", "n_eval": int(len(yev))})
                rows.append(rB)

                if B > 0:
                    Xtr_aug = np.vstack([X_tr, Xtgt]); ytr_aug = np.concatenate([y_tr, ytgt])
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
    out = os.path.join(RESULTS_DIR, "block3_budget_axis_v2.parquet")
    df.to_parquet(out)
    print(f"\nSaved: {out} ({len(df)} rows)")
    print(f"Wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
