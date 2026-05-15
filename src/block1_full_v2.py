"""
Block 1 full grid v2 — adds three new tasks to the original 7.

This script imports task loaders from block1_full (the 7-task panel) and
extra_tasks (the 3 new datasets), then runs the same (3 learners × 4
calibrators × 3 CP × 5 seeds) grid over the union.

Output:
- results/block1_full_v2.parquet  (long-form per-cell metrics, all 10 tasks)
- results/block1_full_v2.log
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from cp_variants import (
    split_cp, mondrian_cp, weighted_cp_from_weights,
    estimate_likelihood_ratio, summarize,
)
from calibrators import apply_calibrator, ece_uniform_width, ece_uniform_mass_jackknife, brier
from block1_full import (
    load_acs_full, task_acs_state, task_acs_temporal, task_acs_sex_ca,
    task_acs_race_ca, task_adult_sex, task_bank_age, task_taiwan_sex,
    get_learner,
)
from extra_tasks import task_diabetes_race, task_speeddating_race, task_onlinenews_channel

ALPHA = 0.10
SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASKS = [
    ("ACSIncome_state",    task_acs_state),
    ("ACSIncome_temporal", task_acs_temporal),
    ("ACSIncome_sex_CA",   task_acs_sex_ca),
    ("ACSIncome_race_CA",  task_acs_race_ca),
    ("Adult_sex",          task_adult_sex),
    ("Bank_age",           task_bank_age),
    ("Taiwan_sex",         task_taiwan_sex),
    # New in v2:
    ("Diabetes_race",       task_diabetes_race),
    ("SpeedDating_race",    task_speeddating_race),
    ("OnlineNews_channel",  task_onlinenews_channel),
]


def main():
    t_start = time.time()
    acs = load_acs_full()
    rows = []
    learners = ["hgb", "xgb", "lgbm"]
    cps = ["split", "weighted", "mondrian"]
    cals = ["uncal", "platt", "isotonic", "temperature"]

    for task_name, task_loader in TASKS:
        for seed in SEEDS:
            t_task = time.time()
            try:
                X_tr, y_tr, X_cal, y_cal, splits = task_loader(acs, seed)
            except Exception as e:
                print(f"[skip] task={task_name} seed={seed}: {type(e).__name__}: {e}", flush=True)
                continue
            density_ratios = {}
            for sn, (Xs, _) in splits.items():
                if sn == "ID":
                    density_ratios[sn] = np.ones(len(X_cal))
                else:
                    density_ratios[sn] = estimate_likelihood_ratio(X_cal, Xs, seed=seed)
            for learner in learners:
                model = get_learner(learner, seed)
                model.fit(X_tr, y_tr)
                p_cal_raw = model.predict_proba(X_cal)[:, 1]
                split_probs = {sn: model.predict_proba(Xs)[:, 1] for sn, (Xs, _) in splits.items()}
                for cal_name in cals:
                    p_full_cal = apply_calibrator(cal_name, p_cal_raw, y_cal, p_cal_raw)
                    for cp_name in cps:
                        rng = default_rng(seed * 991 + hash(cp_name + cal_name) % 1000)
                        for split_name, (Xs, ys) in splits.items():
                            p_s_raw = split_probs[split_name]
                            p_full_s = apply_calibrator(cal_name, p_cal_raw, y_cal, p_s_raw)
                            if cp_name == "split":
                                inc, q_info = split_cp(p_full_cal, y_cal, p_full_s, ALPHA, rng)
                            elif cp_name == "weighted":
                                inc, q_info = weighted_cp_from_weights(
                                    p_full_cal, y_cal, p_full_s,
                                    density_ratios[split_name], ALPHA, rng,
                                )
                            else:
                                inc, q_info = mondrian_cp(p_full_cal, y_cal, p_full_s, ALPHA, rng)
                            s = summarize(inc, ys)
                            rows.append({
                                "task": task_name, "learner": learner, "cp": cp_name,
                                "calibrator": cal_name, "seed": seed, "split": split_name,
                                "n": int(len(ys)),
                                "coverage": s["coverage"], "set_size": s["set_size"],
                                "cc_cov0": s["cc_coverage"].get(0, float("nan")),
                                "cc_cov1": s["cc_coverage"].get(1, float("nan")),
                                "ece_biased": ece_uniform_width(p_full_s[:, 1], ys, n_bins=10),
                                "ece_debiased": ece_uniform_mass_jackknife(p_full_s[:, 1], ys, n_bins=15),
                                "brier": brier(p_full_s[:, 1], ys),
                                "accuracy": float((np.argmax(p_full_s, axis=1) == ys).mean()),
                            })
            print(f"  done task={task_name:22s} seed={seed}  splits={list(splits.keys())}  "
                  f"took {time.time()-t_task:.1f}s  elapsed {time.time()-t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "block1_full_v2.parquet")
    df.to_parquet(out)
    print(f"\nSaved: {out} ({len(df)} rows; {df.task.nunique()} tasks; "
          f"{df['split'].nunique()} unique split names)")
    print(f"Total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
