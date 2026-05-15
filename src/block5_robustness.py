"""
Block 5 — Robustness appendix experiments.

Three small checks that Codex round-3 review identified as essential for arXiv-readiness:

1. OnlineNews channel shift without `kw_*` (keyword-stat) features.
   These columns are computed using shares of other articles and were flagged
   as a mild leakage risk. Rerun split / weighted / Mondrian CP on
   tech → entertainment without kw_* features and verify the headline split-CP
   coverage drop (and weighted-CP under-coverage) survive.

2. Weighted-CP diagnostic: log effective-sample-size (ESS), max weight, and
   weight-99th-percentile per (task, OOD split). These quantify density-ratio
   instability — the suspected mechanism behind weighted CP under-coverage on
   hard-shift tasks.

3. APS-vs-LAC ablation on one ID task and one severely-shifted task
   (ACSIncome_state) to justify the score choice in the paper.

Output:
- results/block5_robustness.json
- results/block5_robustness.log
- results/block5_robustness.md (analysis)
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from cp_variants import (
    split_cp, mondrian_cp, weighted_cp_from_weights,
    estimate_likelihood_ratio, aps_score, aps_inclusion, summarize,
)
from calibrators import ece_uniform_width, brier

ALPHA = 0.10
SEEDS = [0, 1, 2]
import pathlib
# Resolve project root relative to this file (code-release/src/<this>.py → code-release/)
ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _encode_frame(X):
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("object", "category"):
            out[c] = out[c].astype("category").cat.codes.astype(float)
        else:
            out[c] = out[c].astype(float)
    return out.fillna(-1).values


def load_onlinenews_no_kw():
    """OnlineNews loader with all kw_* and data_channel + url + timedelta features dropped."""
    print("[load] OpenML Online News Popularity (id=4545) — DROP kw_* + data_channel + url + timedelta", flush=True)
    b = fetch_openml(data_id=4545, as_frame=True)
    X = b.data.copy()
    y = (b.target > 1400).astype(int).values
    chan_tech = (X["data_channel_is_tech"] > 0.5).values
    chan_ent  = (X["data_channel_is_entertainment"] > 0.5).values
    kw_cols   = [c for c in X.columns if c.startswith("kw_")]
    chan_cols = [c for c in X.columns if c.startswith("data_channel_is_")]
    other_drop = [c for c in ["url", "timedelta"] if c in X.columns]
    drop_cols = kw_cols + chan_cols + other_drop
    print(f"[load] dropping {len(kw_cols)} kw_* + {len(chan_cols)} data_channel + {len(other_drop)} other = {len(drop_cols)} cols; "
          f"remaining feature count = {X.shape[1] - len(drop_cols)}", flush=True)
    X = X.drop(columns=drop_cols)
    X_arr = _encode_frame(X)
    return X_arr[chan_tech], y[chan_tech], X_arr[chan_ent], y[chan_ent]


def compute_aps_summary(p_full_cal, y_cal, p_full_te, y_te, X_cal, X_te, alpha, rng):
    """Run three CP variants on a given (cal, test) pair."""
    out = {}
    rng_split = default_rng(int(rng.integers(0, 10**6)))
    inc_split, _ = split_cp(p_full_cal, y_cal, p_full_te, alpha, rng_split)
    out["split"] = summarize(inc_split, y_te)
    rng_w = default_rng(int(rng.integers(0, 10**6)))
    w_cal = estimate_likelihood_ratio(X_cal, X_te, seed=int(rng.integers(0, 10**6)))
    # ESS and weight diagnostics
    w_norm = w_cal / w_cal.sum()
    ess = float(1.0 / np.sum(w_norm ** 2))
    max_w = float(w_cal.max())
    p99_w = float(np.percentile(w_cal, 99))
    out["_weight_diag"] = {"ess": ess, "max_weight": max_w, "p99_weight": p99_w, "n_cal": int(len(w_cal))}
    inc_w, _ = weighted_cp_from_weights(p_full_cal, y_cal, p_full_te, w_cal, alpha, rng_w)
    out["weighted"] = summarize(inc_w, y_te)
    rng_m = default_rng(int(rng.integers(0, 10**6)))
    inc_m, _ = mondrian_cp(p_full_cal, y_cal, p_full_te, alpha, rng_m)
    out["mondrian"] = summarize(inc_m, y_te)
    return out


def isotonic_calibrate(p_cal, y_cal, p_target):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    return iso.transform(p_target)


def to_pcal(p_pos):
    p_pos = np.clip(p_pos, 1e-7, 1 - 1e-7)
    return np.stack([1 - p_pos, p_pos], axis=1)


def run_onlinenews_robustness():
    print("\n========== ROBUSTNESS 1: OnlineNews without kw_* features ==========\n")
    X_id, y_id, X_ood, y_ood = load_onlinenews_no_kw()
    results = []
    for seed in SEEDS:
        rng = default_rng(seed)
        n = len(X_id)
        idx = rng.permutation(n)
        n_tr = int(0.5 * n); n_cal = int(0.25 * n)
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
        model.fit(X_id[tr], y_id[tr])
        p_cal_raw = model.predict_proba(X_id[cal])[:, 1]
        # Apply isotonic calibration on cal set
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_id[cal])
        # ID test
        p_te_raw = model.predict_proba(X_id[te])[:, 1]
        p_full_cal_iso = to_pcal(iso.transform(p_cal_raw))
        p_full_id_iso  = to_pcal(iso.transform(p_te_raw))
        p_ood_raw = model.predict_proba(X_ood)[:, 1]
        p_full_ood_iso = to_pcal(iso.transform(p_ood_raw))
        rng2 = default_rng(seed * 991)
        id_sum  = compute_aps_summary(p_full_cal_iso, y_id[cal], p_full_id_iso,  y_id[te], X_id[cal], X_id[te], ALPHA, rng2)
        rng3 = default_rng(seed * 991 + 7)
        ood_sum = compute_aps_summary(p_full_cal_iso, y_id[cal], p_full_ood_iso, y_ood,    X_id[cal], X_ood,    ALPHA, rng3)
        for split_name, sm in [("ID", id_sum), ("OOD_entertainment", ood_sum)]:
            for cp_name in ["split", "weighted", "mondrian"]:
                results.append({
                    "task": "OnlineNews_channel_no_kw",
                    "seed": seed, "split": split_name, "cp": cp_name,
                    "coverage": sm[cp_name]["coverage"],
                    "set_size": sm[cp_name]["set_size"],
                })
            # Append weight diagnostics
            wd = ood_sum["_weight_diag"]
            results.append({
                "task": "OnlineNews_channel_no_kw",
                "seed": seed, "split": split_name, "cp": "weighted_diag",
                "coverage": np.nan, "set_size": np.nan,
                "ess": wd["ess"], "max_weight": wd["max_weight"], "p99_weight": wd["p99_weight"],
                "n_cal": wd["n_cal"],
            })
    df = pd.DataFrame(results)
    print(df.groupby(["split", "cp"])[["coverage", "set_size"]].mean().round(3))
    print()
    diags = df[df["cp"] == "weighted_diag"].dropna(subset=["ess"])
    print(f"Weighted CP density-ratio diagnostics on OOD_entertainment (no kw_*):")
    print(diags.groupby("split")[["ess", "max_weight", "p99_weight", "n_cal"]].mean().round(2))
    return df


def run_weight_diagnostics_main():
    """For each main task, compute the ESS and max weight of the density-ratio
    estimator on the headline OOD split (re-using the v2 task loaders)."""
    from block1_full import (load_acs_full, task_acs_state, task_acs_temporal, task_acs_sex_ca,
                             task_acs_race_ca, task_adult_sex, task_bank_age, task_taiwan_sex)
    from extra_tasks import task_diabetes_race, task_speeddating_race, task_onlinenews_channel
    HEADLINE_OOD = {
        "ACSIncome_state": "OOD_MS", "ACSIncome_temporal": "OOD_2018",
        "ACSIncome_sex_CA": "OOD_F", "ACSIncome_race_CA": "OOD_Black",
        "Adult_sex": "OOD_F", "Bank_age": "OOD_age_ge_35", "Taiwan_sex": "OOD_F",
        "Diabetes_race": "OOD_African", "SpeedDating_race": "OOD_Asian",
        "OnlineNews_channel": "OOD_entertainment",
    }
    TASKS = [
        ("ACSIncome_state", task_acs_state), ("ACSIncome_temporal", task_acs_temporal),
        ("ACSIncome_sex_CA", task_acs_sex_ca), ("ACSIncome_race_CA", task_acs_race_ca),
        ("Adult_sex", task_adult_sex), ("Bank_age", task_bank_age),
        ("Taiwan_sex", task_taiwan_sex), ("Diabetes_race", task_diabetes_race),
        ("SpeedDating_race", task_speeddating_race), ("OnlineNews_channel", task_onlinenews_channel),
    ]
    print("\n========== ROBUSTNESS 2: Weighted-CP density-ratio diagnostics per task ==========\n")
    acs = load_acs_full()
    rows = []
    for tname, loader in TASKS:
        X_tr, y_tr, X_cal, y_cal, splits = loader(acs, 0)
        ood_name = HEADLINE_OOD[tname]
        X_ood = splits[ood_name][0]
        w = estimate_likelihood_ratio(X_cal, X_ood, seed=0)
        w_norm = w / w.sum()
        ess = float(1.0 / np.sum(w_norm ** 2))
        rows.append({
            "task": tname, "headline_ood": ood_name,
            "n_cal": int(len(X_cal)),
            "ess": ess,
            "ess_ratio": float(ess / len(X_cal)),
            "max_weight": float(w.max()),
            "p99_weight": float(np.percentile(w, 99)),
            "mean_weight": float(w.mean()),
        })
        print(f"  {tname:22s} ESS/n_cal = {rows[-1]['ess_ratio']:.3f}  max_w = {rows[-1]['max_weight']:.2f}  p99_w = {rows[-1]['p99_weight']:.2f}", flush=True)
    return pd.DataFrame(rows)


def main():
    nokw_df = run_onlinenews_robustness()
    diag_df = run_weight_diagnostics_main()
    nokw_df.to_parquet(f"{ROOT}/results/block5_onlinenews_no_kw.parquet")
    diag_df.to_parquet(f"{ROOT}/results/block5_weight_diagnostics.parquet")
    print("\nSaved: results/block5_onlinenews_no_kw.parquet, results/block5_weight_diagnostics.parquet")


if __name__ == "__main__":
    main()
