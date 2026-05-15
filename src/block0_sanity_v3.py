"""
Block 0 sanity pilot v3.

Findings from v2 forced a redesign:
- 1 - p_y nonconformity gives trivial size-2 prediction sets and 100% coverage on
  binary ACSIncome (HGB is well-enough calibrated that q >= 0.5, so both labels
  end up in the set for the bulk of points). This is a known failure mode of LAC.
- XGBoost 3.0 raises 'base_score must be in (0,1)' when fit with default --
  set base_score=0.5 explicitly.

v3 changes:
- Use APS (Adaptive Prediction Sets, Romano-Sesia-Candes 2020) as the conformal
  score. APS gives non-trivial set sizes on binary problems.
- Track both coverage and set size; either a coverage drop OR a set-size growth
  under shift is a positive sanity signal.
- Pass criterion: max OOD (coverage drop in pp) >= 3 OR (max OOD set size -
  ID set size) >= 0.1 across any (learner, OOD state) cell.

Setup: ACSIncome 2018; ID = California; OOD = TX, FL, MS; 3 learners x 3 seeds.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datasets import load_dataset

ALPHA = 0.10
SEEDS = [0, 1, 2]
ID_STATE = "CA"
OOD_STATES = ["TX", "FL", "MS"]
YEAR = 2018
# NOTE: the birkhoffg/folktables-acs-income HF mirror already binarizes PINCP
# (1 = income > 50k, 0 otherwise) -- it is the label column, NOT raw income.
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_2018():
    print("[load] HF mirror birkhoffg/folktables-acs-income (train split)")
    ds = load_dataset("birkhoffg/folktables-acs-income", split="train")
    df = ds.to_pandas()
    df = df[df["YEAR"] == YEAR].copy()
    print(f"[load] rows for YEAR={YEAR}: {len(df)}")
    df["y"] = df["PINCP"].astype(int)
    print(f"[load] y positive rate (CA): {(df.loc[df['STATE']=='CA','y']).mean():.3f}")
    return df


def aps_scores(p_full, y, rng):
    """Romano-Sesia-Candes 2020 randomized APS score.
    p_full has shape (n, K); y is the true label. Returns s in [0,1]."""
    n, K = p_full.shape
    order = np.argsort(-p_full, axis=1)  # descending
    rank_of_y = np.array([np.where(order[i] == y[i])[0][0] for i in range(n)])
    # prefix-sum of sorted probs
    sorted_p = np.take_along_axis(p_full, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    # sum of probs strictly above true label's rank
    above = np.zeros(n)
    p_y = p_full[np.arange(n), y]
    for i in range(n):
        r = rank_of_y[i]
        if r > 0:
            above[i] = cumsum[i, r - 1]
        # subtract a uniform fraction of the true-class prob (randomization)
    u = rng.uniform(0, 1, size=n)
    s = above + u * p_y
    return s


def aps_inclusion(p_full, q, rng):
    """For each row and each candidate class, decide inclusion in the APS set
    of size threshold q. Returns boolean (n, K) array."""
    n, K = p_full.shape
    order = np.argsort(-p_full, axis=1)
    sorted_p = np.take_along_axis(p_full, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    # For each rank r, score is cumsum[r-1] + u * sorted_p[r]; include iff score <= q
    u = rng.uniform(0, 1, size=(n, K))
    above = np.concatenate([np.zeros((n, 1)), cumsum[:, :-1]], axis=1)  # (n, K)
    scores_sorted = above + u * sorted_p
    inc_sorted = scores_sorted <= q
    # restore original class order
    inc = np.zeros((n, K), dtype=bool)
    for i in range(n):
        inc[i, order[i]] = inc_sorted[i]
    return inc


def coverage_setsize_aps(p_full, y, q, rng):
    inc = aps_inclusion(p_full, q, rng)
    set_sizes = inc.sum(axis=1)
    covered = inc[np.arange(len(y)), y]
    return float(covered.mean()), float(set_sizes.mean())


def run_one(df, learner_name, learner_cls, learner_kwargs, seed):
    rng = default_rng(seed)
    t0 = time.time()
    id_df = df[df["STATE"] == ID_STATE]
    X_id = id_df[FEATURES].values.astype(float)
    y_id = id_df["y"].values.astype(int)
    n = len(X_id)
    idx = rng.permutation(n)
    n_tr = int(0.5 * n); n_cal = int(0.25 * n)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]
    model = learner_cls(**learner_kwargs)
    model.fit(X_id[tr], y_id[tr])
    # calibration scores
    p_cal_full = model.predict_proba(X_id[cal])
    s_cal = aps_scores(p_cal_full, y_id[cal], rng)
    k = int(np.ceil((len(s_cal) + 1) * (1 - ALPHA)))
    q = float(np.sort(s_cal)[min(k - 1, len(s_cal) - 1)])
    # ID test
    p_id_te = model.predict_proba(X_id[te])
    cov_id, sz_id = coverage_setsize_aps(p_id_te, y_id[te], q, rng)
    acc_id = float((np.argmax(p_id_te, axis=1) == y_id[te]).mean())
    rows = [{
        "learner": learner_name, "seed": seed, "split": "ID_test",
        "state": ID_STATE, "n": int(len(te)),
        "acc": acc_id, "coverage": cov_id, "set_size": sz_id, "q": q,
        "coverage_drop_pp": 0.0, "size_increase": 0.0,
    }]
    for st in OOD_STATES:
        sub = df[df["STATE"] == st]
        X_oo = sub[FEATURES].values.astype(float)
        y_oo = sub["y"].values.astype(int)
        p_oo = model.predict_proba(X_oo)
        cov_oo, sz_oo = coverage_setsize_aps(p_oo, y_oo, q, rng)
        acc_oo = float((np.argmax(p_oo, axis=1) == y_oo).mean())
        rows.append({
            "learner": learner_name, "seed": seed, "split": f"OOD_{st}",
            "state": st, "n": int(len(y_oo)),
            "acc": acc_oo, "coverage": cov_oo, "set_size": sz_oo, "q": q,
            "coverage_drop_pp": float((cov_id - cov_oo) * 100),
            "size_increase": float(sz_oo - sz_id),
        })
    print(f"  {learner_name:5s} seed={seed}  q={q:.3f}  "
          f"cov_id={cov_id:.3f} sz_id={sz_id:.2f}  "
          f"TX cov={rows[1]['coverage']:.3f} Δ={rows[1]['coverage_drop_pp']:+.2f}pp sz_inc={rows[1]['size_increase']:+.3f}  "
          f"FL cov={rows[2]['coverage']:.3f} Δ={rows[2]['coverage_drop_pp']:+.2f}pp sz_inc={rows[2]['size_increase']:+.3f}  "
          f"MS cov={rows[3]['coverage']:.3f} Δ={rows[3]['coverage_drop_pp']:+.2f}pp sz_inc={rows[3]['size_increase']:+.3f}  "
          f"({time.time() - t0:.1f}s)")
    return rows


def main():
    df = load_2018()
    print(f"[states] {sorted(df['STATE'].unique())[:10]}...")
    print(f"[counts] ID={ID_STATE}:{(df['STATE']==ID_STATE).sum()} "
          f"TX:{(df['STATE']=='TX').sum()} FL:{(df['STATE']=='FL').sum()} MS:{(df['STATE']=='MS').sum()}")
    learners = [
        ("hgb",   HistGradientBoostingClassifier, {"max_iter": 300, "learning_rate": 0.05}),
        ("xgb",   XGBClassifier,                  {"n_estimators": 300, "learning_rate": 0.05, "tree_method": "hist", "eval_metric": "logloss", "verbosity": 0, "n_jobs": 4, "base_score": 0.5}),
        ("lgbm",  LGBMClassifier,                 {"n_estimators": 300, "learning_rate": 0.05, "verbose": -1, "n_jobs": 4}),
    ]
    all_rows = []
    for learner_name, cls, kw in learners:
        for seed in SEEDS:
            kw_seeded = dict(kw); kw_seeded["random_state"] = seed
            rows = run_one(df, learner_name, cls, kw_seeded, seed)
            all_rows.extend(rows)
    out_df = pd.DataFrame(all_rows)
    id_agg = out_df[out_df["split"] == "ID_test"].groupby("learner").agg({
        "coverage": "mean", "set_size": "mean", "acc": "mean",
    }).round(3)
    ood_agg = out_df[out_df["split"] != "ID_test"].groupby(["learner", "state"]).agg({
        "coverage": "mean", "set_size": "mean", "acc": "mean",
        "coverage_drop_pp": "mean", "size_increase": "mean",
    }).round(3)
    print("\n=== ID test (mean across seeds) ===")
    print(id_agg)
    print("\n=== OOD by (learner, state) ===")
    print(ood_agg)
    out_json = os.path.join(RESULTS_DIR, "block0_sanity_v3.json")
    with open(out_json, "w") as f:
        json.dump({
            "rows": all_rows,
            "id_agg": id_agg.reset_index().to_dict(orient="records"),
            "ood_agg": ood_agg.reset_index().to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved: {out_json}")
    cov_drops = out_df[out_df["split"] != "ID_test"]["coverage_drop_pp"].values
    sz_incs = out_df[out_df["split"] != "ID_test"]["size_increase"].values
    print(f"\n[acceptance] coverage drop: max={cov_drops.max():+.2f}pp, mean={cov_drops.mean():+.2f}pp")
    print(f"[acceptance] set size increase: max={sz_incs.max():+.3f}, mean={sz_incs.mean():+.3f}")
    pass_cov = cov_drops.max() >= 3.0
    pass_sz = sz_incs.max() >= 0.10
    print(f"\nVERDICT: {'PASS' if (pass_cov or pass_sz) else 'FAIL'} "
          f"(coverage-drop>=3pp: {pass_cov}, set-size-inc>=0.10: {pass_sz})")


if __name__ == "__main__":
    main()
