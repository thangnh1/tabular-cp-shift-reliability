"""
Block 4.5 v2 — Deploy-time decision protocol re-fit on the expanded 10-task pool.

Same feature definitions as `src/block4_5_deploy_time_protocol.py`. New task pool
includes the 3 v2 datasets (Diabetes_race, SpeedDating_race, OnlineNews_channel).

Output:
- results/block4_5_v2_deploy_time.parquet  (per-task features, 10 rows)
- results/block4_5_v2_deploy_time.json     (tree + LOO-CV results)
- results/block4_5_v2_deploy_time.log
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from block1_full import (
    load_acs_full, task_acs_state, task_acs_temporal, task_acs_sex_ca,
    task_acs_race_ca, task_adult_sex, task_bank_age, task_taiwan_sex,
)
from extra_tasks import task_diabetes_race, task_speeddating_race, task_onlinenews_channel

import pathlib
# Resolve project root relative to this file (code-release/src/<this>.py → code-release/)
ROOT = str(pathlib.Path(__file__).resolve().parents[1])

# 10 tasks (HELOC excluded as in v2)
TASKS = [
    ("ACSIncome_state",      task_acs_state),
    ("ACSIncome_temporal",   task_acs_temporal),
    ("ACSIncome_sex_CA",     task_acs_sex_ca),
    ("ACSIncome_race_CA",    task_acs_race_ca),
    ("Adult_sex",            task_adult_sex),
    ("Bank_age",             task_bank_age),
    ("Taiwan_sex",           task_taiwan_sex),
    ("Diabetes_race",         task_diabetes_race),
    ("SpeedDating_race",      task_speeddating_race),
    ("OnlineNews_channel",    task_onlinenews_channel),
]

HEADLINE_OOD = {
    "ACSIncome_state":    "OOD_MS",
    "ACSIncome_temporal": "OOD_2018",
    "ACSIncome_sex_CA":   "OOD_F",
    "ACSIncome_race_CA":  "OOD_Black",
    "Adult_sex":          "OOD_F",
    "Bank_age":           "OOD_age_ge_35",
    "Taiwan_sex":         "OOD_F",
    "Diabetes_race":       "OOD_African",
    "SpeedDating_race":    "OOD_Asian",
    "OnlineNews_channel":  "OOD_entertainment",
}

SEED = 0


def task_deploy_features(acs, task_name, loader):
    X_tr, y_tr, X_cal, y_cal, splits = loader(acs, SEED)
    ood_name = HEADLINE_OOD[task_name]
    X_ood = splits[ood_name][0]

    n_id_train = int(len(X_tr))
    n_id_cal   = int(len(X_cal))
    n_features = int(X_tr.shape[1])
    id_class_imb = float(abs(float(y_tr.mean()) - 0.5))

    rng = default_rng(SEED)
    def subsamp(X, k=5000):
        return X[rng.choice(len(X), size=k, replace=False)] if len(X) > k else X
    Xs_cal = subsamp(X_cal); Xs_ood = subsamp(X_ood)
    Xmix = np.vstack([Xs_cal, Xs_ood])
    ymix = np.array([0]*len(Xs_cal) + [1]*len(Xs_ood))
    lr = LogisticRegression(max_iter=200, solver="liblinear").fit(Xmix, ymix)
    p_shift_cal = np.clip(lr.predict_proba(Xs_cal)[:, 1], 1e-6, 1-1e-6)
    log_dr = np.log(p_shift_cal / (1 - p_shift_cal))
    log_dr_max = float(log_dr.max())
    log_dr_p99 = float(np.percentile(log_dr, 99))
    log_dr_std = float(log_dr.std())

    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)
    hgb.fit(X_tr, y_tr)
    p_ood = np.clip(hgb.predict_proba(X_ood), 1e-7, 1-1e-7)
    ood_pred_entropy = float((-(p_ood * np.log(p_ood)).sum(axis=1)).mean())
    p_cal_pos = float(np.clip(hgb.predict_proba(X_cal)[:, 1].mean(), 1e-7, 1-1e-7))
    p_ood_pos = float(np.clip(p_ood[:, 1].mean(), 1e-7, 1-1e-7))
    mean_pred_kl_id_to_ood = (
        p_cal_pos * np.log(p_cal_pos / p_ood_pos)
        + (1 - p_cal_pos) * np.log((1 - p_cal_pos) / (1 - p_ood_pos))
    )

    return {
        "task": task_name,
        "n_id_train": n_id_train,
        "n_id_cal": n_id_cal,
        "n_features": n_features,
        "id_class_imb": id_class_imb,
        "log_dr_max": log_dr_max,
        "log_dr_p99": log_dr_p99,
        "log_dr_std": log_dr_std,
        "ood_pred_entropy": ood_pred_entropy,
        "mean_pred_kl_id_to_ood": float(mean_pred_kl_id_to_ood),
        "mean_pred_id": p_cal_pos,
        "mean_pred_ood": p_ood_pos,
    }


def best_cp_family_per_task(df_v2):
    ood = df_v2[df_v2["split"] != "ID"]
    g = ood.groupby(["task", "cp"])["coverage"].mean().reset_index()
    g["dist"] = (0.90 - g["coverage"]).abs()
    out = {}
    for t in sorted(df_v2["task"].unique()):
        sub = g[g["task"] == t].sort_values("dist")
        out[t] = sub.iloc[0]["cp"]
    return out


def main():
    print("[step 1] computing deploy-time features for 10 tasks (SEED=0)...", flush=True)
    acs = load_acs_full()
    rows = [task_deploy_features(acs, name, loader) for name, loader in TASKS]
    feats = pd.DataFrame(rows).set_index("task")
    print("\n=== Deploy-time features ===")
    print(feats.round(3))
    feats.to_parquet(f"{ROOT}/results/block4_5_v2_deploy_time.parquet")

    print("\n[step 2] loading Block 1 v2 best CP family per task...", flush=True)
    df = pd.read_parquet(f"{ROOT}/results/block1_full_v2.parquet")
    target = best_cp_family_per_task(df)
    print("Best CP family per task:")
    for t, c in target.items():
        print(f"  {t:22s} → {c}")

    X = feats.values
    fnames = list(feats.columns)
    y = np.array([target[t] for t in feats.index])

    print("\n[step 3] full-data depth-2 tree (descriptive)...", flush=True)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    print(export_text(clf, feature_names=fnames))

    print("[step 4] LOO-CV with depth-2 tree...", flush=True)
    loo = LeaveOneOut()
    correct = 0; fold = []
    for tr_idx, te_idx in loo.split(X):
        clf_loo = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X[tr_idx], y[tr_idx])
        pred = clf_loo.predict(X[te_idx])[0]
        actual = y[te_idx][0]
        ok = pred == actual
        correct += int(ok)
        fold.append({"task": feats.index[te_idx[0]], "pred": pred, "actual": actual, "correct": bool(ok)})
    n = len(X)
    print(f"\nLOO-CV depth-2: {correct}/{n} = {100*correct/n:.0f}% (random baseline 33%)")
    for f in fold:
        ok = "✓" if f["correct"] else "✗"
        print(f"  {ok} {f['task']:22s} pred={f['pred']:9s} actual={f['actual']}")

    # Try depth-3 too (more capacity, 10 tasks may support it)
    print("\n[step 5] depth-3 tree LOO-CV...", flush=True)
    correct3 = 0; fold3 = []
    for tr_idx, te_idx in loo.split(X):
        clf_loo = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[tr_idx], y[tr_idx])
        pred = clf_loo.predict(X[te_idx])[0]
        actual = y[te_idx][0]
        ok = pred == actual
        correct3 += int(ok)
        fold3.append({"task": feats.index[te_idx[0]], "pred": pred, "actual": actual, "correct": bool(ok)})
    print(f"LOO-CV depth-3: {correct3}/{n} = {100*correct3/n:.0f}%")

    out = {
        "feature_names": fnames,
        "task_features": feats.reset_index().to_dict("records"),
        "target_cp_family": target,
        "tree_full_data_depth2": export_text(clf, feature_names=fnames),
        "loo_depth2": {"correct": int(correct), "n": int(n), "rate": float(correct/n)},
        "loo_depth3": {"correct": int(correct3), "n": int(n), "rate": float(correct3/n)},
        "fold_results_depth2": fold,
        "fold_results_depth3": fold3,
    }
    with open(f"{ROOT}/results/block4_5_v2_deploy_time.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved: results/block4_5_v2_deploy_time.json")


if __name__ == "__main__":
    main()
