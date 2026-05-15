"""
Block 4.5 — Rebuild the decision protocol using ONLY deploy-time observables.

Codex review identified the previous Block 4 protocol as ill-posed: it used
`shift_cov_drop_pp` and `shift_class_rate_diff`, both of which require OOD labels.
This block recomputes per-task features from quantities a practitioner would have
at deployment with B=0 target labels:

Deploy-time observables (NO OOD labels required):
- n_id_train      : size of ID training set
- n_id_cal        : size of ID calibration set
- n_features      : number of features
- id_class_imb    : |P(y=1 | ID train) - 0.5| (computable from ID labels only)
- log_dr_max      : max log density ratio of OOD/ID covariates (logistic regression)
- log_dr_p99      : 99th-percentile log density ratio (covariate-shift severity proxy)
- log_dr_std      : std of log density ratios (covariate-shift dispersion)
- ood_pred_entropy: mean predictive entropy on X_OOD (model-side shift indicator)
- ood_pred_drift  : mean(|P_id(y=1|x)| - |P_ood(y=1|x)|) wait no — better:
                    KL(model output dist on X_id_cal || model output dist on X_OOD)

We re-fit the depth-2 decision tree on these features (target: top-1 CP-family from
Block 1 OOD coverage) and LOO-CV across 7 tasks. Honest reporting; no goalpost moving.

Output:
- results/block4_5_deploy_time.parquet  (per-task features)
- results/block4_5_deploy_time.json     (tree + LOO-CV results)
- results/block4_5_deploy_time.md       (analysis report)
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
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from block1_full import (
    load_acs_full, task_acs_state, task_acs_temporal, task_acs_sex_ca,
    task_acs_race_ca, task_adult_sex, task_bank_age, task_taiwan_sex,
)

ROOT = "/Users/bee/Desktop/Work/Research/admission-paper-2026"

KEEP_TASKS = [
    "ACSIncome_state",
    "ACSIncome_temporal",
    "ACSIncome_sex_CA",
    "ACSIncome_race_CA",
    "Adult_sex",
    "Bank_age",
    "Taiwan_sex",
]

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

SEED = 0  # one seed for deploy-time feature computation — these are population stats


def task_deploy_features(acs, task_name, loader):
    X_tr, y_tr, X_cal, y_cal, splits = loader(acs, SEED)
    ood_name = HEADLINE_OOD[task_name]
    X_ood = splits[ood_name][0]

    # ID-side observables (no OOD labels needed)
    n_id_train = int(len(X_tr))
    n_id_cal   = int(len(X_cal))
    n_features = int(X_tr.shape[1])
    id_class_imb = float(abs(float(y_tr.mean()) - 0.5))

    # Density-ratio diagnostics (covariates only — no OOD labels)
    rng = default_rng(SEED)
    # subsample each side to 5000
    def subsamp(X, k=5000):
        if len(X) > k:
            return X[rng.choice(len(X), size=k, replace=False)]
        return X
    X_cal_sub = subsamp(X_cal); X_ood_sub = subsamp(X_ood)
    Xmix = np.vstack([X_cal_sub, X_ood_sub])
    ymix = np.array([0] * len(X_cal_sub) + [1] * len(X_ood_sub))
    lr = LogisticRegression(max_iter=200, solver="liblinear").fit(Xmix, ymix)
    p_shift_cal = lr.predict_proba(X_cal_sub)[:, 1]
    p_shift_cal = np.clip(p_shift_cal, 1e-6, 1 - 1e-6)
    log_dr = np.log(p_shift_cal / (1 - p_shift_cal))
    log_dr_max = float(log_dr.max())
    log_dr_p99 = float(np.percentile(log_dr, 99))
    log_dr_std = float(log_dr.std())

    # Train a quick HGB on X_tr, y_tr — purely for diagnostic model entropy on X_OOD
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)
    hgb.fit(X_tr, y_tr)
    p_ood = hgb.predict_proba(X_ood)
    p_ood = np.clip(p_ood, 1e-7, 1 - 1e-7)
    ood_pred_entropy = float((-(p_ood * np.log(p_ood)).sum(axis=1)).mean())  # nats

    # Model-side covariate shift indicator: KL divergence between marginal P(y=1|X) on ID-cal vs OOD
    p_cal = hgb.predict_proba(X_cal)
    p_cal_pos = float(np.clip(p_cal[:, 1].mean(), 1e-7, 1 - 1e-7))
    p_ood_pos = float(np.clip(p_ood[:, 1].mean(),  1e-7, 1 - 1e-7))
    # binary KL using mean positive predictions as Bernoulli probabilities
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


def best_cp_family_per_task(df_block1):
    ood = df_block1[df_block1["split"] != "ID"]
    g = ood.groupby(["task", "cp"])["coverage"].mean().reset_index()
    g["dist"] = (0.90 - g["coverage"]).abs()
    out = {}
    for t in KEEP_TASKS:
        sub = g[g["task"] == t].sort_values("dist")
        out[t] = sub.iloc[0]["cp"]
    return out


def main():
    print("[step 1] computing deploy-time features for 7 tasks (SEED=0)...", flush=True)
    acs = load_acs_full()
    rows = [task_deploy_features(acs, name, loader) for name, loader in TASK_LOADERS]
    feats = pd.DataFrame(rows).set_index("task")
    print("\n=== Deploy-time features (no OOD labels used) ===")
    print(feats.round(3))
    feats.to_parquet(f"{ROOT}/results/block4_5_deploy_time.parquet")

    print("\n[step 2] loading Block 1 best CP family per task...", flush=True)
    df_block1 = pd.read_parquet(f"{ROOT}/results/block1_full.parquet")
    df_block1 = df_block1[df_block1["task"].isin(KEEP_TASKS)]
    target = best_cp_family_per_task(df_block1)
    print("Best CP family per task (from Block 1 OOD coverage):")
    for t, c in target.items():
        print(f"  {t:22s} → {c}")

    # Build classifier on deploy-time features only
    X = feats.values
    feat_names = list(feats.columns)
    y = np.array([target[t] for t in feats.index])

    print("\n[step 3] full-data depth-2 tree (descriptive)...", flush=True)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    print(export_text(clf, feature_names=feat_names))

    print("[step 4] LOO-CV...", flush=True)
    loo = LeaveOneOut()
    correct = 0
    fold = []
    for tr_idx, te_idx in loo.split(X):
        clf_loo = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf_loo.fit(X[tr_idx], y[tr_idx])
        pred = clf_loo.predict(X[te_idx])[0]
        actual = y[te_idx][0]
        ok = pred == actual
        correct += int(ok)
        fold.append({"task": feats.index[te_idx[0]], "pred": pred, "actual": actual, "correct": bool(ok)})
    n = len(X)
    print(f"\nLOO-CV top-1 CP family using ONLY deploy-time features: {correct}/{n} = {100*correct/n:.0f}%")
    print(f"Random baseline (uniform 3-class): {100/3:.0f}%")
    for f in fold:
        ok = "✓" if f["correct"] else "✗"
        print(f"  {ok} {f['task']:22s} pred={f['pred']:9s} actual={f['actual']}")

    out = {
        "feature_names": feat_names,
        "task_features": feats.reset_index().to_dict("records"),
        "target_cp_family": target,
        "tree_full_data": export_text(clf, feature_names=feat_names),
        "loo_correct": int(correct),
        "loo_n": int(n),
        "loo_rate": float(correct / n),
        "fold_results": fold,
    }
    with open(f"{ROOT}/results/block4_5_deploy_time.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved: results/block4_5_deploy_time.json")


if __name__ == "__main__":
    main()
