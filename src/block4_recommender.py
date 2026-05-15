"""
Block 4 — Decision-protocol recommender.

Honest framing: with only 7 tasks, leave-one-task-out cross-validation of a
12-class (CP variant × calibrator) recommender is statistically very thin.
We therefore (a) fit a shallow tree as a sanity tool, (b) report its LOO-CV
top-1 and top-3 accuracy with explicit caveats, and (c) extract the dominant
split rule and present it as a *protocol* — the real paper contribution.

Inputs:
- results/block1_full.parquet  (Block 1: per-cell OOD coverage / set size / ECE)
- results/block3_budget_axis.parquet (Block 3: budget-vs-method)

Per task we compute:
- n_train, n_features
- class_imbalance      (fraction of positive class on ID train)
- shift_class_rate_diff (|P(y=1|ID) - P(y=1|OOD)|, key feature for Mondrian failure)
- shift_acc_drop_pp    (HGB accuracy ID - HGB accuracy worst-OOD)
- shift_dr_max_log     (max log density-ratio std across OOD splits; proxy for covariate shift)
- shift_cov_drop_pp    (split-CP marginal coverage drop ID - worst-OOD, pp)

Target (per task) — top-3 (CP, calibrator) ordered by distance-to-nominal 0.90.

Output:
- results/block4_recommender.json
- results/block4_recommender.md
- src/decision_protocol.py  (the extracted rule as a runnable Python function)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, os.path.dirname(__file__))

ROOT = "/Users/bee/Desktop/Work/Research/admission-paper-2026"

# Keep tasks consistent with revised proposal (HELOC dropped).
KEEP_TASKS = [
    "ACSIncome_state", "ACSIncome_temporal", "ACSIncome_sex_CA",
    "ACSIncome_race_CA", "Adult_sex", "Bank_age", "Taiwan_sex",
]


def load_block1():
    df = pd.read_parquet(f"{ROOT}/results/block1_full.parquet")
    return df[df["task"].isin(KEEP_TASKS)]


def task_features(df_block1):
    feats = {}
    for task in KEEP_TASKS:
        sub = df_block1[df_block1["task"] == task]
        # ID and worst-OOD per task (worst = lowest split-CP coverage)
        id_rows  = sub[sub["split"] == "ID"]
        ood_rows = sub[sub["split"] != "ID"]
        # base-rate / size proxies: use mean n on ID / n on OOD across rows
        n_id = int(id_rows["n"].mean()) if len(id_rows) else 0
        n_oodtot = int(ood_rows.groupby("split")["n"].first().sum())
        # class balance on training set: not directly stored; use cc_cov0 vs cc_cov1
        # as a weak proxy (Block 1 logs the cc_coverage for class 0/1 on each split).
        cc0 = id_rows["cc_cov0"].mean(); cc1 = id_rows["cc_cov1"].mean()
        # Class imbalance proxy: difference between cov of class 0 and class 1 in ID
        # (a perfectly balanced model has both near 0.90)
        class_imb_proxy = abs(cc0 - cc1)
        # OOD class-rate shift proxy: difference in per-class coverage shift relative to ID
        ood_cc0 = ood_rows["cc_cov0"].mean(); ood_cc1 = ood_rows["cc_cov1"].mean()
        shift_class_rate_diff = abs((cc0 - ood_cc0) - (cc1 - ood_cc1))
        # Accuracy drop: ID acc - worst-OOD acc (mean across calibrators/cps/seeds)
        id_acc = id_rows["accuracy"].mean()
        worst_ood_acc = ood_rows.groupby("split")["accuracy"].mean().min()
        shift_acc_drop_pp = (id_acc - worst_ood_acc) * 100
        # Coverage drop: split-CP only, ID - worst-OOD
        split_id_cov = sub[(sub["cp"] == "split") & (sub["split"] == "ID")]["coverage"].mean()
        split_ood_cov = sub[(sub["cp"] == "split") & (sub["split"] != "ID")].groupby("split")["coverage"].mean().min()
        shift_cov_drop_pp = (split_id_cov - split_ood_cov) * 100
        feats[task] = {
            "n_train_est": n_id * 2,  # ID is 25% of source -> train ~ 50% of source ~ 2× n_id
            "n_features": 10,  # all tasks use ~10–24 cols; HGB doesn't care strongly
            "class_imb_proxy": float(class_imb_proxy),
            "shift_class_rate_diff": float(shift_class_rate_diff),
            "shift_acc_drop_pp": float(shift_acc_drop_pp),
            "shift_cov_drop_pp": float(shift_cov_drop_pp),
        }
    return pd.DataFrame(feats).T


def task_targets(df_block1):
    """For each task, rank (cp, calibrator) by mean OOD distance to 0.90."""
    ood = df_block1[df_block1["split"] != "ID"]
    g = ood.groupby(["task", "cp", "calibrator"])["coverage"].mean().reset_index()
    g["dist"] = (0.90 - g["coverage"]).abs()
    g["method"] = g["cp"] + "+" + g["calibrator"]
    ranked = {}
    for task in KEEP_TASKS:
        sub = g[g["task"] == task].sort_values("dist")
        ranked[task] = list(sub["method"])
    return ranked


def main():
    df = load_block1()
    feats = task_features(df)
    print("=== Task descriptors ===")
    print(feats.round(3))
    print()
    ranked = task_targets(df)
    print("=== Top-3 (cp+calibrator) per task by distance to nominal 0.90 ===")
    for t in KEEP_TASKS:
        print(f"  {t:22s} → {ranked[t][:3]}")
    print()

    # Build classification target: predict the top-1 method.
    # 7 tasks, ~12 classes — LOO-CV is heavily over-saturated. Treat as exploration.
    X = feats.values
    feature_names = list(feats.columns)
    y_top1 = np.array([ranked[t][0] for t in feats.index])
    print(f"Unique top-1 methods across 7 tasks: {sorted(set(y_top1))}")
    print()

    # Fit a depth-2 tree on full data (purely descriptive — extract the rule)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y_top1)
    print("=== Full-data tree (descriptive, NOT cross-validated) ===")
    print(export_text(clf, feature_names=feature_names))

    # LOO-CV top-1 and top-3 accuracy
    loo = LeaveOneOut()
    top1_correct = 0
    top3_correct = 0
    fold_results = []
    for tr_idx, te_idx in loo.split(X):
        clf_loo = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf_loo.fit(X[tr_idx], y_top1[tr_idx])
        pred = clf_loo.predict(X[te_idx])[0]
        # top-3 from probability ordering: use predict_proba if available
        try:
            probs = clf_loo.predict_proba(X[te_idx])[0]
            classes = clf_loo.classes_
            top3 = list(classes[np.argsort(-probs)[:3]])
        except Exception:
            top3 = [pred]
        actual_top1 = y_top1[te_idx][0]
        actual_top3 = ranked[feats.index[te_idx[0]]][:3]
        top1_correct += int(pred == actual_top1)
        # "top-3 hit": predicted top-1 is in the actual top-3
        in_top3 = pred in actual_top3
        top3_correct += int(in_top3)
        fold_results.append({
            "task": feats.index[te_idx[0]],
            "predicted_top1": pred,
            "actual_top1": actual_top1,
            "actual_top3": actual_top3,
            "predicted_in_actual_top3": bool(in_top3),
        })

    n = len(X)
    print(f"\n=== LOO-CV (n={n} tasks; 7 tasks ≪ classes ≈ {len(set(y_top1))}; results are exploratory) ===")
    print(f"  top-1 LOO accuracy: {top1_correct}/{n} = {100*top1_correct/n:.0f}%")
    print(f"  predicted-top-1 ∈ actual-top-3: {top3_correct}/{n} = {100*top3_correct/n:.0f}%")
    print()
    for fr in fold_results:
        ok = "✓" if fr["predicted_in_actual_top3"] else "✗"
        print(f"  {ok} {fr['task']:22s} pred={fr['predicted_top1']:22s} actual_top1={fr['actual_top1']}")

    # === Also fit a CP-family-only recommender (3 classes: split/weighted/mondrian) ===
    # This is the practitioner-relevant decision; calibrator choice is a minor refinement.
    y_cp = np.array([ranked[t][0].split("+")[0] for t in feats.index])
    print(f"\n=== CP-family-only recommender (3 classes) ===")
    print(f"Top-1 CP per task: {dict(zip(feats.index, y_cp))}")
    clf_cp_full = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y_cp)
    print("Full-data tree (descriptive):")
    print(export_text(clf_cp_full, feature_names=feature_names))
    top1_cp = 0
    fold_cp = []
    for tr_idx, te_idx in loo.split(X):
        clf_loo = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf_loo.fit(X[tr_idx], y_cp[tr_idx])
        pred = clf_loo.predict(X[te_idx])[0]
        actual = y_cp[te_idx][0]
        ok = pred == actual
        top1_cp += int(ok)
        fold_cp.append({
            "task": feats.index[te_idx[0]],
            "predicted_cp": pred, "actual_cp": actual, "correct": bool(ok),
        })
    print(f"\nLOO-CV top-1 CP family: {top1_cp}/{n} = {100*top1_cp/n:.0f}%")
    print(f"  Random baseline (uniform 3-class): {100/3:.0f}%")
    for fr in fold_cp:
        ok_str = "✓" if fr["correct"] else "✗"
        print(f"  {ok_str} {fr['task']:22s} pred_cp={fr['predicted_cp']:9s} actual_cp={fr['actual_cp']}")

    # Save artifacts
    out = {
        "feature_names": feature_names,
        "task_features": feats.reset_index().rename(columns={"index": "task"}).to_dict("records"),
        "top3_per_task": ranked,
        "top1_cp_family_per_task": dict(zip(list(feats.index), list(y_cp))),
        "tree_full_data_method": export_text(clf, feature_names=feature_names),
        "tree_full_data_cp_family": export_text(clf_cp_full, feature_names=feature_names),
        "loo_top1_method": {"correct": int(top1_correct), "n": int(n), "rate": float(top1_correct/n)},
        "loo_top3_method_in_actual_top3": {"correct": int(top3_correct), "n": int(n), "rate": float(top3_correct/n)},
        "loo_top1_cp_family": {"correct": int(top1_cp), "n": int(n), "rate": float(top1_cp/n)},
        "fold_results_method": fold_results,
        "fold_results_cp_family": fold_cp,
    }
    with open(f"{ROOT}/results/block4_recommender.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/block4_recommender.json")


if __name__ == "__main__":
    main()
