"""
Generate the three paper figures.

Fig 1: Split-CP marginal coverage drop per task, with weighted and Mondrian overlaid.
Fig 2: ESS / n_cal vs weighted-CP coverage gap from nominal 0.90, scatter.
Fig 3: Budget-axis coverage on hard-shift tasks (post-hoc CP vs retrain).

Output PDFs + PNGs in paper/latex/figures/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pathlib
# Resolve project root relative to this file (code-release/src/figures/<this>.py → code-release/)
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
OUT = f"{ROOT}/paper-figures"
os.makedirs(OUT, exist_ok=True)

# Make math labels render without TeX (TeX is not configured for the figure pipeline)
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

TASK_ORDER = [
    "OnlineNews_channel",
    "ACSIncome_state",
    "ACSIncome_race_CA",
    "Adult_sex",
    "ACSIncome_temporal",
    "SpeedDating_race",
    "Diabetes_race",
    "ACSIncome_sex_CA",
    "Bank_age",
    "Taiwan_sex",
]
TASK_LABEL = {
    "OnlineNews_channel":   "OnlineNews\n(channel)",
    "ACSIncome_state":      "ACSIncome\n(state)",
    "ACSIncome_race_CA":    "ACSIncome\n(race, CA)",
    "Adult_sex":            "Adult (sex)",
    "ACSIncome_temporal":   "ACSIncome\n(temporal)",
    "SpeedDating_race":     "SpeedDating\n(race)",
    "Diabetes_race":        "Diabetes 130-US\n(race)",
    "ACSIncome_sex_CA":     "ACSIncome\n(sex, CA)",
    "Bank_age":             "Bank (age)",
    "Taiwan_sex":           "Taiwan credit\n(sex)",
}

# ----------------------------------------------------------------------------
# Fig 1 — Split-CP coverage drop per task, with weighted and Mondrian overlaid
# ----------------------------------------------------------------------------
def make_fig1():
    df = pd.read_parquet(f"{ROOT}/results/block1_full_v2.parquet")
    df = df[df["task"].isin(TASK_ORDER)]
    # ID coverage per (task, cp) mean across calibrators/learners/seeds
    id_cov  = df[df["split"] == "ID"].groupby(["task", "cp"])["coverage"].mean().unstack("cp")
    # Worst-OOD coverage per (task, cp)
    ood_min = df[df["split"] != "ID"].groupby(["task", "cp", "split"])["coverage"].mean()
    ood_min = ood_min.groupby(["task", "cp"]).min().unstack("cp")
    # Drop in pp (positive = under-cover)
    drop = (id_cov - ood_min) * 100
    drop = drop.reindex(TASK_ORDER)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    n = len(TASK_ORDER); x = np.arange(n)
    w = 0.27
    colors = {"split": "#1f77b4", "weighted": "#ff7f0e", "mondrian": "#2ca02c"}
    for i, cp in enumerate(["split", "weighted", "mondrian"]):
        ax.bar(x + (i - 1) * w, drop[cp].values, width=w, color=colors[cp],
               edgecolor="black", linewidth=0.4, label=cp)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(5, color="red", lw=0.7, linestyle="--", label="5 pp threshold")
    ax.axhline(-1, color="grey", lw=0.4, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=0, ha="center")
    ax.set_ylabel("Coverage drop  (pp;  +ve = under-cover)")
    ax.set_title("Worst-OOD coverage drop per task, by CP variant (10 tasks, mean over learners/calibrators/seeds)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_ylim(-3, 10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_coverage_drop.pdf")
    fig.savefig(f"{OUT}/fig1_coverage_drop.png", dpi=160)
    plt.close(fig)
    print("Saved fig1")


# ----------------------------------------------------------------------------
# Fig 2 — ESS/n_cal vs weighted-CP coverage gap from nominal 0.90
# ----------------------------------------------------------------------------
def make_fig2():
    df_diag = pd.read_parquet(f"{ROOT}/results/block5_weight_diagnostics.parquet")
    df_b1 = pd.read_parquet(f"{ROOT}/results/block1_full_v2.parquet")
    # Get weighted CP coverage on the headline OOD per task (mean across cals/learners/seeds)
    HEADLINE_OOD = {
        "ACSIncome_state":    "OOD_MS", "ACSIncome_temporal": "OOD_2018",
        "ACSIncome_sex_CA":   "OOD_F",  "ACSIncome_race_CA":  "OOD_Black",
        "Adult_sex":          "OOD_F",  "Bank_age":           "OOD_age_ge_35",
        "Taiwan_sex":         "OOD_F",  "Diabetes_race":      "OOD_African",
        "SpeedDating_race":   "OOD_Asian", "OnlineNews_channel": "OOD_entertainment",
    }
    wc = []
    for t, sp in HEADLINE_OOD.items():
        sub = df_b1[(df_b1["task"] == t) & (df_b1["split"] == sp) & (df_b1["cp"] == "weighted")]
        if len(sub):
            wc.append({"task": t, "wcp_cov": sub["coverage"].mean()})
    wc = pd.DataFrame(wc).set_index("task")
    diag = df_diag.set_index("task")
    df = diag.join(wc, how="inner")
    df["gap"] = df["wcp_cov"] - 0.90  # positive = over-cover; negative = under-cover
    df["abs_gap"] = df["gap"].abs()
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    colors = ["#d62728" if abs(g) > 0.03 else "#2ca02c" for g in df["gap"]]
    ax.scatter(df["ess_ratio"], df["wcp_cov"], s=60, c=colors, edgecolor="black", linewidth=0.4, zorder=3)
    for t in df.index:
        ax.annotate(TASK_LABEL[t].replace("\n", " "), (df.loc[t, "ess_ratio"], df.loc[t, "wcp_cov"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.90, color="black", lw=0.5, linestyle="--", label="target  0.90")
    ax.axhspan(0.88, 0.92, color="grey", alpha=0.1, label="±0.02 of target")
    ax.axvline(0.05, color="red", lw=0.6, linestyle=":", label=r"ESS/$n_{cal}$ = 0.05 (this study's diagnostic threshold)")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlim(-0.001, 1.05)
    ax.set_ylim(0.55, 1.05)
    ax.set_xlabel(r"ESS / $n_{cal}$ of weighted-CP density-ratio estimator (log scale)")
    ax.set_ylabel("Weighted-CP marginal coverage")
    ax.set_title("Weighted-CP coverage vs density-ratio ESS (10-task panel; one point per task)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_ess_vs_coverage.pdf")
    fig.savefig(f"{OUT}/fig2_ess_vs_coverage.png", dpi=160)
    plt.close(fig)
    print("Saved fig2")


# ----------------------------------------------------------------------------
# Fig 3 — Budget-axis lines for representative tasks
# ----------------------------------------------------------------------------
def make_fig3():
    df1 = pd.read_parquet(f"{ROOT}/results/block3_budget_axis.parquet")
    df2 = pd.read_parquet(f"{ROOT}/results/block3_budget_axis_v2.parquet")
    df = pd.concat([df1, df2], ignore_index=True)
    # Pick representative tasks: hard-shift (OnlineNews, ACSIncome_state), moderate (Adult_sex), no-shift (Taiwan)
    SHOW = ["OnlineNews_channel", "ACSIncome_state", "Adult_sex", "Taiwan_sex"]
    df = df[df["task"].isin(SHOW)]
    g = df.groupby(["task", "method", "B"])["coverage"].mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=True)
    method_colors = {"id_postcp": "#1f77b4", "recalibrate": "#ff7f0e", "retrain": "#2ca02c"}
    method_label  = {"id_postcp": "post-hoc CP (B=0)", "recalibrate": "recalibrate(cal $\\cup$ B)", "retrain": "retrain(ID $\\cup$ B)"}
    for i, t in enumerate(SHOW):
        ax = axes[i]
        sub = g[g["task"] == t]
        for m in ["id_postcp", "recalibrate", "retrain"]:
            sub_m = sub[sub["method"] == m].sort_values("B")
            xs = sub_m["B"].values
            ys = sub_m["coverage"].values
            ax.plot(xs, ys, "-o", color=method_colors[m], markersize=4, label=method_label[m] if i == 0 else None, linewidth=1.5)
        ax.axhline(0.90, color="black", lw=0.5, linestyle="--")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 50, 500, 5000])
        ax.set_xticklabels(["0", "50", "500", "5000"])
        ax.set_xlabel(r"target labels $B$")
        if i == 0:
            ax.set_ylabel("Marginal coverage (target 0.90)")
        ax.set_title(TASK_LABEL[t].replace("\n", " "))
        ax.set_ylim(0.50, 1.00)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    fig.suptitle("Budget-axis: marginal coverage vs target-label budget, 4 representative tasks (HGB, 3 seeds)", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_budget_axis.pdf")
    fig.savefig(f"{OUT}/fig3_budget_axis.png", dpi=160)
    plt.close(fig)
    print("Saved fig3")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
