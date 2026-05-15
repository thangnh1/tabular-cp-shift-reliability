"""
Block 1 full grid -- expansion of src/block1_v1_validation.py.

Tasks (8 total, each binary classification with explicit ID/OOD partitions):
- ACSIncome_state     : ID=CA-2018; OOD={TX,FL,MS}-2018
- ACSIncome_temporal  : ID=CA-2014; OOD=CA-2018
- ACSIncome_sex_CA    : ID=CA-2018 male; OOD=CA-2018 female
- ACSIncome_race_CA   : ID=CA-2018 RAC1P=1 (white-alone); OOD=CA-2018 RAC1P=2 (black), RAC1P=6 (asian)
- Adult_sex           : ID=Adult male; OOD=Adult female (sklearn fetch_openml 'adult')
- Bank_age            : ID=age<35; OOD=age>=35 (OpenML bank-marketing v1, V1=age)
- Taiwan_sex          : ID=SEX=1 male; OOD=SEX=2 female (OpenML 42477)
- HELOC_gender        : ID=majority gender; OOD=minority (OpenML 43890)

Seeds: [0,1,2,3,4]   Learners: HGB, XGB, LGBM
CP variants: split, weighted, Mondrian (all with APS score, from src/cp_variants.py)
Calibrators: uncal, Platt, isotonic, temperature (from src/calibrators.py)

Outputs:
- results/block1_full.parquet  (long-form per-cell metrics)
- results/block1_full.log      (live progress; warnings filtered by the wrapper)

Estimated wall time: 20-30 min on CPU. No GPU.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datasets import load_dataset

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from cp_variants import (
    split_cp, mondrian_cp, weighted_cp_from_weights,
    estimate_likelihood_ratio, summarize,
)
from calibrators import apply_calibrator, ece_uniform_width, ece_uniform_mass_jackknife, brier

ALPHA = 0.10
SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ACS_FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]


# ----------------------------------------------------------------------------
# Task loaders. Each returns a dict: {"X_tr","y_tr","X_cal","y_cal","splits"}
# where splits = {split_name: (X_split, y_split)} including an "ID" key for the
# in-distribution held-out test.
# ----------------------------------------------------------------------------

def _split_id(X_id, y_id, seed):
    rng = default_rng(seed)
    n = len(X_id)
    idx = rng.permutation(n)
    n_tr = int(0.5 * n); n_cal = int(0.25 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]


def load_acs_full():
    print("[load] HF mirror birkhoffg/folktables-acs-income (all years)", flush=True)
    ds = load_dataset("birkhoffg/folktables-acs-income", split="train")
    df = ds.to_pandas()
    df["y"] = df["PINCP"].astype(int)
    return df


def task_acs_state(acs, seed):
    df = acs[acs["YEAR"] == 2018]
    ca = df[df["STATE"] == "CA"]
    X = ca[ACS_FEATURES].values.astype(float); y = ca["y"].values.astype(int)
    tr, cal, te = _split_id(X, y, seed)
    splits = {"ID": (X[te], y[te])}
    for st in ["TX", "FL", "MS"]:
        sub = df[df["STATE"] == st]
        splits[f"OOD_{st}"] = (sub[ACS_FEATURES].values.astype(float), sub["y"].values.astype(int))
    return X[tr], y[tr], X[cal], y[cal], splits


def task_acs_temporal(acs, seed):
    df_14 = acs[(acs["STATE"] == "CA") & (acs["YEAR"] == 2014)]
    df_18 = acs[(acs["STATE"] == "CA") & (acs["YEAR"] == 2018)]
    X14 = df_14[ACS_FEATURES].values.astype(float); y14 = df_14["y"].values.astype(int)
    X18 = df_18[ACS_FEATURES].values.astype(float); y18 = df_18["y"].values.astype(int)
    tr, cal, te = _split_id(X14, y14, seed)
    splits = {"ID": (X14[te], y14[te]), "OOD_2018": (X18, y18)}
    return X14[tr], y14[tr], X14[cal], y14[cal], splits


def task_acs_sex_ca(acs, seed):
    df = acs[(acs["STATE"] == "CA") & (acs["YEAR"] == 2018)]
    male = df[df["SEX"] == 1]
    female = df[df["SEX"] == 2]
    Xm = male[ACS_FEATURES].values.astype(float); ym = male["y"].values.astype(int)
    Xf = female[ACS_FEATURES].values.astype(float); yf = female["y"].values.astype(int)
    tr, cal, te = _split_id(Xm, ym, seed)
    splits = {"ID": (Xm[te], ym[te]), "OOD_F": (Xf, yf)}
    return Xm[tr], ym[tr], Xm[cal], ym[cal], splits


def task_acs_race_ca(acs, seed):
    df = acs[(acs["STATE"] == "CA") & (acs["YEAR"] == 2018)]
    white = df[df["RAC1P"] == 1]
    Xw = white[ACS_FEATURES].values.astype(float); yw = white["y"].values.astype(int)
    tr, cal, te = _split_id(Xw, yw, seed)
    splits = {"ID": (Xw[te], yw[te])}
    for code, name in [(2, "Black"), (6, "Asian")]:
        sub = df[df["RAC1P"] == code]
        if len(sub) > 100:
            splits[f"OOD_{name}"] = (sub[ACS_FEATURES].values.astype(float), sub["y"].values.astype(int))
    return Xw[tr], yw[tr], Xw[cal], yw[cal], splits


# Adult / Bank / Taiwan / HELOC --------------------------------------------------

def _encode_frame(X):
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("object", "category"):
            out[c] = out[c].astype("category").cat.codes.astype(float)
        else:
            out[c] = out[c].astype(float)
    return out.fillna(-1).values


_cache = {}


def task_adult_sex(_unused, seed):
    if "adult" not in _cache:
        print("[load] OpenML adult v2", flush=True)
        b = fetch_openml(name="adult", version=2, as_frame=True)
        y = (b.target == ">50K").astype(int).values
        sex_codes = b.data["sex"].astype("category").cat.codes.values
        X = _encode_frame(b.data)
        _cache["adult"] = (X, y, sex_codes)
    X, y, sex = _cache["adult"]
    male_code = int(np.bincount(sex).argmax())
    is_m = (sex == male_code)
    Xm, ym = X[is_m], y[is_m]; Xf, yf = X[~is_m], y[~is_m]
    tr, cal, te = _split_id(Xm, ym, seed)
    splits = {"ID": (Xm[te], ym[te]), "OOD_F": (Xf, yf)}
    return Xm[tr], ym[tr], Xm[cal], ym[cal], splits


def task_bank_age(_unused, seed):
    if "bank" not in _cache:
        print("[load] OpenML bank-marketing v1", flush=True)
        b = fetch_openml(name="bank-marketing", version=1, as_frame=True)
        y = (b.target == "2").astype(int).values  # '2' is yes-subscription
        # V1 is age in the OpenML rename
        age = b.data["V1"].astype(float).values
        X = _encode_frame(b.data)
        _cache["bank"] = (X, y, age)
    X, y, age = _cache["bank"]
    is_young = (age < 35)
    Xy, yy = X[is_young], y[is_young]; Xo, yo = X[~is_young], y[~is_young]
    tr, cal, te = _split_id(Xy, yy, seed)
    splits = {"ID": (Xy[te], yy[te]), "OOD_age_ge_35": (Xo, yo)}
    return Xy[tr], yy[tr], Xy[cal], yy[cal], splits


def task_taiwan_sex(_unused, seed):
    if "taiwan" not in _cache:
        print("[load] OpenML default-of-credit-card-clients (id=42477)", flush=True)
        b = fetch_openml(data_id=42477, as_frame=True)
        y = b.target.astype(int).values
        # column name for sex: 'X2' or 'SEX' depending on version
        cols = list(b.data.columns)
        sex_col = next((c for c in cols if c.lower() in ("x2", "sex")), None)
        if sex_col is None:
            sex_col = cols[1]  # fallback to 2nd col
        sex = b.data[sex_col].astype(float).astype(int).values
        X = _encode_frame(b.data)
        _cache["taiwan"] = (X, y, sex)
    X, y, sex = _cache["taiwan"]
    is_m = (sex == 1)
    Xm, ym = X[is_m], y[is_m]; Xf, yf = X[~is_m], y[~is_m]
    tr, cal, te = _split_id(Xm, ym, seed)
    splits = {"ID": (Xm[te], ym[te]), "OOD_F": (Xf, yf)}
    return Xm[tr], ym[tr], Xm[cal], ym[cal], splits


def task_heloc_gender(_unused, seed):
    if "heloc" not in _cache:
        print("[load] OpenML HELOC (id=43890)", flush=True)
        b = fetch_openml(data_id=43890, as_frame=True)
        # target is 'TRUE'/'FALSE' string
        y = (b.target.astype(str).str.upper() == "TRUE").astype(int).values
        # find gender col
        gcol = next((c for c in b.data.columns if c.lower() in ("gender",)), None)
        g = b.data[gcol] if gcol else b.data.iloc[:, 0]
        g_codes = g.astype("category").cat.codes.values
        X = _encode_frame(b.data)
        _cache["heloc"] = (X, y, g_codes)
    X, y, g = _cache["heloc"]
    maj = int(np.bincount(g).argmax())
    is_maj = (g == maj)
    Xa, ya = X[is_maj], y[is_maj]; Xb, yb = X[~is_maj], y[~is_maj]
    tr, cal, te = _split_id(Xa, ya, seed)
    splits = {"ID": (Xa[te], ya[te]), "OOD_minority_gender": (Xb, yb)}
    return Xa[tr], ya[tr], Xa[cal], ya[cal], splits


TASKS = [
    ("ACSIncome_state",    task_acs_state),
    ("ACSIncome_temporal", task_acs_temporal),
    ("ACSIncome_sex_CA",   task_acs_sex_ca),
    ("ACSIncome_race_CA",  task_acs_race_ca),
    ("Adult_sex",          task_adult_sex),
    ("Bank_age",           task_bank_age),
    ("Taiwan_sex",         task_taiwan_sex),
    ("HELOC_gender",       task_heloc_gender),
]


def get_learner(name, seed):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist",
                             eval_metric="logloss", verbosity=0, n_jobs=4, base_score=0.5,
                             random_state=seed)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, n_jobs=4,
                              random_state=seed)
    raise ValueError(name)


def run_cell(task_name, task_loader, acs_df, learner, cal_name, cp_name, seed,
             X_tr, y_tr, X_cal, y_cal, splits, density_ratios, p_cal_raw, split_probs):
    rng = default_rng(seed * 991 + hash(cp_name + cal_name) % 1000)
    p_full_cal = apply_calibrator(cal_name, p_cal_raw, y_cal, p_cal_raw)
    rows = []
    for split_name, (Xs, ys) in splits.items():
        p_s_raw = split_probs[split_name]
        p_full_s = apply_calibrator(cal_name, p_cal_raw, y_cal, p_s_raw)
        if cp_name == "split":
            inc, _ = split_cp(p_full_cal, y_cal, p_full_s, ALPHA, rng)
        elif cp_name == "weighted":
            inc, _ = weighted_cp_from_weights(
                p_full_cal, y_cal, p_full_s,
                density_ratios[split_name], ALPHA, rng,
            )
        else:  # mondrian
            inc, _ = mondrian_cp(p_full_cal, y_cal, p_full_s, ALPHA, rng)
        s = summarize(inc, ys)
        rows.append({
            "task": task_name, "learner": learner, "cp": cp_name, "calibrator": cal_name,
            "seed": seed, "split": split_name, "n": int(len(ys)),
            "coverage": s["coverage"], "set_size": s["set_size"],
            "cc_cov0": s["cc_coverage"].get(0, float("nan")),
            "cc_cov1": s["cc_coverage"].get(1, float("nan")),
            "ece_biased": ece_uniform_width(p_full_s[:, 1], ys, n_bins=10),
            "ece_debiased": ece_uniform_mass_jackknife(p_full_s[:, 1], ys, n_bins=15),
            "brier": brier(p_full_s[:, 1], ys),
            "accuracy": float((np.argmax(p_full_s, axis=1) == ys).mean()),
        })
    return rows


def main():
    t_start = time.time()
    # Load ACS once
    acs = load_acs_full()
    rows = []
    learners = ["hgb", "xgb", "lgbm"]
    cps = ["split", "weighted", "mondrian"]
    cals = ["uncal", "platt", "isotonic", "temperature"]
    n_task = len(TASKS); n_tot = n_task * len(SEEDS) * len(learners) * len(cals) * len(cps)
    cell_i = 0

    for task_name, task_loader in TASKS:
        for seed in SEEDS:
            t_task = time.time()
            try:
                X_tr, y_tr, X_cal, y_cal, splits = task_loader(acs, seed)
            except Exception as e:
                print(f"[skip] task={task_name} seed={seed}: {type(e).__name__}: {e}", flush=True)
                continue
            # Cache density ratios per (task, seed, split)
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
                    for cp_name in cps:
                        rs = run_cell(task_name, task_loader, acs, learner, cal_name, cp_name, seed,
                                      X_tr, y_tr, X_cal, y_cal, splits,
                                      density_ratios, p_cal_raw, split_probs)
                        rows.extend(rs)
                        cell_i += 1
            print(f"  done task={task_name:20s} seed={seed}  splits={list(splits.keys())}  "
                  f"{cell_i}/{n_tot} cells  task-took {time.time()-t_task:.1f}s  "
                  f"elapsed {time.time()-t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "block1_full.parquet")
    df.to_parquet(out)
    print(f"\nSaved: {out}  ({len(df)} rows; {len(df.task.unique())} tasks; "
          f"{df['split'].nunique()} unique split names)", flush=True)
    print(f"Total wall time: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
