"""
Three additional binary-classification tabular tasks with defensible OOD splits.

All loaders return the standard signature used by `block1_full.py`:
    (X_tr, y_tr, X_cal, y_cal, splits_dict)
where splits_dict maps split_name -> (X, y).

Tasks:
- `Diabetes_race`:       Diabetes 130-US (OpenML id=4541). y=<30-day readmission.
                          ID = Caucasian; OOD = AfricanAmerican.
- `SpeedDating_race`:    OpenML id=40536. y=match decision.
                          ID = European/Caucasian-American; OOD = Asian/Pacific Islander.
- `OnlineNews_channel`:  OpenML id=4545. y=(shares > 1400).
                          ID = data_channel_is_tech; OOD = data_channel_is_entertainment.

For each task we DROP the grouping column(s) from the feature matrix to avoid
trivial leakage (the model would otherwise just look at the OOD-defining feature).
Encoded categoricals -> integer codes (good enough for tree learners), NaN -> -1.
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import fetch_openml

_cache = {}


def _encode_frame(X):
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("object", "category"):
            out[c] = out[c].astype("category").cat.codes.astype(float)
        else:
            out[c] = out[c].astype(float)
    return out.fillna(-1).values


def _split_id(X_id, y_id, seed):
    rng = default_rng(seed)
    n = len(X_id)
    idx = rng.permutation(n)
    n_tr = int(0.5 * n); n_cal = int(0.25 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]


def task_diabetes_race(acs_unused, seed):
    if "diabetes" not in _cache:
        print("[load] OpenML Diabetes 130-US (id=4541)", flush=True)
        b = fetch_openml(data_id=4541, as_frame=True)
        X = b.data.copy()
        y = (b.target == "<30").astype(int).values
        # Identifiers + grouping column: drop entirely
        drop_cols = ["encounter_id", "patient_nbr", "race"]
        race = X["race"].astype(str).values
        X_drop = X.drop(columns=[c for c in drop_cols if c in X.columns])
        X_arr = _encode_frame(X_drop)
        _cache["diabetes"] = (X_arr, y, race)
    X, y, race = _cache["diabetes"]
    is_id = (race == "Caucasian")
    is_ood = (race == "AfricanAmerican")
    Xid, yid = X[is_id], y[is_id]
    Xood, yood = X[is_ood], y[is_ood]
    tr, cal, te = _split_id(Xid, yid, seed)
    splits = {"ID": (Xid[te], yid[te]), "OOD_African": (Xood, yood)}
    return Xid[tr], yid[tr], Xid[cal], yid[cal], splits


def task_speeddating_race(acs_unused, seed):
    if "speeddating" not in _cache:
        print("[load] OpenML SpeedDating (id=40536)", flush=True)
        b = fetch_openml(data_id=40536, as_frame=True)
        X = b.data.copy()
        y = b.target.astype(int).values
        race = X["race"].astype(str).values
        # Drop grouping columns (race + race_o + samerace), which would trivially
        # reveal which subset a row belongs to:
        drop_cols = [c for c in ["race", "race_o", "samerace", "d_race", "d_race_o"] if c in X.columns]
        X_drop = X.drop(columns=drop_cols)
        X_arr = _encode_frame(X_drop)
        _cache["speeddating"] = (X_arr, y, race)
    X, y, race = _cache["speeddating"]
    is_id = (race == "European/Caucasian-American")
    is_ood = (race == "Asian/Pacific Islander/Asian-American")
    Xid, yid = X[is_id], y[is_id]
    Xood, yood = X[is_ood], y[is_ood]
    tr, cal, te = _split_id(Xid, yid, seed)
    splits = {"ID": (Xid[te], yid[te]), "OOD_Asian": (Xood, yood)}
    return Xid[tr], yid[tr], Xid[cal], yid[cal], splits


def task_onlinenews_channel(acs_unused, seed):
    if "onlinenews" not in _cache:
        print("[load] OpenML Online News Popularity (id=4545)", flush=True)
        b = fetch_openml(data_id=4545, as_frame=True)
        X = b.data.copy()
        y = (b.target > 1400).astype(int).values
        chan_tech = (X["data_channel_is_tech"] > 0.5).values
        chan_ent  = (X["data_channel_is_entertainment"] > 0.5).values
        # Drop all data_channel_is_* and url, timedelta to prevent trivial leakage
        # (timedelta is days since article publication — could correlate with shares-evolution-over-time).
        chan_cols = [c for c in X.columns if c.startswith("data_channel_is_")]
        other_drop = [c for c in ["url", "timedelta"] if c in X.columns]
        drop_cols = chan_cols + other_drop
        X_drop = X.drop(columns=drop_cols)
        X_arr = _encode_frame(X_drop)
        _cache["onlinenews"] = (X_arr, y, chan_tech, chan_ent)
    X, y, chan_tech, chan_ent = _cache["onlinenews"]
    Xid, yid = X[chan_tech], y[chan_tech]
    Xood, yood = X[chan_ent],  y[chan_ent]
    tr, cal, te = _split_id(Xid, yid, seed)
    splits = {"ID": (Xid[te], yid[te]), "OOD_entertainment": (Xood, yood)}
    return Xid[tr], yid[tr], Xid[cal], yid[cal], splits
