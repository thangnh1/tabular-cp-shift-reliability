"""
Post-hoc calibrators with a uniform fit/transform interface.

Each calibrator takes:
- p_cal:    raw probability of the positive class on calibration set (binary)
- y_cal:    true labels on calibration set
- p_target: raw probability of the positive class on target

Returns calibrated probabilities for the binary positive class. The
calling code reconstructs (n, 2) probabilities as np.stack([1-p, p], axis=1).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar


def uncalibrated(p_cal, y_cal, p_target):
    return p_target


def platt(p_cal, y_cal, p_target):
    """Platt scaling on logits."""
    logit_cal = np.log(np.clip(p_cal, 1e-7, 1 - 1e-7) / np.clip(1 - p_cal, 1e-7, 1 - 1e-7))
    logit_target = np.log(np.clip(p_target, 1e-7, 1 - 1e-7) / np.clip(1 - p_target, 1e-7, 1 - 1e-7))
    lr = LogisticRegression().fit(logit_cal.reshape(-1, 1), y_cal)
    return lr.predict_proba(logit_target.reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_target):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    return iso.transform(p_target)


def temperature(p_cal, y_cal, p_target):
    """Temperature scaling on logits."""
    logit_cal = np.log(np.clip(p_cal, 1e-7, 1 - 1e-7) / np.clip(1 - p_cal, 1e-7, 1 - 1e-7))
    logit_target = np.log(np.clip(p_target, 1e-7, 1 - 1e-7) / np.clip(1 - p_target, 1e-7, 1 - 1e-7))

    def nll(T):
        T = max(T, 1e-3)
        p = 1.0 / (1.0 + np.exp(-(logit_cal / T)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p)).mean()

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T = res.x
    return 1.0 / (1.0 + np.exp(-(logit_target / T)))


CALIBRATORS = {
    "uncal": uncalibrated,
    "platt": platt,
    "isotonic": isotonic,
    "temperature": temperature,
}


def apply_calibrator(name, p_cal, y_cal, p_target):
    """Return calibrated (n_target, 2) probability matrix."""
    fn = CALIBRATORS[name]
    p_pos = fn(p_cal, y_cal, p_target)
    p_pos = np.clip(p_pos, 1e-7, 1 - 1e-7)
    return np.stack([1 - p_pos, p_pos], axis=1)


def ece_uniform_width(p, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(p)
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (p >= edges[b]) & (p < edges[b + 1])
        else:
            mask = (p >= edges[b]) & (p <= edges[b + 1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        e += (mask.sum() / n) * abs(conf - acc)
    return float(e)


def ece_uniform_mass_jackknife(p, y, n_bins=15):
    n = len(p)
    if n < n_bins * 4:
        n_bins = max(2, n // 4)
    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    bin_idx = np.array_split(np.arange(n), n_bins)
    per_bin = []
    ece_full = 0.0
    for idx in bin_idx:
        conf = p_s[idx].mean(); acc = y_s[idx].mean()
        per_bin.append((len(idx), conf, acc, abs(conf - acc)))
        ece_full += (len(idx) / n) * abs(conf - acc)
    jk = []
    for k in range(n_bins):
        m = n - per_bin[k][0]
        e = sum((per_bin[j][0] / m) * per_bin[j][3] for j in range(n_bins) if j != k)
        jk.append(e)
    bias = (n_bins - 1) * (np.mean(jk) - ece_full)
    return float(max(0.0, ece_full - bias))


def brier(p, y):
    return float(np.mean((p - y) ** 2))
