"""
Conformal Prediction variants with APS (Romano-Sesia-Candes 2020) nonconformity.

All variants take pre-trained predict_proba on:
- (p_cal, y_cal): calibration features, true labels
- p_te: test features (or evaluated probabilities)

Returns prediction sets as boolean (n_test, n_classes) array, plus per-test
coverage and set size summary.

Implementations:
- split_cp           : standard split conformal (Vovk et al.)
- weighted_cp        : Tibshirani et al. 2019; covariate-shift correction via
                       density-ratio reweighting of calibration scores
- mondrian_cp        : class-conditional; compute a per-class quantile from
                       calibration data (group by true class), apply at test
                       time via predicted class group

LAC (1-p_y) variants are NOT included; APS only -- see exp:003 wiki page for
the design lesson that LAC collapses to trivial size-2 sets on binary tasks.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression


def aps_score(p_full, y, rng):
    """Randomized APS score for true label y. p_full shape (n, K)."""
    n, K = p_full.shape
    order = np.argsort(-p_full, axis=1)
    sorted_p = np.take_along_axis(p_full, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    above = np.zeros(n)
    p_y = p_full[np.arange(n), y]
    for i in range(n):
        r = int(np.where(order[i] == y[i])[0][0])
        if r > 0:
            above[i] = cumsum[i, r - 1]
    u = rng.uniform(0, 1, size=n)
    return above + u * p_y


def aps_inclusion(p_full, q, rng):
    """Vectorized APS inclusion at threshold q. p_full (n, K). Returns (n, K) bool."""
    n, K = p_full.shape
    order = np.argsort(-p_full, axis=1)
    sorted_p = np.take_along_axis(p_full, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    above = np.concatenate([np.zeros((n, 1)), cumsum[:, :-1]], axis=1)
    u = rng.uniform(0, 1, size=(n, K))
    scores_sorted = above + u * sorted_p
    inc_sorted = scores_sorted <= q
    inc = np.zeros((n, K), dtype=bool)
    for i in range(n):
        inc[i, order[i]] = inc_sorted[i]
    return inc


def aps_inclusion_per_class_q(p_full, q_per_class, rng):
    """APS inclusion where each candidate class c uses its own threshold q[c].
    For each row i and candidate class c, include c iff
        aps_score_for_class_c(x_i) <= q_per_class[c]
    """
    n, K = p_full.shape
    order = np.argsort(-p_full, axis=1)
    sorted_p = np.take_along_axis(p_full, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    above = np.concatenate([np.zeros((n, 1)), cumsum[:, :-1]], axis=1)
    u = rng.uniform(0, 1, size=(n, K))
    scores_sorted = above + u * sorted_p  # (n, K) in rank order
    # We need scores_per_class[i, c] = score of class c at row i
    inc = np.zeros((n, K), dtype=bool)
    # Inverse permutation: for each row, rank_of[c] tells us where class c ranks
    rank_of = np.argsort(order, axis=1)  # (n, K)
    for c in range(K):
        scores_c = scores_sorted[np.arange(n), rank_of[:, c]]
        inc[:, c] = scores_c <= q_per_class[c]
    return inc


def split_cp(p_cal_full, y_cal, p_te_full, alpha, rng):
    """Standard split CP with APS score. Marginal coverage guarantee.
    Returns: inclusion bool (n_te, K), quantile q (float)."""
    s_cal = aps_score(p_cal_full, y_cal, rng)
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = float(np.sort(s_cal)[min(k - 1, n - 1)])
    inc = aps_inclusion(p_te_full, q, rng)
    return inc, q


def estimate_likelihood_ratio(X_cal, X_te, max_te_subsample=5000, seed=0):
    """Density-ratio w(x) = P_test(x)/P_cal(x) via logistic regression on
    {(X_cal, label=0), (X_te_sub, label=1)}; w_hat = P(label=1|x) / (1 - P(label=1|x)).
    For large X_te we subsample to max_te_subsample rows -- LR converges quickly
    with a few thousand examples per class. Returns weights on calibration rows."""
    rng = np.random.default_rng(seed)
    n_te = len(X_te)
    if n_te > max_te_subsample:
        idx = rng.choice(n_te, size=max_te_subsample, replace=False)
        X_te_sub = X_te[idx]
    else:
        X_te_sub = X_te
    n_cal = len(X_cal)
    if n_cal > max_te_subsample:
        idx = rng.choice(n_cal, size=max_te_subsample, replace=False)
        X_cal_sub = X_cal[idx]
    else:
        X_cal_sub = X_cal
    Xmix = np.vstack([X_cal_sub, X_te_sub])
    ymix = np.array([0] * len(X_cal_sub) + [1] * len(X_te_sub))
    clf = LogisticRegression(max_iter=200, solver="liblinear").fit(Xmix, ymix)
    # apply to all calibration rows (not the subsample)
    p_shift_cal = clf.predict_proba(X_cal)[:, 1]
    p_shift_cal = np.clip(p_shift_cal, 1e-6, 1 - 1e-6)
    return p_shift_cal / (1 - p_shift_cal)


def weighted_cp_from_weights(p_cal_full, y_cal, p_te_full, w_cal, alpha, rng):
    """Weighted split CP given a pre-computed density-ratio w_cal on calibration
    rows. Caller supplies w_cal so the LR fit is not repeated across
    (calibrator, cp_variant) combos."""
    s_cal = aps_score(p_cal_full, y_cal, rng)
    s = np.concatenate([s_cal, [np.inf]])
    w = np.concatenate([w_cal, [1.0]])
    w = w / w.sum()
    order = np.argsort(s)
    s_s = s[order]; w_s = w[order]
    cum = np.cumsum(w_s)
    idx = int(np.searchsorted(cum, 1 - alpha))
    q = float(s_s[min(idx, len(s_s) - 1)])
    if not np.isfinite(q):
        finite = s_cal[np.isfinite(s_cal)]
        q = float(finite.max() if len(finite) else 1.0)
    inc = aps_inclusion(p_te_full, q, rng)
    return inc, q


def weighted_cp(p_cal_full, y_cal, X_cal, p_te_full, X_te, alpha, rng):
    """Convenience wrapper that estimates the density ratio + applies weighted CP."""
    w_cal = estimate_likelihood_ratio(X_cal, X_te)
    return weighted_cp_from_weights(p_cal_full, y_cal, p_te_full, w_cal, alpha, rng)


def mondrian_cp(p_cal_full, y_cal, p_te_full, alpha, rng):
    """Class-conditional Mondrian split CP with APS.
    Compute per-class q_y on the calibration set where y_cal == y, then for
    each test point and each candidate class c, include c iff
        aps_score_for_class_c(x_test) <= q_c
    This guarantees class-conditional coverage on the calibration distribution."""
    s_cal_full = aps_score(p_cal_full, y_cal, rng)
    K = p_cal_full.shape[1]
    q_per_class = np.zeros(K)
    for c in range(K):
        mask = (y_cal == c)
        if mask.sum() == 0:
            q_per_class[c] = 1.0
            continue
        s_c = s_cal_full[mask]
        n_c = len(s_c)
        k = int(np.ceil((n_c + 1) * (1 - alpha)))
        q_per_class[c] = float(np.sort(s_c)[min(k - 1, n_c - 1)])
    inc = aps_inclusion_per_class_q(p_te_full, q_per_class, rng)
    return inc, q_per_class


def summarize(inc, y):
    """Coverage, set size, class-conditional coverage."""
    K = inc.shape[1]
    covered = inc[np.arange(len(y)), y]
    set_size = inc.sum(axis=1)
    cc = {}
    for c in range(K):
        mask = (y == c)
        if mask.sum() == 0:
            cc[c] = float("nan")
        else:
            cc[c] = float(covered[mask].mean())
    return {
        "coverage": float(covered.mean()),
        "set_size": float(set_size.mean()),
        "cc_coverage": cc,
    }
