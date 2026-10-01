"""Discrimination, calibration, classification and clinical-utility metrics.

Calibration metrics follow Van Calster et al. (2019) and Huang et al. (2020):
mean calibration (O/E), weak calibration (Cox intercept & slope), moderate
calibration (loess curve, ICI / E50 / E90), binned ECE/MCE (equal-frequency bins,
reported only as a secondary measure), Spiegelhalter's z, Brier score and log loss.
Hosmer-Lemeshow is deliberately not used.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6
DCA_THRESHOLDS = np.round(np.concatenate([np.arange(0.0025, 0.05, 0.0025),
                                          np.arange(0.05, 0.51, 0.025)]), 4)


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def fit_offset_intercept(y, lp):
    """Maximum-likelihood a in logit(P(y=1)) = a + lp, by solving the score equation
    sum(y - expit(a + lp)) = 0. Monotone in a, so a bracketed root always converges
    (an iterative GLM fit can diverge when some predictions are exactly 0 or 1)."""
    from scipy.optimize import brentq
    from scipy.special import expit
    score = lambda a: np.sum(y - expit(a + lp))  # noqa: E731
    lo, hi = -50.0, 50.0
    if score(lo) <= 0 or score(hi) >= 0:  # no events or no non-events
        return float("nan")
    return float(brentq(score, lo, hi, xtol=1e-8))


def calibration_intercept_slope(y, p):
    """Cox (1958) recalibration framework.

    Intercept: logit(P(y=1)) = a + 1*logit(p)  (slope fixed at 1, i.e. calibration-in-the-large)
    Slope:     logit(P(y=1)) = a + b*logit(p)
    Intercept < 0 means risks are overestimated; slope < 1 means risks are too extreme.
    """
    lp = logit(_clip(p))
    a = fit_offset_intercept(y, lp)
    if np.ptp(lp) < 1e-9:  # constant predictions: the slope is undefined
        return a, float("nan")
    return a, logistic_slope(y, lp)


def logistic_slope(y, lp, max_iter=50):
    """Slope b of logit(P(y=1)) = c + b*lp by Newton-Raphson with step halving.
    Same estimate as a binomial GLM, but much faster on large test sets."""
    from scipy.special import expit
    X = np.column_stack([np.ones_like(lp), lp])
    w = np.array([0.0, 1.0])

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z)

    f = nll(w)
    for _ in range(max_iter):
        mu = expit(X @ w)
        grad = X.T @ (mu - y)
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return float("nan")
        t = 1.0
        while t > 1e-8 and nll(w - t * step) > f:
            t /= 2
        w_new = w - t * step
        f_new = nll(w_new)
        if abs(f - f_new) < 1e-10 * (1 + abs(f)):
            w = w_new
            break
        w, f = w_new, f_new
    return float(w[1]) if abs(w[1]) < 100 else float("nan")


def loess_curve(y, p, frac=0.75, max_fit=50_000):
    """Flexible calibration curve: lowess of outcome on predicted risk. Returns (p_sorted, smoothed).

    For test sets larger than `max_fit` the curve is fitted on a fixed random subsample and
    evaluated at every point by interpolation (only the 200k-row simulated test sets need this)."""
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    if len(ps) > max_fit:
        sub = np.sort(np.random.default_rng(0).choice(len(ps), max_fit, replace=False))
        fx, fy = ps[sub], ys[sub]
    else:
        fx, fy = ps, ys
    delta = 0.005 * (fx[-1] - fx[0])  # interpolation shortcut so lowess stays fast
    fitted = sm.nonparametric.lowess(fy, fx, frac=frac, it=0, delta=delta, return_sorted=False)
    if len(fx) < len(ps):
        fitted = np.interp(ps, fx, fitted)
    return ps, np.clip(fitted, 0, 1)


def ici_e50_e90(y, p):
    """Integrated Calibration Index (Austin & Steyerberg, 2019) and the median / 90th percentile error."""
    ps, smooth = loess_curve(y, p)
    err = np.abs(smooth - ps)
    return float(err.mean()), float(np.median(err)), float(np.quantile(err, 0.9)), (ps, smooth)


def ece_mce(y, p, n_bins=10):
    """Equal-frequency binned ECE/MCE, comparing mean observed vs mean predicted in each bin."""
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b").agg(o=("y", "mean"), e=("p", "mean"), n=("y", "size"))
    gap = (df.o - df.e).abs()
    return float((gap * df.n).sum() / len(y)), float(gap.max())


def spiegelhalter_z(y, p):
    p = _clip(p)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    return float(num / den)


def net_benefit(y, p, t):
    """Net Benefit (Vickers & Elkin, 2006) of treating everyone with predicted risk >= t."""
    n = len(y)
    pred = p >= t
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def classification(y, p, t):
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    return sens, spec, ppv, mcc


def evaluate(y, p, prevalence_threshold, p_true=None):
    """All scalar metrics for one set of predictions, plus curve data for plotting."""
    p = _clip(np.asarray(p, dtype=float))
    y = np.asarray(y)
    a, b = calibration_intercept_slope(y, p)
    ici, e50, e90, (ps, smooth) = ici_e50_e90(y, p)
    ece, mce = ece_mce(y, p)
    out = {
        "n_test": len(y), "events_test": int(y.sum()),
        "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
        "mean_pred": p.mean(), "obs_rate": y.mean(), "oe_ratio": y.mean() / p.mean(),
        "cal_intercept": a, "cal_slope": b, "ici": ici, "e50": e50, "e90": e90,
        "ece": ece, "mce": mce, "spiegelhalter_z": spiegelhalter_z(y, p),
    }
    # Scaled Brier: 1 - Brier / Brier of a model that always predicts the prevalence.
    out["scaled_brier"] = 1 - out["brier"] / (y.mean() * (1 - y.mean()))
    # Absolute ICI shrinks with prevalence (all risks are small), so also report it relative
    # to the event rate to keep event fractions comparable.
    out["ici_rel"] = ici / y.mean()
    if p_true is not None:
        # Simulation only: distance from each patient's true risk (strong-calibration check).
        out["mae_true"] = float(np.mean(np.abs(p - p_true)))
        out["mae_true_rel"] = out["mae_true"] / float(np.mean(p_true))
    for name, t in [("t050", 0.5), ("tprev", prevalence_threshold)]:
        s, sp, ppv, mcc = classification(y, p, t)
        out.update({f"sens_{name}": s, f"spec_{name}": sp, f"ppv_{name}": ppv, f"mcc_{name}": mcc})

    # Curves: calibration curve on 40 quantiles of predicted risk; decision curve on a fixed grid.
    q = np.quantile(ps, np.linspace(0.01, 0.99, 40))
    cal_curve = np.interp(q, ps, smooth)
    nb = np.array([net_benefit(y, p, t) for t in DCA_THRESHOLDS])
    curves = {"cal_x": q, "cal_y": cal_curve, "dca_t": DCA_THRESHOLDS, "dca_nb": nb}
    return out, curves
