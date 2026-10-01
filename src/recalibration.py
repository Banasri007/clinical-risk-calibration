"""Post-hoc recalibration maps.

Every map except `prior` is fitted on the dedicated calibration partition and never on
the classifier's own training predictions (Silva Filho et al., 2023; Huang et al., 2020).
"""
import numpy as np
from betacal import BetaCalibration
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _lp(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def prior_correction(p, train_prev, resampled_prev):
    """Analytic plug-in correction for a known change in training event fraction.

    Rescales the odds by [pi/(1-pi)] / [pi_s/(1-pi_s)] (Elkan 2001; Piccininni et al., 2024).
    Requires no calibration data. Exact for random under/oversampling; only approximate for
    SMOTE/ADASYN, which also change the predictor distribution of the minority class.
    """
    r = (train_prev / (1 - train_prev)) / (resampled_prev / (1 - resampled_prev))
    odds = np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))
    return odds * r / (1 + odds * r)


class InterceptOnly:
    """logit(p*) = a + logit(p): re-estimates calibration-in-the-large only (as tested by
    van den Goorbergh et al. 2022 and Carriero et al. 2025)."""

    def fit(self, p, y):
        from metrics import fit_offset_intercept
        self.a = fit_offset_intercept(np.asarray(y), _lp(p))
        return self

    def predict(self, p):
        return expit(self.a + _lp(p))


class Platt:
    """logit(p*) = a + b*logit(p): logistic (Platt-type) recalibration on the logit scale,
    i.e. intercept and slope. Can only produce sigmoidal maps."""

    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_lp(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.m.predict_proba(_lp(p).reshape(-1, 1))[:, 1]


class Isotonic:
    """Monotone step function (pool-adjacent-violators). Output is clipped away from 0/1."""

    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self.m.predict(p), 1e-4, 1 - 1e-4)


class Beta:
    """Beta calibration (Kull et al., 2017): sigmoidal, inverse-sigmoidal or identity maps."""

    def fit(self, p, y):
        self.m = BetaCalibration(parameters="abm").fit(np.clip(p, EPS, 1 - EPS).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.m.predict(np.clip(p, EPS, 1 - EPS).reshape(-1, 1))


FITTED = {"intercept": InterceptOnly, "platt": Platt, "isotonic": Isotonic, "beta": Beta}
