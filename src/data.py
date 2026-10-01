"""Cohort construction and preprocessing for the Diabetes 130-US Hospitals dataset.

Cohort rules (following Strack et al., 2014):
  * drop encounters discharged to hospice or expired (no opportunity for readmission)
  * keep the first encounter per patient, so no patient appears in more than one partition
Outcome: readmission within 30 days ("<30").
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HOSPICE_OR_EXPIRED = [11, 13, 14, 19, 20, 21]
NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
DROP = ["encounter_id", "patient_nbr", "weight", "payer_code", "readmitted",
        "examide", "citoglipton"]  # weight/payer >40% missing; two drugs are constant


def icd9_group(code):
    """Map an ICD-9 code to the diagnosis groups used by Strack et al. (2014)."""
    if pd.isna(code) or code == "?":
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


UCI_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def ensure_data():
    """Download and unpack the UCI dataset (CC BY 4.0) into data/ if it is not there yet."""
    csv = DATA_DIR / "diabetic_data.csv"
    if not csv.exists():
        import io
        import urllib.request
        import zipfile
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {UCI_URL}", flush=True)
        with urllib.request.urlopen(UCI_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    return csv


def load_cohort():
    """Return (X dataframe, y array) for the patient-level cohort."""
    df = pd.read_csv(ensure_data(), na_values="?", low_memory=False)
    df = df[~df.discharge_disposition_id.isin(HOSPICE_OR_EXPIRED)]
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    y = (df.readmitted == "<30").astype(int).to_numpy()

    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].astype("string").map(icd9_group)
    top_spec = df.medical_specialty.value_counts().index[:10]
    df["medical_specialty"] = df.medical_specialty.where(df.medical_specialty.isin(top_spec), "Other")

    X = df.drop(columns=DROP)
    categorical = [c for c in X.columns if c not in NUMERIC]
    X[categorical] = X[categorical].astype("string").fillna("Missing")
    return X.reset_index(drop=True), y


def make_preprocessor(X):
    """Preprocessing is always fitted on the training partition only."""
    categorical = [c for c in X.columns if c not in NUMERIC]
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False), categorical),
    ])


def subsample_to_prevalence(y, event_fraction, rng):
    """Indices of a cohort whose event fraction is `event_fraction`.

    All non-events are kept and events are randomly subsampled. If the requested
    fraction is at or above the natural prevalence, the full cohort is returned.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = int(round(event_fraction * len(neg) / (1 - event_fraction)))
    if n_pos >= len(pos):
        return np.arange(len(y))
    keep_pos = rng.choice(pos, size=n_pos, replace=False)
    return np.sort(np.concatenate([neg, keep_pos]))


SIM_FEATURES = 10


def sim_mechanism(event_fraction, rng, n_features=SIM_FEATURES, target_auc=0.75):
    """Logistic data-generating mechanism in the style of van den Goorbergh et al. (2022).

    Independent standard-normal predictors with equal coefficients. The coefficient size is
    solved so the true risk has AUROC = `target_auc`, and the intercept so the marginal event
    fraction equals `event_fraction`. Returns (intercept, coefficients).
    """
    from scipy.optimize import brentq
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score

    Z = rng.standard_normal(400_000)  # the linear predictor is N(0, s^2), so one draw suffices

    def intercept(s):
        return brentq(lambda b: expit(b + s * Z).mean() - event_fraction, -30, 10)

    def auc(s):
        p = expit(intercept(s) + s * Z)
        y = rng.binomial(1, p)
        return roc_auc_score(y, p)

    s = brentq(lambda s: auc(s) - target_auc, 0.2, 3.0, xtol=1e-3)
    return intercept(s), np.full(n_features, s / np.sqrt(n_features))


def sim_draw(n, b0, beta, rng):
    """Draw n patients from the mechanism. Returns (X dataframe, y, true risk)."""
    from scipy.special import expit

    X = rng.standard_normal((n, len(beta)))
    p = expit(b0 + X @ beta)
    y = rng.binomial(1, p)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(len(beta))]), y, p
