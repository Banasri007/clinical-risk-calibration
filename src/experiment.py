"""Factorial experiment: event fraction x imbalance correction x classifier x recalibration.

For every repeat and event fraction:
  1. subsample events to reach the target event fraction (all non-events kept)
  2. stratified split into three equal partitions: train | calibration | test
  3. fit preprocessing on train only
  4. apply the imbalance correction to train only
  5. fit the classifier, predict calibration and test partitions
  6. fit each recalibration map on the calibration partition, apply to test
  7. evaluate on the untouched test partition (which keeps the true event fraction)

With --dataset sim, partitions are drawn independently from a known logistic mechanism
(train 50k, calibration 50k, test 200k) so that event fractions down to 0.1% still leave about
200 test events, and predictions can also be scored against the true risk.

Usage:
  python src/experiment.py --repeats 1 --fractions natural 0.01 --out results/pilot
  python src/experiment.py --repeats 5 --out results/main
  python src/experiment.py --dataset sim --repeats 5 --out results/sim
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # avoids a noisy core-count probe on Windows
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cohort, make_preprocessor, sim_draw, sim_mechanism, subsample_to_prevalence  # noqa: E402
from metrics import evaluate  # noqa: E402
from recalibration import FITTED, prior_correction  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SVM_MAX_TRAIN = 3000  # RBF-SVM is O(n^2); its training set is stratified-subsampled to this size

CORRECTIONS = ["none", "rus", "ros", "smote", "adasyn", "weighted"]


SIM_SIZES = (50_000, 50_000, 200_000)  # train, calibration, test


def make_classifier(name, seed, weighted, continuous=False):
    cw = "balanced" if weighted else None
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight=cw)
    if name == "nb" and continuous:
        return GaussianNB()
    if name == "nb":
        # Features are mostly one-hot; standardized numerics are binarized at their mean.
        return BernoulliNB(alpha=1.0, binarize=0.0)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=50, n_jobs=-1)
    if name == "cart":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight=cw, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight=cw, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                      class_weight=cw, n_jobs=-1, random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
                             eval_metric="logloss")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                             max_iter=300, random_state=seed)
    raise ValueError(name)


# Classifiers with no native class_weight: weighting is done through sample_weight (xgb)
# or is unavailable (nb, knn, mlp -> the "weighted" cell is skipped and reported as missing).
SAMPLE_WEIGHT_OK = {"xgb"}
CLASS_WEIGHT_OK = {"lr", "cart", "svm", "rf"}


def resample(X, y, method, seed):
    if method in ("none", "weighted"):
        return X, y
    sampler = {
        "rus": RandomUnderSampler(random_state=seed),
        "ros": RandomOverSampler(random_state=seed),
        "smote": SMOTE(k_neighbors=5, random_state=seed),
        "adasyn": ADASYN(n_neighbors=5, random_state=seed),
    }[method]
    return sampler.fit_resample(X, y)


def prepare_real(X_all, y_all, ef, rng, seed):
    """Subsample events, split 1/3 : 1/3 : 1/3 (stratified), fit preprocessing on train only."""
    idx = subsample_to_prevalence(y_all, ef, rng)
    X, y = X_all.iloc[idx], y_all[idx]
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=1 / 3, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, train_size=0.5, stratify=y_rest, random_state=seed)
    pre = make_preprocessor(X_tr).fit(X_tr)
    return pre.transform(X_tr), y_tr, pre.transform(X_cal), y_cal, pre.transform(X_te), y_te, None


def prepare_sim(ef, rng):
    """Draw independent train / calibration / test sets from the known mechanism."""
    b0, beta = sim_mechanism(ef, rng)
    (X_tr, y_tr, _), (X_cal, y_cal, _), (X_te, y_te, p_te) = (sim_draw(n, b0, beta, rng) for n in SIM_SIZES)
    pre = StandardScaler().fit(X_tr)
    return pre.transform(X_tr), y_tr, pre.transform(X_cal), y_cal, pre.transform(X_te), y_te, p_te


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sim = args.dataset == "sim"
    if not sim:
        X_all, y_all = load_cohort()
        natural = y_all.mean()
        print(f"cohort: n={len(y_all)}, events={y_all.sum()}, prevalence={natural:.4f}", flush=True)

    rows, curve_rows = [], []
    for rep in range(args.rep_offset, args.rep_offset + args.repeats):
        for frac in args.fractions:
            ef = natural if frac == "natural" else float(frac)
            rng = np.random.default_rng(1000 * rep + int(ef * 1e5) + (7 if sim else 0))
            seed = int(rng.integers(1e9))
            if sim:
                A_tr, y_tr, A_cal, y_cal, A_te, y_te, p_true = prepare_sim(ef, rng)
            else:
                A_tr, y_tr, A_cal, y_cal, A_te, y_te, p_true = prepare_real(X_all, y_all, ef, rng, seed)
            train_prev = y_tr.mean()
            print(f"[rep {rep}] EF={ef:.4f}: train {len(y_tr)} ({y_tr.sum()} ev), "
                  f"cal {len(y_cal)} ({y_cal.sum()} ev), test {len(y_te)} ({y_te.sum()} ev), "
                  f"{A_tr.shape[1]} features", flush=True)

            # RBF-SVM gets a stratified subsample of the *original* training partition, taken before
            # any correction, so every correction condition sees the same events.
            if len(y_tr) > SVM_MAX_TRAIN:
                A_svm, _, y_svm, _ = train_test_split(A_tr, y_tr, train_size=SVM_MAX_TRAIN,
                                                      stratify=y_tr, random_state=seed)
            else:
                A_svm, y_svm = A_tr, y_tr

            for corr in args.corrections:
                try:
                    A_r, y_r = resample(A_tr, y_tr, corr, seed)
                    if "svm" in args.classifiers:
                        A_rs, y_rs = resample(A_svm, y_svm, corr, seed)
                except (ValueError, RuntimeError) as e:  # e.g. ADASYN finding no neighbours
                    print(f"   skip {corr}: {e}", flush=True)
                    continue
                resampled_prev = 0.5 if corr == "weighted" else y_r.mean()

                for clf_name in args.classifiers:
                    weighted = corr == "weighted"
                    if weighted and clf_name not in CLASS_WEIGHT_OK | SAMPLE_WEIGHT_OK:
                        continue
                    t0 = time.time()
                    clf = make_classifier(clf_name, seed, weighted and clf_name in CLASS_WEIGHT_OK, continuous=sim)
                    A_fit, y_fit = (A_rs, y_rs) if clf_name == "svm" else (A_r, y_r)
                    fit_kw = {}
                    if weighted and clf_name in SAMPLE_WEIGHT_OK:
                        w = np.where(y_fit == 1, 0.5 / y_fit.mean(), 0.5 / (1 - y_fit.mean()))
                        fit_kw["sample_weight"] = w
                    clf.fit(A_fit, y_fit, **fit_kw)
                    p_cal = clf.predict_proba(A_cal)[:, 1]
                    p_te = clf.predict_proba(A_te)[:, 1]
                    fit_s = time.time() - t0

                    preds = {"none": p_te}
                    if corr != "none":
                        preds["prior"] = prior_correction(p_te, train_prev, resampled_prev)
                    for name, cls in FITTED.items():
                        try:
                            preds[name] = cls().fit(p_cal, y_cal).predict(p_te)
                        except Exception as e:  # noqa: BLE001 - record and continue
                            print(f"   {clf_name}/{corr}/{name} failed: {e}", flush=True)

                    for recal, p in preds.items():
                        m, curves = evaluate(y_te, p, prevalence_threshold=train_prev, p_true=p_true)
                        key = dict(dataset=args.dataset, rep=rep, event_fraction=round(ef, 4), correction=corr,
                                   classifier=clf_name, recalibration=recal)
                        rows.append({**key, **m, "train_prev": train_prev,
                                     "resampled_prev": resampled_prev, "fit_seconds": fit_s})
                        curve_rows.append({**key, **{k: json.dumps(np.round(v, 5).tolist()) for k, v in curves.items()}})
                    r0 = rows[-len(preds)]
                    print(f"   {corr:8s} {clf_name:5s} AUROC={r0['auroc']:.3f} "
                          f"int={r0['cal_intercept']:+.2f} slope={r0['cal_slope']:.2f} "
                          f"ICI={r0['ici']:.3f} ({fit_s:.0f}s)", flush=True)

            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            pd.DataFrame(curve_rows).to_csv(out / "curves.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0, help="first repeat index (to split runs across processes)")
    ap.add_argument("--dataset", choices=["real", "sim"], default="real")
    ap.add_argument("--fractions", nargs="+", default=None)
    ap.add_argument("--corrections", nargs="+", default=CORRECTIONS)
    ap.add_argument("--classifiers", nargs="+", default=None)
    ap.add_argument("--out", default="results/main")
    a = ap.parse_args()
    if a.fractions is None:
        a.fractions = (["0.02", "0.01", "0.005", "0.002", "0.001"] if a.dataset == "sim"
                       else ["natural", "0.05", "0.02", "0.01", "0.005"])
    if a.classifiers is None:
        # Left out of the simulation: SVM (its 3,000-row training cap would leave 3-60 events) and
        # k-NN (k = 50 neighbours rarely contain an event below 1%, and querying 250k rows is slow).
        a.classifiers = (["lr", "nb", "cart", "rf", "xgb", "mlp"] if a.dataset == "sim"
                         else ["lr", "nb", "knn", "cart", "svm", "rf", "xgb", "mlp"])
    run(a)
