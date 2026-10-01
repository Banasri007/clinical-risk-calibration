"""Summary tables, hypothesis checks and figures from experiment results.

Usage: python src/analyze.py results/real_a results/real_b results/sim --out report
Writes report/real/ and report/sim/. Works on partial results (whatever repeats have finished).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CORR_ORDER = ["none", "rus", "ros", "smote", "adasyn", "weighted"]
CORR_LABEL = {"none": "Uncorrected", "rus": "RUS", "ros": "ROS", "smote": "SMOTE",
              "adasyn": "ADASYN", "weighted": "Class weighting"}
# Validated categorical order (slots 1-6), fixed per entity; markers give a second encoding.
CORR_COLOR = dict(zip(CORR_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CORR_MARKER = dict(zip(CORR_ORDER, ["o", "v", "^", "s", "D", "P"]))
CLF_ORDER = ["lr", "nb", "knn", "cart", "svm", "rf", "xgb", "mlp"]
CLF_LABEL = {"lr": "Logistic regression", "nb": "Naïve Bayes", "knn": "k-NN", "cart": "Decision tree",
             "svm": "SVM (RBF)", "rf": "Random forest", "xgb": "XGBoost", "mlp": "MLP"}
RECAL_ORDER = ["none", "prior", "intercept", "platt", "isotonic", "beta"]
RECAL_LABEL = {"none": "None", "prior": "Prior correction", "intercept": "Intercept-only",
               "platt": "Platt", "isotonic": "Isotonic", "beta": "Beta"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(dirs):
    res = pd.concat([pd.read_csv(Path(d) / "results.csv") for d in dirs], ignore_index=True)
    cur = pd.concat([pd.read_csv(Path(d) / "curves.csv") for d in dirs], ignore_index=True)
    return res, cur


def mean_sd(df, keys, col):
    g = df.groupby(keys)[col].agg(["mean", "std", "count"]).reset_index()
    g["fmt"] = g["mean"].round(2).astype(str) + " ± " + g["std"].fillna(0).round(2).astype(str)
    return g


def fig_intercept_vs_fraction(res, out):
    """H1: calibration intercept (no recalibration) by event fraction, one panel per classifier."""
    d = res[res.recalibration == "none"]
    clfs = [c for c in CLF_ORDER if c in d.classifier.unique()]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, clf in zip(axes.flat, clfs):
        for corr in CORR_ORDER:
            g = d[(d.classifier == clf) & (d.correction == corr)].groupby("event_fraction").cal_intercept.mean()
            if len(g):
                ax.plot(g.index * 100, g.values, color=CORR_COLOR[corr], marker=CORR_MARKER[corr],
                        markersize=5, linewidth=2, label=CORR_LABEL[corr])
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xscale("log"); ax.set_title(CLF_LABEL[clf], fontsize=9, color=INK)
        ticks = sorted(d.event_fraction.unique() * 100)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    for ax in axes.flat[len(clfs):]:
        ax.axis("off")
    fig.supxlabel("Event fraction (%), log scale"); fig.supylabel("Calibration intercept (0 = ideal; < 0 = overestimation)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(out / "fig1_intercept_vs_event_fraction.png"); plt.close(fig)


def fig_auroc_change(res, out):
    """H1: change in AUROC relative to the uncorrected model (same rep, fraction, classifier)."""
    d = res[res.recalibration == "none"]
    base = d[d.correction == "none"].set_index(["rep", "event_fraction", "classifier"]).auroc
    d = d[d.correction != "none"].copy()
    d["d_auroc"] = d.auroc.values - base.reindex(pd.MultiIndex.from_frame(d[["rep", "event_fraction", "classifier"]])).values
    g = d.groupby(["classifier", "correction"]).d_auroc.mean().unstack().reindex(
        index=[c for c in CLF_ORDER if c in d.classifier.unique()], columns=[c for c in CORR_ORDER if c != "none"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(g.index)); w = 0.16
    for i, corr in enumerate(g.columns):
        ax.bar(x + (i - 2) * w, g[corr].values, width=w - 0.02, color=CORR_COLOR[corr], label=CORR_LABEL[corr])
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([CLF_LABEL[c] for c in g.index], rotation=20, ha="right")
    ax.set_ylabel("ΔAUROC vs uncorrected (mean over fractions & repeats)")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out / "fig2_auroc_change.png"); plt.close(fig)
    return g


def fig_recal_heatmap(res, out, ef):
    """H2/H3: |calibration slope - 1| after each recalibration, at one event fraction, for resampled models."""
    d = res[(np.isclose(res.event_fraction, ef)) & (res.correction != "none")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (metric, title, fmt) in zip(axes, [("cal_intercept", "Calibration intercept", "{:+.2f}"),
                                               ("cal_slope", "Calibration slope", "{:.2f}")]):
        g = d.groupby(["classifier", "recalibration"])[metric].mean().unstack()
        g = g.reindex(index=[c for c in CLF_ORDER if c in g.index], columns=[r for r in RECAL_ORDER if r in g.columns])
        dev = (g.abs() if metric == "cal_intercept" else (g - 1).abs())
        cap = 1.0 if metric == "cal_slope" else 2.0
        im = ax.imshow(np.clip(dev.values, 0, cap), cmap="Blues", vmin=0, vmax=cap, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if min(dev.values[i, j], cap) > cap * 0.55 else INK)
        ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([RECAL_LABEL[r] for r in g.columns], rotation=20, ha="right")
        if ax is axes[0]:
            ax.set_yticks(range(g.shape[0])); ax.set_yticklabels([CLF_LABEL[c] for c in g.index])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(f"{title} (darker = further from ideal)", fontsize=9); ax.grid(False)
    fig.suptitle(f"After recalibration, event fraction {ef*100:g}% (mean over resampling methods & repeats)", fontsize=10)
    fig.savefig(out / f"fig3_recalibration_heatmap_ef{ef*100:g}.png"); plt.close(fig)


def fig_calibration_curves(cur, out, clf, ef):
    """Flexible calibration curves for one classifier at one fraction: before and after Beta calibration."""
    d = cur[(cur.classifier == clf) & (np.isclose(cur.event_fraction, ef)) & (cur.rep == cur.rep.min())]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, recal in zip(axes, ["none", "beta"]):
        lim = 0
        for corr in CORR_ORDER:
            r = d[(d.correction == corr) & (d.recalibration == recal)]
            if r.empty:
                continue
            x, y = np.array(json.loads(r.cal_x.iloc[0])), np.array(json.loads(r.cal_y.iloc[0]))
            lim = max(lim, x.max(), y.max())
            ax.plot(x, y, color=CORR_COLOR[corr], linewidth=2, label=CORR_LABEL[corr])
        ax.plot([0, lim], [0, lim], color=INK2, linestyle="--", linewidth=1, label="Perfect calibration")
        ax.set_title(f"{CLF_LABEL[clf]}, recalibration: {RECAL_LABEL[recal]}", fontsize=9)
        ax.set_xlabel("Predicted risk")
    axes[0].set_ylabel("Observed proportion (loess)")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Flexible calibration curves at event fraction {ef*100:g}% (1st–99th percentile of predictions)", fontsize=10)
    fig.savefig(out / f"fig4_calibration_curves_{clf}_ef{ef*100:g}.png"); plt.close(fig)


def fig_decision_curves(cur, res, out, clf, ef):
    """H4: Net Benefit - uncorrected vs corrected (raw and Beta-recalibrated) vs treat-all / treat-none."""
    d = cur[(cur.classifier == clf) & (np.isclose(cur.event_fraction, ef))]
    prev = res[np.isclose(res.event_fraction, ef)].obs_rate.mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    t_max = min(0.5, prev * 4)
    for corr, recal, ls in [("none", "none", "-"), ("ros", "none", "-"), ("ros", "beta", "--"),
                            ("smote", "none", "-"), ("smote", "beta", "--")]:
        r = d[(d.correction == corr) & (d.recalibration == recal)]
        if r.empty:
            continue
        t = np.array(json.loads(r.dca_t.iloc[0]))
        nb = np.mean([json.loads(v) for v in r.dca_nb], axis=0)  # average over repeats
        keep = t <= t_max
        label = CORR_LABEL[corr] + ("" if recal == "none" else " + Beta")
        # The uncorrected curve is drawn wide and underneath so it stays visible where others coincide.
        lw, z = (5, 1) if corr == "none" else (2, 2)
        ax.plot(t[keep], nb[keep], color=CORR_COLOR[corr], linestyle=ls, linewidth=lw, zorder=z,
                alpha=0.6 if corr == "none" else 1, label=label)
    t = np.linspace(0.001, t_max, 100)
    ax.plot(t, prev - (1 - prev) * t / (1 - t), color=INK2, linewidth=1, label="Treat all")
    ax.axhline(0, color=INK, linewidth=1, label="Treat none")
    ax.set_ylim(-prev * 0.2, prev * 1.05)
    ax.set_xlabel("Risk threshold"); ax.set_ylabel("Net Benefit")
    ax.set_title(f"Decision curves, {CLF_LABEL[clf]}, event fraction {ef*100:g}%", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(out / f"fig5_decision_curve_{clf}_ef{ef*100:g}.png"); plt.close(fig)


def hypothesis_summary(res):
    lines = []
    raw = res[res.recalibration == "none"]
    corr = raw[raw.correction.isin(["rus", "ros", "smote", "adasyn", "weighted"])]
    lines.append("H1  share of corrected models (no recalibration) with intercept < 0: "
                 f"{(corr.cal_intercept < 0).mean():.1%}")
    by_ef = corr.groupby("event_fraction").cal_intercept.median()
    lines.append("H1  median intercept by event fraction: " + ", ".join(f"{k*100:g}%: {v:+.2f}" for k, v in by_ef.items()))
    base = raw[raw.correction == "none"].set_index(["rep", "event_fraction", "classifier"]).auroc
    da = corr.auroc.values - base.reindex(pd.MultiIndex.from_frame(corr[["rep", "event_fraction", "classifier"]])).values
    lines.append(f"H1  ΔAUROC corrected - uncorrected: median {np.nanmedian(da):+.3f}, "
                 f"share improved by > 0.01: {(da > 0.01).mean():.1%}")

    rc = res[res.correction.isin(["rus", "ros", "smote", "adasyn", "weighted"])]
    s = rc.groupby("recalibration").agg(abs_int=("cal_intercept", lambda v: v.abs().median()),
                                        abs_slope_dev=("cal_slope", lambda v: (v - 1).abs().median()),
                                        ici_rel=("ici_rel", "median")).reindex(RECAL_ORDER)
    lines.append("H2  corrected models, median over all cells:\n" + s.round(3).to_string())
    pr = rc[rc.recalibration == "prior"].groupby("correction").oe_ratio.median()
    lines.append("H2  prior correction O/E ratio by correction (1 = ideal): " + ", ".join(f"{k}: {v:.2f}" for k, v in pr.items()))

    ext = res[res.classifier.isin(["nb", "cart"]) & res.recalibration.isin(["platt", "isotonic", "beta"])]
    h3 = ext.groupby(["classifier", "recalibration"]).agg(ici_rel=("ici_rel", "median"),
                                                          slope=("cal_slope", "median")).round(3)
    lines.append("H3  extreme-probability classifiers:\n" + h3.to_string())

    u = raw[raw.correction == "none"][["classifier", "event_fraction", "sens_tprev", "spec_tprev"]]
    c = corr[["classifier", "event_fraction", "correction", "sens_t050", "spec_t050"]]
    h4 = pd.concat([u.groupby("classifier")[["sens_tprev", "spec_tprev"]].median(),
                    c.groupby("classifier")[["sens_t050", "spec_t050"]].median()], axis=1).round(3)
    lines.append("H4  uncorrected @ prevalence threshold vs corrected @ 0.5 (medians):\n" + h4.to_string())
    return "\n\n".join(lines)


def analyse(res, cur, out):
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{out.name}] {len(res)} rows, repeats {sorted(res.rep.unique())}, fractions {sorted(res.event_fraction.unique())}")

    res.groupby("event_fraction")[["n_test", "events_test"]].mean().round(0).to_csv(out / "table1_test_partitions.csv")
    raw = res[res.recalibration == "none"]
    for col in ["cal_intercept", "cal_slope", "auroc", "ici_rel"]:
        g = mean_sd(raw, ["event_fraction", "classifier", "correction"], col)
        g.pivot_table(index=["event_fraction", "classifier"], columns="correction", values="fmt", aggfunc="first")          .reindex(columns=CORR_ORDER).to_csv(out / f"table2_{col}_no_recal.csv")
    keys = ["event_fraction", "classifier", "correction", "recalibration"]
    cols = ["auroc", "auprc", "cal_intercept", "cal_slope", "ici", "ici_rel", "ece", "brier", "scaled_brier",
            "spiegelhalter_z", "oe_ratio", "sens_t050", "spec_t050", "sens_tprev", "spec_tprev", "mcc_tprev"]
    cols += [c for c in ["mae_true", "mae_true_rel"] if c in res]
    res.groupby(keys)[cols].mean().round(4).to_csv(out / "table3_all_configurations.csv")

    fig_intercept_vs_fraction(res, out)
    fig_auroc_change(res, out)
    fracs = sorted(res.event_fraction.unique())
    for ef in fracs:
        fig_recal_heatmap(res, out, ef)
    clfs = set(res.classifier.unique())
    for clf in ["lr", "xgb", "nb"]:
        for ef in {fracs[-1], 0.01}:
            if clf in clfs and np.isclose(fracs, ef).any():
                fig_calibration_curves(cur, out, clf, ef)
    for clf in ["lr", "xgb"]:
        for ef in {fracs[-1], fracs[0]}:
            if clf in clfs:
                fig_decision_curves(cur, res, out, clf, ef)

    summary = hypothesis_summary(res)
    (out / "hypotheses.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    dirs = [d for d in a.dirs if (Path(d) / "results.csv").exists()]
    res, cur = load(dirs)
    for df in (res, cur):
        if "dataset" not in df:
            df["dataset"] = "real"
    for ds in sorted(res.dataset.unique()):
        analyse(res[res.dataset == ds].copy(), cur[cur.dataset == ds].copy(), Path(a.out) / ds)


if __name__ == "__main__":
    main()
