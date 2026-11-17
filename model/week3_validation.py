#!/usr/bin/env python3
"""
EE5609 Project 10 - Milestone 2 (Tasks 1-3 of the review)
---------------------------------------------------------
Task 1  Validate the statistical widths: Gaussian prediction (mean w^T mu and
        variance w^T Sigma w, from the 60k TRAINING images) against overflows
        actually OBSERVED during bit-true inference on the 10k TEST images, at
        the SIRA width, SIRA-1 and SIRA-2 (and the Gaussian-model width).
Task 2  Accuracy with reduced widths, using the bit-true fixed-point datapath
        (fixedpoint.py: wrap-around or saturating accumulator, integer
        rescaling with round-half-up). Mismatches against the full-width
        fixed-point network, all-layer configurations and one-layer sweeps.
Task 3  Reliability of the Gaussian model: empirical vs Gaussian vs Bernstein on
        the Week-1 synthetic distributions (independent terms: Bernstein valid),
        and empirical vs Gaussian vs Cantelli on the MNIST layers (terms
        correlated: Bernstein's independence assumption fails, Cantelli needs
        only the mean and variance). Verdict: conservative / optimistic.

Predicted values are model outputs. Observed values are counts with exact
(Clopper-Pearson) 95% intervals; zero observed events are reported as an upper
bound only. No probability below the measurement floor is claimed as observed.

Outputs: results/week3/ (tables as CSV + summary.md, figures as PNG + PDF).
Run (from ~/ee5609):  python model/week3_validation.py        (~2-3 min)
"""

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, ROOT, W_QMAX, A_QMAX, bits_signed, load_mnist, IntMLP
from fixedpoint import FixedPointMLP, WIDE, prefix_extremes, reg_limits, self_test
from week2_accumulators import (B_GRID, cap_at_sira, gauss_tail_grid, mean_cov,
                                sira_bits, width_from_grid)

sys.path.insert(0, str(ROOT / "montecarlo"))
import week1_montecarlo as w1                                           # noqa: E402

OUT = RESULTS_DIR / "week3"
TARGET = 1e-9                     # design target of the Gaussian model (per dot product)
SWEEP_BELOW_SIRA = 6              # Task 2 sweep: SIRA, SIRA-1, ... SIRA-6

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cp_interval(k, n, conf=0.95):
    """Exact Clopper-Pearson interval for k events in n trials."""
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def upper95_zero(n):
    """One-sided 95% upper bound when 0 events are seen in n trials (~3/n)."""
    return 1 - 0.05 ** (1 / n)


def fmt_p(p):
    return "0" if p == 0 else f"{p:.2e}"


def verdict(k, n, pred):
    """Compare an observed count with a predicted probability."""
    lo, hi = cp_interval(k, n)
    if lo > pred:
        return "optimistic (real rate higher)"
    if hi < pred:
        return "conservative (real rate lower)"
    return "consistent"


def width_verdict(k, n, pred):
    """Verdict at one width; 'not testable' when the prediction is far below
    what n samples can resolve and nothing was observed."""
    if k == 0 and pred < 0.1 * upper95_zero(n):
        return "not testable here (prediction below measurement floor)"
    return verdict(k, n, pred)


def cantelli_grid(m, sd, b_sira):
    """Distribution-free bound from mean and variance only (one-sided Chebyshev,
    applied to each side): P(S-m >= t) <= v / (v + t^2)."""
    lo = -(2.0 ** (B_GRID - 1))[:, None]
    hi = (2.0 ** (B_GRID - 1) - 1)[:, None]
    v = (np.asarray(sd) ** 2)[None, :]
    m = np.asarray(m)[None, :]
    tu = hi + 1 - m
    tl = m - (lo - 1)
    pu = np.where(tu > 0, v / (v + tu ** 2), 1.0)
    pl = np.where(tl > 0, v / (v + tl ** 2), 1.0)
    return cap_at_sira(np.minimum(1.0, pu + pl), b_sira)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def md_table(rows, cols, heads=None):
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return out


# ---------------------------------------------------------------------------
# per-layer statistics and widths
# ---------------------------------------------------------------------------
def layer_model(W, x_tr, acc_te):
    """Mean, std (w^T Sigma w) per neuron from training inputs; widths."""
    Wf = W.astype(np.float64)
    mu, Sig = mean_cov(x_tr)
    m = Wf @ mu
    v = np.einsum("nk,kl,nl->n", Wf, Sig, Wf, optimize=True)
    v_diag = (Wf ** 2) @ np.diag(Sig)
    sd = np.sqrt(np.maximum(v, 0))
    b_sira_n, _, _ = sira_bits(W)
    pg = cap_at_sira(gauss_tail_grid(m, sd), b_sira_n)
    pc = cantelli_grid(m, sd, b_sira_n)
    K = W.shape[1]
    return dict(
        m=m, sd=sd, pg=pg, pc=pc,
        offdiag_share=float(np.median(1 - v_diag / np.maximum(v, 1e-12))),
        datatype=bits_signed(-K * W_QMAX * A_QMAX, K * W_QMAX * A_QMAX),
        sira=int(b_sira_n.max()),
        gauss9=int(width_from_grid(pg, 1e-9).max()),
        gauss15=int(width_from_grid(pg, 1e-15).max()),
        cantelli9=int(width_from_grid(pc, 1e-9).max()),
        observed=bits_signed(acc_te.min(), acc_te.max()))


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------
def task1(fp, models, ins_te, acc_te):
    rows, curves = [], []
    for l, name in enumerate(fp.names):
        md = models[l]
        x, s = ins_te[l], acc_te[l]
        n_img, N = s.shape
        K = fp.W[l].shape[1]
        pmin, pmax = prefix_extremes(x, fp.W[l])
        n_dots = s.size
        configs = [("SIRA", md["sira"]), ("SIRA-1", md["sira"] - 1), ("SIRA-2", md["sira"] - 2)]
        if md["gauss9"] not in [c[1] for c in configs]:
            configs.append(("Gaussian model 1e-9", md["gauss9"]))
        for b in range(10, md["sira"] + 2):
            lo, hi = reg_limits(b)
            k_fin = int(((s < lo) | (s > hi)).sum())
            k_any = int(((pmin < lo) | (pmax > hi)).sum())
            gi = int(np.where(B_GRID == b)[0][0])
            pred_mean = float(md["pg"][gi].mean())
            cant_mean = float(md["pc"][gi].mean())
            ci = cp_interval(k_fin, n_dots)
            curves.append(dict(layer=name, width=b, dot_products=n_dots, observed_final=k_fin,
                               observed_partial=k_any, observed_rate=k_fin / n_dots,
                               ci_low=ci[0], ci_high=ci[1], gaussian_pred=pred_mean,
                               cantelli_bound=cant_mean))
        for label, b in configs:
            lo, hi = reg_limits(b)
            fin = (s < lo) | (s > hi)
            anyo = (pmin < lo) | (pmax > hi)
            k_fin, k_any = int(fin.sum()), int(anyo.sum())
            gi = int(np.where(B_GRID == b)[0][0])
            pred = float(md["pg"][gi].mean())
            exp_count = float(md["pg"][gi].sum() * n_img)
            ci = cp_interval(k_fin, n_dots)
            obs = (f"{k_fin / n_dots:.2e} [{ci[0]:.1e}, {ci[1]:.1e}]" if k_fin
                   else f"0 (95% upper bound {upper95_zero(n_dots):.1e})")
            rows.append(dict(
                layer=name, config=label, width_bits=b, dot_products=n_dots,
                mac_operations=n_dots * K,
                predicted_prob_per_dot=fmt_p(pred), predicted_count=f"{exp_count:.3g}",
                observed_final_overflows=k_fin, observed_rate_95ci=obs,
                observed_partial_sum_overflows=k_any,
                gaussian_verdict=("overflow impossible (SIRA guarantee)" if b >= md["sira"]
                                  else width_verdict(k_fin, n_dots, pred)),
                images_affected=int(fin.any(axis=1).sum()),
                neurons_affected=int(fin.any(axis=0).sum())))
    return rows, curves


def fig_task1(curves, models, names, n_img):
    fig, axes = plt.subplots(1, len(names), figsize=(13.5, 4.2), sharey=True)
    for ax, name, md in zip(axes, names, models):
        c = [r for r in curves if r["layer"] == name]
        b = np.array([r["width"] for r in c])
        k = np.array([r["observed_final"] for r in c])
        n = c[0]["dot_products"]
        rate = k / n
        lo = np.array([r["ci_low"] for r in c])
        hi = np.array([r["ci_high"] for r in c])
        ax.semilogy(b, np.maximum([r["cantelli_bound"] for r in c], 1e-30), ls="--", color=INK2,
                    lw=1.3, label="Predicted bound: Cantelli (mean + variance only)")
        ax.semilogy(b, np.maximum([r["gaussian_pred"] for r in c], 1e-30), color=YELLOW, lw=2.2,
                    label="Predicted: Gaussian with $w^T\\mu$, $w^T\\Sigma w$")
        pos = k > 0
        ax.errorbar(b[pos], rate[pos], yerr=[rate[pos] - lo[pos], hi[pos] - rate[pos]], fmt="o",
                    ms=5, color=BLUE, capsize=2, label="Observed on test set (95% CI)")
        ub = upper95_zero(n)
        ax.plot(b[~pos], np.full((~pos).sum(), ub), "v", ms=6, color=BLUE, mfc="white",
                label="Observed 0: 95% upper bound")
        ax.axhline(ub, color=BLUE, lw=0.6, ls=":")
        for off, lab in [(0, "SIRA"), (1, "SIRA-1"), (2, "SIRA-2")]:
            x = md["sira"] - off
            ax.axvline(x, color=ORANGE, lw=1, ls="-" if off == 0 else ":")
            ax.text(x + 0.1, 3e-1, lab, rotation=90, fontsize=7.5, color=INK2, va="top")
        ax.set_title(f"{name}  ({n:,} dot products)")
        ax.set_xlabel("accumulator width (bits)")
        ax.set_ylim(1e-12, 2)
        ax.set_xlim(b[0] - 0.5, b[-1] + 0.5)
    axes[0].set_ylabel("overflow probability per dot product")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Predicted vs observed accumulator overflow, MNIST MLP "
                 "(prediction from training set, observation on 10,000 test images)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    save(fig, "fig1_predicted_vs_observed_overflow")


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------
def task2(fp, xte, yte, models, base_pred):
    names = fp.names
    L = len(names)
    acc_base = float((base_pred == yte).mean())
    rows = [dict(config="Float rescaling (Week 2 reference)", widths="32/32/32", mode="-",
                 accuracy=None, mismatches=None, overflows="-")]
    ref = IntMLP().predict(xte)
    rows[0].update(accuracy=f"{(ref == yte).mean():.4f}", mismatches=int((ref != base_pred).sum()))
    rows.append(dict(config="Fixed-point, full width (baseline)", widths="32/32/32", mode="-",
                     accuracy=f"{acc_base:.4f}", mismatches=0, overflows="0/0/0"))
    tr, _ = fp.forward(xte, [WIDE] * L, rounding="truncate")
    rows.append(dict(config="Fixed-point, truncating rescale", widths="32/32/32", mode="-",
                     accuracy=f"{(tr == yte).mean():.4f}", mismatches=int((tr != base_pred).sum()),
                     overflows="0/0/0"))
    configs = [("Datatype bound", [m["datatype"] for m in models]),
               ("SIRA", [m["sira"] for m in models]),
               ("Statistical (Gaussian model 1e-9)", [m["gauss9"] for m in models]),
               ("SIRA-1 (all layers)", [m["sira"] - 1 for m in models]),
               ("SIRA-2 (all layers)", [m["sira"] - 2 for m in models])]
    for label, ws in configs:
        for mode in ("wrap", "saturate"):
            pred, infos = fp.forward(xte, ws, mode=mode, cache_tag="test")
            rows.append(dict(config=label, widths="/".join(map(str, ws)), mode=mode,
                             accuracy=f"{(pred == yte).mean():.4f}",
                             mismatches=int((pred != base_pred).sum()),
                             overflows="/".join(str(i["final_overflows"]) for i in infos)))
    sweep = []
    for l in range(L):
        for off in range(0, SWEEP_BELOW_SIRA + 1):
            b = models[l]["sira"] - off
            ws = [WIDE] * L
            ws[l] = b
            for mode in ("wrap", "saturate"):
                pred, infos = fp.forward(xte, ws, mode=mode, cache_tag="test")
                sweep.append(dict(layer=names[l], width_bits=b, below_sira=off, mode=mode,
                                  accuracy=float((pred == yte).mean()),
                                  mismatches=int((pred != base_pred).sum()),
                                  overflows_in_layer=infos[l]["final_overflows"],
                                  partial_overflows_in_layer=infos[l]["any_overflows"]))
    return rows, sweep, acc_base


def fig_task2(sweep, models, names, acc_base):
    fig, axes = plt.subplots(1, len(names), figsize=(13.5, 4.0), sharey=True)
    for ax, name, md in zip(axes, names, models):
        for mode, col, mk in [("wrap", ORANGE, "o"), ("saturate", BLUE, "s")]:
            r = [s for s in sweep if s["layer"] == name and s["mode"] == mode]
            b = [s["width_bits"] for s in r]
            ax.plot(b, [100 * s["accuracy"] for s in r], marker=mk, ms=5, lw=2, color=col,
                    label=f"{mode}-around" if mode == "wrap" else "saturating")
            for s in r:
                if s["mismatches"] and mode == "wrap":
                    ax.annotate(f"{s['mismatches']}", (s["width_bits"], 100 * s["accuracy"]),
                                textcoords="offset points",
                                xytext=(7, 2) if s["accuracy"] < 0.9 else (4, -13), ha="left",
                                fontsize=7, color=INK2)
        ax.axhline(100 * acc_base, color=INK2, lw=0.8, ls="--", label="full-width baseline")
        ax.set_ylim(0, 104)
        for off, lab in [(0, "SIRA"), (1, "-1"), (2, "-2")]:
            ax.axvline(md["sira"] - off, color=ORANGE, lw=0.8, ls="-" if off == 0 else ":")
        ax.set_title(f"{name} narrowed, other layers full width")
        ax.set_xlabel("accumulator width of this layer (bits)")
        ax.invert_xaxis()
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Layer sensitivity: accuracy as one layer's accumulator is narrowed below SIRA "
                 "(numbers = mismatches vs baseline, wrap-around)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig2_layer_sensitivity_accuracy")


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------
def task3_synthetic(trials, seed):
    rng = np.random.default_rng(seed)
    wc, _ = w1.worst_case()
    rows, verdict_rows, design = [], [], []
    plot = {}
    for name, (sampler, (xlo, xhi)) in w1.SCENARIOS.items():
        sums, mu, var = w1.monte_carlo(sampler, trials, rng)
        corners = [w1.W_MIN * xlo, w1.W_MIN * xhi, w1.W_MAX * xlo, w1.W_MAX * xhi]
        pmin, pmax = min(corners), max(corners)
        g = lambda b: w1.gaussian_tail(b, mu, var)
        bn = lambda b: w1.bernstein_tail(b, mu, var, pmin, pmax)
        recs = []
        for b in w1.WIDTHS:
            lo, hi = reg_limits(b)
            k = int(((sums < lo) | (sums > hi)).sum())
            ci = cp_interval(k, trials)
            recs.append(dict(scenario=name, width=b, observed=k, trials=trials,
                             observed_rate=k / trials, ci_low=ci[0], ci_high=ci[1],
                             gaussian=g(b), bernstein=bn(b),
                             bernstein_holds=bool(ci[0] <= bn(b) + 1e-15)))
        rows += recs
        tail = [r for r in recs if r["gaussian"] < 0.05 and (r["observed"] > 0 or r["gaussian"] > 1e-12)]
        for r in tail:
            verdict_rows.append(dict(
                scenario=name, width=r["width"], observed=f"{r['observed']} / {trials:,}",
                observed_rate_95ci=(f"{r['observed_rate']:.2e} [{r['ci_low']:.1e}, {r['ci_high']:.1e}]"
                                    if r["observed"] else f"0 (< {upper95_zero(trials):.1e})"),
                gaussian_predicted=fmt_p(r["gaussian"]), bernstein_bound=fmt_p(min(1, r["bernstein"])),
                ratio_observed_to_gaussian=(f"{r['observed_rate'] / r['gaussian']:.1f}x"
                                            if r["observed"] and r["gaussian"] > 0 else "-"),
                gaussian_verdict=verdict(r["observed"], trials, r["gaussian"])))
        obs_needed = bits_signed(int(sums.min()), int(sums.max()))
        design.append(dict(scenario=name, observed_max_width=obs_needed,
                           gaussian_width_1e9=w1.width_for(1e-9, g, wc),
                           bernstein_width_1e9=w1.width_for(1e-9, bn, wc),
                           worst_case=wc,
                           bernstein_never_violated=all(r["bernstein_holds"] for r in recs)))
        plot[name] = recs
    return rows, verdict_rows, design, plot


def fig_task3_synthetic(plot, trials):
    fig, axes = plt.subplots(1, len(plot), figsize=(13.5, 4.0), sharey=True)
    for ax, (name, recs) in zip(axes, plot.items()):
        b = np.array([r["width"] for r in recs])
        k = np.array([r["observed"] for r in recs])
        rate = k / trials
        lo = np.array([r["ci_low"] for r in recs])
        hi = np.array([r["ci_high"] for r in recs])
        ax.semilogy(b, np.maximum([min(1, r["bernstein"]) for r in recs], 1e-30), ls="--",
                    color=AQUA, lw=2, label="Bernstein bound (valid: independent, bounded terms)")
        ax.semilogy(b, np.maximum([r["gaussian"] for r in recs], 1e-30), color=YELLOW, lw=2.2,
                    label="Gaussian prediction")
        pos = k > 0
        ax.errorbar(b[pos], rate[pos], yerr=[rate[pos] - lo[pos], hi[pos] - rate[pos]], fmt="o",
                    ms=5, color=BLUE, capsize=2, label="Observed, 2M trials (95% CI)")
        ub = upper95_zero(trials)
        ax.plot(b[~pos], np.full((~pos).sum(), ub), "v", ms=6, color=BLUE, mfc="white",
                label="Observed 0: 95% upper bound")
        ax.set_title(name)
        ax.set_xlabel("accumulator width (bits)")
        ax.set_ylim(1e-14, 2)
        ax.set_xlim(b[0] - 0.5, 21.5)
    axes[0].set_ylabel("overflow probability per dot product")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Synthetic K = 32 dot products: Gaussian prediction vs Bernstein bound vs observation",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    save(fig, "fig3_synthetic_gaussian_vs_bernstein")


def task3_mnist(models, acc_te, names):
    ks = [2, 3, 4, 5, 6]
    rows, zs = [], {}
    for name, md, s in zip(names, models, acc_te):
        ok = md["sd"] > 0
        z = ((s[:, ok] - md["m"][ok]) / md["sd"][ok]).ravel()
        zs[name] = z
        n = z.size
        for k in ks:
            c = int((np.abs(z) > k).sum())
            pred = 2 * norm.sf(k)
            ci = cp_interval(c, n)
            rows.append(dict(layer=name, threshold_sigma=k, values=n, observed_beyond=c,
                             observed_rate_95ci=(f"{c / n:.2e} [{ci[0]:.1e}, {ci[1]:.1e}]" if c
                                                 else f"0 (< {upper95_zero(n):.1e})"),
                             gaussian_predicted=f"{pred:.2e}",
                             ratio_observed_to_gaussian=f"{c / n / pred:.2f}x" if c else "-",
                             gaussian_verdict=verdict(c, n, pred)))
    return rows, zs


def fig_task3_mnist(zs, models, names):
    fig, axes = plt.subplots(2, len(names), figsize=(13.5, 7.0))
    xx = np.linspace(-8, 8, 400)
    for j, name in enumerate(names):
        z = zs[name]
        ax = axes[0, j]
        ax.hist(z, bins=160, range=(-8, 8), density=True, color=BLUE, alpha=0.75, log=True,
                label="Observed (test set)")
        ax.plot(xx, norm.pdf(xx), color=YELLOW, lw=2, label="Gaussian N(0,1)")
        ax.set_ylim(1e-8, 1)
        ax.set_title(f"{name}: standardised accumulator values")
        ax.set_xlabel("(s - $w^T\\mu$) / $\\sqrt{w^T\\Sigma w}$")
        if j == 0:
            ax.set_ylabel("density (log)")
            ax.legend(fontsize=8)
        ax = axes[1, j]
        ks = np.arange(1, 7.01, 0.25)
        n = z.size
        az = np.abs(z)
        cnt = np.array([(az > k).sum() for k in ks])
        pred = 2 * norm.sf(ks)
        ratio = cnt / n / pred
        lo = np.array([cp_interval(int(c), n)[0] for c in cnt]) / pred
        hi = np.array([cp_interval(int(c), n)[1] for c in cnt]) / pred
        pos = cnt > 0
        ax.fill_between(ks[pos], lo[pos], hi[pos], color=BLUE, alpha=0.18, lw=0)
        ax.semilogy(ks[pos], ratio[pos], color=BLUE, lw=2, label="observed / Gaussian (95% band)")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.text(1.05, 1.15, "Gaussian exact", fontsize=7.5, color=INK2)
        ax.text(1.05, 6, "above 1 = Gaussian optimistic", fontsize=7.5, color=INK2)
        ax.set_ylim(0.05, 200)
        ax.set_xlim(1, 6)
        ax.set_xlabel("threshold k (standard deviations)")
        ax.set_title(f"{name}: P(|z| > k), observed vs Gaussian")
        if j == 0:
            ax.set_ylabel("ratio observed / predicted")
    fig.suptitle("MNIST accumulators vs the Gaussian model (mean and covariance from training set, "
                 "values from 10,000 test images)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig4_mnist_distribution_vs_gaussian")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xte, yte = load_mnist()
    fp = FixedPointMLP()
    self_test(fp, xte)
    print("Bit-true datapath self-test passed (wrap and saturate match term-by-term simulation).")

    base_pred, _, ins_tr, acc_tr = fp.forward(xtr, [WIDE] * fp.L, keep=True)
    base_pred, _, ins_te, acc_te = fp.forward(xte, [WIDE] * fp.L, keep=True)
    print(f"Full-width fixed-point test accuracy {(base_pred == yte).mean():.4f}")

    models = [layer_model(fp.W[l], ins_tr[l], acc_te[l]) for l in range(fp.L)]
    del ins_tr, acc_tr                                     # training data no longer needed
    widths = [dict(layer=n, K=fp.W[l].shape[1], neurons=fp.W[l].shape[0],
                   datatype=m["datatype"], SIRA=m["sira"], SIRA_minus_1=m["sira"] - 1,
                   SIRA_minus_2=m["sira"] - 2, gaussian_model_1e9=m["gauss9"],
                   gaussian_model_1e15=m["gauss15"], cantelli_1e9=m["cantelli9"],
                   observed_max_test=m["observed"],
                   offdiag_share_of_variance=f"{m['offdiag_share']:.2f}")
              for l, (n, m) in enumerate(zip(fp.names, models))]
    write_csv(OUT / "table1_widths.csv", widths)

    print("Task 1: predicted vs observed overflow ...")
    t1, curves = task1(fp, models, ins_te, acc_te)
    write_csv(OUT / "table2_task1_overflow.csv", t1)
    write_csv(OUT / "task1_overflow_curve.csv", curves)
    fig_task1(curves, models, fp.names, len(xte))

    print("Task 2: accuracy with bit-true reduced accumulators ...")
    t2, sweep, acc_base = task2(fp, xte, yte, models, base_pred)
    write_csv(OUT / "table3_task2_accuracy.csv", t2)
    write_csv(OUT / "table4_task2_layer_sweep.csv", sweep)
    fig_task2(sweep, models, fp.names, acc_base)

    print("Task 3: Gaussian reliability ...")
    syn_rows, syn_verdict, syn_design, syn_plot = task3_synthetic(2_000_000, 5609)
    write_csv(OUT / "task3_synthetic_all_widths.csv", syn_rows)
    write_csv(OUT / "table5_task3_synthetic_verdict.csv", syn_verdict)
    write_csv(OUT / "table6_task3_synthetic_design.csv", syn_design)
    fig_task3_synthetic(syn_plot, 2_000_000)
    mn_rows, zs = task3_mnist(models, acc_te, fp.names)
    write_csv(OUT / "table7_task3_mnist_tail.csv", mn_rows)
    fig_task3_mnist(zs, models, fp.names)

    # ---- summary.md -----------------------------------------------------
    md = ["# Milestone 2: validation of statistical accumulator sizing (Tasks 1-3)", "",
          "Predicted = model output (Gaussian from w^T mu and w^T Sigma w on the 60k training "
          "images; Bernstein / Cantelli bounds). Observed = counts from bit-true inference on the "
          "10k test images, with exact 95% (Clopper-Pearson) intervals. Zero observed events only "
          "give an upper bound (about 3/n); no smaller probability is claimed as observed.", "",
          "## Table 1. Accumulator widths per layer (bits)", ""]
    md += md_table(widths, ["layer", "K", "datatype", "SIRA", "SIRA_minus_1", "SIRA_minus_2",
                            "gaussian_model_1e9", "cantelli_1e9", "observed_max_test"],
                   ["Layer", "K", "Datatype", "SIRA", "SIRA-1", "SIRA-2",
                    "Gaussian model (pred. 1e-9)", "Cantelli (pred. 1e-9)", "Largest observed"])
    md += ["", "## Table 2. Task 1: predicted vs observed overflow (test set)", ""]
    md += md_table(t1, ["layer", "config", "width_bits", "dot_products", "mac_operations",
                        "predicted_prob_per_dot", "predicted_count", "observed_final_overflows",
                        "observed_rate_95ci", "observed_partial_sum_overflows", "images_affected",
                        "gaussian_verdict"],
                   ["Layer", "Config", "Bits", "Dot products", "MAC ops", "Predicted P/dot",
                    "Predicted count", "Observed overflows", "Observed rate [95% CI]",
                    "Partial-sum overflows", "Images affected", "Gaussian is"])
    md += ["", "## Table 3. Task 2: accuracy, bit-true fixed point (10,000 test images)", ""]
    md += md_table(t2, ["config", "widths", "mode", "accuracy", "mismatches", "overflows"],
                   ["Configuration", "Widths fc1/fc2/fc3", "Overflow mode", "Accuracy",
                    "Mismatches vs baseline", "Final-sum overflows fc1/fc2/fc3"])
    md += ["", "## Table 4. Task 2: layer sensitivity (one layer narrowed, others full width)", ""]
    sw = [dict(s, accuracy=f"{s['accuracy']:.4f}") for s in sweep]
    md += md_table(sw, ["layer", "width_bits", "below_sira", "mode", "accuracy", "mismatches",
                        "overflows_in_layer"],
                   ["Layer", "Bits", "Below SIRA", "Mode", "Accuracy", "Mismatches", "Overflows"])
    md += ["", "## Table 5. Task 3: synthetic distributions, Gaussian vs observed (2M trials)", ""]
    md += md_table(syn_verdict, ["scenario", "width", "observed_rate_95ci", "gaussian_predicted",
                                 "bernstein_bound", "ratio_observed_to_gaussian", "gaussian_verdict"],
                   ["Distribution", "Bits", "Observed rate [95% CI]", "Gaussian pred.",
                    "Bernstein bound", "Observed / Gaussian", "Gaussian is"])
    md += ["", "## Table 6. Task 3: synthetic design widths at 1e-9 (predicted) vs observed maximum", ""]
    md += md_table(syn_design, ["scenario", "observed_max_width", "gaussian_width_1e9",
                                "bernstein_width_1e9", "worst_case", "bernstein_never_violated"],
                   ["Distribution", "Largest observed (bits)", "Gaussian pred. 1e-9",
                    "Bernstein 1e-9", "Worst case", "Bernstein never below observed"])
    md += ["", "## Table 7. Task 3: MNIST layers, tail beyond k sigma (test set)", ""]
    md += md_table(mn_rows, ["layer", "threshold_sigma", "observed_beyond", "observed_rate_95ci",
                             "gaussian_predicted", "ratio_observed_to_gaussian", "gaussian_verdict"],
                   ["Layer", "k (sigma)", "Count", "Observed rate [95% CI]", "Gaussian pred.",
                    "Observed / Gaussian", "Gaussian is"])
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nAll outputs in {OUT}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
