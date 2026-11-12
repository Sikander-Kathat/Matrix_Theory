#!/usr/bin/env python3
"""
EE5609 Project 10 - Week 2, step 2: real accumulator values and w^T Sigma w
---------------------------------------------------------------------------
Uses the int8 model from week2_train_quantize.py.

1. Runs exact integer inference on all 60k training and 10k test images and
   logs every accumulator value s_j = w_j^T x (test set saved to disk).
2. For every layer, estimates the input mean mu and covariance Sigma from the
   TRAINING images (K x K matrix), and for every output neuron j computes
       mean      m_j  = w_j^T mu
       variance  v_j  = w_j^T Sigma w_j            (quadratic form)
   plus two simpler variances for comparison:
       diagonal  sum_i w_ji^2 Sigma_ii            (ignores correlations)
       eigen     lambda_max(Sigma) * ||w_j||^2     (Rayleigh-quotient bound)
3. Bound ladder per neuron and per layer (bits of a two's-complement register):
       datatype   K * 127 * 255                    (Quiz 2 style, any w, any x)
       SIRA l1    255 * (sum of positive w) / 255 * (sum of negative w)
       Gaussian   smallest B with P(N(m_j, v_j) outside register) <= p,
                  p = 1e-9 and 1e-15 per dot product, and 1e-9 per inference
       empirical  widest value actually seen on the TEST images
   Moments come from the training set; the check is on the unseen test set.
4. Same ladder for K = 32 tiles (the Quiz 2 / Week 1 setting): each dot
   product is split into 32-term partial sums.
5. Tail check: are real accumulators Gaussian far out in the tail?

Outputs in results/week2/:  CSV tables, summary.md, acc_test.npz and figures
(PNG for viewing, PDF for the LaTeX report).

Run (from ~/ee5609):  python model/week2_accumulators.py           (~1 min)
"""

import csv
import math
import time

import numpy as np
from scipy.stats import norm, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (A_QMAX, RESULTS_DIR, W_QMAX, IntMLP, bits_signed,
                    int_matmul, load_mnist)

OUT = RESULTS_DIR / "week2"
TILE = 32
P_DOT = (1e-9, 1e-15)          # per-dot-product targets (as in Week 1)
P_INF = 1e-9                   # per-inference target, union over a layer's neurons
B_GRID = np.arange(2, 41)

# Fixed colours per method (validated categorical palette, fixed order)
C = {"datatype": "#2a78d6", "sira": "#eb6834", "g15": "#1baf7a",
     "g9": "#eda100", "emp": "#e87ba4", "diag": "#1baf7a", "eig": "#eb6834"}
LAYER_C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def mean_cov(x_u8, chunk=10_000):
    """Population mean and covariance (ddof=0) of uint8 rows, float64."""
    n = len(x_u8)
    mu = x_u8.mean(axis=0, dtype=np.float64)
    S = np.zeros((x_u8.shape[1],) * 2)
    for i in range(0, n, chunk):
        xc = x_u8[i:i + chunk].astype(np.float64) - mu
        S += xc.T @ xc
    return mu, S / n


def gauss_tail_grid(m, sd):
    """P(N(m, sd^2) falls outside the B-bit register) for every B in B_GRID.
    Integer-valued sums -> continuity correction of 0.5. Shape [len(B_GRID), N]."""
    lo = -(2.0 ** (B_GRID - 1))[:, None]
    hi = (2.0 ** (B_GRID - 1) - 1)[:, None]
    m, sd = np.asarray(m)[None, :], np.asarray(sd)[None, :]
    safe = np.where(sd > 0, sd, 1.0)
    p = norm.sf((hi + 0.5 - m) / safe) + norm.cdf((lo - 0.5 - m) / safe)
    p_det = ((m < lo) | (m > hi)).astype(float)            # sd == 0: deterministic
    return np.where(sd > 0, p, p_det)


def width_from_grid(pgrid, target):
    """Smallest B in B_GRID with tail <= target, per column."""
    ok = pgrid <= target
    return B_GRID[np.argmax(ok, axis=0)]


def cap_at_sira(pgrid, b_sira):
    """Zero the Gaussian tail for every width >= that neuron's SIRA width."""
    return np.where(B_GRID[:, None] >= np.asarray(b_sira)[None, :], 0.0, pgrid)


def sira_bits(W):
    """Per-neuron SIRA interval-arithmetic width with inputs in [0, 255]."""
    W = W.astype(np.int64)
    hi = A_QMAX * np.where(W > 0, W, 0).sum(1)
    lo = A_QMAX * np.where(W < 0, W, 0).sum(1)
    return np.array([bits_signed(a, b) for a, b in zip(lo, hi)]), lo, hi


def emp_bits(acc):
    return np.array([bits_signed(a, b) for a, b in zip(acc.min(0), acc.max(0))])


# ---------------------------------------------------------------------------
# Analysis of one layer (full dot product)
# ---------------------------------------------------------------------------
def analyse_layer(name, W, x_tr, acc_tr, x_te, acc_te):
    N, K = W.shape
    Wf = W.astype(np.float64)
    mu, Sig = mean_cov(x_tr)
    m = Wf @ mu
    v = np.einsum("nk,kl,nl->n", Wf, Sig, Wf, optimize=True)
    v_diag = (Wf ** 2) @ np.diag(Sig)
    lam = np.linalg.eigvalsh(Sig)[::-1]
    v_eig = lam[0] * (Wf ** 2).sum(1)

    # sanity: quadratic form == empirical variance of the training accumulators
    rel_err_var = np.max(np.abs(v - acc_tr.var(0)) / np.maximum(acc_tr.var(0), 1))
    rel_err_mean = np.max(np.abs(m - acc_tr.mean(0)) / np.maximum(np.abs(acc_tr.mean(0)), 1))

    b_sira, l1_lo, l1_hi = sira_bits(W)
    sd = np.sqrt(np.maximum(v, 0))
    # The Gaussian has unbounded tails, but the true sum can never leave the
    # SIRA interval: at or above the SIRA width the overflow probability is 0.
    pgrid = cap_at_sira(gauss_tail_grid(m, sd), b_sira)
    b_g = {p: width_from_grid(pgrid, p) for p in P_DOT}
    p_inf_curve = pgrid.sum(axis=1)                       # union bound over neurons
    b_inf = int(B_GRID[np.argmax(p_inf_curve <= P_INF)])

    b_dt = bits_signed(-K * W_QMAX * A_QMAX, K * W_QMAX * A_QMAX)
    b_emp_te = emp_bits(acc_te)
    b_emp_tr = emp_bits(acc_tr)

    sd_te = np.where(sd > 0, sd, np.nan)
    z = (acc_te - m) / sd_te
    kurt = kurtosis(acc_te, axis=0, fisher=True)

    # empirical vs Gaussian overflow rate per dot product, test set
    curve = []
    for b in range(6, 30):
        lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
        emp = float(((acc_te < lo) | (acc_te > hi)).mean())
        img = float(((acc_te < lo) | (acc_te > hi)).any(axis=1).mean())
        gi = int(np.where(B_GRID == b)[0][0])
        curve.append((b, emp, float(pgrid[gi].mean()), img, float(min(1, p_inf_curve[gi]))))

    neurons = []
    for j in range(N):
        neurons.append(dict(
            layer=name, neuron=j, K=K, mean=m[j], sd_full=sd[j],
            sd_diag=math.sqrt(v_diag[j]), sd_eig=math.sqrt(v_eig[j]),
            l1_lo=int(l1_lo[j]), l1_hi=int(l1_hi[j]),
            emp_min=int(acc_te[:, j].min()), emp_max=int(acc_te[:, j].max()),
            bits_datatype=b_dt, bits_sira=int(b_sira[j]),
            bits_gauss_1e15=int(b_g[1e-15][j]), bits_gauss_1e9=int(b_g[1e-9][j]),
            bits_emp_test=int(b_emp_te[j]), bits_emp_train=int(b_emp_tr[j]),
            excess_kurtosis=float(kurt[j]) if np.isfinite(kurt[j]) else float("nan")))

    summary = dict(
        layer=name, K=K, N=N, bits_datatype=b_dt, bits_sira=int(b_sira.max()),
        bits_gauss_dot_1e15=int(b_g[1e-15].max()), bits_gauss_dot_1e9=int(b_g[1e-9].max()),
        bits_gauss_inf_1e9=b_inf, bits_emp_test=int(b_emp_te.max()),
        bits_emp_train=int(b_emp_tr.max()),
        max_abs_test=int(np.abs(acc_te).max()),
        sira_over_sd_median=float(np.nanmedian(np.maximum(-l1_lo, l1_hi) / sd_te)),
        diag_over_full_median=float(np.nanmedian(np.sqrt(v_diag) / sd_te)),
        eig_over_full_median=float(np.nanmedian(np.sqrt(v_eig) / sd_te)),
        lambda_max=float(lam[0]), trace=float(lam.sum()),
        top1_eig_share=float(lam[0] / lam.sum()),
        median_excess_kurtosis=float(np.nanmedian(kurt)),
        relerr_quadform_var=float(rel_err_var), relerr_mean=float(rel_err_mean))
    return dict(summary=summary, neurons=neurons, curve=curve, z=z, lam=lam,
                sd=sd, sd_diag=np.sqrt(v_diag), sd_eig=np.sqrt(v_eig),
                b_sira=b_sira, b_g9=b_g[1e-9], b_g15=b_g[1e-15], b_emp=b_emp_te)


# ---------------------------------------------------------------------------
# K = 32 tiles
# ---------------------------------------------------------------------------
def analyse_tiles(name, W, x_tr, x_te):
    N, K = W.shape
    Wf = W.astype(np.float64)
    rows = []
    worst = dict(sira=0, g9=0, g15=0, emp=0)
    for t, k0 in enumerate(range(0, K, TILE)):
        cols = slice(k0, min(k0 + TILE, K))
        Wt = W[:, cols]
        mu, Sig = mean_cov(x_tr[:, cols])
        m = Wf[:, cols] @ mu
        sd = np.sqrt(np.maximum(np.einsum("nk,kl,nl->n", Wf[:, cols], Sig, Wf[:, cols]), 0))
        bs, _, _ = sira_bits(Wt)
        pg = cap_at_sira(gauss_tail_grid(m, sd), bs)
        g9, g15 = width_from_grid(pg, 1e-9), width_from_grid(pg, 1e-15)
        s_te = int_matmul(x_te[:, cols], Wt)
        be = emp_bits(s_te)
        worst = dict(sira=max(worst["sira"], bs.max()), g9=max(worst["g9"], g9.max()),
                     g15=max(worst["g15"], g15.max()), emp=max(worst["emp"], be.max()))
        rows.append(dict(layer=name, tile=t, k_start=k0, k_len=cols.stop - k0,
                         bits_sira_max=int(bs.max()), bits_gauss_1e9_max=int(g9.max()),
                         bits_gauss_1e15_max=int(g15.max()), bits_emp_test_max=int(be.max()),
                         max_abs_test=int(np.abs(s_te).max())))
    b_dt = bits_signed(-TILE * W_QMAX * A_QMAX, TILE * W_QMAX * A_QMAX)
    summary = dict(layer=name, K=TILE, N=N, n_tiles=len(rows), bits_datatype=b_dt,
                   bits_sira=int(worst["sira"]), bits_gauss_dot_1e15=int(worst["g15"]),
                   bits_gauss_dot_1e9=int(worst["g9"]), bits_emp_test=int(worst["emp"]))
    return summary, rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def fig_ladder(full, tiles):
    methods = [("bits_datatype", "Datatype (Quiz 2)", C["datatype"]),
               ("bits_sira", "SIRA $\\ell_1$", C["sira"]),
               ("bits_gauss_dot_1e15", "Gaussian $w^T\\Sigma w$, $10^{-15}$", C["g15"]),
               ("bits_gauss_dot_1e9", "Gaussian $w^T\\Sigma w$, $10^{-9}$", C["g9"]),
               ("bits_emp_test", "Largest value seen (test)", C["emp"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, rows, title in [(axes[0], full, "Full dot product (K = 784 / 256 / 128)"),
                            (axes[1], tiles, "K = 32 tiles (Quiz 2 setting)")]:
        x = np.arange(len(rows))
        wbar = 0.16
        for i, (key, lab, col) in enumerate(methods):
            vals = [r[key] for r in rows]
            bars = ax.bar(x + (i - 2) * wbar, vals, wbar * 0.88, color=col, label=lab)
            for bx, val in zip(bars, vals):
                ax.text(bx.get_x() + bx.get_width() / 2, val + 0.3, str(val),
                        ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"{r['layer']}\nK={r['K']}" for r in rows])
        ax.set_title(title)
        ax.set_ylim(0, 29)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accumulator width (bits)")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Accumulator width needed per layer: worst case vs statistical (MNIST MLP, int8 x uint8)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8)
    save(fig, "fig1_bound_ladder")


def fig_overflow_curves(res, n_test):
    fig, axes = plt.subplots(1, len(res), figsize=(13, 3.8), sharey=True)
    for ax, r in zip(axes, res):
        s = r["summary"]
        b = np.array([c[0] for c in r["curve"]])
        emp = np.array([c[1] for c in r["curve"]])
        gau = np.array([c[2] for c in r["curve"]])
        ax.semilogy(b, np.maximum(gau, 1e-30), "-", color=C["g9"], lw=2,
                    label="Gaussian $w^T\\Sigma w$ (avg over neurons)")
        ax.semilogy(b, np.where(emp > 0, emp, np.nan), "o", ms=5, color=C["datatype"],
                    label="Measured on test images")
        res_line = 1 / (n_test * s["N"])
        ax.axhline(res_line, color=INK2, lw=0.8, ls="-.")
        ax.text(b[0] + 0.2, res_line * 3, "measurement floor 1/(images x neurons)",
                fontsize=7, color=INK2)
        for p in P_DOT:
            ax.axhline(p, color=GRID, lw=1)
            ax.text(b[-1], p * 2, f"{p:.0e}", fontsize=7, color=INK2, ha="right")
        ax.axvline(s["bits_sira"], color=C["sira"], lw=1.2, ls="--")
        ax.text(s["bits_sira"] - 0.2, 1e-26, f"SIRA\n{s['bits_sira']} b", color=INK2,
                fontsize=7.5, ha="right")
        ax.set_title(f"{s['layer']}  (K = {s['K']}, {s['N']} neurons)")
        ax.set_xlabel("accumulator width (bits)")
        ax.set_ylim(1e-30, 2)
        ax.set_xlim(b[0] - 0.5, b[-1] + 0.5)
    axes[0].set_ylabel("P(dot product overflows)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Overflow probability vs width on real MNIST accumulators "
                 "(moments from training set, counts on test set)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig2_overflow_vs_width")


def fig_tail(res):
    fig, ax = plt.subplots(figsize=(6, 4))
    zz = np.linspace(0, 12, 400)
    ax.semilogy(zz, 2 * norm.sf(zz), color=INK, lw=1.5, ls="--", label="Gaussian prediction")
    for r, col in zip(res, LAYER_C):
        z = np.abs(r["z"][np.isfinite(r["z"])]).ravel()
        z.sort()
        ccdf = 1.0 - np.arange(len(z)) / len(z)
        step = max(1, len(z) // 4000)
        ax.semilogy(np.r_[z[::step], z[-1]], np.r_[ccdf[::step], ccdf[-1]], color=col, lw=2,
                    label=f"{r['summary']['layer']} (median excess kurtosis "
                          f"{r['summary']['median_excess_kurtosis']:.1f})")
    ax.set_xlabel("|accumulator - mean| / standard deviation   (in sigmas)")
    ax.set_ylabel("fraction of dot products beyond")
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlim(0, 7)
    ax.set_title("Tail check: real accumulators vs Gaussian (test set)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_tail_vs_gaussian")


def fig_per_neuron(res):
    fig, axes = plt.subplots(1, len(res), figsize=(13, 3.6), sharey=True)
    for ax, r in zip(axes, res):
        s = r["summary"]
        order = np.argsort(r["b_sira"] + 1e-3 * r["b_g9"])
        x = np.arange(len(order))
        ax.axhline(s["bits_datatype"], color=C["datatype"], lw=2, label="Datatype")
        ax.step(x, r["b_sira"][order], where="mid", color=C["sira"], lw=2, label="SIRA $\\ell_1$")
        ax.step(x, r["b_g9"][order], where="mid", color=C["g9"], lw=2,
                label="Gaussian $10^{-9}$")
        ax.step(x, r["b_emp"][order], where="mid", color=C["emp"], lw=1.2,
                label="Largest value seen")
        ax.set_title(f"{s['layer']}  (K = {s['K']})")
        ax.set_xlabel("output neuron (sorted by SIRA width)")
        ax.set_ylim(15, 27)
    axes[0].set_ylabel("bits needed by this neuron")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Per-neuron accumulator width: the layer width is set by its worst neuron",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig4_per_neuron_widths")


def fig_quadform(res):
    fig, axes = plt.subplots(1, len(res) + 1, figsize=(15, 3.8))
    ax = axes[0]
    for r, col in zip(res, LAYER_C):
        lam = r["lam"]
        ax.semilogy(np.arange(1, len(lam) + 1), np.maximum(lam / lam[0], 1e-12), color=col,
                    lw=2, label=f"{r['summary']['layer']} (K = {r['summary']['K']})")
    ax.set_xscale("log")
    ax.set_ylim(1e-8, 2)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("$\\lambda_i / \\lambda_{max}$")
    ax.set_title("Eigenvalues of input covariance $\\Sigma$")
    ax.legend(fontsize=7.5)
    for ax, r in zip(axes[1:], res):
        sd = r["sd"]
        ok = sd > 0
        lim = [sd[ok].min() * 0.8, max(r["sd_eig"].max(), r["sd_diag"].max()) * 1.2]
        ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
        ax.loglog(sd[ok], r["sd_eig"][ok], "o", ms=3.5, color=C["eig"], alpha=0.8,
                  label="$\\sqrt{\\lambda_{max}}\\,\\|w\\|_2$ (eigenvalue bound)")
        ax.loglog(sd[ok], r["sd_diag"][ok], "o", ms=3.5, color=C["diag"], alpha=0.8,
                  label="$\\sqrt{\\sum w_i^2\\Sigma_{ii}}$ (ignore correlation)")
        ax.set_xlabel("true std $\\sqrt{w^T\\Sigma w}$")
        ax.set_title(f"{r['summary']['layer']}: std estimates per neuron")
        if ax is axes[1]:
            ax.set_ylabel("estimated std")
    h, l = axes[1].get_legend_handles_labels()
    fig.suptitle("The quadratic form $w^T\\Sigma w$ vs simpler variance estimates "
                 "(above the dashed line = overestimate, below = underestimate)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, markerscale=1.5)
    save(fig, "fig5_quadratic_form")


def fig_hist(res, acc_te):
    fig, axes = plt.subplots(1, len(res), figsize=(13, 3.6))
    for ax, r, a in zip(axes, res, acc_te):
        s = r["summary"]
        b9 = s["bits_gauss_dot_1e9"]
        lim = 2 ** (b9 - 1)
        ax.hist(a.ravel(), bins=200, range=(-lim, lim), color=C["datatype"], log=True)
        for b, col, lab in [(s["bits_emp_test"], C["emp"], "largest seen"),
                            (b9, C["g9"], "Gaussian $10^{-9}$")]:
            for sgn in (-1, 1):
                ax.axvline(sgn * 2 ** (b - 1), color=col, lw=1.5, ls="--",
                           label=f"{lab}: {b} bits" if sgn > 0 else None)
        ax.set_title(f"{s['layer']}  ({a.size:,} values)")
        ax.set_xlabel("accumulator value")
        ax.text(0.02, 0.97, f"off-scale:\nSIRA {s['bits_sira']} bits (±{2 ** (s['bits_sira'] - 1):,})\n"
                f"datatype {s['bits_datatype']} bits", transform=ax.transAxes, fontsize=7,
                va="top", color=INK2)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("count (log)")
    fig.suptitle("Distribution of every accumulator value on the 10,000 test images "
                 "(dashed = register limits)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig6_accumulator_histograms")


# ---------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    net = IntMLP()
    xtr, ytr, xte, yte = load_mnist()

    # exactness check of the float64 integer matmul on a slice
    ref = xte[:300].astype(np.int64) @ net.W[0].astype(np.int64).T
    assert np.array_equal(ref, int_matmul(xte[:300], net.W[0])), "int matmul not exact"

    print("Integer inference on 60k train + 10k test images ...")
    _, ins_tr, acc_tr = net.forward(xtr, keep=True)
    pred_te, ins_te, acc_te = net.forward(xte, keep=True)
    acc_q = float((net.predict(xte) == yte).mean())
    np.savez_compressed(OUT / "acc_test.npz",
                        **{n: a.astype(np.int32) for n, a in zip(net.names, acc_te)},
                        labels=yte)
    print(f"  int8 test accuracy {acc_q:.4f}; saved all test accumulators to acc_test.npz")

    res, tiles, tile_rows, neurons, curves = [], [], [], [], []
    for l, name in enumerate(net.names):
        print(f"Analysing {name} ...")
        r = analyse_layer(name, net.W[l], ins_tr[l], acc_tr[l], ins_te[l], acc_te[l])
        ts, trows = analyse_tiles(name, net.W[l], ins_tr[l], ins_te[l])
        res.append(r)
        tiles.append(ts)
        tile_rows += trows
        neurons += r["neurons"]
        curves += [dict(layer=name, width=c[0], measured_per_dot=c[1], gaussian_per_dot=c[2],
                        measured_per_image=c[3], gaussian_union_per_image=c[4])
                   for c in r["curve"]]

    layers = [r["summary"] for r in res]
    write_csv(OUT / "layer_summary.csv", layers)
    write_csv(OUT / "tile32_summary.csv", tiles)
    write_csv(OUT / "tile32_detail.csv", tile_rows)
    write_csv(OUT / "per_neuron.csv", neurons)
    write_csv(OUT / "overflow_curve.csv", curves)

    print("Plotting ...")
    fig_ladder(layers, tiles)
    fig_overflow_curves(res, len(xte))
    fig_tail(res)
    fig_per_neuron(res)
    fig_quadform(res)
    fig_hist(res, acc_te)

    # ---- console + markdown summary ------------------------------------
    hdr = ("| layer | K | datatype | SIRA l1 | Gauss 1e-15/dot | Gauss 1e-9/dot | "
           "Gauss 1e-9/inference | largest seen (test) |")
    sep = "|---|---|---|---|---|---|---|---|"
    md = ["# Week 2 results: real accumulators of the int8 MNIST MLP", "",
          f"Int8 test accuracy: {acc_q:.4f}. Widths in bits (two's complement), "
          "layer width = worst neuron.", "", "## Full dot product", "", hdr, sep]
    for s in layers:
        md.append(f"| {s['layer']} | {s['K']} | {s['bits_datatype']} | {s['bits_sira']} | "
                  f"{s['bits_gauss_dot_1e15']} | {s['bits_gauss_dot_1e9']} | "
                  f"{s['bits_gauss_inf_1e9']} | {s['bits_emp_test']} |")
    md += ["", "## K = 32 tiles (Quiz 2 setting)", "",
           "| layer | tiles | datatype | SIRA l1 | Gauss 1e-15/dot | Gauss 1e-9/dot | largest seen (test) |",
           "|---|---|---|---|---|---|---|"]
    for s in tiles:
        md.append(f"| {s['layer']} | {s['n_tiles']} | {s['bits_datatype']} | {s['bits_sira']} | "
                  f"{s['bits_gauss_dot_1e15']} | {s['bits_gauss_dot_1e9']} | {s['bits_emp_test']} |")
    md += ["", "## Quadratic-form diagnostics", "",
           "| layer | max rel. error w^T Sigma w vs measured variance | lambda_max share of trace | "
           "median sqrt(diag)/true std | median sqrt(lambda_max)*||w||/true std | "
           "median SIRA half-range / std | median excess kurtosis |",
           "|---|---|---|---|---|---|---|"]
    for s in layers:
        md.append(f"| {s['layer']} | {s['relerr_quadform_var']:.1e} | {s['top1_eig_share']:.2f} | "
                  f"{s['diag_over_full_median']:.2f} | {s['eig_over_full_median']:.2f} | "
                  f"{s['sira_over_sd_median']:.1f} | {s['median_excess_kurtosis']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nAll outputs in {OUT}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
