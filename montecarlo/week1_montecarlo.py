#!/usr/bin/env python3
"""
EE5609 Project 10 - Statistical Accumulator Sizing
Week 1: worst-case bound vs Monte Carlo vs analytical tail bounds
------------------------------------------------------------------
Setting (Quiz 2, Q4): dot product of K = 32 signed 8-bit weights and 8-bit
activations, accumulated in a B-bit two's-complement register.

What this script does
  1. Reproduces the worst-case width (21 bits; 20 bits with narrow-range int8).
  2. For three input scenarios, simulates millions of dot products (Monte Carlo)
     and measures how often the final sum would NOT fit in B bits.
  3. Computes three analytical tail estimates for the same probability:
       - Gaussian / CLT      : approximation, uses mean and variance
       - Hoeffding (rigorous): uses only the range of each product
       - Bernstein (rigorous): uses range AND variance (much tighter)
  4. Reports the width each method needs for an overflow target of
     1e-9 and 1e-15 per dot product, saves a plot and a CSV.

Only the FINAL sum is checked: with wrap-around two's-complement arithmetic,
intermediate overflows cancel out if the final sum fits.

Run:  python week1_montecarlo.py                  (2 million trials, ~1 min)
      python week1_montecarlo.py --trials 200000  (quick test)
"""

import argparse
import csv
import math
import os

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")           # save to file, no window needed
import matplotlib.pyplot as plt

K = 32                          # dot-product length (Quiz 2)
W_MIN, W_MAX = -128, 127        # signed int8 weights
X_MIN, X_MAX = -128, 127        # activations use the full int8 range in the
                                # uniform case; ReLU cases only use [0, 127]
WIDTHS = range(8, 23)           # accumulator widths to evaluate
TARGETS = (1e-9, 1e-15)         # per-dot-product overflow targets


# ---------------------------------------------------------------------------
# 1. Worst-case width
# ---------------------------------------------------------------------------
def bits_signed(lo, hi):
    """Smallest B such that [lo, hi] fits in a B-bit two's-complement register."""
    b = 1
    while not (-(2 ** (b - 1)) <= lo and hi <= 2 ** (b - 1) - 1):
        b += 1
    return b


def worst_case():
    prods = [W_MIN * X_MIN, W_MIN * X_MAX, W_MAX * X_MIN, W_MAX * X_MAX]
    full = bits_signed(K * min(prods), K * max(prods))
    narrow = bits_signed(K * (-127 * 127), K * (127 * 127))   # -128 never used
    return full, narrow


# ---------------------------------------------------------------------------
# 2. Input scenarios (integer samplers). Edit these to try other assumptions.
# ---------------------------------------------------------------------------
def q_clip(v, lo, hi):
    return np.clip(np.rint(v), lo, hi).astype(np.int32)


def uniform_int8(rng, shape):
    """Pessimistic: every int8 value equally likely, signed activations."""
    w = rng.integers(W_MIN, W_MAX + 1, size=shape, dtype=np.int32)
    x = rng.integers(X_MIN, X_MAX + 1, size=shape, dtype=np.int32)
    return w, x


def gauss_relu(rng, shape):
    """Typical DNN layer: bell-shaped weights (127 ~ 4 sigma),
    ReLU activations (half-Gaussian, ~50% exact zeros, 127 ~ 3 sigma)."""
    w = q_clip(rng.normal(0, 127 / 4, shape), W_MIN, W_MAX)
    x = q_clip(np.maximum(rng.normal(0, 127 / 3, shape), 0), 0, X_MAX)
    return w, x


def laplace_relu(rng, shape):
    """Heavy-tailed weights (Laplacian, 127 ~ 8 scale units) with ReLU
    activations; many small weights, a few large ones."""
    w = q_clip(rng.laplace(0, 127 / 8, shape), W_MIN, W_MAX)
    x = q_clip(np.maximum(rng.normal(0, 127 / 3, shape), 0), 0, X_MAX)
    return w, x


SCENARIOS = {
    "Uniform int8 (pessimistic)": (uniform_int8, (X_MIN, X_MAX)),
    "Gaussian weights + ReLU":    (gauss_relu,   (0, X_MAX)),
    "Laplacian weights + ReLU":   (laplace_relu, (0, X_MAX)),
}


# ---------------------------------------------------------------------------
# 3. Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(sampler, trials, rng, chunk=200_000):
    """Returns the final sums S (int64) and running moments of one product term."""
    sums = np.empty(trials, dtype=np.int64)
    n_terms, s1, s2 = 0, 0.0, 0.0
    done = 0
    while done < trials:
        n = min(chunk, trials - done)
        w, x = sampler(rng, (n, K))
        p = (w * x).astype(np.int64)
        sums[done:done + n] = p.sum(axis=1)
        n_terms += p.size
        s1 += p.sum(dtype=np.float64)
        s2 += (p.astype(np.float64) ** 2).sum()
        done += n
    mu = s1 / n_terms
    var = s2 / n_terms - mu ** 2
    return sums, mu, var


def overflow_prob_empirical(sums, b):
    lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    return np.mean((sums < lo) | (sums > hi))


# ---------------------------------------------------------------------------
# 4. Analytical tail estimates for P(S outside [lo, hi])
#    S = sum of K independent products X_i, each with mean mu, variance var,
#    and range [pmin, pmax].
# ---------------------------------------------------------------------------
def gaussian_tail(b, mu, var):
    m, s = K * mu, math.sqrt(K * var)
    lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    # continuity correction: S is integer-valued
    return norm.sf((hi + 0.5 - m) / s) + norm.cdf((lo - 0.5 - m) / s)


def _one_sided(u, f):
    return 1.0 if u <= 0 else min(1.0, f(u))


def hoeffding_tail(b, mu, pmin, pmax):
    m, R2 = K * mu, (pmax - pmin) ** 2
    lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    f = lambda u: math.exp(-2 * u * u / (K * R2))
    return min(1.0, _one_sided(hi - m, f) + _one_sided(m - lo, f))


def bernstein_tail(b, mu, var, pmin, pmax):
    m, V = K * mu, K * var
    M = max(pmax - mu, mu - pmin)          # bound on |X_i - mu|
    lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    f = lambda u: math.exp(-(u * u / 2) / (V + M * u / 3))
    return min(1.0, _one_sided(hi - m, f) + _one_sided(m - lo, f))


def width_for(target, tail_fn, cap):
    """Smallest width whose overflow probability is <= target (never above
    the worst-case width, which has probability exactly 0)."""
    for b in WIDTHS:
        if b >= cap or tail_fn(b) <= target:
            return min(b, cap)
    return cap


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=5609)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    wc, wc_narrow = worst_case()
    print(f"Worst-case width, K={K}, int8 x int8 : {wc} bits")
    print(f"Worst-case width, narrow-range int8   : {wc_narrow} bits")
    print(f"Monte Carlo trials per scenario       : {args.trials:,} "
          f"(smallest observable probability ~ {1 / args.trials:.0e})\n")

    rows = []
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 4.6), sharey=True)
    for ax, (name, (sampler, (xlo, xhi))) in zip(axes, SCENARIOS.items()):
        sums, mu, var = monte_carlo(sampler, args.trials, rng)
        corners = [W_MIN * xlo, W_MIN * xhi, W_MAX * xlo, W_MAX * xhi]
        pmin, pmax = min(corners), max(corners)

        g = lambda b: gaussian_tail(b, mu, var)
        h = lambda b: hoeffding_tail(b, mu, pmin, pmax)
        br = lambda b: bernstein_tail(b, mu, var, pmin, pmax)

        bs = list(WIDTHS)
        emp = [overflow_prob_empirical(sums, b) for b in bs]

        print(f"--- {name} ---")
        print(f"  per-product mean {mu:9.1f}, std {math.sqrt(var):9.1f};  "
              f"sum std {math.sqrt(K * var):9.0f};  "
              f"largest |S| seen {np.abs(sums).max():,} "
              f"(needs {bits_signed(sums.min(), sums.max())} bits)")
        print(f"  {'width':>5} {'MonteCarlo':>11} {'Gaussian':>11} "
              f"{'Bernstein':>11} {'Hoeffding':>11}")
        for b, e in zip(bs, emp):
            print(f"  {b:5d} {e:11.2e} {g(b):11.2e} {br(b):11.2e} {h(b):11.2e}")
            rows.append([name, b, e, g(b), br(b), h(b)])
        for t in TARGETS:
            print(f"  width for {t:.0e}: Gaussian {width_for(t, g, wc)}, "
                  f"Bernstein {width_for(t, br, wc)}, "
                  f"Hoeffding {width_for(t, h, wc)}  (worst case {wc})")
        print()

        floor = 1e-30
        emp_plot = [e if e > 0 else np.nan for e in emp]
        ax.semilogy(bs, emp_plot, "o", ms=6, label="Monte Carlo")
        ax.semilogy(bs, [max(g(b), floor) for b in bs], "-", label="Gaussian (CLT)")
        ax.semilogy(bs, [max(br(b), floor) for b in bs], "--", label="Bernstein (rigorous)")
        ax.semilogy(bs, [max(h(b), floor) for b in bs], ":", label="Hoeffding (rigorous)")
        for t in TARGETS:
            ax.axhline(t, color="grey", lw=0.8)
            ax.text(bs[0], t * 2, f"{t:.0e}", fontsize=8, color="grey")
        ax.axvline(wc, color="red", lw=1)
        ax.text(wc - 0.2, 1e-26, f"worst case\n{wc} bits", color="red",
                ha="right", fontsize=8)
        ax.axhline(1 / args.trials, color="C0", lw=0.6, ls="-.")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("accumulator width (bits)")
        ax.set_ylim(floor, 2)
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("P(final sum overflows)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Overflow probability vs accumulator width, K={K}, int8 x int8 "
                 f"(dash-dot = Monte Carlo resolution 1/N)", fontsize=11)
    fig.tight_layout()

    png = os.path.join(args.out, "week1_overflow_vs_width.png")
    fig.savefig(png, dpi=150)
    csv_path = os.path.join(args.out, "week1_overflow_vs_width.csv")
    with open(csv_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["scenario", "width", "monte_carlo", "gaussian",
                     "bernstein", "hoeffding"])
        cw.writerows(rows)
    print(f"Saved {png}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
