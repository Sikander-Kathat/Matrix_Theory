"""
EE5609 Project 10 - bit-true fixed-point model of the int8 MLP datapath.

This is the arithmetic the hardware actually performs, with no floating point
anywhere in the forward pass:

  1. MAC      s = sum_k w_jk * x_k      int8 x uint8 products, summed in order
              k = 0..K-1 into a B-bit two's-complement accumulator.
              Overflow behaviour (per layer):
                'wrap'      two's-complement wrap-around (plain adder)
                'saturate'  clamp to [-2^(B-1), 2^(B-1)-1] after EVERY add
                            (order dependent, simulated term by term)
  2. BIAS     z = s + b_j               b_j = round(b / (s_w * s_in)), added in
              a separate 32-bit adder after the accumulator (SIRA style).
  3. RESCALE  y = (z * M0_j + 2^(r_j-1)) >> r_j
              integer multiplier M0_j in [2^30, 2^31) and right shift r_j with
              M0_j * 2^-r_j = s_w_j * s_in / s_out  (rounding = round-half-up;
              'truncate' uses a plain arithmetic shift instead)
  4. ReLU + CLIP to uint8 [0, 255]  (hidden layers)
     last layer: argmax over z_j * L0_j, L0_j = Q31 of s_w_j / max(s_w)

The 'wide' configuration (B = 32 for every layer) can never overflow and is the
reference "original quantized implementation".
"""

import numpy as np

from common import A_QMAX, IntMLP, int_matmul

WIDE = 32


def reg_limits(b):
    return -(1 << (b - 1)), (1 << (b - 1)) - 1


def wrap(s, b):
    """Two's-complement wrap of int64 values to b bits."""
    half = 1 << (b - 1)
    return ((s + half) % (1 << b)) - half


def prefix_extremes(x_u8, w_i8, chunk=2500):
    """Running min and max of the partial sums s_1..s_K for every (image, neuron),
    plus the exact final sum. Needed for saturation and intermediate overflow."""
    n, (N, K) = len(x_u8), w_i8.shape
    pmin = np.empty((n, N), np.int64)
    pmax = np.empty((n, N), np.int64)
    W = w_i8.astype(np.int64)
    for i in range(0, n, chunk):
        x = x_u8[i:i + chunk].astype(np.int64)
        acc = np.zeros((len(x), N), np.int64)
        lo = np.zeros_like(acc)
        hi = np.zeros_like(acc)
        for k in range(K):
            acc += x[:, k:k + 1] * W[None, :, k]
            np.minimum(lo, acc, out=lo)
            np.maximum(hi, acc, out=hi)
        pmin[i:i + chunk], pmax[i:i + chunk] = lo, hi
    return pmin, pmax


def saturate_sequential(x_rows, w_rows, b):
    """Term-by-term saturating accumulation for selected (image, neuron) pairs.
    x_rows, w_rows: [n, K]. Returns the B-bit saturated final value."""
    lo, hi = reg_limits(b)
    P = x_rows.astype(np.int64) * w_rows.astype(np.int64)
    acc = np.zeros(len(P), np.int64)
    for k in range(P.shape[1]):
        acc = np.clip(acc + P[:, k], lo, hi)
    return acc


def saturate_pairs(x_u8, W, ii, jj, b, chunk=20_000):
    """saturate_sequential for (image ii, neuron jj) pairs, in memory-safe chunks."""
    out = np.empty(len(ii), np.int64)
    for s in range(0, len(ii), chunk):
        out[s:s + chunk] = saturate_sequential(x_u8[ii[s:s + chunk]], W[jj[s:s + chunk]], b)
    return out


def wrap_sequential(x_rows, w_rows, b):
    """Term-by-term wrapping accumulation (used only to check wrap())."""
    P = x_rows.astype(np.int64) * w_rows.astype(np.int64)
    acc = np.zeros(len(P), np.int64)
    for k in range(P.shape[1]):
        acc = wrap(acc + P[:, k], b)
    return acc


def requant_params(scale):
    """Real multiplier -> (M0, r) with M0 in [2^30, 2^31) and scale ~= M0 * 2^-r."""
    e = np.floor(np.log2(scale)).astype(np.int64)
    r = 30 - e
    M0 = np.rint(scale * np.exp2(r.astype(float))).astype(np.int64)
    over = M0 >= (1 << 31)
    M0[over] //= 2
    r[over] -= 1
    return M0, r


class FixedPointMLP:
    """Bit-true version of IntMLP (same int8 weights, same scales)."""

    def __init__(self, net=None):
        self.net = net or IntMLP()
        n = self.net
        self.L = n.n_layers
        self.names = n.names
        self.W = n.W
        self.bias = [np.rint(n.b[l] / (n.sw[l] * n.s_in[l])).astype(np.int64) for l in range(self.L)]
        self.M0, self.r = [], []
        for l in range(self.L - 1):
            M0, r = requant_params(n.sw[l] * n.s_in[l] / n.s_out[l])
            self.M0.append(M0)
            self.r.append(r)
        self.L0 = np.rint(n.sw[-1] / n.sw[-1].max() * (1 << 31)).astype(np.int64)
        self._cache = {}

    # -- one layer -----------------------------------------------------------
    def accumulate(self, l, x_u8, b, mode, cache_key=None):
        """B-bit accumulator output for layer l. Returns (acc, info) where info
        counts final-sum overflows and intermediate (partial-sum) overflows."""
        W = self.W[l]
        s = int_matmul(x_u8, W)                               # exact final sums
        lo, hi = reg_limits(b)
        final_ovf = (s < lo) | (s > hi)
        info = {"dots": s.size, "macs": s.size * W.shape[1],
                "final_overflows": int(final_ovf.sum())}
        if b >= WIDE:
            info["any_overflows"] = 0
            return s, info
        key = (cache_key, l) if cache_key else None
        if key and key in self._cache:
            pmin, pmax = self._cache[key]
        else:
            pmin, pmax = prefix_extremes(x_u8, W)
            if key:
                self._cache[key] = (pmin, pmax)
        any_ovf = (pmin < lo) | (pmax > hi)
        info["any_overflows"] = int(any_ovf.sum())
        if mode == "wrap":
            return wrap(s, b), info
        if mode == "saturate":
            out = s.copy()
            ii, jj = np.nonzero(any_ovf)                      # only these can differ
            if len(ii):
                out[ii, jj] = saturate_pairs(x_u8, W, ii, jj, b)
            return out, info
        raise ValueError(mode)

    def requant(self, l, acc, rounding="round"):
        z = acc + self.bias[l]
        p = z * self.M0[l]
        r = self.r[l]
        if rounding == "round":
            y = (p + (np.int64(1) << (r - 1))) >> r
        else:                                                 # truncate
            y = p >> r
        return np.clip(y, 0, A_QMAX).astype(np.uint8)

    def logits(self, acc):
        return (acc + self.bias[-1]) * self.L0

    # -- whole network ---------------------------------------------------------
    def forward(self, x_u8, widths, mode="wrap", rounding="round", keep=False,
                cache_tag=None):
        """widths: list of accumulator widths per layer (32 = wide).
        cache_tag: name of the input set; prefix sums of layer 1 are cached
        under it (layer-1 inputs never change)."""
        a = x_u8
        infos, ins, accs = [], [], []
        pristine = True              # inputs still identical to the wide baseline
        for l in range(self.L):
            key = cache_tag if pristine else None
            acc, info = self.accumulate(l, a, widths[l], mode, key)
            infos.append(info)
            changed = info["any_overflows"] if mode == "saturate" else info["final_overflows"]
            pristine = pristine and changed == 0
            if keep:
                ins.append(a)
                accs.append(acc)
            if l < self.L - 1:
                a = self.requant(l, acc, rounding)
        pred = np.argmax(self.logits(acc), axis=1)
        if keep:
            return pred, infos, ins, accs
        return pred, infos


def self_test(fp, x_u8, n=200, b=14):
    """Checks wrap() against term-by-term wrapping and that saturation only
    changes values whose partial sums overflowed."""
    W = fp.W[0]
    x = x_u8[:n]
    s = int_matmul(x, W)
    ii, jj = np.meshgrid(np.arange(n), np.arange(W.shape[0]), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    seq = wrap_sequential(x[ii], W[jj], b).reshape(n, -1)
    assert np.array_equal(seq, wrap(s, b)), "wrap mismatch"
    sat_full = saturate_sequential(x[ii], W[jj], b).reshape(n, -1)
    sat, _ = fp.accumulate(0, x, b, "saturate")
    assert np.array_equal(sat_full, sat), "saturate mismatch"
    return True
