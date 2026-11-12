"""
EE5609 Project 10 - shared helpers for Week 2 onwards.

  * project paths
  * MNIST loading (local .gz files, downloaded once if missing)
  * bit-width helpers
  * the integer (int8 weight / uint8 activation) MLP and its exact forward pass

Quantization scheme (fixed for the whole project)
-------------------------------------------------
  weights      : signed int8, narrow range [-127, 127], symmetric,
                 one scale per output neuron (per-channel)
  activations  : unsigned uint8 [0, 255]. Layer-1 input is the raw MNIST pixel
                 (already uint8); hidden activations are ReLU outputs.
  accumulator  : holds ONLY the integer dot product  s_j = w_j^T x  (int x int).
                 Bias and rescaling are applied after the MAC, as in SIRA where
                 scale and bias are merged into one multiply and one add.
  requantize   : a = clip(round(relu(s_w*s_in*s + b) / s_out), 0, 255)

With this scheme the datatype (Quiz 2 style) worst case for a K-term dot product
is K * 127 * 255; for K = 32 that is 1,036,320 < 2^20, i.e. 21 bits - the same
21 bits as Quiz 2.
"""

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # ~/ee5609
DATA_DIR = ROOT / "data" / "mnist"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

W_QMAX = 127          # int8 narrow range
A_QMAX = 255          # uint8 activations

# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------
_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]


def _download(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    for base in _MIRRORS:
        try:
            print(f"  downloading {base}{name}")
            urllib.request.urlretrieve(base + name, dest)
            return dest
        except Exception as e:                       # try the next mirror
            print(f"    failed ({e})")
    raise RuntimeError(f"Could not download {name}. Put the four MNIST .gz "
                       f"files in {DATA_DIR} manually.")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist():
    """Returns (x_train uint8 [60000,784], y_train, x_test uint8 [10000,784], y_test)."""
    out = {}
    for key, name in _FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            path = _download(name)
        out[key] = _read_idx(path)
    return (out["train_x"].reshape(-1, 784), out["train_y"].astype(np.int64),
            out["test_x"].reshape(-1, 784), out["test_y"].astype(np.int64))


# ---------------------------------------------------------------------------
# Bit widths
# ---------------------------------------------------------------------------
def bits_signed(lo, hi):
    """Smallest B with -2^(B-1) <= lo and hi <= 2^(B-1)-1 (two's complement)."""
    lo, hi = int(lo), int(hi)
    need = max(hi.bit_length() if hi > 0 else 0,
               (-lo - 1).bit_length() if lo < 0 else 0)
    return need + 1


def reg_range(b):
    return -(2 ** (b - 1)), 2 ** (b - 1) - 1


# ---------------------------------------------------------------------------
# Integer MLP
# ---------------------------------------------------------------------------
def int_matmul(x_u8, w_i8):
    """Exact integer x @ w.T. Uses float64 BLAS: every product and partial sum is
    an integer below 2^53, so the result is exact (checked in week2 script)."""
    return np.rint(x_u8.astype(np.float64) @ w_i8.astype(np.float64).T).astype(np.int64)


class IntMLP:
    """Loaded from model/mlp_int8.npz (written by week2_train_quantize.py)."""

    def __init__(self, path=MODEL_DIR / "mlp_int8.npz"):
        z = np.load(path)
        self.n_layers = int(z["n_layers"])
        self.W = [z[f"W{l}"] for l in range(self.n_layers)]          # int8 [N, K]
        self.sw = [z[f"sw{l}"] for l in range(self.n_layers)]        # [N]
        self.b = [z[f"b{l}"] for l in range(self.n_layers)]          # float [N]
        self.s_in = [float(z[f"s_in{l}"]) for l in range(self.n_layers)]
        self.s_out = [float(z[f"s_out{l}"]) for l in range(self.n_layers - 1)]
        self.names = [f"fc{l + 1}" for l in range(self.n_layers)]

    def requant(self, l, acc):
        z = acc * (self.sw[l] * self.s_in[l]) + self.b[l]
        return np.clip(np.rint(np.maximum(z, 0) / self.s_out[l]), 0, A_QMAX).astype(np.uint8)

    def forward(self, x_u8, keep=False, chunk=10_000):
        """Returns logits-accumulator of the last layer, and if keep=True the
        per-layer (inputs uint8, accumulators int64) lists."""
        ins = [[] for _ in range(self.n_layers)]
        accs = [[] for _ in range(self.n_layers)]
        last = []
        for i in range(0, len(x_u8), chunk):
            a = x_u8[i:i + chunk]
            for l in range(self.n_layers):
                acc = int_matmul(a, self.W[l])
                if keep:
                    ins[l].append(a)
                    accs[l].append(acc)
                if l < self.n_layers - 1:
                    a = self.requant(l, acc)
            last.append(acc)
        out = np.concatenate(last)
        if keep:
            return out, [np.concatenate(v) for v in ins], [np.concatenate(v) for v in accs]
        return out

    def predict(self, x_u8):
        acc = self.forward(x_u8)
        return np.argmax(acc * self.sw[-1] + self.b[-1] / self.s_in[-1], axis=1)
