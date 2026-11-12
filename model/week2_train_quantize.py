#!/usr/bin/env python3
"""
EE5609 Project 10 - Week 2, step 1: train and quantize the MNIST model
-----------------------------------------------------------------------
1. Trains a float MLP 784 -> 256 -> 128 -> 10 (ReLU) on MNIST with plain NumPy
   (Adam, mini-batches). No PyTorch needed, fully reproducible from the seed.
2. Post-training quantization (our own, no library):
     weights      -> int8 [-127, 127], one scale per output neuron
     activations  -> uint8 [0, 255]; layer-1 input = raw pixels, hidden scales
                     calibrated on the training set (99.99th percentile)
   The accumulator holds only the integer dot product w^T x; bias and scales
   are applied after the MAC (see common.py).
3. Reports float vs int8 test accuracy and saves
     model/mlp_float.npz, model/mlp_int8.npz, results/week2/model_summary.txt

Run (from ~/ee5609):  python model/week2_train_quantize.py          (~1-2 min)
"""

import argparse
import time

import numpy as np

from common import (A_QMAX, MODEL_DIR, RESULTS_DIR, W_QMAX, IntMLP, load_mnist)

SIZES = [784, 256, 128, 10]


# ---------------------------------------------------------------------------
# Float MLP in NumPy
# ---------------------------------------------------------------------------
def init_params(rng):
    params = []
    for k, n in zip(SIZES[:-1], SIZES[1:]):
        W = rng.normal(0, np.sqrt(2.0 / k), (n, k)).astype(np.float32)   # He init
        params.append([W, np.zeros(n, np.float32)])
    return params


def forward(params, x):
    acts = [x]
    for l, (W, b) in enumerate(params):
        z = acts[-1] @ W.T + b
        acts.append(np.maximum(z, 0) if l < len(params) - 1 else z)
    return acts


def train(x, y, epochs, batch, lr, seed):
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    m = [[np.zeros_like(p) for p in layer] for layer in params]
    v = [[np.zeros_like(p) for p in layer] for layer in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    n = len(x)
    for ep in range(epochs):
        lr_ep = lr * (0.1 if ep >= int(0.75 * epochs) else 1.0)   # one step decay
        perm = rng.permutation(n)
        loss_sum = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            acts = forward(params, xb)
            logits = acts[-1]
            logits = logits - logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            loss_sum += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
            g = p
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            t += 1
            for l in range(len(params) - 1, -1, -1):
                W, _ = params[l]
                gW = g.T @ acts[l]
                gb = g.sum(0)
                if l > 0:
                    g = (g @ W) * (acts[l] > 0)
                for j, gp in enumerate((gW, gb)):
                    m[l][j] = b1 * m[l][j] + (1 - b1) * gp
                    v[l][j] = b2 * v[l][j] + (1 - b2) * gp * gp
                    mh = m[l][j] / (1 - b1 ** t)
                    vh = v[l][j] / (1 - b2 ** t)
                    params[l][j] -= lr_ep * mh / (np.sqrt(vh) + eps)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss {loss_sum / n:.4f}")
    return params


def float_accuracy(params, x, y):
    return float((forward(params, x)[-1].argmax(1) == y).mean())


# ---------------------------------------------------------------------------
# Post-training quantization
# ---------------------------------------------------------------------------
def quantize(params, x_cal_u8, pct):
    """x_cal_u8: uint8 calibration images. Calibrates each hidden activation
    scale on the outputs of the already-quantized previous layers."""
    out = {"n_layers": len(params)}
    a = x_cal_u8
    s_in = 1.0 / 255.0                     # pixel p means p/255 in the float model
    for l, (W, b) in enumerate(params):
        sw = np.abs(W).max(axis=1) / W_QMAX
        sw[sw == 0] = 1.0
        Wq = np.clip(np.rint(W / sw[:, None]), -W_QMAX, W_QMAX).astype(np.int8)
        out[f"W{l}"], out[f"sw{l}"], out[f"b{l}"] = Wq, sw.astype(np.float64), b.astype(np.float64)
        out[f"s_in{l}"] = s_in
        if l < len(params) - 1:
            acc = np.rint(a.astype(np.float64) @ Wq.astype(np.float64).T)
            z = np.maximum(acc * (sw * s_in) + b, 0)
            s_out = np.percentile(z, pct) / A_QMAX
            out[f"s_out{l}"] = s_out
            a = np.clip(np.rint(z / s_out), 0, A_QMAX).astype(np.uint8)
            s_in = s_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=5609)
    ap.add_argument("--calib-pct", type=float, default=99.99,
                    help="percentile of ReLU outputs mapped to 255")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "week2"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    xtr_u8, ytr, xte_u8, yte = load_mnist()
    xtr = xtr_u8.astype(np.float32) / 255.0
    xte = xte_u8.astype(np.float32) / 255.0
    print(f"MNIST: train {xtr.shape}, test {xte.shape}")

    print(f"Training float MLP {SIZES} ...")
    t0 = time.time()
    params = train(xtr, ytr, args.epochs, args.batch, args.lr, args.seed)
    acc_f = float_accuracy(params, xte, yte)
    print(f"  done in {time.time() - t0:.0f} s, float test accuracy {acc_f:.4f}")
    np.savez(MODEL_DIR / "mlp_float.npz",
             **{f"W{l}": W for l, (W, _) in enumerate(params)},
             **{f"b{l}": b for l, (_, b) in enumerate(params)})

    print("Quantizing (int8 weights per-channel, uint8 activations) ...")
    q = quantize(params, xtr_u8, args.calib_pct)
    np.savez(MODEL_DIR / "mlp_int8.npz", **q)
    net = IntMLP()
    acc_q = float((net.predict(xte_u8) == yte).mean())
    print(f"  int8 test accuracy {acc_q:.4f}  (float {acc_f:.4f}, "
          f"drop {100 * (acc_f - acc_q):.2f} points)")

    lines = [f"MLP {SIZES}, ReLU, NumPy Adam, {args.epochs} epochs, batch {args.batch}, "
             f"lr {args.lr}, seed {args.seed}",
             f"Float test accuracy : {acc_f:.4f}",
             f"Int8  test accuracy : {acc_q:.4f}",
             "Quantization: int8 weights [-127,127] per output neuron; uint8 activations "
             f"[0,255]; hidden scales = {args.calib_pct} percentile of ReLU output / 255",
             ""]
    for l in range(net.n_layers):
        W = net.W[l].astype(np.int64)
        lines.append(f"{net.names[l]}: K={W.shape[1]:4d} N={W.shape[0]:4d}  "
                     f"|w| mean {np.abs(W).mean():6.2f}  zeros {100 * (W == 0).mean():5.1f}%  "
                     f"s_in {net.s_in[l]:.3e}"
                     + (f"  s_out {net.s_out[l]:.3e}" if l < net.n_layers - 1 else ""))
    (out_dir / "model_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {MODEL_DIR / 'mlp_float.npz'}, {MODEL_DIR / 'mlp_int8.npz'}, "
          f"{out_dir / 'model_summary.txt'}")


if __name__ == "__main__":
    main()
