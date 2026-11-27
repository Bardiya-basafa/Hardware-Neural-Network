#!/usr/bin/env python3
"""Bit-exact software model of nn_trainer, for fast hyper-parameter sweeps.

This is a Python port of tb/tb_ref_pkg.vhd (ref_forward / ref_train_step) plus
the datasets, initialisation and success criterion of datasets_pkg.vhd and
nn_pkg.vhd. It reproduces the GHDL results integer for integer (same epoch
count, same weights), but runs a training experiment in seconds instead of
minutes, so it is the tool for trying topologies, learning rates and seeds
before changing datasets_pkg.vhd.

Examples
  python3 tools/nn_model.py XOR                          # dataset defaults -> 22 epochs
  python3 tools/nn_model.py PARITY3 --layers 3,6,6,1 --lr 2
  python3 tools/nn_model.py SEG7 --seeds 1,3,5,7,11      # robustness sweep over seeds
  python3 tools/nn_model.py --all                        # every dataset with its defaults

Anything found here must still be confirmed with the real hardware:
  make train DS=<name>
"""
import argparse
import math

# ----------------------------------------------------------------------------
# integer helpers (same as tb_ref_pkg)
# ----------------------------------------------------------------------------
def sat(v, bits):
    hi, lo = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    return hi if v > hi else lo if v < lo else v

def rshift(v, n):                       # round(v / 2^n), half up
    return v if n == 0 else (v + 2 ** (n - 1)) // 2 ** n

def sigmoid_lut(z):                     # activation_sigmoid.vhd
    return int(math.floor(16.0 / (1.0 + math.exp(-z / 16.0)) + 0.5))

def act(z, a):
    return max(z, 0) if a == "RELU" else sigmoid_lut(z)

def deriv(y, a):                        # activation_deriv.vhd, Q.8
    return (256 if y > 0 else 0) if a == "RELU" else y * (16 - y)

# ----------------------------------------------------------------------------
# nn_pkg: initialisation
# ----------------------------------------------------------------------------
def init_weights(s, seed):
    n = sum(s[l] * s[l + 1] for l in range(len(s) - 1))
    x, r = seed % 65537, []
    for _ in range(n):
        x = (75 * x + 74) % 65537
        mag = 512 + (x % 32) * 64       # 0.125 .. 0.61 in Q4.12
        r.append(-mag if (x // 32) % 2 else mag)
    return r

def init_biases(s):
    r = []
    for l in range(1, len(s)):
        r += [256 if l < len(s) - 1 else 0] * s[l]   # hidden 0.0625, output 0
    return r

# ----------------------------------------------------------------------------
# tb_ref_pkg: forward pass and one SGD step
# ----------------------------------------------------------------------------
def offsets(s):
    wb, bb, ab = [0], [0], [0]
    for l in range(1, len(s)):
        wb.append(wb[-1] + s[l - 1] * s[l]); bb.append(bb[-1] + s[l]); ab.append(ab[-1] + s[l - 1])
    return wb, bb, ab

def forward(s, w, b, x, hid="RELU", out="SIGMOID"):
    L = len(s) - 1
    wb, bb, ab = offsets(s)
    a = list(x)
    for l in range(L):
        for k in range(s[l + 1]):
            acc = sum(a[ab[l] + j] * sat(rshift(w[wb[l] + k * s[l] + j], 8), 8) for j in range(s[l]))
            acc += sat(rshift(b[bb[l] + k], 8), 8) * 16
            a.append(act(sat((acc + 8) // 16, 8), out if l == L - 1 else hid))
    return a

def train_step(s, w, b, x, t, lr, hid="RELU", out="SIGMOID"):
    L = len(s) - 1
    wb, bb, ab = offsets(s)
    a = forward(s, w, b, x, hid, out)
    d = [0] * (sum(s) - s[0])
    for k in range(s[L]):
        y = a[ab[L] + k]
        fd = 256 if out == "SIGMOID" else deriv(y, out)
        d[bb[L - 1] + k] = sat(rshift((y - t[k]) * fd, 4), 16)
    for l in range(L - 1, 0, -1):                       # top down, old weights
        for j in range(s[l]):
            acc = sum(w[wb[l] + k * s[l] + j] * d[bb[l] + k] for k in range(s[l + 1]))
            d[bb[l - 1] + j] = sat(rshift(sat(rshift(acc, 12), 16) * deriv(a[ab[l] + j], hid), 8), 16)
    for l in range(L):
        for k in range(s[l + 1]):
            for j in range(s[l]):
                i = wb[l] + k * s[l] + j
                w[i] = sat(w[i] - rshift(d[bb[l] + k] * a[ab[l] + j], lr), 16)
            i = bb[l] + k
            b[i] = sat(b[i] - rshift(d[bb[l] + k] * 16, lr), 16)

# ----------------------------------------------------------------------------
# datasets_pkg
# ----------------------------------------------------------------------------
SEG7 = ["1111110", "0110000", "1101101", "1111001", "0110011",
        "1011011", "1011111", "1110000", "1111111", "1111011"]
BLOB_C = [(-8, -5), (8, -5), (0, 8)]
BLOB_OFS = [(0, 0), (6, 2), (-5, 5), (2, -6), (-6, -2), (8, -1), (-1, -8), (-3, 7)]
XOR_W = [2048, 1638, -1843, 2253, 2458, -1434, 1229, 2662, 1843, -1638, 1434, -1229]
XOR_B = [410, 205, 205, -410, 0]

# name: (layers, lr_shift, seed, max_epochs, err_tol)
DEFAULTS = {
    "XOR":     ([2, 4, 1],     1, None, 3000, 4),
    "PARITY3": ([3, 8, 8, 1],  3, 11,   500,  8),
    "CIRCLE":  ([2, 8, 8, 1],  3, 7,    500,  8),
    "BLOBS3":  ([2, 6, 3],     2, 3,    500,  8),
    "SEG7":    ([7, 12, 10],   3, 7,    500,  8),
}

def dataset(name):
    if name in ("XOR", "PARITY3"):
        n = 2 if name == "XOR" else 3
        X = [[((s >> (n - 1 - i)) & 1) * 16 for i in range(n)] for s in range(2 ** n)]
        T = [[(bin(s).count("1") % 2) * 16] for s in range(2 ** n)]
    elif name == "CIRCLE":
        X = [[-28 + 8 * (s % 8), -28 + 8 * (s // 8)] for s in range(64)]
        T = [[16 if x * x + y * y < 512 else 0] for x, y in X]
    elif name == "BLOBS3":
        X = [[BLOB_C[s // 8][0] + BLOB_OFS[s % 8][0], BLOB_C[s // 8][1] + BLOB_OFS[s % 8][1]] for s in range(24)]
        T = [[16 if k == s // 8 else 0 for k in range(3)] for s in range(24)]
    elif name == "SEG7":
        X = [[16 * int(c) for c in p] for p in SEG7]
        T = [[16 if k == s else 0 for k in range(10)] for s in range(10)]
    else:
        raise SystemExit("unknown dataset " + name)
    return X, T

def argmax(v):
    m = 0
    for n in range(1, len(v)):
        if v[n] > v[m]:
            m = n
    return m

def correct(y, t):
    return (y[0] >= 8) == (t[0] >= 8) if len(y) == 1 else argmax(y) == argmax(t)

# ----------------------------------------------------------------------------
def train(name, layers, lr, seed, max_epochs, tol):
    """Returns the epoch in which the criterion is met, or None."""
    X, T = dataset(name)
    if seed is None:                                     # XOR: hand-picked weights
        w, b = list(XOR_W), list(XOR_B)
    else:
        w, b = init_weights(layers, seed), init_biases(layers)
    for e in range(1, max_epochs + 1):
        for x, t in zip(X, T):
            train_step(layers, w, b, x, t, lr)
        n_ok, err = 0, 0
        for x, t in zip(X, T):
            y = forward(layers, w, b, x)[-layers[-1]:]
            n_ok += correct(y, t)
            err = max(err, max(abs(p - q) for p, q in zip(y, t)))
        if n_ok == len(X) and err <= tol:
            return e
    return None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="XOR")
    ap.add_argument("--layers", help="topology, e.g. 3,8,8,1 (first/last must match the dataset)")
    ap.add_argument("--lr", type=int, help="LR_SHIFT (learning rate 2^-n)")
    ap.add_argument("--seeds", help="comma separated init_weights seeds")
    ap.add_argument("--epochs", type=int, help="max epochs")
    ap.add_argument("--all", action="store_true", help="run every dataset with its defaults")
    a = ap.parse_args()
    for name in (DEFAULTS if a.all else [a.dataset]):
        layers, lr, seed, epochs, tol = DEFAULTS[name]
        layers = [int(v) for v in a.layers.split(",")] if a.layers else layers
        lr = a.lr if a.lr is not None else lr
        epochs = a.epochs or epochs
        seeds = [int(v) for v in a.seeds.split(",")] if a.seeds else [seed]
        if name == "XOR" and (a.layers or a.seeds):
            seeds = [v if v is not None else 1 for v in seeds]
        for sd in seeds:
            e = train(name, layers, lr, sd, epochs, tol)
            tag = "hand-picked" if sd is None else "seed %d" % sd
            print("%-8s %-12s lr=2^-%d %-12s -> %s" % (name, "-".join(map(str, layers)), lr, tag,
                  "converged after %d epochs" % e if e else "NOT converged in %d epochs" % epochs))

if __name__ == "__main__":
    main()
