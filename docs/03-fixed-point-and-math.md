# 3. Fixed-point arithmetic and the training math

## 3.1 Number formats

`Qm.n` means a two's-complement number with `m` integer bits (including the sign) and `n` fractional bits. Its value is `integer / 2^n`.

| Name | Bits | Format | Range | Step | Used for |
|---|---|---|---|---|---|
| `data_t` | 8 | Q4.4 | −8.0 … 7.9375 | 1/16 | Inputs, targets, activations, forward weights and biases |
| `acc_t` | 24 | Q16.8 | ±32768 | 1/256 | MAC accumulator (product of two Q4.4 numbers is Q.8) |
| `master_t` | 16 | Q4.12 | −8.0 … 7.99976 | 1/4096 | Trainable master copy of every weight and bias |
| `delta_t` | 16 | Q8.8 | −128 … 127.996 | 1/256 | Deltas, activation derivatives |

Some example values:

| Real | Q4.4 integer | Q4.12 integer | Q8.8 integer |
|---|---|---|---|
| 1.0 | 16 | 4096 | 256 |
| 0.5 | 8 | 2048 | 128 |
| −0.45 | −7 (= −0.4375) | −1843 | −115 |
| 0.0625 | 1 | 256 | 16 |

Constants are written as `q(0.5)` (Q4.4) or `qm(0.5)` (Q4.12). The conversion functions round to the nearest value and saturate (`nn_pkg.to_fixed`).

## 3.2 Rounding and saturation

Every operation that drops bits uses **round half up** (`round_shift`, `nn_pkg.vhd:104`):

```
round_shift(x, n) = floor( (x + 2^(n-1)) / 2^n )        e.g. 24/16 = 1.5 -> 2,   -24/16 = -1.5 -> -1
```

Every operation that drops integer bits **saturates** (`saturate`, `nn_pkg.vhd:89`). A value that is too large becomes the largest representable value instead of wrapping around, so a large activation never flips sign.

## 3.3 The forward pass of one neuron

```
acc     = Σ x_i · w_i                       Q4.4 · Q4.4 = Q.8, summed in the 24-bit acc_t
s       = acc + bias · 16                   bias aligned to Q.8
z       = saturate( round_shift(s, 4), 8 )  back to Q4.4          (output_stage)
y       = f(z)                              ReLU: max(0, z)      Sigmoid: LUT[z]
```

The sigmoid LUT holds `round(16 / (1 + e^(−z/16)))` for all 256 patterns of `z`. Its outputs are the 17 values 0/16 … 16/16:

- z ≤ −3.44 gives exactly 0;
- z ≥ 3.44 gives exactly 1.

The ReLU output saturates at 7.9375.

## 3.4 Why the weights have two copies

Training changes a weight by `lr · δ · a`, and these steps are usually **much smaller than 1/16**, the step of Q4.4. If the weights were stored in Q4.4, almost every update would round to zero and nothing would be learned.

Therefore:

- Every weight is stored as a **Q4.12 master copy** (`wm`, `bm` in `nn_trainer`). Its step is 1/4096, so small updates accumulate.
- The forward pass uses the master **rounded to Q4.4**: `to_data(w) = saturate(round_shift(w, 8), 8)`. The layers stay 8-bit and cheap.
- Only the backward pass reads the masters directly. It uses them in `BP_ACC` and updates them in `UPD`.

This is the same idea as mixed-precision training on GPUs: a low-precision forward pass and high-precision weight accumulation.

## 3.5 Back-propagation, derived

Notation:

- layer `l` maps level `l` to level `l+1`;
- `a[l]` holds the values of level `l` (`a[0] = x`, `a[L] = y`);
- `z[l]` is the pre-activation of layer `l`;
- `W[l][k][j]` is the weight from `a[l]_j` to neuron `k`.

**Forward:** `z[l]_k = Σ_j W[l][k][j] · a[l]_j + B[l]_k`, and `a[l+1]_k = f(z[l]_k)`.

**Delta.** Define the delta of neuron k in layer l as the gradient of the loss E with respect to its pre-activation: `d[l]_k = ∂E/∂z[l]_k`. Then the chain rule gives:

```
∂E/∂W[l][k][j] = d[l]_k · a[l]_j            ∂E/∂B[l]_k = d[l]_k                  (1)
d[l-1]_j      = ( Σ_k W[l][k][j] · d[l]_k ) · f'(z[l-1]_j)                       (2)
```

- (1) is the weight gradient. It is the product of the neuron's delta and the value on that weight's input.
- (2) carries the error one layer **down**. Each hidden neuron collects the deltas of every neuron it feeds, weighted by the connecting weights, then multiplies the sum by its own slope.

**Output delta.** The output delta depends on the loss:

| Output | Loss | `d[L-1]_k` |
|---|---|---|
| Sigmoid | Cross-entropy `−t·ln y − (1−t)·ln(1−y)` | **`y_k − t_k`** |
| ReLU | Squared error `½(y − t)²` | `(y_k − t_k) · f'(z)` |

For the sigmoid, `∂E/∂y = (y−t) / (y(1−y))` and `∂y/∂z = y(1−y)`. The `y(1−y)` terms cancel. This matters a lot in 8-bit hardware: the LUT gives *exactly* 0 or 1 for |z| ≥ 3.44, so `y(1−y)` would be exactly 0. A neuron that is confidently wrong would then never get a gradient and would stay wrong forever. With cross-entropy the gradient is `y − t`, which is largest precisely when the output is confidently wrong.

**Derivative from the output.** The derivative is computed from `y`, so `z` never has to be stored (`activation_deriv`):

- ReLU: `f'(z) = 1` if `y > 0`;
- Sigmoid: `f'(z) = y(1−y)`.

**SGD update** with learning rate `lr = 2^−LR_SHIFT`:

```
W[l][k][j] ← W[l][k][j] − lr · d[l]_k · a[l]_j          B[l]_k ← B[l]_k − lr · d[l]_k
```

A power-of-two learning rate turns the multiplication by lr into a shift.

**Order.** Equation (2) needs the weights `W[l]` *from before* their update. `nn_trainer` therefore computes **all** deltas first, from the top layer down, and only then updates every weight.

## 3.6 Every hardware step with its formats

| Step (state) | Operation | Format in → out | Rounding |
|---|---|---|---|
| `OUT_DELTA` | `(y − t) · f'` | Q.4 · Q.8 = Q.12 → **Q8.8** | `round_shift(·, 4)`, saturate 16 bit |
| `BP_ACC` | `bacc += W · d` | Q.12 · Q.8 = Q.20, 40-bit accumulator | none (exact) |
| `BP_MUL` (1) | `bacc → Q8.8` | Q.20 → Q.8 | `round_shift(·, 12)`, saturate 16 bit |
| `BP_MUL` (2) | `· f'(a)` | Q.8 · Q.8 = Q.16 → **Q8.8** | `round_shift(·, 8)`, saturate 16 bit |
| `UPD` weight | `W − d · a · 2^−lr` | Q.8 · Q.4 = **Q.12**, the master format | `round_shift(·, LR_SHIFT)`, saturate 16 bit |
| `UPD` bias | `B − d · 1.0 · 2^−lr` | Q.8 · Q.4(16) = Q.12 | same |

The formats are chosen so that `δ · a` lands **directly in Q4.12**: Q.8 × Q.4 = Q.12, with no extra shift. `nn_trainer` checks this with the assertion `MASTER_FRAC = DELTA_FRAC + FRAC_BITS`.

## 3.7 Worked example: one training step

This is XOR on the 2-4-1 network, sample `x = (1, 1)`, target `t = 0`, from the initial weights in `datasets_pkg`, with lr = 1/2. The numbers come from the bit-exact model and are given as integer (real value).

**1. Forward weights** (masters rounded to Q4.4):

- `W1 = [(8,6), (−7,9), (10,−6), (5,10)]`, i.e. (0.5,0.375), (−0.4375,0.5625), …
- `B1 = (2, 1, 1, −2)`
- `W2 = (7, −6, 6, −5)`
- `B2 = 0`

**2. Hidden layer.** For neuron 0: `acc = 16·8 + 16·6 = 224`, and `+ 2·16 = 256`, so `z = (256+8)/16 = 16` (1.0) and ReLU gives `h0 = 16`.

All four: `h = (16, 3, 5, 13)` = (1.0, 0.1875, 0.3125, 0.8125).

**3. Output.**

- `acc = 16·7 + 3·(−6) + 5·6 + 13·(−5) = 59`, so `z = (59+8)/16 = 4` (0.25).
- `y = LUT[4] = 9` (0.5625). This is wrong, since the target is 0.

**4. Output delta.** `d2 = round((9 − 0) · 256 / 16) = 144` (0.5625 = y − t).

**5. Hidden deltas.** For neuron 0:

- `bacc = W2_master[0] · d2 = 1843 · 144 = 265392`;
- `round(265392 / 4096) = 65`;
- `· f'(h0) = 65 · 256 / 256 = 65` (0.254).

All four: `d1 = (65, −58, 50, −43)`. The hidden neurons whose outgoing weight is positive are blamed in proportion to that weight.

**6. Updates** (lr = 1/2):

- `W2[0] = 1843 − round(144·16 / 2) = 1843 − 1152 = 691`
- `B2 = 0 − 1152 = −1152`
- `W1[0][0] = 2048 − round(65·16 / 2) = 2048 − 520 = 1528`
- … and the same for all 12 weights and 5 biases.

**7. Result.** The same input now gives `y = 5` (0.3125), down from 0.5625, which is closer to the target 0.

The hardware produces exactly these integers. `tb_nn_trainer` checks every weight after every step against `tb_ref_pkg.ref_train_step`, which implements this table in plain integer arithmetic.

## 3.8 Accuracy limits

| Limit | Effect | Mitigation used here |
|---|---|---|
| Q4.4 forward weights | A weight below 1/32 in magnitude acts as 0 | Initial magnitudes ≥ 0.125 (`init_weights`) |
| Q4.4 inputs | Inputs must be multiples of 1/16 within ±8 | Datasets are built on exact grids (CIRCLE step 0.5) |
| Q8.8 deltas | Deltas below 1/512 vanish deep in the network | Moderate depth; cross-entropy output delta stays large |
| ReLU saturation at 7.94 | Large activations clip | Moderate weights and learning rates |
| Dead ReLUs | A neuron with `z ≤ 0` for every sample never learns | Small positive hidden bias at start (0.0625) |
| 17-level sigmoid | Output resolution 1/16 | Success criterion uses thresholds and tolerances in Q4.4 LSBs |
| 40-bit back-sum | At most 256 terms per sum | `BACC_W = 40` |
