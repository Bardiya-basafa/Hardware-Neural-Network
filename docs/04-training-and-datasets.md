# 4. Training: datasets, procedure and results

## 4.1 The five problems

All datasets are defined in `datasets_pkg.vhd` with integer arithmetic. They are chosen by name through the generic `DATASET` of `tb_nn_trainer`, `tb_nn_autotrain` and `nn_train_top`.

| Name | Topology | Hidden layers | Parameters | Samples | Task | Why it is interesting |
|---|---|---|---|---|---|---|
| `XOR` | 2-4-1 | 1 | 12 w + 5 b | 4 | `x0 xor x1` | The classic problem that is not linearly separable |
| `PARITY3` | 3-8-8-1 | **2** | 96 w + 17 b | 8 | `x0 xor x1 xor x2` | Harder than XOR. It exercises back-propagation *between two hidden layers*. |
| `CIRCLE` | 2-8-8-1 | **2** | 88 w + 17 b | 64 | Inside / outside a circle | Curved decision boundary, many samples, deep network |
| `BLOBS3` | 2-6-3 | 1 | 12+18 w + 9 b | 24 | Which of 3 point clouds | **Multi-class** (3 one-hot outputs, argmax), touching classes |
| `SEG7` | 7-12-10 | 1 | 84+120 w + 22 b | 10 | 7-segment pattern → digit 0–9 | **10 classes**, wide layers, 226 parameters |

Every network uses ReLU hidden layers and sigmoid outputs. Targets are 0.0 or 1.0 (Q4.4: 0 or 16).

### XOR and PARITY3

The inputs count in binary over every combination of 0 and 1, and the target is the parity:

```
PARITY3   x2 x1 x0 | t        XOR   x1 x0 | t
          0  0  0  | 0              0  0  | 0
          0  0  1  | 1              0  1  | 1
          0  1  0  | 1              1  0  | 1
          0  1  1  | 0              1  1  | 0
          1  0  0  | 1
          ...      | ...
          1  1  1  | 1
```

### CIRCLE

The 64 inputs form an 8 × 8 grid at −1.75, −1.25, …, 1.75. Every point is exact in Q4.4. The target is 1 where `x² + y² < 2` (24 points, `#`) and 0 elsewhere (40 points, `.`):

```
 y= 1.75   . . . . . . . .
    1.25   . . . # # . . .
    0.75   . . # # # # . .
    0.25   . # # # # # # .
   -0.25   . # # # # # # .
   -0.75   . . # # # # . .
   -1.25   . . . # # . . .
   -1.75   . . . . . . . .
          x=-1.75 ... 1.75
```

### BLOBS3

There are three classes. Each has 8 points: a centre plus a shared set of 8 offsets of up to ±0.5.

| Class | Centre | One-hot target |
|---|---|---|
| A | (−0.5, −0.31) | (1, 0, 0) |
| B | (0.5, −0.31) | (0, 1, 0) |
| C | (0, 0.5) | (0, 0, 1) |

The centres are only about 1 apart, so the clouds touch. A first version with centres 2 apart was learned in 2 epochs, which made it too easy.

### SEG7

The 7 inputs are the segments a–g of a 7-segment display. There are 10 outputs, one per digit (one-hot):

```
 aaa      0: abcdef      5: a cd fg
f   b     1:  bc         6: a cdefg
 ggg      2: ab de g     7: abc
e   c     3: abcd  g     8: abcdefg
 ddd      4:  bc  fg     9: abcd fg
```

## 4.2 The training procedure

One **epoch** in both `tb_nn_trainer` and `nn_autotrain` is:

1. **Training pass.** Present every sample once in ROM order with `train = '1'`. Each sample is one complete forward + backprop + SGD step. This is plain stochastic gradient descent with batch size 1.
2. **Evaluation pass.** Present every sample once with `train = '0'`. This is inference only, and no weights change.
3. **Score.** Count the correct samples and the largest error:

   | Outputs | A sample is correct when |
   |---|---|
   | 1 | `(y ≥ 0.5) = (t ≥ 0.5)` |
   | > 1 | `argmax(y) = argmax(t)` (the first maximum wins on ties) |

4. **Stop** when `n_correct ≥ ds_goal` (all samples) **and** `max |y − t| ≤ ds_err_tol` over all outputs:
   - XOR uses a tolerance of 0.25;
   - the others use 0.5.

   Training also stops at `ds_max_epochs`, which counts as a failure.

The criterion is implemented once, in `datasets_pkg` (`is_correct`, `max_abs_err`). The test benches (software) and `nn_autotrain` (hardware) both use it, so they must stop in the same epoch. `tb_nn_autotrain` checks that they do.

## 4.3 Hyper-parameters

| Name | lr | Initial weights | Max epochs | Tolerance |
|---|---|---|---|---|
| XOR | 1/2 | Hand-picked (asymmetric) | 3000 | 0.25 |
| PARITY3 | 1/8 | `init_weights(·, 11)` | 500 | 0.5 |
| CIRCLE | 1/8 | `init_weights(·, 7)` | 500 | 0.5 |
| BLOBS3 | 1/4 | `init_weights(·, 3)` | 500 | 0.5 |
| SEG7 | 1/8 | `init_weights(·, 7)` | 500 | 0.5 |

These were chosen from sweeps over topology, learning rate and LCG seed, run with **`tools/nn_model.py`**. That script is a bit-exact Python port of `tb_ref_pkg` and of the datasets. It runs a whole training run in well under a second, where GHDL takes seconds to minutes.

- Before the sweeps, it was checked against GHDL on XOR (22 epochs) and CIRCLE (32).
- After the settings were chosen, it also matched the GHDL results for PARITY3 (20), BLOBS3 (46) and SEG7 (29).

`python3 tools/nn_model.py --all` prints all five.

Each chosen configuration converges for **every** seed that was tried, so it is not a lucky pick:

| Dataset / configuration | Seeds tried | Converged | Epochs |
|---|---|---|---|
| PARITY3 3-8-8-1, lr 1/8 | 8 | 8 | 20 – 113 |
| SEG7 7-12-10, lr 1/8 | 5 | 5 | 29 – 89 |
| BLOBS3 2-6-3, lr 1/4 | 4 | 4 | 44 – 46 |
| CIRCLE 2-8-8-1, lr 1/8 | 8 | 8 | 25 – 95 |

Rejected configurations:

- **PARITY3 with 3-6-6-1 at lr 1/4** got stuck for 4 of 8 seeds. One run sat at 6/8 correct with several outputs frozen at 0.6875, which is typical of dead ReLUs.
- **lr 1/2** failed for every seed on PARITY3 and SEG7, because the steps were too large.

## 4.4 Results (GHDL, all with the per-step gradient check passing)

| Dataset | Converged after | Training steps | Clocks / step | Clocks / inference | Whole run in `nn_autotrain` | At 100 MHz |
|---|---|---|---|---|---|---|
| XOR | 22 epochs | 88 | 37 | 11 | 4 599 clocks | 46 µs |
| PARITY3 | 20 epochs | 160 | 228 | 26 | 41 301 clocks | 0.41 ms |
| CIRCLE | 32 epochs | 2 048 | 219 | 25 | 507 937 clocks | 5.1 ms |
| BLOBS3 | 46 epochs | 1 104 | 79 | 13 | 106 031 clocks | 1.1 ms |
| SEG7 | 29 epochs | 290 | 392 | 24 | 121 830 clocks | 1.2 ms |

Notes:

- "At 100 MHz" assumes the design meets 100 MHz. That has not been checked, because no Vivado timing run was possible here.
- The whole-run clock count includes the evaluation passes. The formula is in [01-architecture.md](01-architecture.md#18-nn_autotrain--the-fpga-trains-itself).

XOR after training (`make train DS=XOR`):

```
x = 0.0 0.0  ->  y = 0.25     (target 0.0)
x = 0.0 1.0  ->  y = 0.9375   (target 1.0)
x = 1.0 0.0  ->  y = 0.9375   (target 1.0)
x = 1.0 1.0  ->  y = 0.0625   (target 0.0)
```

The learning rate changes the XOR result as follows (`make train DS=XOR LR=n`):

| LR | Epochs |
|---|---|
| 1 (1/2) | 22 |
| 2 (1/4) | 46 |
| 3 (1/8) | 101 |

## 4.5 Watching it learn

- `tb_nn_trainer` prints `epoch N: correct c/N, max|y-t| = …, sum|y-t| = …` every 50 epochs and at convergence.
- In the waveform:
  - `epoch`, `n_correct`, `max_err` and `sum_abs_err` (Q4.4 LSBs) change once per epoch;
  - ModelSim shows `sum_abs_err` as an analog curve (`sim/modelsim.do`);
  - `u_dut/wm` shows the weights moving.
- `make train DS=CIRCLE CHECK=false` skips the per-step reference comparison. This barely changes the run time (CIRCLE takes about 30 s either way), because simulating the hardware dominates. It is mainly useful while experimenting with a modified trainer whose arithmetic intentionally differs from the reference model.

## 4.6 Adding your own dataset

1. In `datasets_pkg.vhd`, add the name to every `ds_*` function:
   - `ds_layers` (topology) and `ds_samples`;
   - the sample generator in `ds_x` and the targets in `ds_t`, both integer Q4.4 values, sample-major;
   - `ds_w_init` (usually `init_weights(ds_layers(name), seed)`);
   - `ds_lr_shift` and `ds_max_epochs`.

   `ds_b_init`, `ds_goal` and `ds_err_tol` have defaults.
2. Add the name to `DATASETS` in the `Makefile`.
3. Explore first with the fast model:
   - add the dataset to `DEFAULTS` and `dataset()` in `tools/nn_model.py`;
   - sweep with, for example, `python3 tools/nn_model.py MYSET --layers 4,8,8,2 --lr 3 --seeds 1,2,3,5,7`;
   - choose a configuration that converges for most seeds.
4. `make train DS=MYSET` confirms it on the hardware, with the per-step gradient check. If it does not converge, try another seed, a larger hidden layer or a smaller learning rate (larger `LR`). Watch `max_err` and `n_correct`.
5. `make auto DS=MYSET` verifies the self-training hardware on it.
6. Build it into the FPGA: `nn_train_top` with generic `DATASET => "MYSET"`.

Constraints:

- inputs must be Q4.4 (multiples of 1/16 within ±8);
- targets should be 0 / 1 for sigmoid outputs;
- at most 256 terms in any sum (`ACC_GUARD`, `BACC_W`).
