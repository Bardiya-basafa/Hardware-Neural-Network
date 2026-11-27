# 5. Verification and simulation

## 5.1 Strategy

The hardware is never only checked for whether it "seems to learn". Every level is compared **bit for bit** with an independent software model written in plain integer arithmetic (`tb/tb_ref_pkg.vhd`). That model does not use `nn_pkg`'s rounding or saturation functions, so a bug in those helpers cannot hide by being in both the design and its reference.

```
unit level      activation, neuron, layer        vs. ref_relu / ref_sigmoid / ref_neuron
network level   nn_top (fixed weights)          vs. known XOR truth table
training level  nn_trainer, every SGD step      vs. ref_train_step   (all weights, bit-exact)
system level    nn_autotrain, whole run         vs. software run     (epoch, score, weights, answers)
```

## 5.2 Test benches

All of them are self-checking. They report `PASS` or `FAIL` (severity `failure`) and stop with `std.env.finish`, so no run time has to be given.

| Test bench | Generics | What is checked |
|---|---|---|
| `tb_activation` | – | ReLU, sigmoid and both derivatives for **all 256** inputs, against a `real`-number model |
| `tb_neuron` | `N = 5`, `N_TESTS = 200` | 200 random vectors (including saturation) against `ref_neuron`; `done` exactly N+2 clocks after `start` |
| `tb_layer` | – | 100 random vectors on a ReLU and a sigmoid layer; latency N_INPUTS+2; `x_in` is changed right after `start` to prove that the layer latched it |
| `tb_nn_top` | – | The 4 XOR patterns through the fixed-weight network |
| `tb_nn_trainer` | `DATASET`, `LR_SHIFT`, `MAX_EPOCHS`, `CHECK` | Trains a dataset. After **every** training step, every weight and bias is compared with `ref_train_step`. Measures latency. Must converge. |
| `tb_nn_autotrain` | `DATASET` | Only `go` is driven. The DUT must finish in the **same epoch** as a software run of the same loop, with the same `n_correct` / `max_err`, bit-identical weights, and query answers equal to `ref_predict`. |

### The reference model (`tb/tb_ref_pkg.vhd`)

| Subprogram | Purpose |
|---|---|
| `ref_neuron(x, w, b, act)` | One neuron, integer arithmetic |
| `ref_forward(sizes, w, b, x, …, acts)` | Whole network of any depth. Rounds the Q4.12 masters to Q4.4 itself. |
| `ref_train_step(sizes, w, b, x, t, lr, …)` | Forward + backprop + SGD. Updates `w` and `b` in place. |
| `ref_predict` | Outputs of `ref_forward` |
| `ref_sat`, `ref_rshift`, `ref_deriv` | Saturation, round-half-up shift, f'(y) |

The back-sum is accumulated in `real`. Doubles are exact for integers below 2^53, and a 32-bit VHDL `integer` could overflow on `W · d` products.

## 5.3 Evidence that the checks work

**Mutation tests.** A deliberate bug was injected into a scratch copy of the current `nn_trainer`, and the checks had to catch it:

| Mutation | Caught by | When |
|---|---|---|
| ReLU derivative forced to 1 | `tb_nn_trainer` XOR, gradient check | first training step |
| Derivative of the **second** hidden layer forced to 1 (only used by networks with 2 or more hidden layers) | `tb_nn_trainer` PARITY3 and CIRCLE | first training step |
| (same mutation) | XOR | not caught, which is correct: that code path does not exist in a 1-hidden-layer network |

**Golden trace.** When `nn_trainer` was generalised from a fixed 2-layer design to any depth, all 88 weight snapshots of the XOR run were compared with the old implementation. They were identical.

**Single-layer topology.** No dataset uses a network without a hidden layer, so `(2, 1)` was checked once with a scratch test bench that is not part of the regression. It learned AND, with every step matching `ref_train_step` (0 mismatches).

**Python cross-check.** `tools/nn_model.py` is a bit-exact Python port of the reference model, used for hyper-parameter sweeps. It reproduces the GHDL epoch counts exactly for every dataset (22, 20, 32, 46, 29): run `python3 tools/nn_model.py --all`.

## 5.4 Running with GHDL (installed and tested here)

```sh
make                     # full regression: 4 unit TBs + 5 datasets x (trainer + autotrain), about 100 s
make quick               # unit TBs + XOR training, about 1 s
make tb_neuron           # one unit test bench
make train DS=PARITY3    # train one dataset (prints progress, predictions, latency)
make train DS=XOR LR=3 EPOCHS=500
make auto DS=SEG7        # self-training run of one dataset
make wave TB=tb_nn_trainer DS=PARITY3   # -> build/tb_nn_trainer.ghw (about 4 MB; CIRCLE would be far larger), open with gtkwave
make synth DS=SEG7       # GHDL synthesis of nn_top and nn_train_top
make clean
```

Or by hand:

```sh
mkdir -p build
ghdl -a --std=08 --workdir=build nn_pkg.vhd datasets_pkg.vhd weights_pkg.vhd mac_unit.vhd \
     activation_relu.vhd activation_sigmoid.vhd activation_deriv.vhd neuron.vhd layer.vhd \
     nn_top.vhd nn_trainer.vhd nn_autotrain.vhd nn_train_top.vhd tb/tb_ref_pkg.vhd tb/*.vhd
ghdl --elab-run --std=08 --workdir=build tb_nn_trainer -gDATASET=CIRCLE --ieee-asserts=disable-at-0
```

Expected end of `make` (all 14 lines must say PASS):

```
tb_activation: PASS (256 inputs)
tb_neuron: PASS (200 vectors)
tb_layer: PASS (100 vectors)
tb_nn_top: PASS
tb_nn_trainer(XOR): PASS (every weight update matched the reference model)
...
tb_nn_autotrain(SEG7): PASS (same epoch, weights and answers as the reference model)
```

## 5.5 ModelSim / Questa

The script has not been run here (no ModelSim licence on this machine). It uses only standard commands.

```
vsim -do sim/modelsim.do                                         # tb_nn_trainer, XOR
vsim -do "set DS CIRCLE; do sim/modelsim.do"
vsim -do "set TB tb_nn_autotrain; set DS SEG7; do sim/modelsim.do"
vsim -do "set TB tb_layer; do sim/modelsim.do"
```

It compiles everything with `vcom -2008`, starts `vsim -onfinish stop` so the waveform survives `std.env.finish`, and adds prepared wave groups:

- **learning progress**: `sum_abs_err` as an analog curve;
- **trainer FSM / counters**: `state`, `l_cnt`, `k_cnt`, `j_cnt`, `w_addr`;
- **values**: `lstart`, `acts`, `d`, `bacc`, `mul_a`, `mul_b`, `prod`, `wm`, `bm`;
- for `tb_nn_autotrain`: sequencer `state`, `eval_pass`, `s_cnt`, `ep_cnt`, `corr_cnt`, `err_max`.

## 5.6 Vivado (xsim)

This script has not been run here either (Vivado not installed).

```
vivado -mode gui   -source sim/vivado.tcl -tclargs tb_nn_trainer CIRCLE
vivado -mode gui   -source sim/vivado.tcl -tclargs tb_nn_autotrain PARITY3
vivado -mode batch -source sim/vivado.tcl                       # tb_nn_trainer, XOR
```

The script:

- creates `build/vivado_proj`;
- sets **every** VHDL file to *VHDL 2008*, which is required for the `integer_vector` generics, `process(all)` and unary `and`;
- sets the `DATASET` generic on the simulation fileset and runs to the end.

Because the script is untested, the quoting of the string generic (`DATASET=CIRCLE`) may need adjusting for your Vivado version, for example to `DATASET=\"CIRCLE\"`. If it is rejected, change the default of `DATASET` in the test bench instead.

The synthesis top is `nn_top`. To build the self-training system instead:

```tcl
set_property top nn_train_top [current_fileset]
set_property generic {DATASET=PARITY3} [current_fileset]
```

## 5.7 What to look at in the waveform

| Question | Signals |
|---|---|
| Is it learning? | `epoch`, `n_correct`, `max_err`, `sum_abs_err` (once per epoch) |
| Where is the trainer? | `u_dut/state` (`idle` → `fwd` → `out_delta` → `bp_acc`/`bp_mul` → `upd` → `finish`) |
| Which weight is being updated? | `u_dut/l_cnt`, `k_cnt`, `j_cnt`, `w_addr` |
| Forward values | `u_dut/acts` (level 0 = input, last level = output) |
| Layer handshakes | `u_dut/lstart` (bit l = start of layer l, last bit = forward done) |
| Back-prop | `u_dut/d`, `u_dut/bacc`, `u_dut/mul_a`, `mul_b`, `prod` |
| Weights moving | `u_dut/wm`, `u_dut/bm` (Q4.12: divide by 4096) |
| Inside one neuron | `u_dut/g_layer(0)/u_layer/g_neurons(0)/u_neuron/acc`, `state` |

A cycle-by-cycle walk through one training step is in [02-execution-flow.md](02-execution-flow.md#24-level-2-one-sample-inside-nn_trainer).

## 5.8 Known limitations of the verification

- Only GHDL was available. The ModelSim and Vivado scripts are untested, and there are no Vivado utilisation or timing numbers.
- `ghdl --synth` succeeds for both tops. That shows the code is synthesizable, not that it fits or meets timing on a particular FPGA.
- The reference model was written by the same author as the RTL. The mutation tests, the golden trace against the earlier 2-layer implementation, and the agreement with the Python port reduce the risk of a shared misunderstanding, but they do not remove it. The Python port is a translation of the same model, not an independent derivation.
