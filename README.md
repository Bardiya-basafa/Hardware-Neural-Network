# Hardware Neural Network in VHDL

A modular fixed-point neural network for FPGAs that **trains itself on-chip**. Forward pass, back-propagation and the SGD weight update are all done in hardware. The network can have any depth and width. Five datasets are built in, from XOR to 10-class digit recognition, and every one is learned in simulation with each weight update checked bit-exactly against a software model.

All sources are **VHDL-2008**.

## At a glance

```
                        ┌──────────────── nn_train_top (synthesis top, generic DATASET) ───────────────┐
 go ──────────────────► │ nn_autotrain:  sample ROM ─► epoch sequencer ─► score ─► converged? ───────► │──► finished, converged, epoch
                        │                    │                                                         │
                        │                    ▼                                                         │
 q_x, q_start ────────► │ nn_trainer:  x ─► layer 0 ─► layer 1 ─► ... ─► y          (forward)          │──► q_y, q_done
                        │                  ▲  weights (Q4.12 masters, rounded to Q4.4)                 │
                        │                  └── 1 shared multiplier: deltas, back-sums, SGD (backward)  │
                        └──────────────────────────────────────────────────────────────────────────────┘
 layer = N neurons in parallel;  neuron = mac_unit (acc += x·w) + bias + ReLU / sigmoid-LUT
```

| Dataset | Network | Learned in | Clocks for the whole on-chip run |
|---|---|---|---|
| XOR | 2-4-1 | 22 epochs | 4 599 |
| PARITY3 (3-bit parity) | 3-8-8-1 (2 hidden layers) | 20 epochs | 41 301 |
| CIRCLE (inside/outside, 64 points) | 2-8-8-1 (2 hidden layers) | 32 epochs | 507 937 |
| BLOBS3 (3 classes, 24 points) | 2-6-3 | 46 epochs | 106 031 |
| SEG7 (7-segment → digit 0–9) | 7-12-10 | 29 epochs | 121 830 |

## Quick start (GHDL)

```sh
make                      # full regression: 14 self-checking runs (6 test benches), all must print PASS (~100 s)
make quick                # unit tests + XOR training (~1 s)
make train DS=PARITY3     # watch one dataset being learned
make auto  DS=SEG7        # the self-training hardware, checked against the software run
make wave TB=tb_nn_trainer DS=PARITY3 && gtkwave build/tb_nn_trainer.ghw
make synth DS=PARITY3     # GHDL synthesis check
```

For ModelSim run `vsim -do sim/modelsim.do`; for Vivado run `vivado -mode gui -source sim/vivado.tcl -tclargs tb_nn_trainer CIRCLE`. Details are in [docs/05](docs/05-verification-and-simulation.md).

## Where things start and end

1. **Start.** `make` analyzes all sources (packages first) and runs a test bench.
2. **The test bench** (`tb/tb_nn_trainer.vhd`) releases reset, then loops over epochs. For each sample it pulses `start`:
   - with `train = 1` (learn);
   - with `train = 0` (evaluate).
3. **One sample** in `nn_trainer` goes `IDLE → FWD → OUT_DELTA → BP_ACC/BP_MUL → UPD → FINISH`. On `done`, the weights are already updated.
4. **End.** When every sample is correct within tolerance, the test bench prints the predictions and `PASS`, then calls `std.env.finish`. On the FPGA, `nn_autotrain` raises `finished` and `converged`.

The full road map, from the command down to one multiply in one clock, with a cycle-accurate timeline and `file:line` pointers, is in **[docs/02-execution-flow.md](docs/02-execution-flow.md)**.

## Files

| File | Purpose |
|---|---|
| `nn_pkg.vhd` | Types, fixed-point formats, rounding/saturation, topology helpers, weight initialisation |
| `datasets_pkg.vhd` | The five datasets, hyper-parameters, success criterion |
| `weights_pkg.vhd` | Fixed weights of the inference-only XOR network |
| `mac_unit.vhd` | Multiply-accumulate |
| `activation_relu.vhd`, `activation_sigmoid.vhd` | Activations (the sigmoid is a 256-entry LUT) |
| `activation_deriv.vhd` | f'(y) for backprop |
| `neuron.vhd` | One neuron (sequential MAC, latency N+2) |
| `layer.vhd` | N neurons in parallel (latency N_INPUTS+2), weights through ports |
| `nn_top.vhd` | Fixed-weight 2-layer inference network |
| `nn_trainer.vhd` | Trainable network of any depth: forward + backprop + SGD |
| `nn_autotrain.vhd` | Sample ROM + epoch sequencer: trains without outside help |
| `nn_train_top.vhd` | Synthesis top of the self-training system |
| `tb/` | Reference model + 6 self-checking test benches |
| `sim/` | ModelSim and Vivado scripts |
| `tools/nn_model.py` | Bit-exact Python model of the trainer for fast hyper-parameter sweeps |
| `docs/` | Full documentation |

## Number formats (short)

| Format | Used for |
|---|---|
| Q4.4 (8 bit) | Inputs, activations, forward weights |
| Q4.12 (16 bit) | Trainable master weights |
| Q8.8 (16 bit) | Deltas |

Rounding is round-half-up, and every narrowing step saturates. The reasons are in [docs/03](docs/03-fixed-point-and-math.md).

## Documentation

See **[docs/README.md](docs/README.md)**:

- architecture and module reference;
- execution and data-flow road map;
- fixed-point math and backprop derivation, with a worked example;
- datasets and results;
- verification and tool usage;
- extending the design.
