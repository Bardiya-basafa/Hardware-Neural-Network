# CLAUDE.md

This file guides Claude Code (claude.ai/code) when it works in this repository.

## What this is

A fixed-point neural network written in VHDL-2008 that trains itself on an FPGA. It does the forward pass, backpropagation and the SGD weight update in hardware, and the network can have any depth. Only **GHDL** is installed on this machine. The ModelSim script (`sim/modelsim.do`) and the Vivado script (`sim/vivado.tcl`) exist but have never been run.

Full documentation is in `docs/`. Start with `docs/02-execution-flow.md`, which follows a run from start to end.

## Commands

```sh
make                        # full regression: 14 runs, every one must print PASS (~100 s)
make quick                  # unit test benches + XOR training (~1 s)
make tb_layer               # one unit test bench (tb_activation, tb_neuron, tb_layer, tb_nn_top)
make train DS=PARITY3 [LR=3] [EPOCHS=200] [CHECK=false]   # tb_nn_trainer on one dataset
make auto DS=SEG7           # tb_nn_autotrain: the self-training run, checked against software
make wave TB=tb_nn_trainer DS=PARITY3   # writes build/<TB>.ghw for GTKWave (CIRCLE makes a very large file)
make synth DS=PARITY3       # ghdl --synth of nn_top and nn_train_top
python3 tools/nn_model.py --all         # fast bit-exact model; use it for hyper-parameter sweeps
```

- Datasets: `XOR`, `PARITY3`, `CIRCLE`, `BLOBS3`, `SEG7`.
- `make` only re-analyses when a source file changes. If GHDL reports "file has changed and must be reanalysed", run `make analyze`.

## Architecture

- **Module chain:** `nn_train_top` → `nn_autotrain` (sample ROM + epoch sequencer) → `nn_trainer` (forward + backprop + SGD) → `layer` → `neuron` → `mac_unit` + activation.
- **`nn_top`** is a separate inference-only network with fixed weights. Its weights come from `weights_pkg`.
- **Topology and parameter layout:**
  - A topology is an `integer_vector` generic, e.g. `(2, 8, 8, 1)`.
  - All parameters live in two flat arrays, with layer 0 first: weight `W(neuron*N_in + input)`.
  - The offset tables come from `nn_pkg`: `w_base`, `b_base` and `a_base`.
- **`datasets_pkg`** chooses everything by a name string: topology, samples, initial weights, learning rate and success criterion. It uses integers only, because the same code also builds the on-chip ROM.
- **`nn_trainer`'s datapath:**
  - All backprop arithmetic shares one multiplier; a state machine selects its operands each clock (`IDLE, FWD, OUT_DELTA, BP_ACC, BP_MUL, UPD, FINISH`).
  - Addresses come from incrementing registers, so there is no run-time multiply in the address path.
  - All deltas are computed before any weight is updated.

## Invariants (break these and the checks fail)

- **Number formats:**
  - forward values are Q4.4 (`data_t`);
  - master weights are Q4.12 (`master_t`), and the forward pass uses them rounded to Q4.4 with `to_data`;
  - deltas are Q8.8 (`delta_t`).

  Two relations are checked by assertions: `MASTER_FRAC = DELTA_FRAC + FRAC_BITS` and `DELTA_FRAC = 2*FRAC_BITS`.
- **Rounding:** round half up (`round_shift`), and every narrowing step saturates.
- **Sigmoid output:** the output delta is `y - t` (cross-entropy), not `(y - t)·y(1-y)`. The sigmoid look-up table saturates to exactly 0 or 1.
- **Keep the copies of the arithmetic identical.** Any change to the arithmetic in `nn_trainer` must be mirrored in `tb/tb_ref_pkg.vhd` (`ref_train_step`) and in `tools/nn_model.py`. `tb_nn_trainer` compares every weight bit for bit after every training step.
- **Shared success criterion:** `is_correct` / `max_abs_err` in `datasets_pkg` are used by both the testbench and `nn_autotrain`. `tb_nn_autotrain` requires that both stop in the same epoch.
- **File lists:** a new source file must be added in compile order (packages first) to three places — the `Makefile` (`RTL`/`TBS`), `sim/modelsim.do` and `sim/vivado.tcl`.

## VHDL / GHDL gotchas

- Always use VHDL-2008 (`--std=08`). The code relies on `integer_vector` generics, `process(all)`, unary `and` and `std.env.finish`.
- VHDL names are case-insensitive. A loop variable `n` clashes with a generic `N`, and `t` with a constant `T`.
- A `declare` block inside a subprogram is VHDL-2019. Size variables with functions instead.
- `ghdl --synth` rejects integer `abs` computed at run time. Use `numeric_std` `abs` on `signed` instead.
- Make targets are `.PHONY`, so they use static pattern rules (`$(addprefix train-,…): train-%:`), not implicit `%` rules.
- Simulations run with `--ieee-asserts=disable-at-0` to hide warnings about `'U'` values at time 0.
