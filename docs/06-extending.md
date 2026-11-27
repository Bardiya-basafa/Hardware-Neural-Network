# 6. Extending the design

## 6.1 A different topology

Topologies are just `integer_vector` generics, so no RTL changes are needed:

```vhdl
u_net : entity work.nn_trainer
  generic map (
    LAYER_SIZES => (4, 16, 8, 8, 3),               -- 3 hidden layers
    HIDDEN_ACT  => "RELU",
    OUTPUT_ACT  => "SIGMOID",
    LR_SHIFT    => 3,
    W_INIT      => init_weights((4, 16, 8, 8, 3), 42),
    B_INIT      => init_biases((4, 16, 8, 8, 3)))
  port map (...);
```

- `w_out` and `b_out` get `n_weights` and `n_neurons` entries.
- The latency follows the formula in [01-architecture.md](01-architecture.md#17-nn_trainer--trainable-network-of-any-depth).
- To use the new topology for a dataset, change `ds_layers` in `datasets_pkg`.

**Limits:**

- `LAYER_SIZES` needs at least 2 entries. `(N, M)` is a single-layer network: `OUT_DELTA` goes straight to `UPD`.
- At most 256 inputs per neuron (`ACC_GUARD = 8`).
- At most 256 neurons in the layer above any hidden neuron (`BACC_W = PROD_W + 8`).

## 6.2 A deeper fixed-weight inference network

`nn_top` chains two `layer` instances by hand. To add a layer:

- put `L3_WEIGHTS` / `L3_BIASES` into `weights_pkg`;
- add a third `layer` with `start => l2_done` and `x_in => l2_out`.

Alternatively, take the `g_layer` generate loop from `nn_trainer` (`nn_trainer.vhd:175`), which chains any number of layers automatically.

## 6.3 Other activation functions

1. Add an entity with the same ports as `activation_relu` (`x_in`, `y_out : data_t`).
2. Add a `generate` branch for it in `neuron.vhd`, next to `g_relu` / `g_sigmoid`.
3. Add its derivative, expressed **in terms of the output y**, to `activation_deriv.vhd`, and the same formula to `tb_ref_pkg.ref_deriv`.

For example, tanh: `f' = 1 − y²`. In integers that is `256 − y·y` (Q.8, with y in Q.4).

## 6.4 Other number formats

All widths are constants in `nn_pkg`. The constraints are:

- `MASTER_FRAC = DELTA_FRAC + FRAC_BITS` (asserted in `nn_trainer`), so `δ·a` lands in the master format without shifting;
- `DELTA_FRAC = 2·FRAC_BITS` (asserted in `activation_deriv`), so `y·(1−y)` lands in the delta format;
- the sigmoid LUT is built for Q4.4 only (asserted). For another format, regenerate the table with `round(2^F / (1 + e^(−x/2^F)))`;
- `tb_ref_pkg` hard-codes the Q4.4 / Q4.12 / Q8.8 constants (16, 256, 4096, the shifts 4, 8, 12) and must be updated in step.

## 6.5 Larger networks: where the limits are

The current design keeps every weight in flip-flops, because each `layer` reads the weights of **all** its neurons in parallel every clock. That is fine for hundreds of parameters, like the datasets here. For thousands, two changes would be needed:

1. **Weights in block RAM.** Store W column-wise (all neurons' weights for input `idx` in one wide word), so a layer reads one word per clock. The trainer's `UPD` would then write back one weight at a time with a read-modify-write.
2. **Parallel backprop.** One multiplier makes a training step O(#weights) clocks. Several multipliers, one per neuron of a layer, would make `UPD` and `BP_ACC` O(#inputs) per layer, at the cost of more DSPs.

## 6.6 Mini-batches or momentum

Both fit into the existing structure:

- **Mini-batch.** Add a gradient-accumulator array beside `wm`. In `UPD` accumulate instead of subtract, and apply the sum every B samples.
- **Momentum.** Add a velocity array `v`. `UPD` computes `v ← v·β − lr·g` and then `w ← w + v`, with β a power of two so it can be a shift.

Each would need a matching change in `tb_ref_pkg.ref_train_step` to keep the bit-exact check working.

## 6.7 Putting it on a board

`nn_train_top` (generic `DATASET`) is the synthesis top. A minimal board hookup:

| Port | Board |
|---|---|
| `clk`, `rst` | Clock, reset button |
| `go` | Push button (debounce it, and make it a one-clock pulse) |
| `training`, `finished`, `converged` | LEDs |
| `epoch` | 7-segment display or LEDs |
| `q_x` / `q_start` → `q_y` / `q_done` | Switches and a button for the input, LEDs for the output, or a UART bridge |

The `data_array_t` ports would usually be flattened to `std_logic_vector` in a small board-specific wrapper, because pin constraints are written per bit.
