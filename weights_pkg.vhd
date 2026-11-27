--------------------------------------------------------------------------------
-- weights_pkg.vhd
-- Fixed (pre-trained) weights of the inference-only network nn_top.
-- (The trainable networks get their topology and initial weights from
--  datasets_pkg.)
--
-- Inference network used by nn_top (fixed weights, Q4.4)
--      2 inputs -> 2 hidden (ReLU) -> 1 output (ReLU), solves XOR exactly:
--        h0 = relu(x0 + x1)
--        h1 = relu(x0 + x1 - 1)
--        y  = relu(h0 - 2*h1)
--
-- Weight layout: W(neuron * N_INPUTS + input), index from 0.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

package weights_pkg is

  constant NN_INPUTS  : positive := 2;
  constant NN_HIDDEN  : positive := 2;
  constant NN_OUTPUTS : positive := 1;

  constant L1_WEIGHTS : data_array_t(0 to NN_INPUTS * NN_HIDDEN - 1) := (
    q(1.0), q(1.0),         -- hidden neuron 0
    q(1.0), q(1.0)          -- hidden neuron 1
  );
  constant L1_BIASES  : data_array_t(0 to NN_HIDDEN - 1) := (
    q(0.0), q(-1.0)
  );
  constant L2_WEIGHTS : data_array_t(0 to NN_HIDDEN * NN_OUTPUTS - 1) := (
    q(1.0), q(-2.0)         -- output neuron 0
  );
  constant L2_BIASES  : data_array_t(0 to NN_OUTPUTS - 1) := (
    0 => q(0.0)
  );

end package weights_pkg;
