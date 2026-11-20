--------------------------------------------------------------------------------
-- nn_top.vhd
-- Top level: input vector -> layer 1 -> layer 2 -> output vector.
-- Topology and weights come from weights_pkg.
--
-- Usage
--   1. put the input vector on x_in
--   2. pulse start for one clock (while done / idle)
--   3. wait for done = '1'  -> y_out holds the result
--
-- Extending the network: add NN_xxx sizes and Lk_WEIGHTS / Lk_BIASES to
-- weights_pkg, then add one more "layer" instance and chain it (start of the
-- next layer = done of the previous one, x_in = y_out of the previous one).
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity nn_top is
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    start : in  std_logic;
    x_in  : in  data_array_t(0 to NN_INPUTS - 1);
    y_out : out data_array_t(0 to NN_OUTPUTS - 1);
    done  : out std_logic
  );
end entity nn_top;

architecture rtl of nn_top is
  signal l1_out  : data_array_t(0 to NN_HIDDEN - 1);
  signal l1_done : std_logic;
begin

  u_layer1 : entity work.layer
    generic map (
      N_INPUTS   => NN_INPUTS,
      N_NEURONS  => NN_HIDDEN,
      WEIGHTS    => L1_WEIGHTS,
      BIASES     => L1_BIASES,
      ACTIVATION => "RELU"
    )
    port map (
      clk   => clk,
      rst   => rst,
      start => start,
      x_in  => x_in,
      y_out => l1_out,
      done  => l1_done
    );

  u_layer2 : entity work.layer
    generic map (
      N_INPUTS   => NN_HIDDEN,
      N_NEURONS  => NN_OUTPUTS,
      WEIGHTS    => L2_WEIGHTS,
      BIASES     => L2_BIASES,
      ACTIVATION => "RELU"
    )
    port map (
      clk   => clk,
      rst   => rst,
      start => l1_done,
      x_in  => l1_out,
      y_out => y_out,
      done  => done
    );

end architecture rtl;