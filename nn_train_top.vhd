--------------------------------------------------------------------------------
-- nn_train_top.vhd
-- Synthesis top for on-chip training: nn_autotrain with one dataset of
-- datasets_pkg built into ROM. Pulse go and the FPGA trains itself; when
-- finished = '1' the trained network answers q_x -> q_y.
--
--   generic DATASET : "XOR", "PARITY3", "CIRCLE", "BLOBS3", "SEG7"
--
-- The weight read-out ports of the trainer are left internal (a real board
-- would not have enough pins); training status and the query port are
-- enough to drive LEDs / switches or a UART.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.datasets_pkg.all;

entity nn_train_top is
  generic (DATASET : string := "PARITY3");
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    go        : in  std_logic;
    training  : out std_logic;
    finished  : out std_logic;
    converged : out std_logic;
    epoch     : out unsigned(15 downto 0);
    q_start   : in  std_logic;
    q_x       : in  data_array_t(0 to ds_inputs(DATASET) - 1);
    q_y       : out data_array_t(0 to ds_outputs(DATASET) - 1);
    q_done    : out std_logic
  );
end entity nn_train_top;

architecture rtl of nn_train_top is
  constant SIZES : integer_vector := ds_layers(DATASET);
  signal ep_i    : natural range 0 to ds_max_epochs(DATASET);
  signal w_int   : master_array_t(0 to n_weights(SIZES) - 1);
  signal b_int   : master_array_t(0 to n_neurons(SIZES) - 1);
begin

  u_auto : entity work.nn_autotrain
    generic map (
      LAYER_SIZES => SIZES,
      HIDDEN_ACT  => "RELU",
      OUTPUT_ACT  => "SIGMOID",
      LR_SHIFT    => ds_lr_shift(DATASET),
      W_INIT      => ds_w_init(DATASET),
      B_INIT      => ds_b_init(DATASET),
      N_SAMPLES   => ds_samples(DATASET),
      X_ROM       => ds_x(DATASET),
      T_ROM       => ds_t(DATASET),
      MAX_EPOCHS  => ds_max_epochs(DATASET),
      GOAL        => ds_goal(DATASET),
      ERR_TOL     => ds_err_tol(DATASET)
    )
    port map (
      clk => clk, rst => rst, go => go,
      training => training, finished => finished, converged => converged,
      epoch => ep_i, n_correct => open, max_err => open,
      q_start => q_start, q_x => q_x, q_y => q_y, q_done => q_done,
      w_out => w_int, b_out => b_int
    );

  epoch <= to_unsigned(ep_i, 16);

end architecture rtl;
