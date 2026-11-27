--------------------------------------------------------------------------------
-- tb_nn_autotrain.vhd
-- Self-training FPGA test: the test bench only drives clk, rst and one "go"
-- pulse; nn_autotrain reads its own sample ROM and trains until it converges.
--
-- Independent check: the same training run is computed in software with the
-- reference model of tb_ref_pkg (same sample order, same criterion). The DUT
-- must finish in exactly the same epoch, with the same score, bit-identical
-- weights, and its query port must give the reference predictions.
--
--   generic DATASET : any name of datasets_pkg (default "PARITY3")
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.datasets_pkg.all;
use work.tb_ref_pkg.all;

entity tb_nn_autotrain is
  generic (DATASET : string := "PARITY3");
end entity tb_nn_autotrain;

architecture sim of tb_nn_autotrain is
  constant T : time := 10 ns;

  constant SIZES  : integer_vector := ds_layers(DATASET);
  constant NI     : positive := ds_inputs(DATASET);
  constant NO     : positive := ds_outputs(DATASET);
  constant NS     : positive := ds_samples(DATASET);
  constant N_W    : positive := n_weights(SIZES);
  constant N_B    : positive := n_neurons(SIZES);
  constant X_ALL  : data_array_t(0 to NS * NI - 1) := ds_x(DATASET);
  constant T_ALL  : data_array_t(0 to NS * NO - 1) := ds_t(DATASET);
  constant W0     : master_array_t(0 to N_W - 1)   := ds_w_init(DATASET);
  constant B0     : master_array_t(0 to N_B - 1)   := ds_b_init(DATASET);
  constant LR     : natural  := ds_lr_shift(DATASET);
  constant EPOCHS : positive := ds_max_epochs(DATASET);

  signal clk       : std_logic := '0';
  signal rst       : std_logic := '1';
  signal go        : std_logic := '0';
  signal training  : std_logic;
  signal finished  : std_logic;
  signal converged : std_logic;
  signal epoch     : natural range 0 to EPOCHS;
  signal n_correct : natural range 0 to NS;
  signal max_err   : natural range 0 to 255;
  signal q_start   : std_logic := '0';
  signal q_x       : data_array_t(0 to NI - 1) := (others => (others => '0'));
  signal q_y       : data_array_t(0 to NO - 1);
  signal q_done    : std_logic;
  signal w_out     : master_array_t(0 to N_W - 1);
  signal b_out     : master_array_t(0 to N_B - 1);
  signal running   : boolean := true;

  function to_ints (v : data_array_t) return int_array_t is
    variable r : int_array_t(0 to v'length - 1);
  begin
    for n in r'range loop r(n) := to_integer(v(v'low + n)); end loop;
    return r;
  end function;
  function to_ints (v : master_array_t) return int_array_t is
    variable r : int_array_t(0 to v'length - 1);
  begin
    for n in r'range loop r(n) := to_integer(v(v'low + n)); end loop;
    return r;
  end function;
  function to_data (v : int_array_t) return data_array_t is
    variable r : data_array_t(0 to v'length - 1);
  begin
    for n in r'range loop r(n) := to_signed(v(v'low + n), DATA_WIDTH); end loop;
    return r;
  end function;
begin

  clk <= not clk after T / 2 when running;

  u_dut : entity work.nn_autotrain
    generic map (
      LAYER_SIZES => SIZES, HIDDEN_ACT => "RELU", OUTPUT_ACT => "SIGMOID",
      LR_SHIFT => LR, W_INIT => W0, B_INIT => B0,
      N_SAMPLES => NS, X_ROM => X_ALL, T_ROM => T_ALL,
      MAX_EPOCHS => EPOCHS, GOAL => ds_goal(DATASET), ERR_TOL => ds_err_tol(DATASET))
    port map (
      clk => clk, rst => rst, go => go, training => training, finished => finished,
      converged => converged, epoch => epoch, n_correct => n_correct, max_err => max_err,
      q_start => q_start, q_x => q_x, q_y => q_y, q_done => q_done,
      w_out => w_out, b_out => b_out);

  process
    variable rw : int_array_t(0 to N_W - 1) := to_ints(W0);
    variable rb : int_array_t(0 to N_B - 1) := to_ints(B0);
    variable y  : int_array_t(0 to NO - 1);
    variable ref_epoch, ref_corr, ref_err : natural := 0;
    variable ref_conv : boolean := false;
    variable corr, merr : natural;
    variable errors : natural := 0;
    variable t0 : time;

    procedure check (cond : boolean; msg : string) is
    begin
      if not cond then
        errors := errors + 1;
        report "MISMATCH: " & msg severity error;
      end if;
    end procedure;
  begin
    -- 1) software run of the complete training (expected result) ------------
    for e in 1 to EPOCHS loop
      for s in 0 to NS - 1 loop
        ref_train_step(SIZES, rw, rb, to_ints(X_ALL(s * NI to s * NI + NI - 1)),
                       to_ints(T_ALL(s * NO to s * NO + NO - 1)), LR, "RELU", "SIGMOID");
      end loop;
      corr := 0; merr := 0;
      for s in 0 to NS - 1 loop
        ref_predict(SIZES, rw, rb, to_ints(X_ALL(s * NI to s * NI + NI - 1)), "RELU", "SIGMOID", y);
        if is_correct(to_data(y), T_ALL(s * NO to s * NO + NO - 1)) then corr := corr + 1; end if;
        if max_abs_err(to_data(y), T_ALL(s * NO to s * NO + NO - 1)) > merr then
          merr := max_abs_err(to_data(y), T_ALL(s * NO to s * NO + NO - 1));
        end if;
      end loop;
      ref_epoch := e; ref_corr := corr; ref_err := merr;
      if corr >= ds_goal(DATASET) and merr <= ds_err_tol(DATASET) then
        ref_conv := true;
        exit;
      end if;
    end loop;
    report "reference model: " & DATASET & " converged=" & boolean'image(ref_conv) &
           " after " & integer'image(ref_epoch) & " epochs";

    -- 2) hardware: reset, one go pulse, wait --------------------------------
    wait for 3 * T;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);
    go <= '1';
    wait until rising_edge(clk);
    go <= '0';
    t0 := now;
    wait until rising_edge(clk) and finished = '1';
    report "nn_autotrain: finished after " & integer'image(epoch) & " epochs, " &
           integer'image((now - t0) / T) & " clocks, converged = " & std_logic'image(converged) &
           ", correct " & integer'image(n_correct) & "/" & integer'image(NS);

    check(converged = '1', "did not converge");
    check(ref_conv, "reference model did not converge either");
    check(epoch = ref_epoch, "epoch " & integer'image(epoch) & " /= reference " & integer'image(ref_epoch));
    check(n_correct = ref_corr, "n_correct differs from reference");
    check(max_err = ref_err, "max_err differs from reference");
    for n in rw'range loop check(to_integer(w_out(n)) = rw(n), "weight " & integer'image(n)); end loop;
    for n in rb'range loop check(to_integer(b_out(n)) = rb(n), "bias " & integer'image(n)); end loop;

    -- 3) query port: the trained network answers like the reference ----------
    for s in 0 to NS - 1 loop
      q_x     <= X_ALL(s * NI to s * NI + NI - 1);
      q_start <= '1';
      wait until rising_edge(clk);
      q_start <= '0';
      wait until rising_edge(clk) and q_done = '1';
      ref_predict(SIZES, rw, rb, to_ints(X_ALL(s * NI to s * NI + NI - 1)), "RELU", "SIGMOID", y);
      check(to_ints(q_y) = y, "query result of sample " & integer'image(s));
    end loop;

    if errors = 0 then
      report "tb_nn_autotrain(" & DATASET & "): PASS (same epoch, weights and answers as the reference model)";
    else
      report "tb_nn_autotrain(" & DATASET & "): FAIL, " & integer'image(errors) & " mismatches"
        severity failure;
    end if;
    running <= false;
    std.env.finish;
  end process;

end architecture sim;
