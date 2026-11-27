--------------------------------------------------------------------------------
-- tb_nn_trainer.vhd
-- On-chip training test for any dataset of datasets_pkg.
--
--   generic DATASET    : "XOR" (default), "PARITY3", "CIRCLE", "BLOBS3", "SEG7"
--   generic LR_SHIFT   : learning rate 2^-LR_SHIFT, -1 = dataset default
--   generic MAX_EPOCHS : -1 = dataset default
--   generic CHECK      : compare every weight with the reference model after
--                        every training step (the gradient check)
--
-- One epoch = every sample once with train = '1' (one SGD step each, in ROM
-- order), then every sample once with train = '0' to measure the result.
-- Convergence criterion: see datasets_pkg (same as nn_autotrain).
--
-- Gradient check: tb_ref_pkg.ref_train_step - an independent integer model of
-- forward pass + back-propagation + SGD - runs next to the hardware; after
-- EVERY training step all weights and biases must be bit-identical.
--
-- Waveform: epoch, n_correct, max_err and sum_abs_err show the learning
-- progress; u_dut/state, u_dut/d, u_dut/wm the back-propagation itself.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.datasets_pkg.all;
use work.tb_ref_pkg.all;

entity tb_nn_trainer is
  generic (
    DATASET    : string  := "XOR";
    LR_SHIFT   : integer := -1;
    MAX_EPOCHS : integer := -1;
    CHECK      : boolean := true
  );
end entity tb_nn_trainer;

architecture sim of tb_nn_trainer is
  constant T : time := 10 ns;

  -- everything below is derived from the dataset
  constant SIZES  : integer_vector := ds_layers(DATASET);
  constant NI     : positive := SIZES(SIZES'low);
  constant NO     : positive := SIZES(SIZES'high);
  constant NS     : positive := ds_samples(DATASET);
  constant N_W    : positive := n_weights(SIZES);
  constant N_B    : positive := n_neurons(SIZES);
  constant X_ALL  : data_array_t(0 to NS * NI - 1) := ds_x(DATASET);
  constant T_ALL  : data_array_t(0 to NS * NO - 1) := ds_t(DATASET);
  constant W0     : master_array_t(0 to N_W - 1)   := ds_w_init(DATASET);
  constant B0     : master_array_t(0 to N_B - 1)   := ds_b_init(DATASET);
  constant HID    : string := "RELU";
  constant OUTA   : string := "SIGMOID";

  function pick (v, dflt : integer) return integer is
  begin
    if v < 0 then return dflt; else return v; end if;
  end function;
  constant LR     : natural  := pick(LR_SHIFT, ds_lr_shift(DATASET));
  constant EPOCHS : positive := pick(MAX_EPOCHS, ds_max_epochs(DATASET));
  constant GOAL   : natural  := ds_goal(DATASET);
  constant TOL    : natural  := ds_err_tol(DATASET);

  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal start : std_logic := '0';
  signal train : std_logic := '0';
  signal x_in  : data_array_t(0 to NI - 1) := (others => (others => '0'));
  signal t_in  : data_array_t(0 to NO - 1) := (others => (others => '0'));
  signal y_out : data_array_t(0 to NO - 1);
  signal done  : std_logic;
  signal busy  : std_logic;
  signal w_out : master_array_t(0 to N_W - 1);
  signal b_out : master_array_t(0 to N_B - 1);
  signal running : boolean := true;

  -- monitoring (waveform)
  signal epoch       : natural := 0;
  signal n_correct   : natural := 0;
  signal max_err     : natural := 0;   -- Q4.4 LSBs (16 = 1.0)
  signal sum_abs_err : natural := 0;   -- Q4.4 LSBs, all samples and outputs

  function x_of (s : natural) return data_array_t is
  begin
    return X_ALL(s * NI to s * NI + NI - 1);
  end function;
  function t_of (s : natural) return data_array_t is
  begin
    return T_ALL(s * NO to s * NO + NO - 1);
  end function;
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
  function img (v : data_array_t) return string is
  begin
    if v'length = 0 then return ""; end if;
    if v'length = 1 then return real'image(real(to_integer(v(v'low))) / 16.0); end if;
    return real'image(real(to_integer(v(v'low))) / 16.0) & " " & img(v(v'low + 1 to v'high));
  end function;
begin

  clk <= not clk after T / 2 when running;

  u_dut : entity work.nn_trainer
    generic map (
      LAYER_SIZES => SIZES,
      HIDDEN_ACT  => HID,
      OUTPUT_ACT  => OUTA,
      LR_SHIFT    => LR,
      W_INIT      => W0,
      B_INIT      => B0
    )
    port map (clk => clk, rst => rst, start => start, train => train,
              x_in => x_in, t_in => t_in, y_out => y_out, done => done, busy => busy,
              w_out => w_out, b_out => b_out);

  process
    variable rw : int_array_t(0 to N_W - 1) := to_ints(W0);   -- reference model state
    variable rb : int_array_t(0 to N_B - 1) := to_ints(B0);
    variable grad_errors : natural := 0;
    variable correct, merr, serr : natural;
    variable ok : boolean := false;
    variable t_start : time;

    procedure run_sample (s : natural; tr : std_logic) is
    begin
      x_in  <= x_of(s);
      t_in  <= t_of(s);
      train <= tr;
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      wait until rising_edge(clk) and done = '1';
    end procedure;

    procedure compare_weights (e, s : natural) is
      variable bad : boolean := false;
    begin
      for n in rw'range loop bad := bad or to_integer(w_out(n)) /= rw(n); end loop;
      for n in rb'range loop bad := bad or to_integer(b_out(n)) /= rb(n); end loop;
      if bad then
        grad_errors := grad_errors + 1;
        report "weights differ from reference model after epoch " & integer'image(e) &
               ", sample " & integer'image(s) severity error;
      end if;
    end procedure;
  begin
    report "tb_nn_trainer: dataset " & DATASET & ", " & integer'image(NS) & " samples, " &
           integer'image(N_W) & " weights + " & integer'image(N_B) & " biases, lr = 2^-" &
           integer'image(LR);
    wait for 3 * T;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);

    -- latency (clock edges from the start clock to the clock that sets done)
    for tr in std_logic range '0' to '1' loop
      x_in <= x_of(0); t_in <= t_of(0); train <= tr; start <= '1';
      wait until rising_edge(clk);
      t_start := now;
      start <= '0';
      wait until done = '1';
      if tr = '1' then
        report "latency of one training step: " & integer'image((now - t_start) / T) & " clocks";
      else
        report "latency of one inference: " & integer'image((now - t_start) / T) & " clocks";
      end if;
      wait until rising_edge(clk);
    end loop;
    -- start again from the initial weights for the actual training run
    rst <= '1';
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);
    rw := to_ints(W0);
    rb := to_ints(B0);

    for e in 1 to EPOCHS loop
      epoch <= e;
      -- training pass (+ gradient check against the reference model)
      for s in 0 to NS - 1 loop
        run_sample(s, '1');
        if CHECK then
          ref_train_step(SIZES, rw, rb, to_ints(x_of(s)), to_ints(t_of(s)), LR, HID, OUTA);
          compare_weights(e, s);
        end if;
      end loop;
      -- evaluation pass
      correct := 0; merr := 0; serr := 0;
      for s in 0 to NS - 1 loop
        run_sample(s, '0');
        if is_correct(y_out, t_of(s)) then correct := correct + 1; end if;
        if max_abs_err(y_out, t_of(s)) > merr then merr := max_abs_err(y_out, t_of(s)); end if;
        for k in 0 to NO - 1 loop
          serr := serr + abs (to_integer(y_out(k)) - to_integer(T_ALL(s * NO + k)));
        end loop;
      end loop;
      n_correct   <= correct;
      max_err     <= merr;
      sum_abs_err <= serr;

      if e mod 50 = 0 or e = 1 then
        report "epoch " & integer'image(e) & ": correct " & integer'image(correct) & "/" &
               integer'image(NS) & ", max|y-t| = " & real'image(real(merr) / 16.0) &
               ", sum|y-t| = " & real'image(real(serr) / 16.0);
      end if;
      if correct >= GOAL and merr <= TOL then
        report "converged after " & integer'image(e) & " epochs (" & integer'image(e * NS) &
               " training steps): correct " & integer'image(correct) & "/" & integer'image(NS) &
               ", max|y-t| = " & real'image(real(merr) / 16.0);
        ok := true;
        exit;
      end if;
    end loop;

    -- final predictions (short datasets only)
    if NS <= 10 then
      for s in 0 to NS - 1 loop
        run_sample(s, '0');
        report "x = " & img(x_of(s)) & "  ->  y = " & img(y_out) & "   (target " & img(t_of(s)) & ")";
      end loop;
    end if;

    if grad_errors /= 0 then
      report "tb_nn_trainer(" & DATASET & "): FAIL, backprop differs from the reference model in " &
             integer'image(grad_errors) & " steps" severity failure;
    elsif not ok then
      report "tb_nn_trainer(" & DATASET & "): FAIL, did not converge in " & integer'image(EPOCHS) &
             " epochs" severity failure;
    elsif CHECK then
      report "tb_nn_trainer(" & DATASET & "): PASS (every weight update matched the reference model)";
    else
      report "tb_nn_trainer(" & DATASET & "): PASS (converged; gradient check disabled)";
    end if;
    running <= false;
    std.env.finish;
  end process;

end architecture sim;
