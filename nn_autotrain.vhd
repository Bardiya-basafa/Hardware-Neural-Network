--------------------------------------------------------------------------------
-- nn_autotrain.vhd
-- Self-contained on-chip training: sample ROM + epoch sequencer + nn_trainer.
-- After a "go" pulse the FPGA trains by itself until the network has learned
-- the dataset (or MAX_EPOCHS is reached); nothing outside has to feed samples.
--
--   go -> [ epoch: train on sample 0..N-1 (one SGD step each)
--                  evaluate sample 0..N-1 (inference, score the result)
--           converged? -> finished ] -> repeat
--
-- Success criterion (the same as tb_nn_trainer, see datasets_pkg):
--   correct(sample) : 1 output -> (y >= 0.5) = (t >= 0.5); else argmax(y) = argmax(t)
--   converged       : n_correct >= GOAL and max |y - t| <= ERR_TOL
--
-- While not training (idle or finished) the network can be used for
-- inference through q_start / q_x / q_y / q_done.
--
-- Interface
--   go        : pulse while not training: start a training run from the
--               current weights (rst restores the initial weights)
--   training  : '1' while a training run is in progress
--   finished  : '1' after a run ended, until the next go
--   converged : valid while finished = '1': the criterion was met
--   epoch     : epochs completed in this run
--   n_correct, max_err : score of the last evaluation pass (max_err in Q4.4 LSBs)
--   q_start   : pulse while not training: run one inference on q_x
--   q_y       : result, valid when q_done = '1'
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.datasets_pkg.all;

entity nn_autotrain is
  generic (
    LAYER_SIZES : integer_vector;
    HIDDEN_ACT  : string  := "RELU";
    OUTPUT_ACT  : string  := "SIGMOID";
    LR_SHIFT    : natural := 2;
    W_INIT      : master_array_t;
    B_INIT      : master_array_t;
    N_SAMPLES   : positive;
    X_ROM       : data_array_t;          -- N_SAMPLES * inputs,  sample-major
    T_ROM       : data_array_t;          -- N_SAMPLES * outputs, sample-major
    MAX_EPOCHS  : positive := 1000;
    GOAL        : natural;               -- correct samples needed
    ERR_TOL     : natural                -- max |y - t| allowed (Q4.4 LSBs)
  );
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    go        : in  std_logic;
    training  : out std_logic;
    finished  : out std_logic;
    converged : out std_logic;
    epoch     : out natural range 0 to MAX_EPOCHS;
    n_correct : out natural range 0 to N_SAMPLES;
    max_err   : out natural range 0 to 255;
    q_start   : in  std_logic;
    q_x       : in  data_array_t(0 to LAYER_SIZES(LAYER_SIZES'low) - 1);
    q_y       : out data_array_t(0 to LAYER_SIZES(LAYER_SIZES'high) - 1);
    q_done    : out std_logic;
    w_out     : out master_array_t(0 to n_weights(LAYER_SIZES) - 1);
    b_out     : out master_array_t(0 to n_neurons(LAYER_SIZES) - 1)
  );
end entity nn_autotrain;

architecture rtl of nn_autotrain is

  constant NI : positive := LAYER_SIZES(LAYER_SIZES'low);
  constant NO : positive := LAYER_SIZES(LAYER_SIZES'high);
  constant XR : data_array_t(0 to X_ROM'length - 1) := X_ROM;   -- normalise ranges
  constant TR : data_array_t(0 to T_ROM'length - 1) := T_ROM;

  -- S_IDLE   : waiting for go / q_start (also "finished")
  -- S_ISSUE  : start the trainer on sample s_cnt (train or evaluate)
  -- S_WAIT   : wait for the trainer; score the sample in the evaluation pass
  -- S_CHECK  : end of epoch: converged / out of epochs / next epoch
  -- S_QUERY  : external inference in progress
  type state_t is (S_IDLE, S_ISSUE, S_WAIT, S_CHECK, S_QUERY);
  signal state : state_t := S_IDLE;

  signal eval_pass : std_logic := '0';                     -- '0' training pass, '1' evaluation pass
  signal s_cnt     : integer range 0 to N_SAMPLES - 1 := 0;
  signal x_base    : integer range 0 to XR'length := 0;   -- = s_cnt * NI (no multiplier)
  signal t_base    : integer range 0 to TR'length := 0;   -- = s_cnt * NO
  signal ep_cnt    : integer range 0 to MAX_EPOCHS := 0;
  signal corr_cnt  : integer range 0 to N_SAMPLES := 0;
  signal err_max   : integer range 0 to 255 := 0;
  signal fin, conv : std_logic := '0';

  -- trainer interface
  signal tr_start, tr_train, tr_done, tr_busy : std_logic;
  signal tr_x, rom_x : data_array_t(0 to NI - 1);
  signal tr_t, rom_t : data_array_t(0 to NO - 1);
  signal tr_y        : data_array_t(0 to NO - 1);

begin

  assert XR'length = N_SAMPLES * NI and TR'length = N_SAMPLES * NO
    report "nn_autotrain: X_ROM / T_ROM size does not match N_SAMPLES and LAYER_SIZES"
    severity failure;

  -- sample ROM read (address = base register + constant offset)
  g_rx : for i in 0 to NI - 1 generate rom_x(i) <= XR(x_base + i); end generate;
  g_rt : for k in 0 to NO - 1 generate rom_t(k) <= TR(t_base + k); end generate;

  -- trainer inputs: ROM while training, q_x for an external query
  tr_x     <= q_x when state = S_IDLE else rom_x;
  tr_t     <= rom_t;
  tr_train <= '1' when state = S_ISSUE and eval_pass = '0' else '0';
  tr_start <= '1' when state = S_ISSUE or (state = S_IDLE and q_start = '1' and go = '0') else '0';

  u_trainer : entity work.nn_trainer
    generic map (LAYER_SIZES => LAYER_SIZES, HIDDEN_ACT => HIDDEN_ACT, OUTPUT_ACT => OUTPUT_ACT,
                 LR_SHIFT => LR_SHIFT, W_INIT => W_INIT, B_INIT => B_INIT)
    port map (clk => clk, rst => rst, start => tr_start, train => tr_train,
              x_in => tr_x, t_in => tr_t, y_out => tr_y, done => tr_done, busy => tr_busy,
              w_out => w_out, b_out => b_out);

  q_y       <= tr_y;
  q_done    <= tr_done when state = S_QUERY else '0';
  training  <= '0' when state = S_IDLE or state = S_QUERY else '1';
  finished  <= fin;
  converged <= conv;
  epoch     <= ep_cnt;
  n_correct <= corr_cnt;
  max_err   <= err_max;

  process (clk)
    variable e : natural;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state     <= S_IDLE;
        eval_pass <= '0';
        s_cnt     <= 0;
        x_base    <= 0;
        t_base    <= 0;
        ep_cnt    <= 0;
        corr_cnt  <= 0;
        err_max   <= 0;
        fin       <= '0';
        conv      <= '0';
      else
        case state is

          when S_IDLE =>
            if go = '1' then                       -- new training run
              fin       <= '0';
              conv      <= '0';
              ep_cnt    <= 0;
              eval_pass <= '0';
              s_cnt     <= 0;
              x_base    <= 0;
              t_base    <= 0;
              state     <= S_ISSUE;
            elsif q_start = '1' then               -- external inference (started above)
              state <= S_QUERY;
            end if;

          when S_QUERY =>
            if tr_done = '1' then
              state <= S_IDLE;
            end if;

          when S_ISSUE =>                          -- tr_start = '1' in this clock
            state <= S_WAIT;

          when S_WAIT =>
            if tr_done = '1' then
              if eval_pass = '1' then              -- score this sample
                if is_correct(tr_y, rom_t) then
                  corr_cnt <= corr_cnt + 1;
                end if;
                e := max_abs_err(tr_y, rom_t);
                if e > err_max then
                  err_max <= e;
                end if;
              end if;
              if s_cnt = N_SAMPLES - 1 then        -- end of a pass
                s_cnt  <= 0;
                x_base <= 0;
                t_base <= 0;
                if eval_pass = '0' then            -- training pass done -> evaluate
                  eval_pass <= '1';
                  corr_cnt  <= 0;
                  err_max   <= 0;
                  state     <= S_ISSUE;
                else
                  state <= S_CHECK;
                end if;
              else
                s_cnt  <= s_cnt + 1;
                x_base <= x_base + NI;
                t_base <= t_base + NO;
                state  <= S_ISSUE;
              end if;
            end if;

          when S_CHECK =>                          -- corr_cnt / err_max are final here
            ep_cnt    <= ep_cnt + 1;
            eval_pass <= '0';
            if corr_cnt >= GOAL and err_max <= ERR_TOL then
              fin   <= '1';
              conv  <= '1';
              state <= S_IDLE;
            elsif ep_cnt + 1 = MAX_EPOCHS then
              fin   <= '1';
              state <= S_IDLE;
            else
              state <= S_ISSUE;
            end if;

        end case;
      end if;
    end if;
  end process;

end architecture rtl;
