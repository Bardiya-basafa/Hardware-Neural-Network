--------------------------------------------------------------------------------
-- nn_trainer.vhd
-- Trainable multi-layer perceptron of any depth:
--   forward pass + back-propagation + SGD weight update, all in hardware.
--
--   LAYER_SIZES = (N0, N1, ..., NL)    e.g. (2, 4, 1) or (2, 8, 8, 1)
--
--   x -> layer 0 -> level 1 -> layer 1 -> ... -> layer L-1 -> level L = y
--        (HIDDEN_ACT for layers 0 .. L-2, OUTPUT_ACT for layer L-1)
--
-- Forward pass: one "layer" instance per layer (layer -> neuron -> mac_unit),
-- chained with done -> start, exactly like nn_top. The weights are kept here
-- in registers as Q4.12 master copies; the layers see them rounded to Q4.4.
--
-- Back-propagation (one sample = one SGD step):
--   output delta   d[L-1]_k = (y_k - t_k)                 OUTPUT_ACT = SIGMOID
--                  d[L-1]_k = (y_k - t_k) * f'(y_k)       OUTPUT_ACT = RELU
--                  (sigmoid output = cross-entropy loss, whose gradient w.r.t.
--                   the pre-activation is y - t; avoids the zero slope of the
--                   saturated LUT)
--   hidden delta   d[l-1]_j = (sum_k W[l][k][j] * d[l]_k) * f'(a[l]_j)
--                  for l = L-1 down to 1 - ALL deltas are computed before any
--                  weight changes, so every layer uses its old weights
--   update         W[l][k][j] -= lr * d[l]_k * a[l]_j      (a[0] = x)
--                  B[l][k]    -= lr * d[l]_k
--   lr = 2 ** -LR_SHIFT
--
-- All back-prop arithmetic goes through ONE shared 16x16 multiplier, one
-- operation per clock. Weight addresses come from an incrementing address
-- register (no run-time multiplication in the address path).
--
-- Interface
--   start : one-clock pulse while idle (busy = '0'); x_in, t_in, train are
--           sampled on that clock
--   train : '1' = forward + backprop + update, '0' = forward (inference) only
--   y_out : network output of the forward pass, valid when done = '1' (and
--           until the next forward pass ends)
--   done  : one-clock pulse when everything (incl. weight update) is finished
--   busy  : '1' while working
--   w_out / b_out : current master weights / biases (Q4.12), flat layout of
--           nn_pkg (layer 0 first; W(neuron*N_IN + input))
--
-- Clocks from the start clock to done = '1'  (N0..NL = LAYER_SIZES):
--   forward    sum over layers l       (N[l] + 2)
--   out delta  NL
--   hidden     sum over l = 1 .. L-1   N[l] * (N[l+1] + 1)
--   update     sum over layers l       N[l+1] * (N[l] + 1)
--   finish     1
--   inference only (train = '0'): forward + 1
--   e.g. 2-4-1   :  37 per training step, 11 per inference
--        2-8-8-1 : 219 per training step, 25 per inference   (measured in tb_nn_trainer)
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity nn_trainer is
  generic (
    LAYER_SIZES : integer_vector;          -- (inputs, hidden ..., outputs), >= 2 entries
    HIDDEN_ACT  : string  := "RELU";
    OUTPUT_ACT  : string  := "SIGMOID";
    LR_SHIFT    : natural := 2;
    W_INIT      : master_array_t;          -- n_weights(LAYER_SIZES) entries
    B_INIT      : master_array_t           -- n_neurons(LAYER_SIZES) entries
  );
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    start : in  std_logic;
    train : in  std_logic;
    x_in  : in  data_array_t(0 to LAYER_SIZES(LAYER_SIZES'low) - 1);
    t_in  : in  data_array_t(0 to LAYER_SIZES(LAYER_SIZES'high) - 1);
    y_out : out data_array_t(0 to LAYER_SIZES(LAYER_SIZES'high) - 1);
    done  : out std_logic;
    busy  : out std_logic;
    w_out : out master_array_t(0 to n_weights(LAYER_SIZES) - 1);
    b_out : out master_array_t(0 to n_neurons(LAYER_SIZES) - 1)
  );
end entity nn_trainer;

architecture rtl of nn_trainer is

  -- topology ------------------------------------------------------------------
  constant SZ   : integer_vector(0 to LAYER_SIZES'length - 1) := LAYER_SIZES;
  constant NL   : positive := SZ'length - 1;      -- number of layers
  constant NI   : positive := SZ(0);              -- inputs
  constant NO   : positive := SZ(NL);             -- outputs
  constant N_W  : positive := n_weights(SZ);
  constant N_B  : positive := n_neurons(SZ);
  constant N_A  : positive := n_values(SZ);
  constant MAXS : positive := max_size(SZ);
  constant WB   : integer_vector(0 to NL - 1) := w_base(SZ);   -- first weight of layer l
  constant BB   : integer_vector(0 to NL - 1) := b_base(SZ);   -- first bias / delta of layer l
  constant AB   : integer_vector(0 to NL)     := a_base(SZ);   -- first value of level l

  constant ONE_DATA  : signed(DELTA_WIDTH - 1 downto 0) := to_signed(2 ** FRAC_BITS,  DELTA_WIDTH);
  constant ONE_DELTA : delta_t                          := to_signed(2 ** DELTA_FRAC, DELTA_WIDTH);
  constant PROD_W    : positive := 2 * DELTA_WIDTH;
  constant BACC_W    : positive := PROD_W + 8;        -- up to 256 terms per back-sum

  function act_of (l : natural) return string is
  begin
    if l = NL - 1 then return OUTPUT_ACT; else return HIDDEN_ACT; end if;
  end function;

  type state_t is (IDLE, FWD, OUT_DELTA, BP_ACC, BP_MUL, UPD, FINISH);
  signal state : state_t := IDLE;

  -- parameters (master copies, Q4.12)
  signal wm : master_array_t(0 to N_W - 1) := W_INIT;
  signal bm : master_array_t(0 to N_B - 1) := B_INIT;

  -- every value of the forward pass: level 0 = latched x, level l = layer l-1 output
  signal acts   : data_array_t(0 to N_A - 1);
  -- lstart(l) starts layer l; lstart(l+1) is the done of layer l; lstart(NL) = forward done
  signal lstart : std_logic_vector(0 to NL);

  -- latched sample
  signal x_lat     : data_array_t(0 to NI - 1) := (others => (others => '0'));
  signal t_lat     : data_array_t(0 to NO - 1) := (others => (others => '0'));
  signal train_lat : std_logic := '0';

  -- back-propagation
  signal d    : delta_array_t(0 to N_B - 1) := (others => (others => '0'));  -- same layout as biases
  signal bacc : signed(BACC_W - 1 downto 0) := (others => '0');

  -- loop counters
  --   l_cnt  : layer
  --   k_cnt  : neuron of layer l (output side)
  --   j_cnt  : input of layer l  (j_cnt = N[l] selects the bias)
  --   w_addr : flat address of W[l][k][j]
  signal l_cnt  : integer range 0 to NL - 1 := 0;
  signal k_cnt  : integer range 0 to MAXS - 1 := 0;
  signal j_cnt  : integer range 0 to MAXS := 0;
  signal w_addr : integer range 0 to N_W := 0;

  -- operand selection
  signal y_k, a_j  : data_t;      -- output k / value j of level l
  signal d_k       : delta_t;     -- delta of neuron k of layer l
  signal fd_hid    : delta_t;     -- f'(a_j)   (hidden activation)
  signal fd_out    : delta_t;     -- f'(y_k)   (1.0 for a sigmoid output)
  signal fd_out_raw: delta_t;

  -- shared multiplier
  signal mul_a, mul_b : signed(DELTA_WIDTH - 1 downto 0);
  signal prod         : signed(PROD_W - 1 downto 0);

  -- new_w = w - round(prod * 2^-LR_SHIFT), saturated
  function sgd_step (w : master_t; p : signed) return master_t is
    variable s : signed(PROD_W downto 0);
  begin
    s := resize(w, PROD_W + 1) - resize(round_shift(p, LR_SHIFT), PROD_W + 1);
    return saturate(s, MASTER_WIDTH);
  end function;

begin

  assert SZ'length >= 2
    report "nn_trainer: LAYER_SIZES needs at least an input and an output size"
    severity failure;
  assert W_INIT'length = N_W and B_INIT'length = N_B
    report "nn_trainer: W_INIT / B_INIT length does not match LAYER_SIZES"
    severity failure;
  assert MASTER_FRAC = DELTA_FRAC + FRAC_BITS
    report "nn_trainer: fixed-point formats do not line up (MASTER_FRAC /= DELTA_FRAC + FRAC_BITS)"
    severity failure;

  ---------------------------------------------------------------------------
  -- Forward pass: one layer instance per layer, chained done -> start
  ---------------------------------------------------------------------------
  lstart(0) <= '1' when state = IDLE and start = '1' else '0';
  acts(0 to NI - 1) <= x_lat;

  g_layer : for l in 0 to NL - 1 generate
    constant LI : positive := SZ(l);
    constant LO : positive := SZ(l + 1);
    signal xin : data_array_t(0 to LI - 1);
    signal wq  : data_array_t(0 to LI * LO - 1);
    signal bq  : data_array_t(0 to LO - 1);
    signal yo  : data_array_t(0 to LO - 1);
  begin
    -- Q4.12 master -> Q4.4 forward weights
    g_w : for n in 0 to LI * LO - 1 generate wq(n) <= to_data(wm(WB(l) + n)); end generate;
    g_b : for n in 0 to LO - 1      generate bq(n) <= to_data(bm(BB(l) + n)); end generate;

    -- layer 0 reads the port (it latches x_in on the start clock itself)
    g_in0 : if l = 0 generate xin <= x_in; end generate;
    g_inl : if l > 0 generate xin <= acts(AB(l) to AB(l) + LI - 1); end generate;

    u_layer : entity work.layer
      generic map (N_INPUTS => LI, N_NEURONS => LO, ACTIVATION => act_of(l))
      port map (clk => clk, rst => rst, start => lstart(l), x_in => xin,
                w_in => wq, b_in => bq, y_out => yo, done => lstart(l + 1));

    acts(AB(l + 1) to AB(l + 1) + LO - 1) <= yo;
  end generate g_layer;

  y_out <= acts(AB(NL) to AB(NL) + NO - 1);
  busy  <= '0' when state = IDLE else '1';
  w_out <= wm;
  b_out <= bm;

  ---------------------------------------------------------------------------
  -- Operand selection + activation derivatives
  ---------------------------------------------------------------------------
  y_k <= acts(AB(NL) + k_cnt)     when k_cnt < NO              else (others => '0');
  a_j <= acts(AB(l_cnt) + j_cnt)  when j_cnt < SZ(l_cnt)       else (others => '0');
  d_k <= d(BB(l_cnt) + k_cnt)     when k_cnt < SZ(l_cnt + 1)   else (others => '0');

  u_dhid : entity work.activation_deriv
    generic map (ACTIVATION => HIDDEN_ACT)
    port map (y_in => a_j, d_out => fd_hid);

  u_dout : entity work.activation_deriv
    generic map (ACTIVATION => OUTPUT_ACT)
    port map (y_in => y_k, d_out => fd_out_raw);

  fd_out <= ONE_DELTA when OUTPUT_ACT = "SIGMOID" else fd_out_raw;

  ---------------------------------------------------------------------------
  -- Shared multiplier: operands per state
  --   OUT_DELTA : (y_k - t_k)   [Q.4]  * f'(y_k)  [Q.8]   = Q.12
  --   BP_ACC    : W[l][k][j]    [Q.12] * d[l]_k   [Q.8]   = Q.20
  --   BP_MUL    : backsum_j     [Q.8]  * f'(a_j)  [Q.8]   = Q.16
  --   UPD       : d[l]_k        [Q.8]  * a_j or 1 [Q.4]   = Q.12
  ---------------------------------------------------------------------------
  process (all)
  begin
    mul_a <= (others => '0');
    mul_b <= (others => '0');
    case state is
      when OUT_DELTA =>
        if k_cnt < NO then
          mul_a <= resize(y_k, DELTA_WIDTH) - resize(t_lat(k_cnt), DELTA_WIDTH);
        end if;
        mul_b <= fd_out;
      when BP_ACC =>
        if w_addr < N_W then
          mul_a <= wm(w_addr);
        end if;
        mul_b <= d_k;
      when BP_MUL =>
        mul_a <= saturate(round_shift(bacc, MASTER_FRAC), DELTA_WIDTH);
        mul_b <= fd_hid;
      when UPD =>
        mul_a <= d_k;
        if j_cnt = SZ(l_cnt) then
          mul_b <= ONE_DATA;
        else
          mul_b <= resize(a_j, DELTA_WIDTH);
        end if;
      when others =>
        null;
    end case;
  end process;

  prod <= mul_a * mul_b;

  ---------------------------------------------------------------------------
  -- Control FSM
  ---------------------------------------------------------------------------
  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state     <= IDLE;
        done      <= '0';
        train_lat <= '0';
        l_cnt     <= 0;
        k_cnt     <= 0;
        j_cnt     <= 0;
        w_addr    <= 0;
        bacc      <= (others => '0');
        wm        <= W_INIT;
        bm        <= B_INIT;
      else
        done <= '0';
        case state is

          when IDLE =>
            if start = '1' then
              x_lat     <= x_in;
              t_lat     <= t_in;
              train_lat <= train;
              state     <= FWD;
            end if;

          -- wait for the last layer
          when FWD =>
            if lstart(NL) = '1' then
              k_cnt <= 0;
              l_cnt <= NL - 1;
              if train_lat = '1' then
                state <= OUT_DELTA;
              else
                state <= FINISH;
              end if;
            end if;

          -- d[L-1]_k = (y_k - t_k) * f'(y_k)             Q.12 -> Q.8
          when OUT_DELTA =>
            d(BB(NL - 1) + k_cnt) <= saturate(round_shift(prod, MASTER_FRAC - DELTA_FRAC), DELTA_WIDTH);
            if k_cnt = NO - 1 then
              k_cnt  <= 0;
              j_cnt  <= 0;
              bacc   <= (others => '0');
              w_addr <= WB(NL - 1);
              if NL > 1 then
                state <= BP_ACC;       -- there are hidden layers to back-propagate into
              else
                state <= UPD;          -- single layer network
              end if;
            else
              k_cnt <= k_cnt + 1;
            end if;

          -- backsum_j = sum_k W[l][k][j] * d[l]_k   (Q.20 accumulator)
          -- W[l][k][j] is at WB(l) + k*N[l] + j -> step N[l] per k
          when BP_ACC =>
            bacc <= bacc + resize(prod, BACC_W);
            if k_cnt = SZ(l_cnt + 1) - 1 then
              k_cnt <= 0;
              state <= BP_MUL;
            else
              k_cnt  <= k_cnt + 1;
              w_addr <= w_addr + SZ(l_cnt);
            end if;

          -- d[l-1]_j = backsum_j * f'(a[l]_j)              Q.16 -> Q.8
          when BP_MUL =>
            d(BB(l_cnt - 1) + j_cnt) <= saturate(round_shift(prod, DELTA_FRAC), DELTA_WIDTH);
            bacc <= (others => '0');
            if j_cnt = SZ(l_cnt) - 1 then
              j_cnt <= 0;
              if l_cnt = 1 then                      -- all deltas known -> update
                l_cnt  <= NL - 1;
                w_addr <= WB(NL - 1);
                state  <= UPD;
              else                                   -- next layer down
                l_cnt  <= l_cnt - 1;
                w_addr <= WB(l_cnt - 1);
                state  <= BP_ACC;
              end if;
            else
              j_cnt  <= j_cnt + 1;
              w_addr <= WB(l_cnt) + j_cnt + 1;
              state  <= BP_ACC;
            end if;

          -- W[l][k][j] -= lr * d[l]_k * a[l]_j   (j = N[l] -> bias)
          -- layers L-1 down to 0; inside a layer the weights are sequential
          when UPD =>
            if j_cnt = SZ(l_cnt) then
              bm(BB(l_cnt) + k_cnt) <= sgd_step(bm(BB(l_cnt) + k_cnt), prod);
              j_cnt <= 0;
              if k_cnt = SZ(l_cnt + 1) - 1 then
                k_cnt <= 0;
                if l_cnt = 0 then
                  state <= FINISH;
                else
                  l_cnt  <= l_cnt - 1;
                  w_addr <= WB(l_cnt - 1);
                end if;
              else
                k_cnt <= k_cnt + 1;
              end if;
            else
              wm(w_addr) <= sgd_step(wm(w_addr), prod);
              w_addr     <= w_addr + 1;
              j_cnt      <= j_cnt + 1;
            end if;

          when FINISH =>
            done  <= '1';
            state <= IDLE;

        end case;
      end if;
    end if;
  end process;

end architecture rtl;
