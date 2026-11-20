--------------------------------------------------------------------------------
-- layer.vhd
-- A fully connected layer: N_NEURONS neurons working in parallel on the same
-- input vector. Inputs are fed to the neurons one per clock (sequential MAC),
-- every neuron gets its own weight for that input.
--
-- Generics
--   N_INPUTS, N_NEURONS : layer size
--   WEIGHTS : flattened weights, WEIGHTS(neuron * N_INPUTS + input), index from 0
--   BIASES  : one bias per neuron, index from 0
--   ACTIVATION : "RELU" or "SIGMOID"
--
-- Interface
--   start : one-clock pulse while idle; x_in is sampled on that clock
--   done  : one-clock pulse, y_out is valid while done = '1' (and afterwards
--           until the next result is produced)
-- Latency: done is high N_INPUTS + 2 clocks after the clock in which start = '1'.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity layer is
  generic (
    N_INPUTS   : positive;
    N_NEURONS  : positive;
    WEIGHTS    : data_array_t;
    BIASES     : data_array_t;
    ACTIVATION : string := "RELU"
  );
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    start : in  std_logic;
    x_in  : in  data_array_t(0 to N_INPUTS - 1);
    y_out : out data_array_t(0 to N_NEURONS - 1);
    done  : out std_logic
  );
end entity layer;

architecture rtl of layer is

  type state_t is (IDLE, RUN, WAIT_DONE);
  signal state : state_t := IDLE;

  signal x_reg  : data_array_t(0 to N_INPUTS - 1) := (others => (others => '0'));
  signal idx    : integer range 0 to N_INPUTS - 1 := 0;

  signal neuron_start : std_logic;
  signal x_valid      : std_logic;
  signal x_last       : std_logic;
  signal x_sel        : data_t;
  signal w_sel        : data_array_t(0 to N_NEURONS - 1);
  signal n_done       : std_logic_vector(0 to N_NEURONS - 1);

begin

  assert WEIGHTS'length = N_INPUTS * N_NEURONS
    report "layer: WEIGHTS must have N_INPUTS * N_NEURONS entries"
    severity failure;
  assert BIASES'length = N_NEURONS
    report "layer: BIASES must have N_NEURONS entries"
    severity failure;

  ---------------------------------------------------------------------------
  -- Control signals for the neurons (derived from the layer FSM)
  ---------------------------------------------------------------------------
  neuron_start <= '1' when (state = IDLE and start = '1')  else '0';
  x_valid      <= '1' when  state = RUN                    else '0';
  x_last       <= '1' when (state = RUN and idx = N_INPUTS - 1) else '0';
  x_sel        <= x_reg(idx);

  ---------------------------------------------------------------------------
  -- Neurons
  ---------------------------------------------------------------------------
  g_neurons : for j in 0 to N_NEURONS - 1 generate
    w_sel(j) <= WEIGHTS(WEIGHTS'low + j * N_INPUTS + idx);

    u_neuron : entity work.neuron
      generic map (ACTIVATION => ACTIVATION)
      port map (
        clk     => clk,
        rst     => rst,
        start   => neuron_start,
        x_valid => x_valid,
        x_last  => x_last,
        x_in    => x_sel,
        w_in    => w_sel(j),
        bias    => BIASES(BIASES'low + j),
        y_out   => y_out(j),
        done    => n_done(j)
      );
  end generate g_neurons;

  done <= and n_done;     -- all neurons finish in the same clock

  ---------------------------------------------------------------------------
  -- Layer FSM: latch inputs, step through the inputs, wait for the neurons
  ---------------------------------------------------------------------------
  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state <= IDLE;
        idx   <= 0;
        x_reg <= (others => (others => '0'));
      else
        case state is
          when IDLE =>
            if start = '1' then
              x_reg <= x_in;
              idx   <= 0;
              state <= RUN;
            end if;

          when RUN =>
            if idx = N_INPUTS - 1 then
              idx   <= 0;
              state <= WAIT_DONE;
            else
              idx <= idx + 1;
            end if;

          when WAIT_DONE =>
            if (and n_done) = '1' then
              state <= IDLE;
            end if;
        end case;
      end if;
    end if;
  end process;

end architecture rtl;