--------------------------------------------------------------------------------
-- neuron.vhd
-- One neuron, sequential MAC:  y = activation( sum(x_i * w_i) + bias )
--
-- Handshake / timing
--   cycle 0     : start = '1' (while idle)  -> accumulator cleared
--   cycle 1..N  : x_valid = '1' with x_in / w_in (one input per clock);
--                 x_last = '1' together with the final input
--   cycle N+1   : bias, rescale, saturation, activation are computed
--   cycle N+2   : done = '1' for one clock, y_out holds the result
--                 (y_out keeps its value until the next computation ends)
--
-- Generic ACTIVATION : "RELU" or "SIGMOID"
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity neuron is
  generic (
    ACTIVATION : string := "RELU"
  );
  port (
    clk     : in  std_logic;
    rst     : in  std_logic;
    start   : in  std_logic;
    x_valid : in  std_logic;
    x_last  : in  std_logic;
    x_in    : in  data_t;
    w_in    : in  data_t;
    bias    : in  data_t;
    y_out   : out data_t;
    done    : out std_logic
  );
end entity neuron;

architecture rtl of neuron is

  type state_t is (IDLE, ACCUM, FINISH);
  signal state     : state_t := IDLE;

  signal mac_clear : std_logic;
  signal mac_en    : std_logic;
  signal acc       : acc_t  := (others => '0');
  signal act_in    : data_t := (others => '0');   -- biased, rescaled, saturated sum
  signal act_out   : data_t := (others => '0');   -- after the activation function

begin

  ---------------------------------------------------------------------------
  -- Datapath
  ---------------------------------------------------------------------------
  mac_clear <= '1' when (state = IDLE  and start   = '1') else '0';
  mac_en    <= '1' when (state = ACCUM and x_valid = '1') else '0';

  u_mac : entity work.mac_unit
    port map (
      clk   => clk,
      rst   => rst,
      clear => mac_clear,
      en    => mac_en,
      a     => x_in,
      b     => w_in,
      acc   => acc
    );

  act_in <= output_stage(acc, bias);

  g_relu : if ACTIVATION = "RELU" generate
    u_act : entity work.activation_relu
      port map (x_in => act_in, y_out => act_out);
  end generate g_relu;

  g_sigmoid : if ACTIVATION = "SIGMOID" generate
    u_act : entity work.activation_sigmoid
      port map (x_in => act_in, y_out => act_out);
  end generate g_sigmoid;

  g_bad : if ACTIVATION /= "RELU" and ACTIVATION /= "SIGMOID" generate
    assert false
      report "neuron: ACTIVATION must be ""RELU"" or ""SIGMOID"""
      severity failure;
  end generate g_bad;

  ---------------------------------------------------------------------------
  -- Control FSM + output register
  ---------------------------------------------------------------------------
  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state <= IDLE;
        done  <= '0';
        y_out <= (others => '0');
      else
        done <= '0';
        case state is
          when IDLE =>
            if start = '1' then
              state <= ACCUM;
            end if;

          when ACCUM =>
            if x_valid = '1' and x_last = '1' then
              state <= FINISH;
            end if;

          when FINISH =>
            y_out <= act_out;
            done  <= '1';
            state <= IDLE;
        end case;
      end if;
    end if;
  end process;

end architecture rtl;