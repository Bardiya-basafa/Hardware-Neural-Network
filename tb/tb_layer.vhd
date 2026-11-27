--------------------------------------------------------------------------------
-- tb_layer.vhd
-- Random test of a fully connected layer (ReLU and Sigmoid instances):
--   * every neuron output compared with the reference model
--   * done must come exactly N_INPUTS + 2 clocks after start
--   * x_in is changed right after start to check that the layer latched it
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.tb_ref_pkg.all;

entity tb_layer is
  generic (N_INPUTS : positive := 3; N_NEURONS : positive := 4; N_TESTS : positive := 100);
end entity tb_layer;

architecture sim of tb_layer is
  constant T : time := 10 ns;

  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal start : std_logic := '0';
  signal x_in  : data_array_t(0 to N_INPUTS - 1) := (others => (others => '0'));
  signal w_in  : data_array_t(0 to N_INPUTS * N_NEURONS - 1) := (others => (others => '0'));
  signal b_in  : data_array_t(0 to N_NEURONS - 1) := (others => (others => '0'));
  signal y_relu, y_sig       : data_array_t(0 to N_NEURONS - 1);
  signal done_relu, done_sig : std_logic;
  signal running : boolean := true;
begin

  clk <= not clk after T / 2 when running;

  u_relu : entity work.layer
    generic map (N_INPUTS => N_INPUTS, N_NEURONS => N_NEURONS, ACTIVATION => "RELU")
    port map (clk, rst, start, x_in, w_in, b_in, y_relu, done_relu);
  u_sig : entity work.layer
    generic map (N_INPUTS => N_INPUTS, N_NEURONS => N_NEURONS, ACTIVATION => "SIGMOID")
    port map (clk, rst, start, x_in, w_in, b_in, y_sig, done_sig);

  process
    variable s1 : positive := 7;
    variable s2 : positive := 1234;
    variable xs : int_array_t(0 to N_INPUTS - 1);
    variable ws : int_array_t(0 to N_INPUTS * N_NEURONS - 1);
    variable bs : int_array_t(0 to N_NEURONS - 1);
    variable cycles : integer;
    variable errors : natural := 0;
    variable e_relu, e_sig : integer;
  begin
    wait for 3 * T;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);

    for tc in 1 to N_TESTS loop
      for n in xs'range loop rand_int(s1, s2, -40, 40, xs(n)); x_in(n) <= to_signed(xs(n), DATA_WIDTH); end loop;
      for n in ws'range loop rand_int(s1, s2, -40, 40, ws(n)); w_in(n) <= to_signed(ws(n), DATA_WIDTH); end loop;
      for n in bs'range loop rand_int(s1, s2, -32, 32, bs(n)); b_in(n) <= to_signed(bs(n), DATA_WIDTH); end loop;

      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      x_in  <= (others => to_signed(-1, DATA_WIDTH));   -- must not affect the result

      cycles := 0;
      loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
        exit when done_relu = '1' or cycles > N_INPUTS + 10;
      end loop;

      if cycles /= N_INPUTS + 2 or done_sig /= '1' then
        report "latency " & integer'image(cycles) & ", expected " & integer'image(N_INPUTS + 2) severity error;
        errors := errors + 1;
      end if;
      for j in 0 to N_NEURONS - 1 loop
        e_relu := ref_neuron(xs, ws(j * N_INPUTS to j * N_INPUTS + N_INPUTS - 1), bs(j), "RELU");
        e_sig  := ref_neuron(xs, ws(j * N_INPUTS to j * N_INPUTS + N_INPUTS - 1), bs(j), "SIGMOID");
        if to_integer(y_relu(j)) /= e_relu or to_integer(y_sig(j)) /= e_sig then
          report "test " & integer'image(tc) & " neuron " & integer'image(j) &
                 ": relu " & integer'image(to_integer(y_relu(j))) & "/" & integer'image(e_relu) &
                 ", sigmoid " & integer'image(to_integer(y_sig(j))) & "/" & integer'image(e_sig)
                 severity error;
          errors := errors + 1;
        end if;
      end loop;
      wait until rising_edge(clk);      -- layer back in IDLE -> next start
    end loop;

    if errors = 0 then
      report "tb_layer: PASS (" & integer'image(N_TESTS) & " vectors)" severity note;
    else
      report "tb_layer: FAIL, " & integer'image(errors) & " errors" severity failure;
    end if;
    running <= false;
    std.env.finish;
  end process;

end architecture sim;
