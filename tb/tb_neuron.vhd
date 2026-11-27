--------------------------------------------------------------------------------
-- tb_neuron.vhd
-- Random test of one neuron (ReLU and Sigmoid instances side by side):
--   * result compared with the reference model
--   * done must come exactly N + 2 clocks after start
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.tb_ref_pkg.all;

entity tb_neuron is
  generic (N : positive := 5; N_TESTS : positive := 200);
end entity tb_neuron;

architecture sim of tb_neuron is
  constant T : time := 10 ns;

  signal clk     : std_logic := '0';
  signal rst     : std_logic := '1';
  signal start   : std_logic := '0';
  signal x_valid : std_logic := '0';
  signal x_last  : std_logic := '0';
  signal x_in    : data_t := (others => '0');
  signal w_in    : data_t := (others => '0');
  signal bias    : data_t := (others => '0');
  signal y_relu, y_sig       : data_t;
  signal done_relu, done_sig : std_logic;
  signal running : boolean := true;
begin

  clk <= not clk after T / 2 when running;

  u_relu : entity work.neuron generic map (ACTIVATION => "RELU")
    port map (clk, rst, start, x_valid, x_last, x_in, w_in, bias, y_relu, done_relu);
  u_sig  : entity work.neuron generic map (ACTIVATION => "SIGMOID")
    port map (clk, rst, start, x_valid, x_last, x_in, w_in, bias, y_sig, done_sig);

  process
    variable s1 : positive := 11;
    variable s2 : positive := 29;
    variable xs, ws : int_array_t(0 to N - 1);
    variable b, cycles : integer;
    variable errors : natural := 0;
  begin
    wait for 3 * T;
    wait until rising_edge(clk);
    rst <= '0';

    for tc in 1 to N_TESTS loop
      -- small values most of the time, full range sometimes (tests saturation)
      for m in 0 to N - 1 loop
        if tc mod 10 = 0 then
          rand_int(s1, s2, -128, 127, xs(m)); rand_int(s1, s2, -128, 127, ws(m));
        else
          rand_int(s1, s2, -32, 32, xs(m));   rand_int(s1, s2, -32, 32, ws(m));
        end if;
      end loop;
      rand_int(s1, s2, -32, 32, b);

      bias  <= to_signed(b, DATA_WIDTH);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      for m in 0 to N - 1 loop
        x_valid <= '1';
        x_in    <= to_signed(xs(m), DATA_WIDTH);
        w_in    <= to_signed(ws(m), DATA_WIDTH);
        if m = N - 1 then x_last <= '1'; end if;
        wait until rising_edge(clk);
      end loop;
      x_valid <= '0';
      x_last  <= '0';

      cycles := N;
      loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
        exit when done_relu = '1' or cycles > N + 10;
      end loop;

      if cycles /= N + 2 then
        report "latency " & integer'image(cycles) & ", expected " & integer'image(N + 2) severity error;
        errors := errors + 1;
      end if;
      if done_sig /= '1' then
        report "sigmoid neuron done not aligned with relu neuron" severity error;
        errors := errors + 1;
      end if;
      if to_integer(y_relu) /= ref_neuron(xs, ws, b, "RELU") then
        report "ReLU neuron: got " & integer'image(to_integer(y_relu)) & ", expected " &
               integer'image(ref_neuron(xs, ws, b, "RELU")) severity error;
        errors := errors + 1;
      end if;
      if to_integer(y_sig) /= ref_neuron(xs, ws, b, "SIGMOID") then
        report "Sigmoid neuron: got " & integer'image(to_integer(y_sig)) & ", expected " &
               integer'image(ref_neuron(xs, ws, b, "SIGMOID")) severity error;
        errors := errors + 1;
      end if;
      wait until rising_edge(clk);
    end loop;

    if errors = 0 then
      report "tb_neuron: PASS (" & integer'image(N_TESTS) & " vectors)" severity note;
    else
      report "tb_neuron: FAIL, " & integer'image(errors) & " errors" severity failure;
    end if;
    running <= false;
    std.env.finish;
  end process;

end architecture sim;
