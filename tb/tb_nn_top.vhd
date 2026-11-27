--------------------------------------------------------------------------------
-- tb_nn_top.vhd
-- Inference test of the complete network (weights from weights_pkg: XOR).
-- Applies the four XOR input patterns (0.0 / 1.0 in Q4.4) and checks y.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity tb_nn_top is
end entity tb_nn_top;

architecture sim of tb_nn_top is
  constant T : time := 10 ns;

  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal start : std_logic := '0';
  signal x_in  : data_array_t(0 to NN_INPUTS - 1) := (others => (others => '0'));
  signal y_out : data_array_t(0 to NN_OUTPUTS - 1);
  signal done  : std_logic;
  signal running : boolean := true;

  type pattern_t is array (0 to 3, 0 to 2) of real;   -- x0, x1, expected y
  constant PATTERNS : pattern_t := (
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0)
  );
begin

  clk <= not clk after T / 2 when running;

  u_dut : entity work.nn_top
    port map (clk => clk, rst => rst, start => start, x_in => x_in, y_out => y_out, done => done);

  process
    variable errors : natural := 0;
  begin
    wait for 3 * T;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);

    for p in 0 to 3 loop
      x_in  <= (q(PATTERNS(p, 0)), q(PATTERNS(p, 1)));
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      wait until rising_edge(clk) and done = '1';
      report "XOR(" & real'image(PATTERNS(p, 0)) & ", " & real'image(PATTERNS(p, 1)) & ") = " &
             real'image(real(to_integer(y_out(0))) / 16.0);
      if y_out(0) /= q(PATTERNS(p, 2)) then
        report "wrong result, expected " & real'image(PATTERNS(p, 2)) severity error;
        errors := errors + 1;
      end if;
      wait until rising_edge(clk);
    end loop;

    if errors = 0 then
      report "tb_nn_top: PASS" severity note;
    else
      report "tb_nn_top: FAIL, " & integer'image(errors) & " errors" severity failure;
    end if;
    running <= false;
    std.env.finish;
  end process;

end architecture sim;
