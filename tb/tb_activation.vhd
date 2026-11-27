--------------------------------------------------------------------------------
-- tb_activation.vhd
-- Exhaustive test (all 256 Q4.4 inputs) of activation_relu, activation_sigmoid
-- and activation_deriv against a real-number reference.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.nn_pkg.all;
use work.tb_ref_pkg.all;

entity tb_activation is
end entity tb_activation;

architecture sim of tb_activation is
  signal x        : data_t := (others => '0');
  signal y_relu   : data_t;
  signal y_sig    : data_t;
  signal d_relu   : delta_t;
  signal d_sig    : delta_t;
begin

  u_relu : entity work.activation_relu    port map (x_in => x, y_out => y_relu);
  u_sig  : entity work.activation_sigmoid port map (x_in => x, y_out => y_sig);
  -- derivative units are fed with the activation OUTPUT
  u_drelu : entity work.activation_deriv generic map ("RELU")    port map (y_in => y_relu, d_out => d_relu);
  u_dsig  : entity work.activation_deriv generic map ("SIGMOID") port map (y_in => y_sig,  d_out => d_sig);

  process
    variable errors : natural := 0;
    variable xi, er, es, edr, eds : integer;
  begin
    for v in -128 to 127 loop
      x <= to_signed(v, DATA_WIDTH);
      wait for 10 ns;
      xi  := v;
      er  := ref_relu(xi);
      es  := ref_sigmoid(xi);
      if er > 0 then edr := 256; else edr := 0; end if;   -- 1.0 in Q.8
      eds := es * (16 - es);                               -- y(1-y) in Q.8
      if to_integer(y_relu) /= er then
        report "ReLU(" & integer'image(xi) & ") = " & integer'image(to_integer(y_relu)) &
               ", expected " & integer'image(er) severity error;
        errors := errors + 1;
      end if;
      if to_integer(y_sig) /= es then
        report "Sigmoid(" & integer'image(xi) & ") = " & integer'image(to_integer(y_sig)) &
               ", expected " & integer'image(es) severity error;
        errors := errors + 1;
      end if;
      if to_integer(d_relu) /= edr or to_integer(d_sig) /= eds then
        report "derivative mismatch at x = " & integer'image(xi) severity error;
        errors := errors + 1;
      end if;
    end loop;

    if errors = 0 then
      report "tb_activation: PASS (256 inputs)" severity note;
    else
      report "tb_activation: FAIL, " & integer'image(errors) & " errors" severity failure;
    end if;
    std.env.finish;
  end process;

end architecture sim;
