--------------------------------------------------------------------------------
-- activation_deriv.vhd
-- Derivative of the activation function, expressed in terms of the activation
-- OUTPUT y (so no pre-activation value has to be stored):
--   RELU    : f'(z) = 1 if z > 0 else 0       ->  1 if y > 0 else 0
--   SIGMOID : f'(z) = s(z) * (1 - s(z))       ->  y * (1 - y)
-- Input  y     : Q4.4 (data_t)
-- Output d_out : Q8.8 (delta_t, DELTA_FRAC = 2 * FRAC_BITS)  (purely combinational)
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity activation_deriv is
  generic (
    ACTIVATION : string := "RELU"
  );
  port (
    y_in  : in  data_t;
    d_out : out delta_t
  );
end entity activation_deriv;

architecture rtl of activation_deriv is
  constant ONE_Q4 : data_t  := to_signed(2 ** FRAC_BITS, DATA_WIDTH);
  constant ONE_D  : delta_t := to_signed(2 ** DELTA_FRAC, DELTA_WIDTH);
begin

  assert DELTA_FRAC = 2 * FRAC_BITS
    report "activation_deriv: DELTA_FRAC must be 2 * FRAC_BITS"
    severity failure;

  g_relu : if ACTIVATION = "RELU" generate
    d_out <= ONE_D when y_in > 0 else (others => '0');
  end generate g_relu;

  g_sigmoid : if ACTIVATION = "SIGMOID" generate
    -- y (Q.4) * (1 - y) (Q.4) = Q.8 ; y is in 0 .. 1 so the product is 0 .. 0.25
    d_out <= resize(y_in * (ONE_Q4 - y_in), DELTA_WIDTH);
  end generate g_sigmoid;

  g_bad : if ACTIVATION /= "RELU" and ACTIVATION /= "SIGMOID" generate
    assert false
      report "activation_deriv: ACTIVATION must be ""RELU"" or ""SIGMOID"""
      severity failure;
  end generate g_bad;

end architecture rtl;
