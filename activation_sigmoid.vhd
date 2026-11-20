--------------------------------------------------------------------------------
-- activation_sigmoid.vhd
-- Sigmoid via a 256-entry look-up table (combinational ROM).
--   address = two's-complement bit pattern of the Q4.4 input
--   content = round(16 / (1 + exp(-x/16)))   -> Q4.4 value in 0.0 .. 1.0
-- The LUT is built for DATA_WIDTH = 8, FRAC_BITS = 4 (checked by an assertion).
-- Same port interface as activation_relu.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity activation_sigmoid is
  port (
    x_in  : in  data_t;
    y_out : out data_t
  );
end entity activation_sigmoid;

architecture rtl of activation_sigmoid is

  type lut_t is array (0 to 255) of integer range 0 to 16;

  constant SIGMOID_LUT : lut_t := (
         8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,
         4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8
  );

begin

  assert DATA_WIDTH = 8 and FRAC_BITS = 4
    report "activation_sigmoid: LUT is built for Q4.4 (8 bit / 4 fractional bits)"
    severity failure;

  y_out <= to_signed(SIGMOID_LUT(to_integer(unsigned(x_in))), DATA_WIDTH);

end architecture rtl;