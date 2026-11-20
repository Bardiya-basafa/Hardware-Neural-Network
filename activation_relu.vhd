--------------------------------------------------------------------------------
-- activation_relu.vhd
-- ReLU:  y = max(0, x)     (purely combinational)
-- Same port interface as activation_sigmoid so the two are interchangeable.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity activation_relu is
  port (
    x_in  : in  data_t;
    y_out : out data_t
  );
end entity activation_relu;

architecture rtl of activation_relu is
begin
  y_out <= (others => '0') when x_in(x_in'high) = '1' else x_in;
end architecture rtl;