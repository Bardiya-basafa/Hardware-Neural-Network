--------------------------------------------------------------------------------
-- mac_unit.vhd
-- Multiply-Accumulate:  acc <= acc + a * b
--   clear : synchronous clear of the accumulator (priority over en)
--   en    : perform one multiply-accumulate on this clock edge
-- The accumulator is wide enough to avoid overflow (see ACC_WIDTH in nn_pkg).
-- Synthesises into a single DSP block on most FPGAs.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity mac_unit is
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    clear : in  std_logic;
    en    : in  std_logic;
    a     : in  data_t;
    b     : in  data_t;
    acc   : out acc_t
  );
end entity mac_unit;

architecture rtl of mac_unit is
  signal acc_r : acc_t := (others => '0');
begin

  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or clear = '1' then
        acc_r <= (others => '0');
      elsif en = '1' then
        acc_r <= acc_r + resize(a * b, ACC_WIDTH);
      end if;
    end if;
  end process;

  acc <= acc_r;

end architecture rtl;