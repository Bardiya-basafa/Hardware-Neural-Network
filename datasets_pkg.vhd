--------------------------------------------------------------------------------
-- datasets_pkg.vhd
-- Training problems for nn_trainer / nn_autotrain, selected by name.
-- Everything is computed with integers only, so the same functions build the
-- on-chip sample ROM in synthesis and the stimuli in simulation.
--
--   name      topology       samples  task
--   XOR       2-4-1          4        y = x0 xor x1
--   PARITY3   3-8-8-1        8        y = x0 xor x1 xor x2          (2 hidden layers)
--   CIRCLE    2-8-8-1        64       8x8 grid in [-1.75, 1.75]^2,
--                                     y = 1 inside x^2 + y^2 < 2     (2 hidden layers)
--   BLOBS3    2-6-3          24       3 touching point clouds,
--                                     one-hot class                  (multi-class)
--   SEG7      7-12-10        10       7-segment pattern -> digit, one-hot
--
-- Values are Q4.4 (data_t); targets are 0.0 / 1.0 (0 / 16).
-- Sample s, input i is at ds_x(name)(s * N_INPUTS + i); targets likewise.
--
-- Success criterion (identical in the test bench and in nn_autotrain):
--   a sample is "correct" when
--     1 output : (y >= 0.5) = (t >= 0.5)
--     n outputs: argmax(y) = argmax(t)        (first maximum wins)
--   training has converged when
--     number of correct samples >= ds_goal(name)
--     and max |y - t| over all samples / outputs <= ds_err_tol(name)  (Q4.4 LSBs)
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

package datasets_pkg is

  function ds_layers     (name : string) return integer_vector;   -- topology
  function ds_inputs     (name : string) return positive;         -- first size
  function ds_outputs    (name : string) return positive;         -- last size
  function ds_samples    (name : string) return positive;
  function ds_x          (name : string) return data_array_t;     -- samples * inputs
  function ds_t          (name : string) return data_array_t;     -- samples * outputs
  function ds_w_init     (name : string) return master_array_t;
  function ds_b_init     (name : string) return master_array_t;
  function ds_lr_shift   (name : string) return natural;          -- lr = 2^-n
  function ds_max_epochs (name : string) return positive;
  function ds_goal       (name : string) return natural;          -- correct samples needed
  function ds_err_tol    (name : string) return natural;          -- max |y - t| (Q4.4 LSBs)

  -- the success criterion shared by hardware and test bench
  function argmax       (v : data_array_t) return natural;
  function is_correct   (y, t : data_array_t) return boolean;
  function max_abs_err  (y, t : data_array_t) return natural;

end package datasets_pkg;

package body datasets_pkg is

  constant ONE  : integer := 2 ** FRAC_BITS;   -- 1.0 in Q4.4

  procedure unknown (name : string) is
  begin
    report "datasets_pkg: unknown dataset """ & name & """" severity failure;
  end procedure;

  -- 7-segment patterns, segments a b c d e f g, digits 0 .. 9
  type seg7_t is array (0 to 9) of std_logic_vector(0 to 6);
  constant SEG7_PAT : seg7_t := (
    "1111110", "0110000", "1101101", "1111001", "0110011",
    "1011011", "1011111", "1110000", "1111111", "1111011");

  -- BLOBS3: class centres and a common set of offsets (Q4.4 integers)
  type pt_t is array (0 to 1) of integer;
  type pts_t is array (natural range <>) of pt_t;
  constant BLOB_C   : pts_t(0 to 2) := ((-8, -5), (8, -5), (0, 8));   -- (-0.5,-0.31) (0.5,-0.31) (0,0.5)
  constant BLOB_OFS : pts_t(0 to 7) := ((0, 0), (6, 2), (-5, 5), (2, -6),
                                        (-6, -2), (8, -1), (-1, -8), (-3, 7));

  -----------------------------------------------------------------------------
  function ds_layers (name : string) return integer_vector is
  begin
    if    name = "XOR"     then return (2, 4, 1);
    elsif name = "PARITY3" then return (3, 8, 8, 1);
    elsif name = "CIRCLE"  then return (2, 8, 8, 1);
    elsif name = "BLOBS3"  then return (2, 6, 3);
    elsif name = "SEG7"    then return (7, 12, 10);
    end if;
    unknown(name);
    return (1, 1);
  end function;

  function ds_inputs (name : string) return positive is
    constant L : integer_vector := ds_layers(name);
  begin
    return L(L'low);
  end function;

  function ds_outputs (name : string) return positive is
    constant L : integer_vector := ds_layers(name);
  begin
    return L(L'high);
  end function;

  function ds_samples (name : string) return positive is
  begin
    if    name = "XOR"     then return 4;
    elsif name = "PARITY3" then return 8;
    elsif name = "CIRCLE"  then return 64;
    elsif name = "BLOBS3"  then return 24;
    elsif name = "SEG7"    then return 10;
    end if;
    unknown(name);
    return 1;
  end function;

  function ds_x (name : string) return data_array_t is
    constant L  : integer_vector := ds_layers(name);
    constant NI : positive := L(L'low);
    constant NS : positive := ds_samples(name);
    variable r  : data_array_t(0 to NS * NI - 1) := (others => (others => '0'));
    variable c  : integer;
  begin
    for s in 0 to NS - 1 loop
      if name = "XOR" or name = "PARITY3" then       -- binary counting
        for i in 0 to NI - 1 loop
          r(s * NI + i) := to_signed(((s / 2 ** (NI - 1 - i)) mod 2) * ONE, DATA_WIDTH);
        end loop;
      elsif name = "CIRCLE" then                     -- grid -1.75 .. 1.75, step 0.5
        r(s * 2)     := to_signed(-28 + 8 * (s mod 8), DATA_WIDTH);
        r(s * 2 + 1) := to_signed(-28 + 8 * (s / 8),   DATA_WIDTH);
      elsif name = "BLOBS3" then                     -- 8 points per class
        c := s / 8;
        r(s * 2)     := to_signed(BLOB_C(c)(0) + BLOB_OFS(s mod 8)(0), DATA_WIDTH);
        r(s * 2 + 1) := to_signed(BLOB_C(c)(1) + BLOB_OFS(s mod 8)(1), DATA_WIDTH);
      elsif name = "SEG7" then
        for i in 0 to 6 loop
          if SEG7_PAT(s)(i) = '1' then
            r(s * 7 + i) := to_signed(ONE, DATA_WIDTH);
          end if;
        end loop;
      end if;
    end loop;
    return r;
  end function;

  function ds_t (name : string) return data_array_t is
    constant L  : integer_vector := ds_layers(name);
    constant NO : positive := L(L'high);
    constant NS : positive := ds_samples(name);
    variable r  : data_array_t(0 to NS * NO - 1) := (others => (others => '0'));
    variable p, xx, yy : integer;
  begin
    for s in 0 to NS - 1 loop
      if name = "XOR" or name = "PARITY3" then       -- parity of the bits of s
        p := 0;
        for b in 0 to 2 loop p := p + (s / 2 ** b) mod 2; end loop;
        r(s) := to_signed((p mod 2) * ONE, DATA_WIDTH);
      elsif name = "CIRCLE" then                     -- x^2 + y^2 < 2.0  (Q.8: 512)
        xx := -28 + 8 * (s mod 8);
        yy := -28 + 8 * (s / 8);
        if xx * xx + yy * yy < 2 * ONE * ONE then
          r(s) := to_signed(ONE, DATA_WIDTH);
        end if;
      elsif name = "BLOBS3" then                     -- one-hot class
        r(s * 3 + s / 8) := to_signed(ONE, DATA_WIDTH);
      elsif name = "SEG7" then                       -- one-hot digit
        r(s * 10 + s) := to_signed(ONE, DATA_WIDTH);
      end if;
    end loop;
    return r;
  end function;

  function ds_w_init (name : string) return master_array_t is
  begin
    if name = "XOR" then                             -- hand-picked, asymmetric
      return (qm( 0.50), qm( 0.40),  qm(-0.45), qm( 0.55),
              qm( 0.60), qm(-0.35),  qm( 0.30), qm( 0.65),
              qm( 0.45), qm(-0.40),  qm( 0.35), qm(-0.30));
    elsif name = "PARITY3" then return init_weights(ds_layers(name), 11);
    elsif name = "CIRCLE"  then return init_weights(ds_layers(name), 7);
    elsif name = "BLOBS3"  then return init_weights(ds_layers(name), 3);
    elsif name = "SEG7"    then return init_weights(ds_layers(name), 7);
    end if;
    unknown(name);
    return (0 => qm(0.0));
  end function;

  function ds_b_init (name : string) return master_array_t is
  begin
    if name = "XOR" then
      return (qm(0.10), qm(0.05), qm(0.05), qm(-0.10), qm(0.0));
    end if;
    return init_biases(ds_layers(name));
  end function;

  function ds_lr_shift (name : string) return natural is
  begin
    if    name = "XOR"     then return 1;
    elsif name = "PARITY3" then return 3;
    elsif name = "CIRCLE"  then return 3;
    elsif name = "BLOBS3"  then return 2;
    elsif name = "SEG7"    then return 3;
    end if;
    unknown(name);
    return 2;
  end function;

  function ds_max_epochs (name : string) return positive is
  begin
    if    name = "XOR"     then return 3000;
    elsif name = "PARITY3" then return 500;
    elsif name = "CIRCLE"  then return 500;
    elsif name = "BLOBS3"  then return 500;
    elsif name = "SEG7"    then return 500;
    end if;
    unknown(name);
    return 1;
  end function;

  function ds_goal (name : string) return natural is
  begin
    return ds_samples(name);          -- every sample right
  end function;

  function ds_err_tol (name : string) return natural is
  begin
    if name = "XOR" then return 4;     -- 0.25
    end if;
    return 8;                          -- 0.5
  end function;

  -----------------------------------------------------------------------------
  function argmax (v : data_array_t) return natural is
    variable m : natural := 0;
  begin
    for n in 1 to v'length - 1 loop
      if v(v'low + n) > v(v'low + m) then m := n; end if;
    end loop;
    return m;
  end function;

  function is_correct (y, t : data_array_t) return boolean is
    constant HALF : data_t := to_signed(ONE / 2, DATA_WIDTH);
  begin
    if y'length = 1 then
      return (y(y'low) >= HALF) = (t(t'low) >= HALF);
    end if;
    return argmax(y) = argmax(t);
  end function;

  function max_abs_err (y, t : data_array_t) return natural is
    variable m, e : signed(DATA_WIDTH downto 0) := (others => '0');   -- one extra bit
  begin
    for n in 0 to y'length - 1 loop
      e := abs (resize(y(y'low + n), DATA_WIDTH + 1) - resize(t(t'low + n), DATA_WIDTH + 1));
      if e > m then m := e; end if;
    end loop;
    return to_integer(m);
  end function;

end package body datasets_pkg;
