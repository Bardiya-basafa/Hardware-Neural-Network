--------------------------------------------------------------------------------
-- nn_pkg.vhd
-- Common types, fixed-point formats and helper functions.
--
-- Fixed-point formats (all two's complement)
--   data_t   : Q4.4   (DATA_WIDTH = 8,  FRAC_BITS = 4)  inputs, weights and
--              activations of the forward (inference) datapath
--   acc_t    : MAC accumulator, 2*FRAC_BITS fractional bits, ACC_GUARD extra
--              integer bits so that up to 2**ACC_GUARD products never overflow
--   master_t : Q4.12  (MASTER_WIDTH = 16, MASTER_FRAC = 12) high precision
--              "master" copy of a trainable weight. The forward pass uses the
--              weight rounded to Q4.4 (to_data); the update is applied to the
--              master copy so that small gradient steps are not lost.
--   delta_t  : Q8.8   (DELTA_WIDTH = 16, DELTA_FRAC = 8) back-propagated errors
--              and activation derivatives
--
-- Network topology ("layer sizes")
--   A network is described by an integer_vector of sizes, e.g. (2, 8, 8, 1):
--     level 0      = the inputs            (here 2)
--     level 1..L-1 = hidden layers         (here 8 and 8)
--     level L      = the outputs           (here 1)
--   Layer l (l = 0 .. L-1) maps level l to level l+1. All parameters of the
--   whole network live in two flat arrays:
--     weights : layer 0, layer 1, ... ; inside a layer W(neuron*N_IN + input)
--     biases  : layer 0, layer 1, ... ; one per neuron
--   The *_base functions return where each layer / level starts.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package nn_pkg is

  -- forward datapath ---------------------------------------------------------
  constant DATA_WIDTH : positive := 8;
  constant FRAC_BITS  : natural  := 4;
  constant ACC_GUARD  : natural  := 8;
  constant ACC_WIDTH  : positive := 2 * DATA_WIDTH + ACC_GUARD;

  subtype data_t is signed(DATA_WIDTH - 1 downto 0);
  subtype acc_t  is signed(ACC_WIDTH - 1 downto 0);
  type data_array_t is array (natural range <>) of data_t;

  -- training datapath --------------------------------------------------------
  constant MASTER_WIDTH : positive := 16;
  constant MASTER_FRAC  : natural  := 12;
  constant DELTA_WIDTH  : positive := 16;
  constant DELTA_FRAC   : natural  := 8;

  subtype master_t is signed(MASTER_WIDTH - 1 downto 0);
  subtype delta_t  is signed(DELTA_WIDTH - 1 downto 0);
  type master_array_t is array (natural range <>) of master_t;
  type delta_array_t  is array (natural range <>) of delta_t;

  -- helpers ------------------------------------------------------------------
  -- saturate a signed value of any width into 'width' bits
  function saturate (x : signed; width : positive) return signed;
  -- arithmetic shift right by 'n' with round-half-up (n = 0 -> unchanged)
  function round_shift (x : signed; n : natural) return signed;
  -- MAC result (2*FRAC_BITS fractional bits) + bias -> rounded, saturated Q4.4
  function output_stage (acc : acc_t; bias : data_t) return data_t;
  -- Q4.12 master weight -> Q4.4 forward weight (rounded, saturated)
  function to_data (w : master_t) return data_t;
  -- convert real -> fixed point (for constants / test benches)
  function to_fixed (r : real; width : positive; frac : natural) return signed;
  function q  (r : real) return data_t;     -- real -> Q4.4
  function qm (r : real) return master_t;   -- real -> Q4.12

  -- topology helpers (see header) --------------------------------------------
  function n_layers  (s : integer_vector) return positive;  -- L = s'length - 1
  function n_weights (s : integer_vector) return positive;  -- sum s(l) * s(l+1)
  function n_neurons (s : integer_vector) return positive;  -- sum s(1 .. L) (= biases = deltas)
  function n_values  (s : integer_vector) return positive;  -- sum s(0 .. L) (inputs + all outputs)
  function max_size  (s : integer_vector) return positive;  -- largest level
  function w_base (s : integer_vector) return integer_vector;  -- first weight of layer l  (0..L-1)
  function b_base (s : integer_vector) return integer_vector;  -- first bias of layer l    (0..L-1)
  function a_base (s : integer_vector) return integer_vector;  -- first value of level l   (0..L)

  -- deterministic pseudo-random initial parameters (integer only, synthesizable)
  --   weights : |w| in 0.125 .. 0.61, random sign (never rounds to 0 in Q4.4)
  --   biases  : hidden neurons 0.0625 (keeps ReLUs alive), output neurons 0
  function init_weights (s : integer_vector; seed : positive) return master_array_t;
  function init_biases  (s : integer_vector) return master_array_t;

end package nn_pkg;

package body nn_pkg is

  function saturate (x : signed; width : positive) return signed is
    constant MAXV : signed(width - 1 downto 0) := '0' & (width - 2 downto 0 => '1');
    constant MINV : signed(width - 1 downto 0) := '1' & (width - 2 downto 0 => '0');
  begin
    if x'length <= width then
      return resize(x, width);
    elsif x > resize(MAXV, x'length) then
      return MAXV;
    elsif x < resize(MINV, x'length) then
      return MINV;
    else
      return resize(x, width);
    end if;
  end function;

  function round_shift (x : signed; n : natural) return signed is
    variable t : signed(x'length downto 0);   -- one extra bit for the +0.5
  begin
    if n = 0 then
      return x;
    end if;
    t := resize(x, x'length + 1) + shift_left(to_signed(1, x'length + 1), n - 1);
    return resize(shift_right(t, n), x'length);
  end function;

  function output_stage (acc : acc_t; bias : data_t) return data_t is
    variable s : signed(ACC_WIDTH downto 0);
  begin
    s := resize(acc, ACC_WIDTH + 1)
         + shift_left(resize(bias, ACC_WIDTH + 1), FRAC_BITS);
    return saturate(round_shift(s, FRAC_BITS), DATA_WIDTH);
  end function;

  function to_data (w : master_t) return data_t is
  begin
    return saturate(round_shift(w, MASTER_FRAC - FRAC_BITS), DATA_WIDTH);
  end function;

  function to_fixed (r : real; width : positive; frac : natural) return signed is
    variable v : integer;
  begin
    v := integer(r * real(2 ** frac));   -- integer() rounds to nearest
    if v > 2 ** (width - 1) - 1 then
      v := 2 ** (width - 1) - 1;
    elsif v < -(2 ** (width - 1)) then
      v := -(2 ** (width - 1));
    end if;
    return to_signed(v, width);
  end function;

  function q (r : real) return data_t is
  begin
    return to_fixed(r, DATA_WIDTH, FRAC_BITS);
  end function;

  function qm (r : real) return master_t is
  begin
    return to_fixed(r, MASTER_WIDTH, MASTER_FRAC);
  end function;

  -----------------------------------------------------------------------------
  -- topology helpers
  -----------------------------------------------------------------------------
  function n_layers (s : integer_vector) return positive is
  begin
    return s'length - 1;
  end function;

  function n_weights (s : integer_vector) return positive is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : natural := 0;
  begin
    for l in 0 to z'high - 1 loop r := r + z(l) * z(l + 1); end loop;
    return r;
  end function;

  function n_neurons (s : integer_vector) return positive is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : natural := 0;
  begin
    for l in 1 to z'high loop r := r + z(l); end loop;
    return r;
  end function;

  function n_values (s : integer_vector) return positive is
    variable r : natural := 0;
  begin
    for l in s'range loop r := r + s(l); end loop;
    return r;
  end function;

  function max_size (s : integer_vector) return positive is
    variable r : natural := 1;
  begin
    for l in s'range loop
      if s(l) > r then r := s(l); end if;
    end loop;
    return r;
  end function;

  function w_base (s : integer_vector) return integer_vector is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : integer_vector(0 to z'high - 1);
    variable o : natural := 0;
  begin
    for l in 0 to z'high - 1 loop r(l) := o; o := o + z(l) * z(l + 1); end loop;
    return r;
  end function;

  function b_base (s : integer_vector) return integer_vector is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : integer_vector(0 to z'high - 1);
    variable o : natural := 0;
  begin
    for l in 0 to z'high - 1 loop r(l) := o; o := o + z(l + 1); end loop;
    return r;
  end function;

  function a_base (s : integer_vector) return integer_vector is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : integer_vector(0 to z'high);
    variable o : natural := 0;
  begin
    for l in 0 to z'high loop r(l) := o; o := o + z(l); end loop;
    return r;
  end function;

  function init_weights (s : integer_vector; seed : positive) return master_array_t is
    variable r   : master_array_t(0 to n_weights(s) - 1);
    variable x   : natural := seed mod 65537;
    variable mag : natural;
  begin
    for n in r'range loop
      x   := (75 * x + 74) mod 65537;               -- small LCG, fits in 32 bit
      mag := 2 ** (MASTER_FRAC - 3) + (x mod 32) * 2 ** (MASTER_FRAC - 6);  -- 0.125 .. 0.61
      if (x / 32) mod 2 = 1 then
        r(n) := to_signed(-mag, MASTER_WIDTH);
      else
        r(n) := to_signed(mag, MASTER_WIDTH);
      end if;
    end loop;
    return r;
  end function;

  function init_biases (s : integer_vector) return master_array_t is
    alias z : integer_vector(0 to s'length - 1) is s;
    variable r : master_array_t(0 to n_neurons(s) - 1) := (others => (others => '0'));
    variable o : natural := 0;
  begin
    for l in 1 to z'high - 1 loop                     -- hidden levels only
      for n in 0 to z(l) - 1 loop
        r(o + n) := to_signed(2 ** (MASTER_FRAC - 4), MASTER_WIDTH);   -- 0.0625
      end loop;
      o := o + z(l);
    end loop;
    return r;
  end function;

end package body nn_pkg;
