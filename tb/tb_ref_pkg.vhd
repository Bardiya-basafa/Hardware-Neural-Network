--------------------------------------------------------------------------------
-- tb_ref_pkg.vhd  (simulation only)
-- Bit-accurate software reference model of the neuron, written independently
-- of nn_pkg (plain integer arithmetic) so the test benches are not circular.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.nn_pkg.all;

package tb_ref_pkg is
  type int_array_t is array (natural range <>) of integer;

  function floor_div (a, b : integer) return integer;
  function ref_relu (x : integer) return integer;
  function ref_sigmoid (x : integer) return integer;   -- Q4.4 in, Q4.4 out
  function ref_neuron (x, w : int_array_t; b : integer; act : string) return integer;
  -- random integer in lo..hi
  procedure rand_int (variable s1, s2 : inout positive; lo, hi : integer; variable r : out integer);

  ---------------------------------------------------------------------------
  -- Reference model of the whole trainable network (any depth), used to
  -- check nn_trainer bit by bit. Plain integers; parameters are flat arrays
  -- in the layout documented in nn_pkg (Q4.12 master values).
  ---------------------------------------------------------------------------
  function ref_sat (v, bits : integer) return integer;          -- saturate to 'bits'
  function ref_rshift (v, n : integer) return integer;          -- round(v / 2^n), half up
  function ref_deriv (y : integer; act : string) return integer; -- f'(y), Q.8
  -- forward pass; acts = all values, level 0 (= x) first
  procedure ref_forward (sizes : integer_vector; w, b, x : int_array_t;
                         hid_act, out_act : string; acts : out int_array_t);
  -- one SGD step (forward + backprop + update), w / b are updated in place
  procedure ref_train_step (sizes : integer_vector; w, b : inout int_array_t;
                            x, t : int_array_t; lr_shift : natural;
                            hid_act, out_act : string);
  -- the outputs (last level) of ref_forward
  procedure ref_predict (sizes : integer_vector; w, b, x : int_array_t;
                         hid_act, out_act : string; y : out int_array_t);
end package tb_ref_pkg;

package body tb_ref_pkg is

  function floor_div (a, b : integer) return integer is
    variable q : integer := a / b;
  begin
    if (a mod b) /= 0 and ((a < 0) /= (b < 0)) then
      q := q - 1;
    end if;
    return q;
  end function;

  function ref_relu (x : integer) return integer is
  begin
    if x < 0 then return 0; else return x; end if;
  end function;

  function ref_sigmoid (x : integer) return integer is
  begin
    return integer(round(16.0 / (1.0 + exp(-real(x) / 16.0))));
  end function;

  function ref_neuron (x, w : int_array_t; b : integer; act : string) return integer is
    variable s : integer := 0;
    variable r : integer;
  begin
    for n in x'range loop
      s := s + x(n) * w(w'low + n - x'low);
    end loop;
    s := s + b * 16;                      -- bias aligned to Q.8
    r := floor_div(s + 8, 16);            -- round half up, back to Q.4
    if r > 127 then r := 127; elsif r < -128 then r := -128; end if;
    if act = "RELU" then
      return ref_relu(r);
    else
      return ref_sigmoid(r);
    end if;
  end function;

  procedure rand_int (variable s1, s2 : inout positive; lo, hi : integer; variable r : out integer) is
    variable u : real;
  begin
    uniform(s1, s2, u);
    r := lo + integer(floor(u * real(hi - lo + 1)));
    if r > hi then r := hi; end if;
  end procedure;

  -- number of values (inputs + all neuron outputs) of a topology
  function ref_nvals (sizes : integer_vector) return natural is
    variable r : natural := 0;
  begin
    for l in sizes'range loop r := r + sizes(l); end loop;
    return r;
  end function;

  function ref_sat (v, bits : integer) return integer is
  begin
    if v > 2 ** (bits - 1) - 1 then return 2 ** (bits - 1) - 1;
    elsif v < -(2 ** (bits - 1)) then return -(2 ** (bits - 1));
    else return v; end if;
  end function;

  function ref_rshift (v, n : integer) return integer is
  begin
    if n = 0 then return v; end if;
    return floor_div(v + 2 ** (n - 1), 2 ** n);
  end function;

  function ref_deriv (y : integer; act : string) return integer is
  begin
    if act = "RELU" then
      if y > 0 then return 256; else return 0; end if;
    else
      return y * (16 - y);               -- y(1-y): Q.4 * Q.4 = Q.8
    end if;
  end function;

  procedure ref_forward (sizes : integer_vector; w, b, x : int_array_t;
                         hid_act, out_act : string; acts : out int_array_t) is
    alias z : integer_vector(0 to sizes'length - 1) is sizes;
    constant NL : natural := z'length - 1;
    variable a  : int_array_t(0 to acts'length - 1);
    variable wo, bo, ai, ao : natural := 0;   -- offsets: weights, biases, level in, level out
    variable s, r : integer;
  begin
    for i in 0 to z(0) - 1 loop a(i) := x(x'low + i); end loop;
    ao := z(0);
    for l in 0 to NL - 1 loop
      for k in 0 to z(l + 1) - 1 loop
        s := 0;
        for j in 0 to z(l) - 1 loop       -- Q4.4 forward weight = round(master / 256)
          s := s + a(ai + j) * ref_sat(ref_rshift(w(w'low + wo + k * z(l) + j), 8), 8);
        end loop;
        s := s + ref_sat(ref_rshift(b(b'low + bo + k), 8), 8) * 16;
        r := ref_sat(floor_div(s + 8, 16), 8);
        if l = NL - 1 then
          if out_act = "RELU" then a(ao + k) := ref_relu(r); else a(ao + k) := ref_sigmoid(r); end if;
        else
          if hid_act = "RELU" then a(ao + k) := ref_relu(r); else a(ao + k) := ref_sigmoid(r); end if;
        end if;
      end loop;
      wo := wo + z(l) * z(l + 1);
      bo := bo + z(l + 1);
      ai := ao;
      ao := ao + z(l + 1);
    end loop;
    acts := a;
  end procedure;

  procedure ref_predict (sizes : integer_vector; w, b, x : int_array_t;
                         hid_act, out_act : string; y : out int_array_t) is
    alias z : integer_vector(0 to sizes'length - 1) is sizes;
    constant NV : natural := ref_nvals(sizes);
    variable a  : int_array_t(0 to NV - 1);
  begin
    ref_forward(sizes, w, b, x, hid_act, out_act, a);
    for k in 0 to z(z'high) - 1 loop y(y'low + k) := a(NV - z(z'high) + k); end loop;
  end procedure;

  procedure ref_train_step (sizes : integer_vector; w, b : inout int_array_t;
                            x, t : int_array_t; lr_shift : natural;
                            hid_act, out_act : string) is
    alias z : integer_vector(0 to sizes'length - 1) is sizes;
    constant NL : natural := z'length - 1;
    constant NV : natural := ref_nvals(sizes);
    variable a  : int_array_t(0 to NV - 1);                 -- all values
    variable d  : int_array_t(0 to NV - z(0) - 1);          -- deltas (bias layout)
    variable wb, bb, ab : int_array_t(0 to NL);             -- layer / level offsets
    variable fd, bs : integer;
    variable acc : real;                                    -- exact: |values| << 2^53
  begin
    wb(0) := 0; bb(0) := 0; ab(0) := 0;
    for l in 1 to NL loop
      wb(l) := wb(l - 1) + z(l - 1) * z(l);
      bb(l) := bb(l - 1) + z(l);
      ab(l) := ab(l - 1) + z(l - 1);
    end loop;
    ref_forward(sizes, w, b, x, hid_act, out_act, a);
    -- output deltas
    for k in 0 to z(NL) - 1 loop
      if out_act = "SIGMOID" then fd := 256; else fd := ref_deriv(a(ab(NL) + k), out_act); end if;
      d(bb(NL - 1) + k) := ref_sat(ref_rshift((a(ab(NL) + k) - t(t'low + k)) * fd, 4), 16);
    end loop;
    -- hidden deltas, top down, with the old weights
    for l in NL - 1 downto 1 loop
      for j in 0 to z(l) - 1 loop
        acc := 0.0;
        for k in 0 to z(l + 1) - 1 loop
          acc := acc + real(w(w'low + wb(l) + k * z(l) + j)) * real(d(bb(l) + k));
        end loop;
        acc := floor((acc + 2048.0) / 4096.0);       -- round half up, Q.20 -> Q.8
        if acc > 32767.0 then bs := 32767; elsif acc < -32768.0 then bs := -32768;
        else bs := integer(acc); end if;
        d(bb(l - 1) + j) := ref_sat(ref_rshift(bs * ref_deriv(a(ab(l) + j), hid_act), 8), 16);
      end loop;
    end loop;
    -- SGD update
    for l in 0 to NL - 1 loop
      for k in 0 to z(l + 1) - 1 loop
        for j in 0 to z(l) - 1 loop
          w(w'low + wb(l) + k * z(l) + j) :=
            ref_sat(w(w'low + wb(l) + k * z(l) + j) - ref_rshift(d(bb(l) + k) * a(ab(l) + j), lr_shift), 16);
        end loop;
        b(b'low + bb(l) + k) := ref_sat(b(b'low + bb(l) + k) - ref_rshift(d(bb(l) + k) * 16, lr_shift), 16);
      end loop;
    end loop;
  end procedure;

end package body tb_ref_pkg;
