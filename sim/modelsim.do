# ModelSim / Questa script. From the project root:
#   vsim -do sim/modelsim.do                                   (tb_nn_trainer, XOR)
#   vsim -do "set DS PARITY3; do sim/modelsim.do"              (another dataset)
#   vsim -do "set TB tb_nn_autotrain; set DS SEG7; do sim/modelsim.do"
#   vsim -do "set TB tb_layer; do sim/modelsim.do"             (a unit test bench)
# All sources are VHDL-2008 (integer_vector generics, unary "and", process(all)).
if {![info exists TB]} { set TB tb_nn_trainer }
if {![info exists DS]} { set DS XOR }

if {[file exists work]} { vdel -lib work -all }
vlib work
# compile order: packages first, then bottom-up
vcom -2008 -work work \
  nn_pkg.vhd datasets_pkg.vhd weights_pkg.vhd mac_unit.vhd activation_relu.vhd \
  activation_sigmoid.vhd activation_deriv.vhd neuron.vhd layer.vhd \
  nn_top.vhd nn_trainer.vhd nn_autotrain.vhd nn_train_top.vhd \
  tb/tb_ref_pkg.vhd tb/tb_activation.vhd tb/tb_neuron.vhd tb/tb_layer.vhd \
  tb/tb_nn_top.vhd tb/tb_nn_trainer.vhd tb/tb_nn_autotrain.vhd

if {$TB eq "tb_nn_trainer" || $TB eq "tb_nn_autotrain"} {
  vsim -onfinish stop -voptargs=+acc -gDATASET=$DS work.$TB
} else {
  vsim -onfinish stop -voptargs=+acc work.$TB
}
add wave -radix decimal /$TB/*

if {$TB eq "tb_nn_trainer"} {
  set D /tb_nn_trainer/u_dut
  add wave -divider "learning progress"
  add wave -format analog-step -height 80 -min 0 -max 64 /tb_nn_trainer/sum_abs_err
  add wave -divider "trainer FSM / counters"
  add wave $D/state
  add wave -radix unsigned $D/l_cnt $D/k_cnt $D/j_cnt $D/w_addr
  add wave -divider "forward values, deltas, weights"
  add wave -radix decimal $D/lstart $D/acts $D/d $D/bacc $D/mul_a $D/mul_b $D/prod $D/wm $D/bm
}
if {$TB eq "tb_nn_autotrain"} {
  set D /tb_nn_autotrain/u_dut
  add wave -divider "autotrain sequencer"
  add wave $D/state $D/eval_pass
  add wave -radix unsigned $D/s_cnt $D/ep_cnt $D/corr_cnt $D/err_max
  add wave -divider "trainer"
  add wave $D/u_trainer/state
  add wave -radix decimal $D/u_trainer/acts $D/u_trainer/wm
}
run -all
wave zoom full
