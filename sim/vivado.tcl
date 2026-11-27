# Vivado (xsim) script. From the project root:
#   vivado -mode batch -source sim/vivado.tcl                              (tb_nn_trainer, XOR)
#   vivado -mode gui   -source sim/vivado.tcl -tclargs tb_nn_trainer CIRCLE
#   vivado -mode gui   -source sim/vivado.tcl -tclargs tb_nn_autotrain SEG7
#   vivado -mode gui   -source sim/vivado.tcl -tclargs tb_layer
# Creates build/vivado_proj; every source is set to VHDL 2008.
# Synthesis top: nn_top (fixed weights). For the self-training design set
#   set_property top nn_train_top [current_fileset]
#   set_property generic {DATASET=PARITY3} [current_fileset]
set tb   [expr {[llength $argv] > 0 ? [lindex $argv 0] : "tb_nn_trainer"}]
set ds   [expr {[llength $argv] > 1 ? [lindex $argv 1] : "XOR"}]
set root [file normalize [file join [file dirname [info script]] ..]]

create_project -force nn_sim $root/build/vivado_proj
# compile order: packages first, then bottom-up
set rtl {nn_pkg datasets_pkg weights_pkg mac_unit activation_relu activation_sigmoid
         activation_deriv neuron layer nn_top nn_trainer nn_autotrain nn_train_top}
set tbs {tb_ref_pkg tb_activation tb_neuron tb_layer tb_nn_top tb_nn_trainer tb_nn_autotrain}

foreach f $rtl { add_files -norecurse $root/$f.vhd }
foreach f $tbs { add_files -fileset sim_1 -norecurse $root/tb/$f.vhd }
set_property file_type {VHDL 2008} [get_files -of_objects [get_filesets {sources_1 sim_1}] -filter {FILE_TYPE == VHDL}]

set_property top nn_top [current_fileset]
set_property top $tb [get_filesets sim_1]
if {$tb eq "tb_nn_trainer" || $tb eq "tb_nn_autotrain"} {
  set_property generic "DATASET=$ds" [get_filesets sim_1]
}
set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

launch_simulation
