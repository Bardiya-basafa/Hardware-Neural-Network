# GHDL flow (VHDL-2008). Build products go to build/.
#
#   make                 full regression: unit tests + every dataset, trained
#                        through nn_trainer (with gradient check) and through
#                        the self-training nn_autotrain            (~2 minutes)
#   make quick           unit tests + XOR training                (~2 seconds)
#   make tb_layer        one unit test bench
#   make train DS=CIRCLE [LR=3] [EPOCHS=200] [CHECK=false]
#                        train one dataset with tb_nn_trainer
#   make auto  DS=SEG7   self-training run of one dataset (tb_nn_autotrain)
#   make wave TB=tb_nn_trainer [DS=PARITY3]
#                        run and dump build/<TB>.ghw (open with GTKWave)
#   make synth [DS=PARITY3]  GHDL synthesis check of nn_top and nn_train_top
#   make clean
#
# Datasets: XOR PARITY3 CIRCLE BLOBS3 SEG7 (see datasets_pkg.vhd)

GHDL      ?= ghdl
GHDLFLAGS := --std=08 --workdir=build
RUNFLAGS  := --ieee-asserts=disable-at-0

# compile order matters: packages first, then bottom-up
RTL := nn_pkg.vhd datasets_pkg.vhd weights_pkg.vhd mac_unit.vhd \
       activation_relu.vhd activation_sigmoid.vhd activation_deriv.vhd \
       neuron.vhd layer.vhd nn_top.vhd nn_trainer.vhd nn_autotrain.vhd \
       nn_train_top.vhd
UNIT_TBS := tb_activation tb_neuron tb_layer tb_nn_top
TBS      := $(UNIT_TBS) tb_nn_trainer tb_nn_autotrain
SRC      := $(RTL) tb/tb_ref_pkg.vhd $(addprefix tb/,$(addsuffix .vhd,$(TBS)))
DATASETS := XOR PARITY3 CIRCLE BLOBS3 SEG7

DS     ?= XOR
TB     ?= tb_nn_trainer
LR     ?= -1
EPOCHS ?= -1
CHECK  ?= true

.PHONY: all quick analyze train auto wave synth clean $(UNIT_TBS) \
        $(addprefix train-,$(DATASETS)) $(addprefix auto-,$(DATASETS))

all: $(UNIT_TBS) $(addprefix train-,$(DATASETS)) $(addprefix auto-,$(DATASETS))
quick: $(UNIT_TBS) train-XOR

build/.analyzed: $(SRC)
	@mkdir -p build
	$(GHDL) -a $(GHDLFLAGS) $(SRC)
	@touch $@
analyze: build/.analyzed

$(UNIT_TBS): build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) $@ $(RUNFLAGS)

$(addprefix train-,$(DATASETS)): train-%: build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) tb_nn_trainer -gDATASET=$* $(RUNFLAGS)

$(addprefix auto-,$(DATASETS)): auto-%: build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) tb_nn_autotrain -gDATASET=$* $(RUNFLAGS)

train: build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) tb_nn_trainer -gDATASET=$(DS) -gLR_SHIFT=$(LR) \
	  -gMAX_EPOCHS=$(EPOCHS) -gCHECK=$(CHECK) $(RUNFLAGS)

auto: build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) tb_nn_autotrain -gDATASET=$(DS) $(RUNFLAGS)

# DS is passed only to the test benches that have a DATASET generic
wave: build/.analyzed
	$(GHDL) --elab-run $(GHDLFLAGS) $(TB) \
	  $(if $(filter tb_nn_trainer tb_nn_autotrain,$(TB)),-gDATASET=$(DS)) \
	  $(RUNFLAGS) --wave=build/$(TB).ghw

synth: build/.analyzed
	$(GHDL) --synth $(GHDLFLAGS) nn_top > build/synth_nn_top.vhd
	$(GHDL) --synth $(GHDLFLAGS) -gDATASET=$(DS) nn_train_top > build/synth_nn_train_top.vhd
	@echo "synthesis OK: build/synth_nn_top.vhd build/synth_nn_train_top.vhd"

clean:
	rm -rf build
