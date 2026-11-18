SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(PWD)/../rtl/mac.sv
COCOTB_TOPLEVEL = mac
COCOTB_TEST_MODULES = test_mac
EXTRA_ARGS += --trace-fst --trace-structs
include $(shell cocotb-config --makefiles)/Makefile.sim