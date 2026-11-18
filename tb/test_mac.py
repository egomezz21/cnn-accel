import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

@cocotb.test()
async def test_random_mac(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    # TODO: reset the DUT for a couple of cycles
    expected = 0
    for _ in range(1000):
        # TODO: pick random a, b in [-128, 127] and random en/clear
        # TODO: drive them, wait a clock edge, update `expected` the same way the hardware should
        # TODO: assert the DUT's acc matches expected (hint: read it as a signed value)
        pass