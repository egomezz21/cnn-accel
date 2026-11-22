import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.triggers import FallingEdge

@cocotb.test()
async def test_random_mac(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # TODO: reset the DUT for a couple of cycles
    dut.rst.value = 1
    dut.clear.value = 0
    dut.en.value = 0
    dut.a.value = 0
    dut.b.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst.value = 0

    await FallingEdge(dut.clk)

    expected = 0

    for _ in range(1000):
        # TODO: pick random a, b in [-128, 127] and random en/clear

        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        en = random.randint(0,1)
        
        if random.random() < 0.05:
            clear = 1
        else:
            clear = 0
        
        # TODO: drive them, wait a clock edge, update `expected` the same way the hardware should

        dut.clear.value = clear
        dut.en.value = en
        dut.a.value = a
        dut.b.value = b

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        
        if clear:
            expected = 0
        elif en :
            expected += a*b

        result = dut.acc.value.to_signed()

        assert result == expected, f"a={a} b={b} en={en} clear={clear}: result {result}, expected {expected}"

