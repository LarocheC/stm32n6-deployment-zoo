"""Drivers for the ST Edge AI Core toolchain.

Everything that shells out to `stedgeai`, `atonn`, or the STM32Cube tools
lives here. Nothing else in the zoo is allowed to build an ST command line by
hand -- that habit is exactly what left both prior projects with their only
record of a compile being the `Parameters:` line inside a generated report.
"""
