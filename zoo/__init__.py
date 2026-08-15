"""STM32N6 deployment zoo.

Push arbitrary models through the ST Edge AI toolchain onto an STM32N6570-DK
and record, reproducibly, what survives and what the toolchain did to it.

Layout rule: nothing under `zoo/` may import from `lab/`, `models/` or
`firmware/`. The core stays reusable; the lab notebook stays prose.
"""

__version__ = "0.1.0"
