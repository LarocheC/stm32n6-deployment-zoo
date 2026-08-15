"""Getting numbers off the board, and refusing to when they would be wrong.

The board stage is where a zoo loses its credibility if it is careless, and
the ways that happens are known rather than hypothetical:

- A wedged ST-LINK, a stale gdbserver or a detached USB device produces a
  failure that looks exactly like a model failure. Every one of those is
  classified as infra and excluded from verdicts.
- Worse, and this is the one to design against: if the loader does not
  actually install the network, the previous firmware is still resident and
  the measurement succeeds — against the wrong model. It yields a plausible
  number. So nothing is measured unless the loader reported success *and* the
  device reports the network we just compiled.

Note that `n6_loader.py` copies the generated `network.c` into ST's bundled
NPU_Validation application and builds it there. That mutates the vendor
install. It is how ST's own flow works and there is no supported alternative,
so it is done deliberately and recorded, not hidden.
"""

from zoo.board.link import BoardError, attach, preflight  # noqa: F401
from zoo.board.measure import load_network, validate  # noqa: F401
