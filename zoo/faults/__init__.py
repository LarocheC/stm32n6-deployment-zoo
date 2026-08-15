"""Fault classification: turning noisy tool output into countable constraints.

Thirty heterogeneous models produce thirty heterogeneous failures, but they do
not represent thirty problems. Normalising each error into a stable signature
and looking it up in a catalogue collapses them: the same limitation hit by six
models becomes one row with a count of six, and the *unmatched* signatures
become the day's discovery queue.

The other job here is the infra/model distinction. A wedged ST-LINK and an
unsupported operator are not the same kind of event, and conflating them is
how a zoo ends up publishing a board outage as a modelling result.
"""

from zoo.faults.signatures import classify, fingerprint  # noqa: F401
from zoo.faults.taxonomy import INFRA_CLASSES, FailureClass  # noqa: F401
