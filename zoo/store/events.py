"""Append-only event log.

Append-only is not a stylistic preference. Three properties fall out of it:

- Parallel screening workers can write concurrently without coordination. A
  single `write()` of a line under the platform's atomic-append semantics
  interleaves cleanly; a mutable results file would need locking.
- Git merges of the results file are trivial, because concurrent work appends
  to the end rather than rewriting rows.
- **Failed attempts survive.** A store that overwrote a failure with a later
  success would erase exactly the material the failure atlas is made of, and
  would hide flapping — a model that passes one run in three is a finding, not
  a pass.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from zoo.config import RESULTS_DIR
from zoo.store.schema import Event

DEFAULT_PATH = RESULTS_DIR / "events.jsonl"


class EventLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH

    # -- writing ----------------------------------------------------------

    def append(self, event: Event, *, sync: bool = True) -> None:
        """Append one event.

        `sync` forces the record to disk. On by default because the events
        most worth keeping are the ones written just before something crashed
        or the board wedged.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = event.to_json() + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            if sync:
                os.fsync(fh.fileno())

    def extend(self, events: list[Event]) -> None:
        for event in events:
            self.append(event, sync=False)
        if events and self.path.is_file():
            with self.path.open("a", encoding="utf-8") as fh:
                os.fsync(fh.fileno())

    # -- reading ----------------------------------------------------------

    def __iter__(self) -> Iterator[Event]:
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line can happen if a process died mid-write.
                    # Skipping it is right: the alternative is refusing to read
                    # an otherwise intact log because of one bad byte.
                    continue
                # Records from an older schema are yielded rather than dropped;
                # the snapshot decides what it can still use. Losing history to
                # a schema bump would defeat the point of an append-only log.
                try:
                    yield Event.from_dict(doc)
                except TypeError:
                    continue

    def read_all(self) -> list[Event]:
        return list(self)

    def __len__(self) -> int:
        return sum(1 for _ in self)
