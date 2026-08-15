"""The results store: an append-only event log, and everything derived from it.

Neither prior project on this machine had one. Both wrote their results as
prose Markdown, by hand, which meant a number could not be traced back to the
command that produced it, a failure could not be counted across models, and
re-running anything meant re-reading paragraphs. The store is what turns a pile
of attempts into a zoo.

`results/events.jsonl` is the source of truth and it is append-only. That
choice buys three things: concurrent screening workers never conflict, git
merges of the results file are trivial, and every *failed* attempt and every
retry is preserved rather than overwritten by the eventual success — which
matters when the product is a failure atlas.

Everything else here is a fold over that log.
"""

from zoo.store.events import EventLog  # noqa: F401
from zoo.store.schema import SCHEMA_VERSION, Event, Stage, Status, Verdict  # noqa: F401
