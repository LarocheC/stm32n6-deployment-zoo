"""Fold the event log into current state.

The fold is where the zoo's most important editorial rule lives: **infra
events never become a model verdict.** A wedged ST-LINK, a serial timeout, a
stale gdbserver — these are facts about the afternoon, not about the model.
They stay in the log so board reliability can be measured, and they are
skipped here so a model is never blamed for the bench.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from zoo.config import RESULTS_DIR
from zoo.store.events import EventLog
from zoo.store.schema import Event, Stage, Status, Verdict


@dataclass
class VariantState:
    """Latest known state of one variant of one graph."""

    model_id: str
    graph_id: str
    variant_id: str
    verdict: str = Verdict.UNTRIED
    deepest_stage: str = ""
    blocked_by: str = ""
    failure_class: str = ""
    signature: str = ""
    is_infra_last: bool = False
    metrics: dict = field(default_factory=dict)
    ts: str = ""
    #: When each stage last passed. The fold merges every stage's metrics into
    #: one row, which is what makes the leaderboard readable and also what lets
    #: a stale board measurement sit next to a fresh quantisation as though the
    #: two describe the same artifact. They do not: re-quantising produces a
    #: different `int8.onnx`, and a latency taken before that is a latency for
    #: a model that no longer exists. Keeping the timestamps lets the report
    #: say so instead of quietly presenting the pair as one result.
    stage_ts: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return Verdict.rank(self.verdict)


@dataclass
class GraphState:
    model_id: str
    graph_id: str
    best: VariantState | None = None
    variants: dict[str, VariantState] = field(default_factory=dict)
    #: Infra events seen for this graph. Counted, never used as a verdict.
    infra_events: int = 0

    @property
    def verdict(self) -> str:
        return self.best.verdict if self.best else Verdict.UNTRIED


@dataclass
class Snapshot:
    graphs: dict[tuple[str, str], GraphState] = field(default_factory=dict)
    #: Every failure signature seen, with counts and affected models. This is
    #: what collapses thirty heterogeneous failures into ten real constraints.
    signatures: dict[str, dict] = field(default_factory=dict)
    total_events: int = 0
    infra_events: int = 0

    def rows(self) -> list[GraphState]:
        return [self.graphs[k] for k in sorted(self.graphs)]

    def to_json(self) -> dict:
        return {
            "graphs": [
                {
                    "model_id": g.model_id,
                    "graph_id": g.graph_id,
                    "verdict": g.verdict,
                    "infra_events": g.infra_events,
                    "best": asdict(g.best) if g.best else None,
                    "variants": {k: asdict(v) for k, v in sorted(g.variants.items())},
                }
                for g in self.rows()
            ],
            "signatures": self.signatures,
            "total_events": self.total_events,
            "infra_events": self.infra_events,
        }


def fold(events: list[Event] | EventLog) -> Snapshot:
    snap = Snapshot()
    latest: dict[tuple, Event] = {}

    for event in events:
        snap.total_events += 1
        if event.is_infra:
            snap.infra_events += 1
            key = (event.model_id, event.graph_id)
            gs = snap.graphs.setdefault(key, GraphState(event.model_id, event.graph_id))
            gs.infra_events += 1
            # Deliberately not folded into any verdict: see module docstring.
            continue

        # Last write wins per (model, graph, variant, stage). Earlier attempts
        # stay in the log; only the current state is folded.
        prev = latest.get(event.key)
        if prev is None or event.ts >= prev.ts:
            latest[event.key] = event

        if event.status == Status.FAIL and event.signature:
            sig = snap.signatures.setdefault(
                event.signature,
                {
                    "signature": event.signature,
                    "failure_class": event.failure_class,
                    "known_issue": event.known_issue,
                    "count": 0,
                    "models": [],
                    "stages": [],
                    "example_error": event.error[:400],
                },
            )
            sig["count"] += 1
            if event.model_id not in sig["models"]:
                sig["models"].append(event.model_id)
            if event.stage not in sig["stages"]:
                sig["stages"].append(event.stage)

    # Group the surviving events per variant before deciding anything, because
    # a verdict is a property of the whole history, not of the last record.
    per_variant: dict[tuple[str, str, str], list[Event]] = {}
    for event in latest.values():
        per_variant.setdefault(
            (event.model_id, event.graph_id, event.variant_id), []
        ).append(event)

    for (model_id, graph_id, variant), evs in per_variant.items():
        gs = snap.graphs.setdefault((model_id, graph_id), GraphState(model_id, graph_id))
        vs = gs.variants.setdefault(variant, VariantState(model_id, graph_id, variant))

        # The verdict is the deepest stage actually CLEARED. A model that
        # compiles and then fails on the board has still compiled — recording
        # that as REJECTED would erase the most useful thing known about it,
        # and would make a board outage look like a modelling result. Where it
        # stopped is recorded separately, in `blocked_by`.
        for ev in evs:
            if ev.status != Status.PASS:
                continue
            earned = Verdict.BY_STAGE.get(ev.stage)
            if earned and Verdict.rank(earned) > Verdict.rank(vs.verdict):
                vs.verdict = earned
            if Stage.index(ev.stage) > Stage.index(vs.deepest_stage):
                vs.deepest_stage = ev.stage
            vs.metrics.update(ev.metrics or {})
            vs.ts = max(vs.ts, ev.ts)
            vs.stage_ts[ev.stage] = max(vs.stage_ts.get(ev.stage, ""), ev.ts)

        # The earliest non-passing stage is where the funnel stopped.
        stopped = sorted(
            (e for e in evs if e.status != Status.PASS),
            key=lambda e: Stage.index(e.stage),
        )
        if stopped:
            first = stopped[0]
            vs.blocked_by = first.stage
            vs.failure_class = first.failure_class
            vs.signature = first.signature
            vs.ts = max(vs.ts, first.ts)
            if vs.verdict == Verdict.UNTRIED:
                vs.verdict = (
                    Verdict.SKIPPED if first.status == Status.SKIP else Verdict.REJECTED
                )

    for gs in snap.graphs.values():
        if gs.variants:
            gs.best = max(gs.variants.values(), key=lambda v: (v.rank, v.ts))

    return snap


DEFAULT_PATH = RESULTS_DIR / "snapshot.json"


def write(snap: Snapshot, path: Path | None = None) -> Path:
    path = path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap.to_json(), indent=1, sort_keys=True) + "\n")
    return path


def load_and_fold(log_path: Path | None = None) -> Snapshot:
    return fold(EventLog(log_path))
