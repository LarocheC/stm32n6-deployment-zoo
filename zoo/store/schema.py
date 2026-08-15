"""Event schema. One record per stage attempt.

Deliberately flat and JSON-native: the log has to stay readable with `jq` and
diffable in git, and a schema that needs the zoo's own code to interpret would
defeat the point of writing it down.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1


class Stage:
    """Funnel stages, in cost order. The order is meaningful: a model's
    `deepest_stage` is how far down this list it got."""

    FETCH = "fetch"
    PROBE = "probe"
    LINT = "lint"
    SHAPE = "shape"
    PATCH = "patch"
    BUDGET = "budget"
    ANALYZE = "analyze"
    QUANTIZE = "quantize"
    GENERATE = "generate"
    VALIDATE_HOST = "validate_host"
    BOARD = "board"
    PROFILE = "profile"
    DEMO = "demo"

    ORDER = (
        FETCH, PROBE, LINT, SHAPE, PATCH, BUDGET, ANALYZE,
        QUANTIZE, GENERATE, VALIDATE_HOST, BOARD, PROFILE, DEMO,
    )

    @classmethod
    def index(cls, stage: str) -> int:
        try:
            return cls.ORDER.index(stage)
        except ValueError:
            return -1


class Status:
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class Verdict:
    """How far a graph got. Ordered from best to worst.

    `SCREENED` is a real, respectable outcome: it means the graph passed every
    board-free check. Most of the zoo's value is produced before the board is
    ever touched, and a verdict vocabulary that only rewards `MEASURED` would
    misrepresent that.
    """

    DEPLOYED = "DEPLOYED"      # running in firmware with live I/O
    MEASURED = "MEASURED"      # real on-board latency recorded
    COMPILES = "COMPILES"      # stedgeai generate produced network.c
    QUANTIZES = "QUANTIZES"    # int8 QDQ produced and validated on host
    IMPORTS = "IMPORTS"        # the ST front end accepted the graph
    SCREENED = "SCREENED"      # passed lint/shape/budget, not yet compiled
    REJECTED = "REJECTED"      # a stage failed for a reason about the model
    SKIPPED = "SKIPPED"        # deliberately not attempted, with a reason
    UNTRIED = "UNTRIED"

    ORDER = (
        UNTRIED, SKIPPED, REJECTED, SCREENED, IMPORTS,
        QUANTIZES, COMPILES, MEASURED, DEPLOYED,
    )

    #: The verdict a graph earns by clearing each stage.
    BY_STAGE = {
        Stage.BUDGET: SCREENED,
        Stage.ANALYZE: IMPORTS,
        Stage.QUANTIZE: QUANTIZES,
        Stage.GENERATE: COMPILES,
        Stage.VALIDATE_HOST: COMPILES,
        Stage.BOARD: MEASURED,
        Stage.PROFILE: MEASURED,
        Stage.DEMO: DEPLOYED,
    }

    @classmethod
    def rank(cls, verdict: str) -> int:
        try:
            return cls.ORDER.index(verdict)
        except ValueError:
            return 0


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Event:
    """One attempt at one stage, for one variant of one graph of one model."""

    stage: str
    status: str
    model_id: str
    graph_id: str = ""
    variant_id: str = ""

    run_id: str = ""
    ts: str = field(default_factory=utcnow)
    schema: int = SCHEMA_VERSION

    #: Provenance. `recipe_sha` and `artifact_sha` make a row reproducible;
    #: `zoo_commit` records which version of the harness produced it.
    recipe_sha: str = ""
    artifact_sha: str = ""
    zoo_commit: str = ""

    #: Tool versions, including `atonn_shim`, because whether the shim is
    #: installed changes results and must never be inferred after the fact.
    toolchain: dict[str, Any] = field(default_factory=dict)

    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    #: Failure detail. `is_infra` is the load-bearing flag: an infra event is
    #: kept for board-reliability statistics but is excluded from the verdict
    #: fold, so a wedged ST-LINK can never be recorded as a model's fault.
    failure_class: str = ""
    signature: str = ""
    known_issue: str = ""
    is_infra: bool = False
    error: str = ""

    artifacts: list[str] = field(default_factory=list)
    log: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, doc: dict) -> Event:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in doc.items() if k in known})

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.graph_id, self.variant_id, self.stage)

    @property
    def ok(self) -> bool:
        return self.status == Status.PASS
