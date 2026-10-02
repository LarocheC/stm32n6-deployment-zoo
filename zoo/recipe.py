"""The recipe schema: one TOML file per model attempt.

TOML because `tomllib` is stdlib on 3.11+, it is strict about types, and
array-of-tables maps cleanly onto heterogeneous graph I/O.

Two schema decisions carry all the weight.

**The unit of compilation is a graph, not a model.** Whisper is an encoder and
a decoder; a diarization pipeline is segmentation and embedding. Each funnels
independently and earns its own row. Without this, "Whisper doesn't work"
would be the only expressible answer, when the useful answer is "the encoder
compiles at a frozen 5 s window and the decoder cannot exist here at all". A
graph may be `enabled = false` with a `skip_reason` and still appear in the
results — recorded and explained, rather than quietly dropped.

**Every input carries a role.** Roles are what let one schema absorb
multi-input, recurrent-state, encoder/decoder and dynamic-shape models without
a special case per family:

  feature   real activation input; calibrated, fed real data, counted in RTF
  state     recurrent state; `feeds_from` names the paired output, so the
            firmware glue knows this is a feedback edge and the calibration
            reader knows to thread real state rather than zeros
  constant  a config scalar (`sr`, `num_beams`). Folded into an initializer at
            patch time, which by itself removes the rank-1-graph-input
            rejection for a large class of Hugging Face exports
  token     integer ids; excluded from activation calibration
  mask      attention/padding mask; excluded from calibration, pinned
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ROLES = ("feature", "state", "constant", "token", "mask")
TIERS = (1, 2, 3)  # 1 = screen only, 2 = measure on board, 3 = firmware demo

#: Marker for a dynamic axis that carries no name in the ONNX file.
ANONYMOUS_AXIS = "?"


#: The policy `[quantize]` keys a recipe may override. A typo must fail loudly:
#: a silently ignored `activation_symetric = true` would ship the wrong scheme.
QUANTIZE_OVERRIDES = ("activation_symmetric", "weight_symmetric")


class RecipeError(ValueError):
    """A recipe is malformed. Always names the file and the offending key."""


@dataclass
class IOSpec:
    name: str
    role: str = "feature"
    shape: list[int | str] = field(default_factory=list)
    dtype: str = "float32"
    #: `state` only: the output tensor whose value feeds this input next step.
    feeds_from: str | None = None
    #: `state` only: how to initialise on the first frame.
    init: str = "zeros"
    #: `constant` only: the value to fold in.
    value: Any = None
    #: `feature` only: calibration provider key.
    calib: str | None = None

    def validate(self, where: str) -> None:
        if self.role not in ROLES:
            raise RecipeError(f"{where}: role {self.role!r} not in {ROLES}")
        if self.role == "state" and not self.feeds_from:
            raise RecipeError(
                f"{where}: input {self.name!r} has role 'state' but no `feeds_from`. "
                "A state input without its paired output cannot be wired as feedback."
            )
        if self.role == "constant" and self.value is None:
            raise RecipeError(
                f"{where}: input {self.name!r} has role 'constant' but no `value` to fold in."
            )

    @property
    def dynamic_axes(self) -> list[int]:
        return [i for i, d in enumerate(self.shape) if not isinstance(d, int)]

    @property
    def anonymous_axes(self) -> list[int]:
        """Axes with no name, rendered as `"?"` by the drafter.

        These are worse than named symbolic dims, not merely equivalent:
        `--fix-parametric-shapes` addresses dimensions *by name*, so an
        anonymous axis cannot be pinned at the command line at all. Fixing one
        means re-exporting the model or rewriting its `value_info`. Keeping
        them out of `unpinned` stops the funnel from suggesting a remedy that
        does not exist.
        """
        return [i for i, d in enumerate(self.shape) if d == ANONYMOUS_AXIS]


@dataclass
class GraphSpec:
    id: str
    file: str
    #: Wall-clock covered by one invocation, for the real-time factor. A model
    #: without this gets a latency but no verdict on whether it keeps up.
    realtime_ms: float | None = None
    #: Symbolic dim name -> fixed value. Becomes --fix-parametric-shapes.
    pin: dict[str, int] = field(default_factory=dict)
    #: Input name -> a different *static* shape to re-resolve the graph at.
    #: Distinct from `pin`, which fills in dimensions the exporter left
    #: symbolic; this replaces dimensions the exporter fixed. Two graph entries
    #: pointing at the same file with different `resolution` are two rows on
    #: the leaderboard, which is the point: input size is a deployment
    #: decision, and on this part it is usually the decisive one.
    resolution: dict[str, list[int]] = field(default_factory=dict)
    inputs: list[IOSpec] = field(default_factory=list)
    outputs: list[IOSpec] = field(default_factory=list)
    #: Patches to attempt in addition to whatever lint suggests. Lint proposes a
    #: patch when it can name the violation the patch fixes; a rewrite that is
    #: needed as a *consequence* of a recipe decision — pinning a window shorter
    #: than the positional embedding the model was exported with — has no
    #: violation to attach to, because the graph as downloaded was fine. Those
    #: are declared here, next to the decision that made them necessary. Each
    #: still passes through the same parity gate.
    patches: list[str] = field(default_factory=list)
    enabled: bool = True
    skip_reason: str | None = None

    def validate(self, where: str) -> None:
        where = f"{where} graph[{self.id}]"
        if not self.enabled and not self.skip_reason:
            raise RecipeError(
                f"{where}: disabled graphs must carry a `skip_reason`. "
                "A silently dropped graph is indistinguishable from one nobody tried."
            )
        seen: set[str] = set()
        for spec in self.inputs:
            if spec.name in seen:
                raise RecipeError(f"{where}: duplicate input {spec.name!r}")
            seen.add(spec.name)
            spec.validate(where)

        out_names = {o.name for o in self.outputs}
        for spec in self.inputs:
            if spec.role == "state" and spec.feeds_from not in out_names:
                raise RecipeError(
                    f"{where}: input {spec.name!r} feeds_from {spec.feeds_from!r}, "
                    f"which is not one of this graph's outputs ({sorted(out_names)})"
                )

    @property
    def state_pairs(self) -> list[tuple[str, str]]:
        return [(s.name, s.feeds_from) for s in self.inputs if s.role == "state" and s.feeds_from]

    @property
    def unpinned(self) -> list[str]:
        """Named symbolic dims still lacking a value — fixable via `pin`."""
        return sorted(
            {
                d
                for spec in self.inputs
                for d in spec.shape
                if isinstance(d, str) and d != ANONYMOUS_AXIS and d not in self.pin
            }
        )

    @property
    def anonymous_axes(self) -> list[tuple[str, int]]:
        """`(input_name, axis)` pairs that cannot be pinned by name at all."""
        return [(s.name, ax) for s in self.inputs for ax in s.anonymous_axes]

    @property
    def is_compilable(self) -> bool:
        """Every dimension fixed, so `stedgeai` has a static graph to work with."""
        return not self.unpinned and not self.anonymous_axes

    def fix_parametric_shapes(self) -> str | None:
        """The literal `--fix-parametric-shapes` argument, or None."""
        if not self.pin:
            return None
        body = ",".join(f"'{k}':{v}" for k, v in sorted(self.pin.items()))
        return "{" + body + "}"


@dataclass
class Variants:
    profile: list[str] = field(default_factory=lambda: ["onchip", "extflash", "allmems"])
    precision: list[str] = field(default_factory=lambda: ["int8"])
    input_data_type: list[str] = field(default_factory=lambda: ["float32"])
    #: `npu` is the accelerator; `m55` omits --st-neural-art and compiles for
    #: the Cortex-M55 alone, giving the NPU-vs-CPU speedup for one extra run.
    backend: list[str] = field(default_factory=lambda: ["npu", "m55"])
    policy: str = "first_success"  # or "all"


@dataclass
class Calibration:
    provider: str = "synthetic"
    dataset: str | None = None
    split: str | None = None
    preprocessor: str | None = None
    n: int = 128
    seed: int = 0
    #: Provider knobs — the input convention a corpus has to be put through to
    #: match what this model was exported for (`scale`, `mean`, `std`, `bgr`).
    #: Kept per recipe rather than per provider because it is a fact about the
    #: model, and getting it wrong sets every activation scale wrong.
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        """Synthetic calibration makes fidelity numbers meaningless.

        Reported with a marker rather than suppressed, and never sufficient
        for promotion to tier 2.
        """
        return self.provider == "synthetic"


@dataclass
class Recipe:
    id: str
    source: str
    path: Path
    domain: str = "misc"
    task: str = "unknown"
    tier: int = 1
    license: str | None = None
    notes: str | None = None
    fetch_provider: str = "hf_onnx"
    revision: str = "main"
    graphs: list[GraphSpec] = field(default_factory=list)
    variants: Variants = field(default_factory=Variants)
    calibration: Calibration = field(default_factory=Calibration)
    #: Per-model overrides of policy `[quantize]`. Only the keys in
    #: QUANTIZE_OVERRIDES: a property of the model, not of the bench.
    quantize: dict[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    def validate(self) -> None:
        where = str(self.path)
        if self.schema != SCHEMA_VERSION:
            raise RecipeError(
                f"{where}: schema {self.schema}, this build understands {SCHEMA_VERSION}"
            )
        if self.tier not in TIERS:
            raise RecipeError(f"{where}: tier {self.tier} not in {TIERS}")
        if not self.graphs:
            raise RecipeError(f"{where}: no [[graph]] entries")
        ids = [g.id for g in self.graphs]
        if len(ids) != len(set(ids)):
            raise RecipeError(f"{where}: duplicate graph ids in {ids}")
        for graph in self.graphs:
            graph.validate(where)
        unknown = sorted(set(self.quantize) - set(QUANTIZE_OVERRIDES))
        if unknown:
            raise RecipeError(
                f"{where}: [quantize] keys {unknown} not in {list(QUANTIZE_OVERRIDES)}"
            )

    @property
    def enabled_graphs(self) -> list[GraphSpec]:
        return [g for g in self.graphs if g.enabled]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _io_from_toml(doc: dict) -> IOSpec:
    return IOSpec(
        name=doc["name"],
        role=doc.get("role", "feature"),
        shape=list(doc.get("shape", [])),
        dtype=doc.get("dtype", "float32"),
        feeds_from=doc.get("feeds_from"),
        init=doc.get("init", "zeros"),
        value=doc.get("value"),
        calib=doc.get("calib"),
    )


def load(path: Path) -> Recipe:
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"{path}: {exc}") from None

    graphs = []
    for gdoc in doc.get("graph", []):
        graphs.append(
            GraphSpec(
                id=gdoc["id"],
                file=gdoc["file"],
                realtime_ms=gdoc.get("realtime_ms"),
                pin={k: int(v) for k, v in gdoc.get("pin", {}).items()},
                resolution={
                    k: [int(d) for d in v] for k, v in gdoc.get("resolution", {}).items()
                },
                patches=list(gdoc.get("patches", [])),
                inputs=[_io_from_toml(d) for d in gdoc.get("input", [])],
                outputs=[_io_from_toml(d) for d in gdoc.get("output", [])],
                enabled=gdoc.get("enabled", True),
                skip_reason=gdoc.get("skip_reason"),
            )
        )

    vdoc = doc.get("variants", {})
    cdoc = doc.get("calibration", {})
    fdoc = doc.get("fetch", {})

    recipe = Recipe(
        id=doc.get("id") or path.stem,
        source=doc.get("source", ""),
        path=path,
        domain=doc.get("domain", "misc"),
        task=doc.get("task", "unknown"),
        tier=int(doc.get("tier", 1)),
        license=doc.get("license"),
        notes=doc.get("notes"),
        fetch_provider=fdoc.get("provider", "hf_onnx"),
        revision=fdoc.get("revision", "main"),
        graphs=graphs,
        variants=Variants(
            profile=vdoc.get("profile", Variants().profile),
            precision=vdoc.get("precision", Variants().precision),
            input_data_type=vdoc.get("input_data_type", Variants().input_data_type),
            backend=vdoc.get("backend", Variants().backend),
            policy=vdoc.get("policy", "first_success"),
        ),
        calibration=Calibration(
            provider=cdoc.get("provider", "synthetic"),
            dataset=cdoc.get("dataset"),
            split=cdoc.get("split"),
            preprocessor=cdoc.get("preprocessor"),
            n=int(cdoc.get("n", 128)),
            seed=int(cdoc.get("seed", 0)),
            options=dict(cdoc.get("options", {})),
        ),
        quantize=dict(doc.get("quantize", {})),
        schema=int(doc.get("schema", SCHEMA_VERSION)),
    )
    recipe.validate()
    return recipe


def discover(models_dir: Path) -> list[Recipe]:
    """Load every recipe under `models/`, sorted by id."""
    found = [load(p) for p in sorted(models_dir.rglob("*.toml")) if p.stem != "_template"]
    return sorted(found, key=lambda r: r.id)
