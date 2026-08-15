"""The operator oracle: which ops reach the NPU, which fall back, which are out.

This is the single highest-value screening primitive in the zoo, because on
this part **latency is epoch-bound rather than MAC-bound** below roughly a
megabyte of weights. Every operator that cannot be mapped to the Neural-ART
accelerator becomes a Cortex-M55 software epoch, and every software epoch
costs an NPU pipeline teardown, a memory round-trip and cache maintenance.
Counting them in the ONNX graph predicts the shape of the result before the
compiler runs.

Two sources, and using only one of them is a mistake:

**ST's operator-mapping documentation** (`stneuralart_operator_support.html`)
is the accelerator's own table — every ONNX op tagged `HW`, `SW_INT` or
`SW_FLOAT`, with the caveats that decide which. This is authoritative, and it
contradicts folklore that both prior projects on this machine operated under:
`PRelu` *is* hardware-mapped (its slope must be quantised), `Softmax` *is*
hardware-mapped once `--expand-softmax` is passed, `Gelu` *is* hardware. Most
importantly it states that `MatMul` and `Gemm` reach hardware **only when the
second input is constant** — which is the one line that explains why
transformer attention, where both operands are activations, is slow here.

**`stedgeai supported-ops`** is a trap if used alone. It returns ~320
operators including `Einsum`, `Where`, `ScatterND`, `TopK`,
`LayerNormalization`, `LSTM` and `GRU`. That is the generic ST Edge AI
*front-end parser* vocabulary — what the importer will accept — not what the
`--st-neural-art` backend will map. Screening on it would pass models that
cannot possibly run well.

Crossing the two yields four tiers, and the gap between them is itself
informative: an op the parser accepts but the mapping table never mentions has
undocumented behaviour on the NPU, which is a risk worth naming rather than
silently treating as supported.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

from zoo.config import RUNS_DIR, Toolchain
from zoo.st.run import run

# -- tiers -------------------------------------------------------------------

HW = "HW"
SW_INT = "SW_INT"
SW_FLOAT = "SW_FLOAT"
SW = "SW"  # ST writes a bare "SW" for a few ops (e.g. DepthToSpace)
MIXED = "MIXED"  # e.g. Swish: "SW_FLOAT / HW" depending on recognition passes
FRONTEND_ONLY = "FRONTEND_ONLY"
UNSUPPORTED = "UNSUPPORTED"
PLUMBING = "PLUMBING"

#: Tiers that execute on the Cortex-M55 rather than the accelerator.
SOFTWARE_TIERS = frozenset({SW, SW_INT, SW_FLOAT})

#: Graph-construction ops that never execute: the front end folds or drops
#: them. Given their own tier because they are numerous — a Whisper encoder
#: carries 164 `Constant` nodes — and counting them as "undocumented on the
#: NPU" would bury the handful of ops that genuinely are.
PLUMBING_OPS = frozenset({"Constant", "ConstantOfShape", "Identity", "Dropout"})

#: Ops with no usable mapping, whatever the front-end parser claims, each with
#: the reason it is blocked. These are asserted from evidence rather than
#: inferred from the table's silence, so the lint stage can quote a cause
#: instead of shrugging.
HARD_BLOCKED: dict[str, str] = {
    "If": (
        "control flow cannot be expressed in a statically scheduled epoch graph. "
        "This is why onnx-community/silero-vad is unusable: its entire network "
        "sits inside two If branches selecting 8 kHz vs 16 kHz."
    ),
    "Loop": "control flow; iteration count is not known at compile time.",
    "Scan": "control flow; iteration count is not known at compile time.",
    "NonZero": "output shape is data-dependent, so no buffer can be allocated.",
    "Einsum": (
        "observed to fail the ST front end with 'Error in computation of shapes'. "
        "Rewrite as explicit Slice/MatMul/Concat, which does compile."
    ),
    "GridSample": "no Neural-ART mapping and no documented software fallback.",
}
HARD_BLOCKED_OPS = frozenset(HARD_BLOCKED)

#: ST's HTML has a handful of typos and spelling variants in the operator
#: column. Mapping them to canonical ONNX names here keeps the lookup honest
#: rather than silently reporting a real op as undocumented.
_DOC_ALIASES = {
    "hardswitch": "HardSwish",  # ST doc typo for HardSwish
    "softsign": "Softsign",  # doc writes "SoftSign"
    "sum": "Sum",  # from the "Add/sum" cell
    "swish": "Swish",  # not a real ONNX op; recognised as a pattern
}


@dataclass(frozen=True)
class OpInfo:
    """One row of ST's mapping table, with its caveats made machine-readable.

    The `Mapped On` column alone is misleading. ST qualifies many rows in the
    comment, and two kinds of qualification change the answer:

    `unlocked_by` — the hardware path exists but only when an atonn
    recognition pass is enabled, and the row's headline tier assumes it is.
    `Softmax` is the clearest case: the table says `HW`, and the comment says
    "SW_INT otherwise". None of ST's shipped profiles pass
    `--expand-softmax`, so the *default* answer for Softmax is SW_INT.

    `requires_constant_input` — the hardware path depends on the graph, not on
    a flag. `MatMul` and `Gemm` reach hardware only when their second operand
    is constant. A weight matrix qualifies; an attention score matrix does
    not. This single distinction is why a convolutional encoder is fast here
    and self-attention is not, and it cannot be resolved from an op histogram
    alone — `zoo.graph.lint` resolves it by inspecting the actual operands.
    """

    name: str
    tier: str
    comment: str = ""
    #: Compiler flag that unlocks the hardware path, if the tier is gated.
    unlocked_by: str | None = None
    #: Documented tier when that flag is absent.
    fallback_tier: str | None = None
    #: True when hardware mapping requires a constant (initializer) operand.
    requires_constant_input: bool = False

    @property
    def is_software(self) -> bool:
        return self.effective_tier(unlocked=False) in SOFTWARE_TIERS

    def effective_tier(self, *, unlocked: bool = False) -> str:
        """Tier actually obtained, given whether recognition passes are on."""
        if self.unlocked_by and self.fallback_tier and not unlocked:
            return self.fallback_tier
        return self.tier


#: Ops whose hardware mapping is gated behind an atonn recognition pass that
#: none of ST's shipped profiles enable. Reporting the pessimistic default and
#: naming the flag that fixes it is the honest presentation; silently assuming
#: either one misleads.
_UNLOCK_FLAGS = {
    "Softmax": "--expand-softmax",
    "Gelu": "--GELU-recognition",
    "Swish": "--SWISH-recognition",
}

#: "... SW_INT otherwise", "fallback to float", "SW fallback is considered".
_FALLBACK_RE = re.compile(r"\b(SW_INT|SW_FLOAT)\b[^.]*\botherwise\b", re.I)
_SW_FALLBACK_RE = re.compile(r"\bSW\s+fallback\b|\bfallback to float\b", re.I)
_CONST_INPUT_RE = re.compile(
    r"(second input|params input|weights tensor)\s+should be constant", re.I
)


# -- HTML table extraction ---------------------------------------------------


class _TableGrabber(HTMLParser):
    """Collect every <table> in the document as a list of rows of cell text.

    Written against stdlib `html.parser` rather than adding a dependency: the
    file is plain pandoc-generated markup with `<thead>/<tbody>/<tr>/<td>` and
    no scripting, so there is nothing to be clever about.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _normalise_tier(raw: str) -> str:
    value = raw.strip().upper().replace(" ", "")
    if not value:
        return UNSUPPORTED
    if "/" in value:
        return MIXED
    if value in (HW, SW, SW_INT, SW_FLOAT):
        return value
    # Unexpected spelling: treat as software rather than claiming hardware.
    return SW


def parse_operator_support(html_path: Path) -> dict[str, dict[str, OpInfo]]:
    """Parse ST's mapping doc into `{framework: {op_name: OpInfo}}`.

    Returns entries for `"onnx"` and `"tflite"`. Keys are the canonical
    operator names as ST writes them; use `lookup()` for tolerant matching.
    """
    grabber = _TableGrabber()
    grabber.feed(html_path.read_text(encoding="utf-8", errors="replace"))

    out: dict[str, dict[str, OpInfo]] = {}
    for table in grabber.tables:
        if not table:
            continue
        header = [c.lower() for c in table[0]]
        if "mapped on" not in header:
            continue
        try:
            name_col = next(i for i, c in enumerate(header) if "operator" in c or "operation" in c)
            tier_col = header.index("mapped on")
        except (StopIteration, ValueError):
            continue
        comment_col = header.index("comment") if "comment" in header else None

        framework = "onnx" if "onnx" in header[name_col] else "tflite"
        table_ops: dict[str, OpInfo] = {}

        for row in table[1:]:
            if len(row) <= max(name_col, tier_col):
                continue
            raw_name = row[name_col].strip()
            if not raw_name:
                continue
            tier = _normalise_tier(row[tier_col])
            comment = row[comment_col].strip() if comment_col is not None and len(row) > comment_col else ""

            # Cells like "Add/sum" list an ONNX op and its aliases together.
            for part in (p.strip() for p in raw_name.split("/")):
                if not part:
                    continue
                canonical = _DOC_ALIASES.get(part.lower(), part)
                unlocked_by = _UNLOCK_FLAGS.get(canonical)

                fallback: str | None = None
                if unlocked_by:
                    m = _FALLBACK_RE.search(comment)
                    if m:
                        fallback = m.group(1).upper()
                    elif _SW_FALLBACK_RE.search(comment):
                        fallback = SW_FLOAT

                table_ops[canonical] = OpInfo(
                    name=canonical,
                    tier=tier,
                    comment=comment,
                    unlocked_by=unlocked_by,
                    fallback_tier=fallback,
                    requires_constant_input=bool(_CONST_INPUT_RE.search(comment)),
                )
        # A framework's table may be split across several <table> elements.
        out.setdefault(framework, {}).update(table_ops)
    return out


# -- front-end parser vocabulary --------------------------------------------


def frontend_ops(tc: Toolchain, *, model_type: str = "onnx") -> set[str]:
    """Ops the ST Edge AI *importer* accepts — not what the NPU maps.

    Kept strictly separate from the mapping table. Their difference is the
    `FRONTEND_ONLY` tier.
    """
    workdir = RUNS_DIR / "_toolchain" / tc.core_tag
    res = run(
        [tc.stedgeai, "supported-ops", "--target", "stm32n6", "--type", model_type],
        cwd=workdir,
        timeout_s=300,
    )
    if not res.ok:
        raise RuntimeError(f"supported-ops failed: {res.combined[-2000:]}")

    # With `--type onnx` the output is a bare comma-separated list between a
    # "<N> operators found" line and the trailing "elapsed time" line. (Without
    # --type, every name instead carries a "(ONNX)"/"(TFLITE)" suffix — hence
    # the two accepted shapes below.)
    text = res.stdout
    start = re.search(r"^\s*(\d+)\s+operators found\s*$", text, flags=re.M)
    declared = int(start.group(1)) if start else None
    body = text[start.end() :] if start else text
    body = re.split(r"^\s*elapsed time", body, flags=re.M)[0]

    names: set[str] = set()
    for token in body.split(","):
        token = token.strip()
        if not token:
            continue
        token = re.sub(r"\s*\([A-Z]+\)\s*$", "", token)  # drop any (ONNX) suffix
        if re.fullmatch(r"[A-Za-z_][\w.]*", token):
            names.add(token)

    if declared is not None and len(names) != declared:
        raise RuntimeError(
            f"supported-ops parse mismatch: tool declared {declared} operators, "
            f"parsed {len(names)}. Output format may have changed:\n{text[:600]}"
        )
    return names


# -- the combined oracle ------------------------------------------------------


@dataclass
class OpTable:
    core_version: str
    mapping: dict[str, OpInfo]
    frontend: set[str]

    # ---- lookup ----

    def _canonical(self, op: str) -> str | None:
        if op in self.mapping:
            return op
        lowered = {k.lower(): k for k in self.mapping}
        return lowered.get(op.lower())

    def tier(self, op: str, *, unlocked: bool = False) -> str:
        """Tier for an ONNX op_type, resolving the four-way taxonomy.

        `unlocked=True` reports the tier obtainable with the `transformer`
        profile's recognition passes enabled.
        """
        if op in HARD_BLOCKED_OPS:
            return UNSUPPORTED
        if op in PLUMBING_OPS:
            return PLUMBING
        key = self._canonical(op)
        if key is not None:
            return self.mapping[key].effective_tier(unlocked=unlocked)
        if op in self.frontend:
            return FRONTEND_ONLY
        return UNSUPPORTED

    def info(self, op: str) -> OpInfo | None:
        key = self._canonical(op)
        return self.mapping[key] if key else None

    def census(self, op_types: dict[str, int], *, unlocked: bool = False) -> dict[str, dict[str, int]]:
        """Group an `{op_type: count}` histogram by tier."""
        grouped: dict[str, dict[str, int]] = {}
        for op, count in sorted(op_types.items()):
            grouped.setdefault(self.tier(op, unlocked=unlocked), {})[op] = count
        return grouped

    def software_op_count(self, op_types: dict[str, int], *, unlocked: bool = False) -> int:
        """How many node instances would land on the Cortex-M55.

        This is a *lower bound*. Ops whose hardware path is conditional on an
        operand being constant (`MatMul`, `Gemm`, `Conv`) are counted as
        hardware here because an op histogram cannot tell the difference;
        resolving them needs the real graph, which `zoo.graph.lint` does.
        """
        return sum(
            count
            for op, count in op_types.items()
            if self.tier(op, unlocked=unlocked) in SOFTWARE_TIERS
        )

    def conditional_ops(self, op_types: dict[str, int]) -> dict[str, int]:
        """Ops present whose hardware mapping depends on a constant operand.

        Their presence is a flag to go look at the graph: these are hardware
        for a convolution's weights and software for attention's scores, and
        the difference dominates the result.
        """
        return {
            op: count
            for op, count in op_types.items()
            if (info := self.info(op)) is not None and info.requires_constant_input
        }

    def gated_ops(self, op_types: dict[str, int]) -> dict[str, str]:
        """Ops present that a recognition flag would move to hardware."""
        out: dict[str, str] = {}
        for op in op_types:
            info = self.info(op)
            if info and info.unlocked_by and info.fallback_tier:
                out[op] = info.unlocked_by
        return out

    # ---- persistence ----

    def to_json(self) -> dict:
        return {
            "core_version": self.core_version,
            "mapping": {k: asdict(v) for k, v in sorted(self.mapping.items())},
            "frontend": sorted(self.frontend),
        }

    @classmethod
    def from_json(cls, doc: dict) -> OpTable:
        return cls(
            core_version=doc["core_version"],
            mapping={k: OpInfo(**v) for k, v in doc["mapping"].items()},
            frontend=set(doc["frontend"]),
        )


def cache_path(tc: Toolchain) -> Path:
    return RUNS_DIR / "_toolchain" / tc.core_tag / "optable.json"


def build(tc: Toolchain) -> OpTable:
    """Parse the doc, query the front end, and combine."""
    mapping = parse_operator_support(tc.operator_support_html).get("onnx", {})
    if not mapping:
        raise RuntimeError(
            f"no ONNX mapping table found in {tc.operator_support_html}; "
            "ST may have changed the document structure"
        )
    return OpTable(
        core_version=tc.stedgeai_expect_version,
        mapping=mapping,
        frontend=frontend_ops(tc),
    )


def load(tc: Toolchain, *, refresh: bool = False) -> OpTable:
    """Cached oracle, keyed by core version.

    The cache is invalidated when the pinned version changes, because the
    mapping genuinely moves between releases.
    """
    path = cache_path(tc)
    if not refresh and path.is_file():
        doc = json.loads(path.read_text())
        if doc.get("core_version") == tc.stedgeai_expect_version:
            return OpTable.from_json(doc)

    table = build(tc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table.to_json(), indent=1))
    return table
