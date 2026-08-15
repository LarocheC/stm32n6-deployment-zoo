"""Extract what the funnel needs to know from an ONNX graph.

Two modes, because the cheapest useful screen should not require a download.

`probe_url` issues a single HTTP Range request for the head of a remote file
and walks the protobuf by hand. This works because of how ONNX serialises:
`GraphProto.node` is field 1 and `GraphProto.initializer` is field 5, so the
nodes — the part that decides whether a model is viable at all — are laid down
before the weights. A few megabytes therefore yields the complete operator
histogram of a model that may be hundreds of megabytes on disk. Graph I/O
(fields 11 and 12) sits *after* the initializers and is not recoverable this
way, which is fine: the op histogram alone rejects most candidates.

`probe_file` is the full parse, used once a model has earned a download. It
adds I/O signatures, symbolic dimensions, weight accounting, and the one
question an op histogram fundamentally cannot answer — whether each
`MatMul`/`Gemm`/`Conv` has a constant second operand, which is what decides
between a hardware epoch and a Cortex-M55 fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal protobuf scanner
# ---------------------------------------------------------------------------

_WIRE_VARINT, _WIRE_64, _WIRE_LEN, _WIRE_32 = 0, 1, 2, 5

# Field numbers, from onnx.proto.
_MODEL_IR_VERSION = 1
_MODEL_GRAPH = 7
_MODEL_OPSET = 8
_GRAPH_NODE = 1
_NODE_INPUT = 1
_NODE_OUTPUT = 2
_NODE_OP_TYPE = 4
_OPSET_DOMAIN = 1
_OPSET_VERSION = 2


class _Truncated(Exception):
    """Ran off the end of the available bytes. Expected for head-only reads."""


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise _Truncated
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long — not a protobuf stream")


def _skip(buf: bytes, pos: int, wire: int) -> int:
    if wire == _WIRE_VARINT:
        return _varint(buf, pos)[1]
    if wire == _WIRE_64:
        pos += 8
    elif wire == _WIRE_32:
        pos += 4
    elif wire == _WIRE_LEN:
        length, pos = _varint(buf, pos)
        pos += length
    else:
        raise ValueError(f"unsupported wire type {wire}")
    if pos > len(buf):
        raise _Truncated
    return pos


def _fields(buf: bytes, start: int, end: int):
    """Yield `(field_number, wire_type, position)` for one message body."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        yield key >> 3, key & 0x07, pos
        pos = _skip(buf, pos, key & 0x07)


@dataclass
class ScanResult:
    """What a head-only protobuf walk can establish."""

    op_types: Counter = field(default_factory=Counter)
    ir_version: int | None = None
    opsets: dict[str, int] = field(default_factory=dict)
    node_count: int = 0
    #: The read ended before the file did. Expected and harmless for a head
    #: scan — it usually means we stopped inside the weights.
    truncated: bool = False
    #: The node list was read to its end, so `op_types` is the *complete*
    #: census rather than a prefix. Determined by observing a graph field with
    #: a number above `node`: ONNX serialises fields in order, so anything
    #: after field 1 proves no nodes remain. Without this, a head scan could
    #: silently report a partial histogram as if it were the whole model.
    nodes_complete: bool = False
    bytes_scanned: int = 0

    @property
    def default_opset(self) -> int | None:
        return self.opsets.get("") or self.opsets.get("ai.onnx")

    @property
    def census_is_trustworthy(self) -> bool:
        return self.nodes_complete and self.node_count > 0

    @property
    def control_flow_ops(self) -> dict[str, int]:
        """Top-level control flow, which hides the real graph in subgraphs.

        The scanner counts top-level nodes only — subgraphs live inside
        `NodeProto.attribute` and are skipped. When this is non-empty the
        census is a view of the wrapper, not the model. Two headline cases:
        `onnx-community/silero-vad` is one `If` selecting 8 kHz vs 16 kHz, and
        `onnx-community/whisper-tiny.en`'s merged decoder is one `If`
        selecting the with-past branch. Both are unusable regardless of what
        the branches contain, so this is a stop, not a prompt to dig deeper.
        """
        return {op: n for op, n in self.op_types.items() if op in ("If", "Loop", "Scan")}


def scan_model_bytes(buf: bytes) -> ScanResult:
    """Walk a (possibly truncated) serialised ModelProto for its op types."""
    out = ScanResult(bytes_scanned=len(buf))

    def scan_node(start: int, end: int) -> None:
        op_type: str | None = None
        try:
            for fnum, wire, pos in _fields(buf, start, end):
                if fnum == _NODE_OP_TYPE and wire == _WIRE_LEN:
                    length, vpos = _varint(buf, pos)
                    if vpos + length > len(buf):
                        raise _Truncated
                    op_type = buf[vpos : vpos + length].decode("utf-8", "replace")
        except _Truncated:
            out.truncated = True
        if op_type:
            out.op_types[op_type] += 1
            out.node_count += 1

    def scan_graph(start: int, end: int) -> None:
        try:
            for fnum, wire, pos in _fields(buf, start, min(end, len(buf))):
                if fnum == _GRAPH_NODE and wire == _WIRE_LEN:
                    length, npos = _varint(buf, pos)
                    scan_node(npos, min(npos + length, len(buf)))
                    if npos + length > len(buf):
                        raise _Truncated
                elif fnum > _GRAPH_NODE:
                    # Fields are serialised in number order, so reaching any
                    # field past `node` proves the node list ended.
                    out.nodes_complete = True
        except _Truncated:
            out.truncated = True

    def scan_opset(start: int, end: int) -> None:
        domain, version = "", None
        for fnum, wire, pos in _fields(buf, start, end):
            if fnum == _OPSET_DOMAIN and wire == _WIRE_LEN:
                length, vpos = _varint(buf, pos)
                domain = buf[vpos : vpos + length].decode("utf-8", "replace")
            elif fnum == _OPSET_VERSION and wire == _WIRE_VARINT:
                version, _ = _varint(buf, pos)
        if version is not None:
            out.opsets[domain] = version

    try:
        for fnum, wire, pos in _fields(buf, 0, len(buf)):
            if fnum == _MODEL_IR_VERSION and wire == _WIRE_VARINT:
                out.ir_version, _ = _varint(buf, pos)
            elif fnum == _MODEL_GRAPH and wire == _WIRE_LEN:
                length, gpos = _varint(buf, pos)
                scan_graph(gpos, gpos + length)
                if gpos + length > len(buf):
                    raise _Truncated
            elif fnum == _MODEL_OPSET and wire == _WIRE_LEN:
                length, opos = _varint(buf, pos)
                if opos + length > len(buf):
                    raise _Truncated
                scan_opset(opos, opos + length)
    except _Truncated:
        out.truncated = True
    return out


#: 8 MB of head is generous. `GraphProto.node` precedes `GraphProto.initializer`
#: in field order, so this captures every node of models far larger than this.
DEFAULT_HEAD_BYTES = 8 * 1024 * 1024


def probe_url(url: str, *, head_bytes: int = DEFAULT_HEAD_BYTES, timeout: float = 60.0) -> ScanResult:
    """Operator histogram for a remote ONNX file, without downloading it.

    Servers that ignore `Range` will send the whole body; the scan is capped
    either way so a misbehaving host costs bandwidth, not correctness.
    """
    import requests

    resp = requests.get(
        url, headers={"Range": f"bytes=0-{head_bytes - 1}"}, timeout=timeout, stream=True
    )
    resp.raise_for_status()
    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        chunks.append(chunk)
        total += len(chunk)
        if total >= head_bytes:
            break
    resp.close()
    return scan_model_bytes(b"".join(chunks))


# ---------------------------------------------------------------------------
# Full local probe
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    name: str
    dtype: str
    #: Ints for fixed dims, strings for symbolic ones, None for rank-unknown.
    shape: list[int | str | None]

    @property
    def symbolic_dims(self) -> list[str]:
        return [d for d in self.shape if isinstance(d, str)]

    @property
    def dynamic_axes(self) -> list[tuple[int, str | None]]:
        """`(axis, name)` for every dim that is not a fixed integer.

        Unnamed axes (`None`) count. ST's exporter-agnostic requirement is
        that *every* dimension be static before compilation, and an anonymous
        dim is no more compilable than a named one — it is merely harder to
        address, since `--fix-parametric-shapes` keys on the name. Anonymous
        axes have to be pinned by re-exporting or by rewriting the value_info.
        """
        return [
            (i, d if isinstance(d, str) else None)
            for i, d in enumerate(self.shape)
            if not isinstance(d, int)
        ]

    @property
    def is_static(self) -> bool:
        return all(isinstance(d, int) for d in self.shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def numel(self) -> int | None:
        if not self.is_static:
            return None
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class ConditionalOp:
    """A `MatMul`/`Gemm`/`Conv` whose hardware mapping we resolved by hand.

    `directly_constant` means the operand is an initializer or a `Constant`
    node output — the compiler sees a constant with no help. `foldable` means
    it is *derived* from constants through value-independent ops (a weight put
    through a `Reshape`, say), so it becomes constant after constant folding
    but may not be recognised as such beforehand. The distinction is
    actionable: foldable operands are an argument for running the simplifier,
    not evidence that the model is attention-bound.
    """

    node_name: str
    op_type: str
    operand: str
    directly_constant: bool
    foldable: bool

    @property
    def constant(self) -> bool:
        return self.directly_constant or self.foldable

    @property
    def maps_to_hardware(self) -> bool:
        return self.constant

    @property
    def needs_folding(self) -> bool:
        return self.foldable and not self.directly_constant


@dataclass
class Probe:
    """Everything the static funnel needs from a local ONNX file."""

    path: Path
    #: Top-level graph nodes only.
    op_types: Counter
    #: Ops nested inside `If`/`Loop`/`Scan` subgraph attributes, counted
    #: separately so a control-flow wrapper cannot make a large model look
    #: like a one-node graph.
    subgraph_op_types: Counter
    ir_version: int
    opsets: dict[str, int]
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    #: Graph inputs that are really initializers (old IR-v3 style exports).
    initializer_inputs: list[str]
    initializer_bytes: int
    max_tensor_dim: int
    max_rank: int
    conditional: list[ConditionalOp]
    has_external_data: bool

    @property
    def default_opset(self) -> int | None:
        return self.opsets.get("") or self.opsets.get("ai.onnx")

    @property
    def symbolic_dims(self) -> list[str]:
        seen: list[str] = []
        for spec in self.inputs + self.outputs:
            for dim in spec.symbolic_dims:
                if dim not in seen:
                    seen.append(dim)
        return seen

    @property
    def control_flow_ops(self) -> dict[str, int]:
        return {op: n for op, n in self.op_types.items() if op in ("If", "Loop", "Scan")}

    @property
    def dynamic_conditional_ops(self) -> list[ConditionalOp]:
        """The ones that will fall back to the Cortex-M55.

        Nonzero here on a `MatMul`-heavy graph means self-attention, and means
        the model is software-bound however good the rest of it looks.
        """
        return [c for c in self.conditional if not c.constant]


_ELEM_TYPE = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
    7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 16: "bfloat16",
}

#: Which operand must be constant for the op to reach hardware. ST's table:
#: Conv "weights tensor should be constant", Gemm "params input should be
#: constant", MatMul "second input should be constant".
_CONDITIONAL_OPERAND = {"Conv": 1, "Gemm": 1, "MatMul": 1}


def _value_info_spec(vi) -> TensorSpec:  # noqa: ANN001
    tt = vi.type.tensor_type
    shape: list[int | str | None] = []
    for dim in tt.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return TensorSpec(
        name=vi.name,
        dtype=_ELEM_TYPE.get(tt.elem_type, f"elem{tt.elem_type}"),
        shape=shape,
    )


def probe_file(path: Path) -> Probe:
    """Full local probe. Loads weights only when they are already inline."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    # Ops hidden inside control-flow subgraphs, counted separately. Without
    # this, a merged encoder/decoder export reads as a one-node graph.
    subgraph_ops: Counter = Counter()

    def _walk_subgraphs(g) -> None:  # noqa: ANN001
        for node in g.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    subgraph_ops.update(n.op_type for n in attr.g.node)
                    _walk_subgraphs(attr.g)
                for sub in attr.graphs:
                    subgraph_ops.update(n.op_type for n in sub.node)
                    _walk_subgraphs(sub)

    _walk_subgraphs(graph)

    initializers = {init.name for init in graph.initializer}

    # Constant-valued tensors also arise from `Constant` nodes, which some
    # exporters emit instead of initializers. Both count as constant operands.
    const_node_outputs = {
        out for node in graph.node if node.op_type == "Constant" for out in node.output
    }
    direct_constants = initializers | const_node_outputs

    # Transitive constant propagation. A weight reshaped before a MatMul is
    # still a weight; without this the operand reads as dynamic and the model
    # looks attention-bound when it is not. Iterated to a fixed point because
    # graph.node is only topologically sorted by convention.
    foldable = set(direct_constants)
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.op_type in ("Constant", "ConstantOfShape"):
                continue
            if not node.input:
                continue
            if all(inp in foldable or inp == "" for inp in node.input):
                for out in node.output:
                    if out and out not in foldable:
                        foldable.add(out)
                        changed = True

    conditional: list[ConditionalOp] = []
    for node in graph.node:
        idx = _CONDITIONAL_OPERAND.get(node.op_type)
        if idx is None or len(node.input) <= idx:
            continue
        operand = node.input[idx]
        conditional.append(
            ConditionalOp(
                node_name=node.name or f"{node.op_type}_{len(conditional)}",
                op_type=node.op_type,
                operand=operand,
                directly_constant=operand in direct_constants,
                foldable=operand in foldable,
            )
        )

    inputs = [_value_info_spec(vi) for vi in graph.input]
    outputs = [_value_info_spec(vi) for vi in graph.output]

    # Old IR-v3 exports declare every weight as a graph input as well as an
    # initializer. Confirmed in opencv/object_detection_nanodet (~140 of them)
    # and opencv/facial_expression_recognition (36). Left alone, stedgeai sees
    # a model with 140 inputs; `onnxsim` removes them.
    initializer_inputs = [spec.name for spec in inputs if spec.name in initializers]

    init_bytes = 0
    has_external = False
    for init in graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            has_external = True
            continue
        try:
            init_bytes += numpy_helper.to_array(init).nbytes
        except Exception:  # noqa: BLE001 - malformed tensors must not kill a probe
            continue

    dims: list[int] = []
    ranks: list[int] = []
    for spec in inputs + outputs:
        ranks.append(spec.rank)
        dims += [d for d in spec.shape if isinstance(d, int)]
    for init in graph.initializer:
        ranks.append(len(init.dims))
        dims += [int(d) for d in init.dims]

    return Probe(
        path=path,
        op_types=Counter(node.op_type for node in graph.node),
        subgraph_op_types=subgraph_ops,
        ir_version=int(model.ir_version),
        opsets={op.domain: int(op.version) for op in model.opset_import},
        inputs=inputs,
        outputs=outputs,
        initializer_inputs=initializer_inputs,
        initializer_bytes=init_bytes,
        max_tensor_dim=max(dims) if dims else 0,
        max_rank=max(ranks) if ranks else 0,
        conditional=conditional,
        has_external_data=has_external,
    )
