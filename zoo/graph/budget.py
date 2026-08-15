"""Memory and compute accounting, read off the artifact rather than estimated.

Three numbers decide whether a model is worth compiling, and they are not
equally important.

**Weights** are the forgiving one. They can stream from the 112 MB of external
octoFlash at roughly a 1.6x latency cost — measured on this board: the same
1.4 MB network ran at 4.40 ms with weights on-chip and 7.14 ms from flash.

**Peak activation is the cliff.** ST's own published measurements on the
STM32N6570-DK: DeepLabv3-MobileNetV2 at 320x320 keeps 2421 KB internal and
runs in 40.83 ms; at 416x416 it spills 2028 KB to external memory and takes
227.02 ms. That is 1.7x the pixels for 5.6x the time. FastDepth is worse —
224 to 320 costs 19.5x. So the question is never "does it fit" but "does peak
activation stay under the on-chip pool".

**MACs** matter least. Below roughly a megabyte of weights latency is
epoch-bound rather than MAC-bound, so the MAC count is context for the other
two rather than a predictor on its own.

A warning this module repeats because it was learned expensively: **the
analytic peak is a lower bound, not an allocation.** A prior project computed
1.95 MB and concluded a model would fit the 2.8 MB on-chip pool; the real
allocator reported 116 MB unallocatable and needed 23.3 MB across pools. This
stage may reject. It must never certify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

#: Bytes per element, by ONNX elem_type.
_ELEM_BYTES = {
    1: 4,   # float32
    2: 1,   # uint8
    3: 1,   # int8
    4: 2,   # uint16
    5: 2,   # int16
    6: 4,   # int32
    7: 8,   # int64
    9: 1,   # bool
    10: 2,  # float16
    11: 8,  # float64
    12: 4,  # uint32
    13: 8,  # uint64
    16: 2,  # bfloat16
}

#: Ops whose output is a view or a requantisation of their input rather than a
#: new buffer the accelerator must materialise.
_QDQ = ("QuantizeLinear", "DequantizeLinear")


@dataclass
class Budget:
    weight_bytes: int = 0
    #: Weight bytes counted at their *quantised* element size. A QDQ graph
    #: stores int8 weights next to float scales; counting the raw initializers
    #: alone would report the deployed size as far larger than it is.
    quantised_weight_bytes: int = 0
    macs: int = 0
    #: Peak live activation as ONNX Runtime materialises it — every
    #: dequantised float intermediate is a real buffer.
    peak_activation_raw: int = 0
    #: Peak as a QDQ-fusing backend sees it. Neural-ART folds Q/DQ into the
    #: consuming integer kernel, so the float intermediate never exists. This
    #: is the number that decides the on-chip fit, and it is usually several
    #: times smaller. Never quote it as measured.
    peak_activation_fused: int = 0
    io_bytes: int = 0
    unresolved_tensors: int = 0
    is_quantised: bool = False
    notes: list[str] = field(default_factory=list)

    def fits(self, pool_bytes: int, *, fused: bool = True) -> bool:
        peak = self.peak_activation_fused if fused else self.peak_activation_raw
        return peak <= pool_bytes

    def placement(self, policy: dict) -> str:
        """Which pool the weights and activations would plausibly land in.

        Advisory. The compiler's allocator is the authority, and it has been
        observed to disagree by orders of magnitude.
        """
        if self.unresolved_tensors:
            # The peak was computed over an incomplete graph. Answering anyway
            # would be the exact failure this module exists to warn about.
            return "unknown-unpinned-shapes"

        cfg = policy.get("budget", {})
        onchip = int(cfg.get("onchip_bytes", 2_883_576))
        psram = int(cfg.get("hyperram_bytes", 33_554_432))
        flash = int(cfg.get("octoflash_bytes", 117_440_512))

        weights = self.quantised_weight_bytes or self.weight_bytes
        peak = self.peak_activation_fused

        if peak > psram:
            return "does-not-fit"
        if peak > onchip:
            return "activations-in-psram"
        if weights + peak <= onchip:
            return "all-on-chip"
        if weights <= flash:
            return "weights-in-flash"
        return "does-not-fit"

    def metrics(self) -> dict[str, Any]:
        return {
            "weight_bytes": self.weight_bytes,
            "quantised_weight_bytes": self.quantised_weight_bytes,
            "macs": self.macs,
            "peak_activation_raw": self.peak_activation_raw,
            "peak_activation_fused": self.peak_activation_fused,
            "io_bytes": self.io_bytes,
            "unresolved_tensors": self.unresolved_tensors,
            "is_quantised": self.is_quantised,
        }


# ---------------------------------------------------------------------------


def _elem_bytes(elem_type: int) -> int:
    return _ELEM_BYTES.get(elem_type, 4)


def _tensor_bytes(shape: list[int], elem_type: int) -> int:
    n = 1
    for dim in shape:
        n *= max(int(dim), 1)
    return n * _elem_bytes(elem_type)


def _shapes(model: Any) -> dict[str, tuple[list[int] | None, int]]:
    """`{tensor_name: (shape or None, elem_type)}` after shape inference."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        graph = inferred.graph
    except Exception:  # noqa: BLE001 - a graph we cannot infer is still worth counting
        graph = model.graph

    out: dict[str, tuple[list[int] | None, int]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        dims: list[int] = []
        static = True
        for dim in tt.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                static = False
                break
        out[vi.name] = (dims if static else None, tt.elem_type)

    for init in graph.initializer:
        out[init.name] = ([int(d) for d in init.dims], init.data_type)
    return out


def weight_bytes(model: Any) -> tuple[int, int, bool]:
    """`(raw_bytes, quantised_bytes, is_quantised)` for the initializers.

    A QDQ graph stores int8 weights alongside float scales and zero-points.
    Summing raw initializer bytes is right for "how big is this file"; the
    deployed footprint is the int8 payload, so both are reported. Resolving
    this correctly matters — a naive walk that looks only for float weights
    reports a quantised model as having none.
    """
    raw = 0
    quantised = 0
    is_quantised = any(node.op_type in _QDQ for node in model.graph.node)

    # Tensors consumed by DequantizeLinear are the quantised payload.
    dq_inputs = {
        node.input[0]
        for node in model.graph.node
        if node.op_type == "DequantizeLinear" and node.input
    }

    for init in model.graph.initializer:
        try:
            nbytes = int(numpy_helper.to_array(init).nbytes)
        except Exception:  # noqa: BLE001
            nbytes = _tensor_bytes([int(d) for d in init.dims], init.data_type)
        raw += nbytes
        if not is_quantised or init.name in dq_inputs or _elem_bytes(init.data_type) == 1:
            quantised += nbytes
    return raw, (quantised if is_quantised else raw), is_quantised


def macs(model: Any) -> int:
    """Multiply-accumulates for Conv / ConvTranspose / Gemm / MatMul.

    Derived from inferred output shapes and weight shapes, so the number moves
    when the graph does. Ops whose shapes cannot be resolved are skipped rather
    than guessed, which makes this a lower bound on an under-inferred graph.
    """
    shapes = _shapes(model)
    initializers = {init.name: [int(d) for d in init.dims] for init in model.graph.initializer}
    total = 0

    for node in model.graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            if len(node.input) < 2 or not node.output:
                continue
            out_shape, _ = shapes.get(node.output[0], (None, 1))
            w_shape = initializers.get(node.input[1]) or (shapes.get(node.input[1], (None, 1))[0])
            if not out_shape or not w_shape:
                continue
            # output elements x (kernel volume x input channels per group)
            out_elems = int(np.prod(out_shape))
            per_output = int(np.prod(w_shape[1:])) if len(w_shape) > 1 else 1
            total += out_elems * per_output

        elif node.op_type == "Gemm":
            if len(node.input) < 2 or not node.output:
                continue
            out_shape, _ = shapes.get(node.output[0], (None, 1))
            b_shape = initializers.get(node.input[1]) or (shapes.get(node.input[1], (None, 1))[0])
            if not out_shape or not b_shape:
                continue
            k = b_shape[0] if len(b_shape) > 1 else b_shape[0]
            total += int(np.prod(out_shape)) * int(k)

        elif node.op_type == "MatMul":
            if len(node.input) < 2 or not node.output:
                continue
            out_shape, _ = shapes.get(node.output[0], (None, 1))
            a_shape, _ = shapes.get(node.input[0], (None, 1))
            if not out_shape or not a_shape:
                continue
            k = a_shape[-1]
            total += int(np.prod(out_shape)) * int(k)

    return total


def peak_activation(model: Any, *, fused_qdq: bool = False) -> tuple[int, int]:
    """`(peak_bytes, unresolved_tensor_count)` by liveness over the node order.

    A tensor is live from the moment it is produced until its last consumer
    has run; the peak is the largest total of live tensors at any point. This
    is the standard lower bound on what an allocator must find room for —
    lower because a real allocator also honours alignment, cannot always reuse
    a freed buffer, and may keep scratch space the graph does not name.

    `fused_qdq=True` models the accelerator rather than ONNX Runtime. In a QDQ
    graph the pattern `Q -> DQ -> Conv` makes ORT materialise a float
    intermediate; Neural-ART folds the Q/DQ into an integer kernel and never
    creates it. Counting those outputs at their quantised element size is what
    turns an apparently-7.8 MB graph into a 1.95 MB one, which is the
    difference between external memory and on-chip.
    """
    shapes = _shapes(model)
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {vi.name for vi in model.graph.input}
    graph_outputs = {vi.name for vi in model.graph.output}

    # Element size the accelerator would actually hold each tensor at.
    elem_override: dict[str, int] = {}
    if fused_qdq:
        for node in model.graph.node:
            if node.op_type == "DequantizeLinear" and node.input and node.output:
                src = shapes.get(node.input[0])
                if src:
                    elem_override[node.output[0]] = src[1]

    def size_of(name: str) -> int | None:
        entry = shapes.get(name)
        if not entry:
            return None
        shape, elem_type = entry
        if shape is None:
            return None
        return _tensor_bytes(shape, elem_override.get(name, elem_type))

    # Last use of each tensor, in node order.
    last_use: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                last_use[name] = index
    for name in graph_outputs:
        last_use[name] = len(model.graph.node)

    live: dict[str, int] = {}
    unresolved = 0
    peak = 0

    for name in graph_inputs:
        if name in initializers:
            continue
        size = size_of(name)
        if size is None:
            unresolved += 1
        else:
            live[name] = size

    for index, node in enumerate(model.graph.node):
        for name in node.output:
            if not name or name in initializers:
                continue
            size = size_of(name)
            if size is None:
                unresolved += 1
                continue
            live[name] = size

        peak = max(peak, sum(live.values()))

        for name in list(live):
            if last_use.get(name, -1) <= index and name not in graph_outputs:
                del live[name]

    return peak, unresolved


def analyse(path: Path | Any) -> Budget:
    """Full accounting for one graph."""
    model = onnx.load(str(path), load_external_data=False) if isinstance(path, (str, Path)) else path

    raw, quantised, is_quantised = weight_bytes(model)
    peak_raw, unresolved_raw = peak_activation(model, fused_qdq=False)
    peak_fused, _ = peak_activation(model, fused_qdq=True)

    shapes = _shapes(model)
    io_bytes = 0
    for vi in list(model.graph.input) + list(model.graph.output):
        entry = shapes.get(vi.name)
        if entry and entry[0] is not None:
            io_bytes += _tensor_bytes(entry[0], entry[1])

    budget = Budget(
        weight_bytes=raw,
        quantised_weight_bytes=quantised,
        macs=macs(model),
        peak_activation_raw=peak_raw,
        peak_activation_fused=peak_fused,
        io_bytes=io_bytes,
        unresolved_tensors=unresolved_raw,
        is_quantised=is_quantised,
    )
    if unresolved_raw:
        budget.notes.append(
            f"{unresolved_raw} tensor(s) had no inferable static shape — the peak is "
            "an under-count, not merely a lower bound"
        )
    if not is_quantised:
        budget.notes.append(
            "graph is float; after int8 quantisation expect roughly a quarter of these "
            "weight bytes, and activations to shrink similarly"
        )
    return budget
