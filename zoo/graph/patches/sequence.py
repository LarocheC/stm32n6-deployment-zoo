"""Rewrites for running a sequence model over a shorter window than it was
exported for.

Freezing a symbolic sequence dimension is necessary but not sufficient. A
transformer encoder with *learned absolute* positional embeddings carries a
constant sized for the export-time window, and shortening the input alone
leaves that constant unchanged — so the graph stops being valid rather than
becoming smaller.

`onnx-community/whisper-tiny.en` is the case that motivated this. Pinning
`encoder_sequence_length` from 3000 to 500 fails at load with

    Node (/Add_2) Op (Add) [ShapeInferenceError] Incompatible dimensions

because `embed_positions.weight` is a `(1500, 384)` initializer added to a
`(1, 250, 384)` activation. Slicing it to `(250, 384)` is exact: position *i*
of a shorter window is the same position *i*, so the retained rows are the
ones the model would have used anyway.

This matters beyond tidiness: the window length is what decides whether the
model is deployable at all. Measured peak activation for this encoder, fp32,
after both pinning and this patch (so every shape resolves and the graph
actually runs):

    30 s (3000 frames)   112.61 MB   beyond even the 32 MB of PSRAM
    10 s (1000 frames)    13.54 MB   PSRAM
     5 s ( 500 frames)     4.99 MB   PSRAM
     2 s ( 200 frames)      2.00 MB  fits the ~2.88 MB on-chip pool

Note that these are float. int8 activations should be roughly four times
smaller, which would bring the 5-second window on-chip — but that is an
expectation to be measured after quantisation, not an entitlement. Before this
patch existed the 5-second figure appeared to be 2.3 MB; that was an
under-count produced by a graph too broken to shape-infer, and it is exactly
the kind of number that survives into a conclusion if nothing checks whether
the model still loads.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from zoo.faults.taxonomy import FailureClass as FC
from zoo.graph.patches import PatchResult, applies_when, register


def _inferred_shapes(model: Any) -> dict[str, list[int | None]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        graph = inferred.graph
    except Exception:  # noqa: BLE001
        graph = model.graph
    out: dict[str, list[int | None]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        out[vi.name] = [
            int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else None
            for d in vi.type.tensor_type.shape.dim
        ]
    return out


def _has_oversized_positional_constant(model: Any) -> bool:
    """Cheap predicate: a 2-D initializer added to something shorter."""
    shapes = _inferred_shapes(model)
    initializers = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        for const_side, act_side in ((0, 1), (1, 0)):
            init = initializers.get(node.input[const_side])
            if init is None or len(init.dims) != 2:
                continue
            act = shapes.get(node.input[act_side])
            if not act or len(act) < 2 or act[-1] != int(init.dims[1]):
                continue
            if act[-2] is not None and act[-2] < int(init.dims[0]):
                return True
    return False


@register(
    "slice_positional_embedding",
    fixes=FC.SHAPE_DYNAMIC_UNPINNABLE,
    summary="truncate learned positional-embedding constants to the pinned window",
)
@applies_when(_has_oversized_positional_constant)
def slice_positional_embedding(
    model: Any, *, length: int | None = None
) -> tuple[Any, PatchResult]:
    """Truncate learned positional-embedding constants to the actual window.

    With `length=None` the target is inferred per-site from the activation the
    embedding is added to, which is the robust choice: the encoder's sequence
    length is usually half the input frame count after the stride-2
    convolution, and hard-coding that relationship would be wrong for the next
    architecture.
    """
    shapes = _inferred_shapes(model)
    initializers = {init.name: init for init in model.graph.initializer}

    edits: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    out = copy.deepcopy(model)
    out_inits = {init.name: init for init in out.graph.initializer}

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        for const_side, act_side in ((0, 1), (1, 0)):
            name = node.input[const_side]
            init = initializers.get(name)
            if init is None or len(init.dims) != 2:
                continue

            rows, width = int(init.dims[0]), int(init.dims[1])
            act_shape = shapes.get(node.input[act_side])
            if not act_shape or len(act_shape) < 2:
                continue
            # The activation is (..., seq, width); its last dim must match the
            # embedding width or this is not a positional add.
            if act_shape[-1] != width:
                continue

            target = length if length is not None else act_shape[-2]
            if target is None or target >= rows:
                continue

            array = numpy_helper.to_array(init)
            sliced = np.ascontiguousarray(array[:target])
            out_inits[name].CopyFrom(numpy_helper.from_array(sliced, name=name))
            edits.append((name, array.shape, sliced.shape))
            break

    if not edits:
        return model, PatchResult(
            "slice_positional_embedding",
            applied=False,
            note="no positional-embedding constant needed truncating",
        )

    detail = {name: {"from": list(a), "to": list(b)} for name, a, b in edits}
    return out, PatchResult(
        "slice_positional_embedding",
        applied=True,
        changed=len(edits),
        note="; ".join(f"{n} {tuple(a)} → {tuple(b)}" for n, a, b in edits),
        detail=detail,
    )
