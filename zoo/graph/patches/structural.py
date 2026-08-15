"""Structural rewrites: version downgrades, input hygiene, shape pinning.

These are the patches whose semantics are unambiguous. Each corresponds to a
lint violation, so a rejected model can be repaired and retried rather than
merely reported.

The ST-front-end-specific rewrites — Slice sentinel bounds, Pad-to-Concat,
Clip-through-DequantizeLinear, rank-1 input promotion — live separately,
because each works around a specific compiler behaviour and needs its exact
precondition recorded alongside it rather than guessed at.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from zoo.faults.taxonomy import FailureClass as FC
from zoo.graph.patches import PatchResult, applies_when, register

#: ONNX IR version the ST front end is happy with. Newer files are usually
#: fine graph-wise; it is the declared version alone that is refused.
TARGET_IR = 8


# ---------------------------------------------------------------------------


@register(
    "downgrade_ir",
    fixes=FC.IR_TOO_HIGH,
    summary=f"lower ir_version to {TARGET_IR} without touching the graph",
)
@applies_when(lambda m: m.ir_version > TARGET_IR)
def downgrade_ir(model: Any) -> tuple[Any, PatchResult]:
    """Lower the declared IR version.

    IR version records which ONNX *container* features a file may use, not
    what the graph computes. Exporters stamp whatever version they were built
    against, so a graph using nothing newer than IR 8 routinely arrives
    declaring IR 10 and is refused on that basis alone. Lowering the number is
    a no-op numerically — hence the parity check will pass — but it is not
    always safe, so the caller must still run it: a file that genuinely uses a
    newer container feature will fail ONNX's own checker afterwards.
    """
    original = model.ir_version
    out = copy.deepcopy(model)
    out.ir_version = TARGET_IR
    try:
        onnx.checker.check_model(out)
        note = f"ir_version {original} → {TARGET_IR}"
    except onnx.checker.ValidationError as exc:
        return model, PatchResult(
            "downgrade_ir",
            applied=False,
            note=f"graph genuinely needs IR {original}: {exc}",
        )
    return out, PatchResult("downgrade_ir", applied=True, changed=1, note=note,
                            detail={"from": original, "to": TARGET_IR})


# ---------------------------------------------------------------------------


def _initializer_inputs(model: Any) -> list[str]:
    names = {init.name for init in model.graph.initializer}
    return [vi.name for vi in model.graph.input if vi.name in names]


@register(
    "strip_initializer_inputs",
    fixes=FC.INITIALIZERS_AS_INPUTS,
    summary="remove graph inputs that are also initializers (IR-v3 style export)",
    changes_signature=True,
)
@applies_when(lambda m: bool(_initializer_inputs(m)))
def strip_initializer_inputs(model: Any) -> tuple[Any, PatchResult]:
    """Drop weights that are declared as graph inputs as well as initializers.

    Older exporters list every initializer in `graph.input` too. ONNX Runtime
    tolerates it; the ST front end sees a model with 140 inputs and behaves
    accordingly. Confirmed in `opencv/object_detection_nanodet` (~140 of them)
    and `opencv/facial_expression_recognition` (36).

    This changes the graph's declared signature — that is the entire point —
    so the caller cannot parity-check it by feeding identical inputs to both
    versions. The rewrite removes no data: every stripped name remains present
    as an initializer.
    """
    doomed = _initializer_inputs(model)
    out = copy.deepcopy(model)
    keep = [vi for vi in out.graph.input if vi.name not in set(doomed)]
    del out.graph.input[:]
    out.graph.input.extend(keep)
    return out, PatchResult(
        "strip_initializer_inputs",
        applied=True,
        changed=len(doomed),
        note=f"removed {len(doomed)} initializer(s) from graph.input",
        detail={"removed": doomed[:20], "total": len(doomed)},
    )


# ---------------------------------------------------------------------------


def pin_dims(model: Any, pin: dict[str, int]) -> tuple[Any, PatchResult]:
    """Replace named symbolic dimensions with fixed values, everywhere.

    `--fix-parametric-shapes` does this inside the compiler, but doing it in
    the file first means every downstream stage — quantisation calibration,
    ONNX Runtime parity, the budget estimate — sees the same static graph the
    compiler will. Keeping those in agreement is worth more than the one
    command-line argument it saves.

    Only *named* dims can be pinned this way. An anonymous axis carries no key
    to address it by, which is why lint reports the two cases differently.
    """
    if not pin:
        return model, PatchResult("pin_dims", applied=False, note="nothing to pin")

    out = copy.deepcopy(model)
    changed = 0
    unmatched = set(pin)

    # The dim_param has to be read before it is cleared, so the value is
    # captured first rather than referenced through the field being mutated.
    for collection in (out.graph.input, out.graph.output, out.graph.value_info):
        for vi in collection:
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_param") and dim.dim_param in pin:
                    pin_value = int(pin[dim.dim_param])
                    unmatched.discard(dim.dim_param)
                    dim.ClearField("dim_param")
                    dim.dim_value = pin_value
                    changed += 1

    note = f"pinned {changed} dimension slot(s)"
    if unmatched:
        note += f"; no such dim in the graph: {sorted(unmatched)}"
    return out, PatchResult(
        "pin_dims",
        applied=changed > 0,
        changed=changed,
        note=note,
        detail={"pin": pin, "unmatched": sorted(unmatched)},
    )


# ---------------------------------------------------------------------------


def fold_const_inputs(model: Any, values: dict[str, Any]) -> tuple[Any, PatchResult]:
    """Turn named graph inputs into initializers holding a fixed value.

    This is how a recipe's `role = "constant"` inputs are honoured, and it is
    worth more than it looks. A rank-0 or rank-1 integer input — a sample rate,
    a beam count, a flag — is a configuration scalar, not data, and ST's front
    end rejects rank-1 graph inputs outright. Folding removes the input
    entirely, so the rejection has nothing left to fire on, and constant
    propagation downstream can then simplify whatever the scalar controlled.
    """
    if not values:
        return model, PatchResult("fold_const_inputs", applied=False, note="no constants given")

    out = copy.deepcopy(model)
    by_name = {vi.name: vi for vi in out.graph.input}
    existing = {init.name for init in out.graph.initializer}

    folded: list[str] = []
    missing: list[str] = []
    for name, value in values.items():
        vi = by_name.get(name)
        if vi is None or name in existing:
            missing.append(name)
            continue

        elem_type = vi.type.tensor_type.elem_type or TensorProto.FLOAT
        np_dtype = helper.tensor_dtype_to_np_dtype(elem_type)
        shape = [
            d.dim_value if d.HasField("dim_value") else 1
            for d in vi.type.tensor_type.shape.dim
        ]
        array = np.asarray(value, dtype=np_dtype)
        if array.shape != tuple(shape):
            array = np.broadcast_to(array, shape).astype(np_dtype) if shape else array
        out.graph.initializer.append(numpy_helper.from_array(array.copy(), name=name))
        folded.append(name)

    keep = [vi for vi in out.graph.input if vi.name not in set(folded)]
    del out.graph.input[:]
    out.graph.input.extend(keep)

    note = f"folded {len(folded)} input(s) into initializers"
    if missing:
        note += f"; not a graph input: {missing}"
    return out, PatchResult(
        "fold_const_inputs",
        applied=bool(folded),
        changed=len(folded),
        note=note,
        detail={"folded": folded, "missing": missing},
    )


# ---------------------------------------------------------------------------


def _simplifiable(model: Any) -> bool:
    """The simplifier needs static shapes and straight-line control flow.

    Given neither, it does not merely fail — it thrashes. Run against
    `onnx-community/silero-vad`, whose graph has fifteen `If` nodes and three
    anonymous dynamic axes, onnxsim tried to fold through every branch and
    reported "Simplified model larger than 2GB. Trying to save as external
    data..." before producing something unusable. Refusing up front is both
    faster and quieter than discarding the result afterwards.
    """
    if any(node.op_type in ("If", "Loop", "Scan") for node in model.graph.node):
        return False
    for vi in model.graph.input:
        for dim in vi.type.tensor_type.shape.dim:
            if not (dim.HasField("dim_value") and dim.dim_value > 0):
                return False
    return True


@register(
    "constant_fold",
    fixes="",
    summary="run the ONNX simplifier: fold Shape/Gather plumbing and constant subgraphs",
)
@applies_when(_simplifiable)
def constant_fold(model: Any) -> tuple[Any, PatchResult]:
    """Constant-fold and simplify.

    Two distinct wins, both documented rather than speculative. `Shape`,
    `Gather` and `ReduceProd` chains left behind by the tracer have no place in
    a statically shaped graph and are folded away entirely. And an operand that
    is constant only *after* folding — a weight put through a `Reshape` before
    a `MatMul` — becomes visibly constant, which is what moves that `MatMul`
    from a Cortex-M55 fallback onto the accelerator.
    """
    try:
        from onnxsim import simplify
    except ImportError:
        return model, PatchResult(
            "constant_fold", applied=False, note="onnxsim not installed"
        )

    before = len(model.graph.node)
    try:
        simplified, ok = simplify(model)
    except Exception as exc:  # noqa: BLE001 - a failed simplify must not kill the run
        return model, PatchResult(
            "constant_fold", applied=False, note=f"{type(exc).__name__}: {exc}"
        )
    if not ok:
        return model, PatchResult(
            "constant_fold", applied=False, note="simplifier reported the result unvalidated"
        )

    after = len(simplified.graph.node)
    return simplified, PatchResult(
        "constant_fold",
        applied=after != before,
        changed=abs(before - after),
        note=f"{before} → {after} nodes",
        detail={"nodes_before": before, "nodes_after": after},
    )
