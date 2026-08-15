"""Graph patches and the parity gate that guards them.

Every patch here claims to preserve semantics. The parity gate is what turns
that from a claim into a check, and its own failure modes matter: it must say
"unknown" when it cannot evaluate, never "fine".
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from zoo.graph import parity, patches
from zoo.graph.patches import structural


def _tiny_model(*, ir_version: int = 8, weights_as_inputs: bool = False):
    """y = relu(x @ W + b), with x [1,4] and W [4,3]."""
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["pre"]),
        helper.make_node("Relu", ["pre"], ["y"]),
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])]
    if weights_as_inputs:
        inputs.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 3]))
        inputs.append(helper.make_tensor_value_info("B", TensorProto.FLOAT, [3]))

    graph = helper.make_graph(
        nodes, "tiny", inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = ir_version
    return model


def _dynamic_model():
    """Same graph with a named symbolic batch dimension."""
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "dyn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


# -- registry ---------------------------------------------------------------


def test_registry_is_populated_and_addressable() -> None:
    names = {p.name for p in patches.available()}
    assert {"downgrade_ir", "strip_initializer_inputs", "constant_fold"} <= names
    assert patches.get("downgrade_ir").fixes == "IR_TOO_HIGH"


def test_unknown_patch_name_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="available:"):
        patches.get("no_such_patch")


# -- downgrade_ir -----------------------------------------------------------


def test_downgrade_ir_only_fires_when_needed() -> None:
    patch = patches.get("downgrade_ir")
    assert not patch.applies(_tiny_model(ir_version=8))
    assert patch.applies(_tiny_model(ir_version=10))


def test_downgrade_ir_lowers_the_version_and_preserves_the_graph() -> None:
    model = _tiny_model(ir_version=10)
    out, result = patches.get("downgrade_ir").run(model)
    assert result.applied
    assert out.ir_version == structural.TARGET_IR
    assert len(out.graph.node) == len(model.graph.node)
    assert parity.compare(model, out).ok is True


# -- strip_initializer_inputs ----------------------------------------------


def test_strip_initializer_inputs_removes_only_the_duplicates() -> None:
    """Old exports list every weight as a graph input too. Left alone,
    stedgeai sees a model with that many inputs."""
    model = _tiny_model(weights_as_inputs=True)
    assert len(model.graph.input) == 3
    patch = patches.get("strip_initializer_inputs")
    assert patch.applies(model)

    out, result = patch.run(model)
    assert result.applied
    assert result.changed == 2
    assert [vi.name for vi in out.graph.input] == ["x"]
    # The data is not lost — it remains as an initializer.
    assert {i.name for i in out.graph.initializer} == {"W", "B"}


def test_strip_initializer_inputs_declares_that_it_changes_the_signature() -> None:
    """It cannot be parity-checked by feeding identical inputs to both
    versions, and the registry has to say so rather than let a caller assume."""
    assert patches.get("strip_initializer_inputs").changes_signature


def test_strip_initializer_inputs_is_a_noop_on_a_clean_graph() -> None:
    assert not patches.get("strip_initializer_inputs").applies(_tiny_model())


# -- constant_fold ----------------------------------------------------------


def test_constant_fold_preserves_semantics_exactly() -> None:
    model = _tiny_model()
    out, _ = patches.get("constant_fold").run(model)
    check = parity.compare(model, out)
    assert check.ok is True
    assert check.max_abs == 0.0


# -- pin_dims ---------------------------------------------------------------


def test_pin_dims_makes_a_dynamic_graph_static() -> None:
    model = _dynamic_model()
    out, result = structural.pin_dims(model, {"batch": 1})
    assert result.applied
    dims = out.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_value == 1
    assert not dims[0].dim_param
    # And it is now runnable, which the original was not.
    assert parity.compare(out, out).ok is True


def test_pin_dims_reports_names_that_are_not_in_the_graph() -> None:
    """A typo'd pin must not look like a successful one."""
    _, result = structural.pin_dims(_dynamic_model(), {"batch": 1, "seq_len": 128})
    assert result.detail["unmatched"] == ["seq_len"]
    assert "no such dim" in result.note


# -- fold_const_inputs ------------------------------------------------------


def test_fold_const_inputs_removes_the_input_and_keeps_the_value() -> None:
    """This is how a recipe's role='constant' inputs are honoured. Folding a
    rank-0/1 scalar removes the input, so ST's rank-1-input rejection has
    nothing left to fire on."""
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "gain"], ["y"])],
        "g",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("gain", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    out, result = structural.fold_const_inputs(model, {"gain": 2.0})
    assert result.applied
    assert [vi.name for vi in out.graph.input] == ["x"]
    folded = next(i for i in out.graph.initializer if i.name == "gain")
    assert float(numpy_helper.to_array(folded)) == 2.0
    onnx.checker.check_model(out)


def test_fold_const_inputs_reports_names_that_are_not_inputs() -> None:
    _, result = structural.fold_const_inputs(_tiny_model(), {"nope": 1})
    assert result.detail["missing"] == ["nope"]
    assert not result.applied


# -- the parity gate itself -------------------------------------------------


def test_parity_detects_a_semantics_changing_edit() -> None:
    """The whole point. A patch that quietly changes the maths must be caught,
    because everything downstream of it would read as success."""
    model = _tiny_model()
    broken = onnx.ModelProto()
    broken.CopyFrom(model)
    for init in broken.graph.initializer:
        if init.name == "W":
            array = numpy_helper.to_array(init) * 2.0
            init.CopyFrom(numpy_helper.from_array(array, name="W"))
    check = parity.compare(model, broken)
    assert check.ok is False
    assert check.max_abs and check.max_abs > 0


def test_parity_says_unknown_rather_than_fine_when_it_cannot_run() -> None:
    """A graph with unpinned dimensions cannot be fed. Reporting that as a
    pass would make the gate worse than useless."""
    model = _dynamic_model()
    check = parity.compare(model, model)
    assert check.ok is None
    assert not check.evaluated
    assert "static" in check.note


def test_parity_flags_a_changed_output_signature() -> None:
    model = _tiny_model()
    trimmed = onnx.ModelProto()
    trimmed.CopyFrom(model)
    trimmed.graph.output[0].name = "renamed"
    trimmed.graph.node[-1].output[0] = "renamed"
    assert parity.compare(model, trimmed).ok is False
