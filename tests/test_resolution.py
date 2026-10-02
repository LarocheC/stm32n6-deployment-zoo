"""Re-resolving a statically shaped graph, and knowing when not to."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from zoo.graph.patches import structural


def _conv_graph(*, reshape_target=None, resize_sizes=False, size=64):
    """A tiny fully-convolutional net ending in a Reshape, like a detector head."""
    weight = numpy_helper.from_array(
        np.ones((4, 3, 3, 3), dtype=np.float32), name="w"
    )
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["conv"], pads=[1, 1, 1, 1], name="Conv_0"),
    ]
    initializers = [weight]

    if resize_sizes:
        sizes = numpy_helper.from_array(
            np.array([1, 4, size, size], dtype=np.int64), name="sizes"
        )
        initializers.append(sizes)
        nodes.append(
            helper.make_node(
                "Resize", ["conv", "", "", "sizes"], ["resized"], name="Resize_1",
                mode="nearest",
            )
        )
        source = "resized"
    else:
        scales = numpy_helper.from_array(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), name="scales"
        )
        initializers.append(scales)
        nodes.append(
            helper.make_node(
                "Resize", ["conv", "", "scales"], ["resized"], name="Resize_1",
                mode="nearest",
            )
        )
        source = "resized"

    target = reshape_target if reshape_target is not None else [1, -1, 4]
    initializers.append(
        numpy_helper.from_array(np.array(target, dtype=np.int64), name="shape")
    )
    nodes.append(helper.make_node("Reshape", [source, "shape"], ["output"], name="Reshape_2"))

    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, size * size, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _output_shape(model, name="output"):
    for vi in model.graph.output:
        if vi.name == name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None


def test_halving_the_input_re_derives_every_downstream_shape():
    model = _conv_graph(size=64)
    out, result = structural.retarget_input_resolution(model, {"input": [1, 3, 32, 32]})

    assert result.applied
    assert [d.dim_value for d in out.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 32, 32]
    # 32x32 -> 1024 rows, not the 4096 the file was exported with.
    assert _output_shape(out) == [1, 1024, 4]
    onnx.checker.check_model(out)


def test_stale_interior_shapes_do_not_survive_the_retarget():
    """A file carrying two resolutions at once infers cleanly and is wrong."""
    model = _conv_graph(size=64)
    model.graph.value_info.append(
        helper.make_tensor_value_info("conv", TensorProto.FLOAT, [1, 4, 64, 64])
    )
    out, result = structural.retarget_input_resolution(model, {"input": [1, 3, 32, 32]})
    assert result.applied
    conv = {vi.name: vi for vi in out.graph.value_info}.get("conv")
    assert [d.dim_value for d in conv.type.tensor_type.shape.dim] == [1, 4, 32, 32]


def test_a_literal_reshape_target_blocks_the_retarget():
    """Reshape [1, 4096, 4] is tied to 64x64; -1 is what makes it portable."""
    model = _conv_graph(size=64, reshape_target=[1, 4096, 4])
    out, result = structural.retarget_input_resolution(model, {"input": [1, 3, 32, 32]})
    assert not result.applied
    assert "literal" in result.note
    # Refused means unchanged, not half-applied.
    assert [d.dim_value for d in out.graph.input[0].type.tensor_type.shape.dim][2] == 64


def test_a_sizes_driven_resize_blocks_the_retarget():
    model = _conv_graph(size=64, resize_sizes=True)
    _, result = structural.retarget_input_resolution(model, {"input": [1, 3, 32, 32]})
    assert not result.applied
    assert "sizes" in result.note


def test_an_unknown_input_name_is_refused_with_the_available_names():
    model = _conv_graph()
    _, result = structural.retarget_input_resolution(model, {"pixels": [1, 3, 32, 32]})
    assert not result.applied
    assert "input" in result.note


def test_a_rank_mismatch_is_refused():
    model = _conv_graph()
    _, result = structural.retarget_input_resolution(model, {"input": [3, 32, 32]})
    assert not result.applied
    assert "rank" in result.note


def test_no_targets_is_a_no_op():
    model = _conv_graph()
    out, result = structural.retarget_input_resolution(model, {})
    assert not result.applied
    assert out is model
