"""Quantisation scheme knobs that a deployment contract depends on."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("onnxruntime")

from zoo.quant import qdq  # noqa: E402
from zoo.quant.calib import CalibrationSpec  # noqa: E402


def _conv_relu(path):
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((4, 3, 3, 3)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "B")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _activation_zero_points(path) -> list[int]:
    model = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    return [
        int(np.abs(inits[n.input[2]].astype(np.int64)).max(initial=0))
        for n in model.graph.node
        if n.op_type == "QuantizeLinear" and len(n.input) > 2
    ]


@pytest.mark.parametrize("symmetric", [False, True])
def test_activation_symmetry_reaches_the_quantiser(tmp_path, symmetric):
    """stm32n6-stt's M55 front end quantises with no offset term; a graph with a
    non-zero input zero-point would bias every feature it is fed."""
    src = _conv_relu(tmp_path / "m.onnx")
    out = tmp_path / f"m_sym{int(symmetric)}.onnx"
    result = qdq.quantize(
        src, out, calibration=CalibrationSpec(provider="synthetic", n=8, seed=0),
        activation_symmetric=symmetric,
    )
    assert result.path == out, result.error
    assert result.metrics()["activation_symmetric"] is symmetric
    zero_points = _activation_zero_points(out)
    assert zero_points
    # Relu's output is one-sided, so the asymmetric scheme gives it a non-zero
    # zero-point; the symmetric one may not give anything one.
    assert (max(zero_points) == 0) is symmetric
