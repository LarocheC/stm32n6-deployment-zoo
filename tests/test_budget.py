"""Memory accounting, and its refusal to answer when it cannot.

The most valuable behaviour tested here is the negative one: an analytic peak
computed over a graph whose shapes did not resolve must not be dressed up as a
placement. That guard is what caught a Whisper window that appeared to fit
on-chip and did not.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from zoo.graph import budget as bmod
from zoo.graph.patches import sequence, structural

POLICY = {
    "budget": {
        "onchip_bytes": 2_883_576,
        "hyperram_bytes": 33_554_432,
        "octoflash_bytes": 117_440_512,
    }
}


def _conv_model(*, channels: int = 8, size: int = 32, dynamic: bool = False):
    w = numpy_helper.from_array(
        np.ones((channels, 3, 3, 3), dtype=np.float32), name="W"
    )
    batch = "batch" if dynamic else 1
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "W"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, channels, size, size])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def test_weight_bytes_and_macs_are_read_off_the_graph() -> None:
    model = _conv_model(channels=8, size=32)
    b = bmod.analyse(model)
    assert b.weight_bytes == 8 * 3 * 3 * 3 * 4
    # 32x32 output positions x 8 channels x (3x3x3) per output
    assert b.macs == 32 * 32 * 8 * 27
    assert not b.is_quantised


def test_peak_activation_scales_with_the_activation_not_the_weights() -> None:
    """The cliff is activations. Doubling the spatial size quadruples the peak
    while the weights are unchanged — which is the whole reason resolution is
    the first knob to reach for."""
    small = bmod.analyse(_conv_model(size=32))
    large = bmod.analyse(_conv_model(size=64))
    assert large.weight_bytes == small.weight_bytes
    assert large.peak_activation_fused == pytest.approx(
        small.peak_activation_fused * 4, rel=0.05
    )


def test_unresolved_shapes_block_a_placement_claim() -> None:
    """The guard that matters.

    A peak computed over a graph with unresolvable shapes is an under-count,
    and naming a pool on that basis is how a model that does not fit gets
    reported as fitting.
    """
    b = bmod.analyse(_conv_model(dynamic=True))
    assert b.unresolved_tensors > 0
    assert b.placement(POLICY) == "unknown-unpinned-shapes"
    assert any("under-count" in n for n in b.notes)


def test_placement_is_named_once_shapes_resolve() -> None:
    b = bmod.analyse(_conv_model(size=32))
    assert b.unresolved_tensors == 0
    assert b.placement(POLICY) == "all-on-chip"


def test_quantised_graph_is_recognised_and_not_reported_as_weightless() -> None:
    """A naive walk looking for float weights reports a QDQ graph as having
    none. Both the raw and the quantised payload have to be counted."""
    q_w = numpy_helper.from_array(np.ones((4, 3), dtype=np.int8), name="Wq")
    scale = numpy_helper.from_array(np.float32(0.02), name="s")
    zp = numpy_helper.from_array(np.int8(0), name="z")
    graph = helper.make_graph(
        [
            helper.make_node("DequantizeLinear", ["Wq", "s", "z"], ["W"]),
            helper.make_node("MatMul", ["x", "W"], ["y"]),
        ],
        "qdq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[q_w, scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    b = bmod.analyse(model)
    assert b.is_quantised
    assert b.quantised_weight_bytes >= 12  # the int8 payload, not zero


# -- the sequence patch -----------------------------------------------------


def _positional_model(*, positions: int = 1500, seq: int = 1500, width: int = 8):
    emb = numpy_helper.from_array(
        np.arange(positions * width, dtype=np.float32).reshape(positions, width),
        name="embed_positions.weight",
    )
    graph = helper.make_graph(
        [helper.make_node("Add", ["h", "embed_positions.weight"], ["y"])],
        "pos",
        [helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, seq, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, seq, width])],
        initializer=[emb],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def test_positional_patch_does_not_fire_on_a_matched_window() -> None:
    model = _positional_model(positions=1500, seq=1500)
    assert not sequence._has_oversized_positional_constant(model)
    _, result = sequence.slice_positional_embedding(model)
    assert not result.applied


def test_positional_patch_truncates_to_the_actual_sequence() -> None:
    """Pinning a shorter window leaves the learned positional constant at its
    export-time size, and the graph stops loading. Slicing is exact: position i
    of a shorter window is still position i."""
    model = _positional_model(positions=1500, seq=250)
    assert sequence._has_oversized_positional_constant(model)

    out, result = sequence.slice_positional_embedding(model)
    assert result.applied
    emb = next(i for i in out.graph.initializer if i.name == "embed_positions.weight")
    assert list(emb.dims) == [250, 8]

    # Exact: the retained rows are unchanged, not resampled.
    original = numpy_helper.to_array(
        next(i for i in model.graph.initializer if i.name == "embed_positions.weight")
    )
    np.testing.assert_array_equal(numpy_helper.to_array(emb), original[:250])
    onnx.checker.check_model(out)


def test_positional_patch_makes_a_pinned_graph_loadable() -> None:
    """End to end: pin, patch, and the result actually runs — which the
    pinned-but-unpatched graph does not."""
    import onnxruntime as ort

    model = _positional_model(positions=1500, seq=1500)
    # Re-declare the input shorter, as pinning a symbolic dim would.
    pinned = onnx.ModelProto()
    pinned.CopyFrom(model)
    for vi in list(pinned.graph.input) + list(pinned.graph.output):
        vi.type.tensor_type.shape.dim[1].dim_value = 250

    options = ort.SessionOptions()
    options.log_severity_level = 4
    with pytest.raises(Exception):  # noqa: B017 - ORT raises its own Fail type
        ort.InferenceSession(pinned.SerializeToString(), sess_options=options,
                             providers=["CPUExecutionProvider"])

    fixed, result = sequence.slice_positional_embedding(pinned)
    assert result.applied
    session = ort.InferenceSession(
        fixed.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    out = session.run(None, {"h": np.zeros((1, 250, 8), dtype=np.float32)})[0]
    assert out.shape == (1, 250, 8)
