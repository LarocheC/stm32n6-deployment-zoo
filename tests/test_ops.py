"""The op oracle is the zoo's most load-bearing static component.

These tests pin the facts that decide verdicts. If ST restructures the
documentation or changes a mapping in a future core release, these fail loudly
rather than letting the funnel quietly start screening against a different
reality.
"""

from __future__ import annotations

import pytest

from zoo.config import ConfigError, load_toolchain
from zoo.graph import ops


@pytest.fixture(scope="module")
def table() -> ops.OpTable:
    try:
        tc = load_toolchain()
    except ConfigError as exc:
        pytest.skip(f"toolchain not configured: {exc}")
    if not tc.operator_support_html.is_file():
        pytest.skip("ST operator-support documentation not installed")
    return ops.load(tc)


def test_mapping_table_is_substantial(table: ops.OpTable) -> None:
    # ST's ONNX mapping table carries ~101 rows for core 4.0.1. A parse that
    # silently returns a handful means the document structure moved.
    assert len(table.mapping) >= 90


def test_frontend_vocabulary_is_larger_than_the_mapping(table: ops.OpTable) -> None:
    """The whole reason both sources are needed.

    `supported-ops` is the importer's vocabulary, not the accelerator's. It is
    strictly larger, and screening on it alone would pass models that cannot
    run. If these ever converge, the FRONTEND_ONLY tier has become meaningless
    and the oracle needs revisiting.
    """
    assert len(table.frontend) > len(table.mapping)
    frontend_only = {o for o in table.frontend if table.tier(o) == ops.FRONTEND_ONLY}
    assert {"LayerNormalization", "Where", "Expand", "ScatterND", "TopK"} <= frontend_only


@pytest.mark.parametrize(
    "op",
    ["Conv", "ConvTranspose", "Gemm", "MatMul", "Relu", "Clip", "Sigmoid", "Tanh",
     "MaxPool", "AveragePool", "GlobalAveragePool", "Add", "Mul", "Concat",
     "Reshape", "Transpose", "PRelu", "HardSwish", "BatchNormalization"],
)
def test_core_conv_vocabulary_is_hardware(table: ops.OpTable, op: str) -> None:
    """The ops a conv-style encoder is made of must all reach hardware.

    `PRelu` is in this list deliberately: both prior projects on this machine
    treated it as unsupported, and ST's table says otherwise (the slope
    attribute must be quantised). That folklore cost real re-export work.
    """
    assert table.tier(op) == ops.HW


@pytest.mark.parametrize(
    "op", ["ReduceSum", "Gather", "Resize", "Tile", "Log", "InstanceNormalization"]
)
def test_known_software_epoch_ops(table: ops.OpTable, op: str) -> None:
    assert table.tier(op) in ops.SOFTWARE_TIERS


@pytest.mark.parametrize("op", ["Einsum", "If", "Loop", "NonZero", "GridSample"])
def test_hard_blocked_ops(table: ops.OpTable, op: str) -> None:
    assert table.tier(op) == ops.UNSUPPORTED


def test_softmax_is_gated_not_free(table: ops.OpTable) -> None:
    """ST's table says HW; its comment says "SW_INT otherwise".

    None of ST's shipped profiles pass `--expand-softmax`, so the honest
    default for Softmax is a software epoch. Taking the headline column at
    face value would make every attention block look cheaper than it is.
    """
    info = table.info("Softmax")
    assert info is not None
    assert info.unlocked_by == "--expand-softmax"
    assert table.tier("Softmax") == ops.SW_INT
    assert table.tier("Softmax", unlocked=True) == ops.HW


def test_matmul_hardware_is_conditional_on_a_constant_operand(table: ops.OpTable) -> None:
    """The one line that explains why transformers are slow on this part.

    A convolution's weights are constant, so `MatMul`/`Gemm` reach hardware.
    Self-attention multiplies two activations, so it does not. An op histogram
    cannot tell these apart, which is why the flag exists for lint to resolve.
    """
    for op in ("MatMul", "Gemm", "Conv"):
        info = table.info(op)
        assert info is not None, op
        assert info.requires_constant_input, op

    whisper_encoder = {"MatMul": 32, "Softmax": 4, "Conv": 2}
    assert table.conditional_ops(whisper_encoder) == {"MatMul": 32, "Conv": 2}


def test_gating_changes_the_software_count(table: ops.OpTable) -> None:
    hist = {"Softmax": 4, "Conv": 10, "Relu": 10}
    assert table.software_op_count(hist) == 4
    assert table.software_op_count(hist, unlocked=True) == 0


def test_census_partitions_every_op(table: ops.OpTable) -> None:
    hist = {"Conv": 3, "Softmax": 1, "Einsum": 1, "LayerNormalization": 2, "Gather": 4}
    census = table.census(hist)
    assert sum(sum(g.values()) for g in census.values()) == sum(hist.values())
    assert census[ops.UNSUPPORTED] == {"Einsum": 1}
    assert census[ops.FRONTEND_ONLY] == {"LayerNormalization": 2}


def test_table_round_trips_through_json(table: ops.OpTable) -> None:
    restored = ops.OpTable.from_json(table.to_json())
    assert restored.mapping == table.mapping
    assert restored.frontend == table.frontend
