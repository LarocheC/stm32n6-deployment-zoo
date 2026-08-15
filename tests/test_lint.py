"""The static screen.

Built against a hand-made op table and synthetic probes so these run in
milliseconds and do not need the ST install — the lint rules are policy, and
policy deserves tests that always run.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from zoo.faults.taxonomy import FailureClass as FC
from zoo.graph import lint as lmod
from zoo.graph import ops as omod
from zoo.graph.probe import ConditionalOp, Probe, TensorSpec

POLICY = {
    "lint": {
        "max_opset": 20,
        "preferred_opset": 13,
        "max_ir_version": 8,
        "max_tensor_dim": 65535,
        "max_rank": 4,
        "forbidden_ops": ["If", "Loop", "Scan", "NonZero"],
        "warn_sw_ops": 12,
        "max_sw_ops": 40,
    }
}


@pytest.fixture
def table() -> omod.OpTable:
    mapping = {
        "Conv": omod.OpInfo("Conv", omod.HW, requires_constant_input=True),
        "MatMul": omod.OpInfo("MatMul", omod.HW, requires_constant_input=True),
        "Relu": omod.OpInfo("Relu", omod.HW),
        "Add": omod.OpInfo("Add", omod.HW),
        "Gather": omod.OpInfo("Gather", omod.SW_INT),
        "ReduceSum": omod.OpInfo("ReduceSum", omod.SW_FLOAT),
        "Softmax": omod.OpInfo(
            "Softmax", omod.HW, unlocked_by="--expand-softmax", fallback_tier=omod.SW_INT
        ),
    }
    return omod.OpTable(
        core_version="test",
        mapping=mapping,
        frontend=set(mapping) | {"LayerNormalization", "Where", "Einsum"},
    )


def _probe(**kw) -> Probe:
    defaults = dict(
        path=Path("test.onnx"),
        op_types=Counter({"Conv": 10, "Relu": 10}),
        subgraph_op_types=Counter(),
        ir_version=8,
        opsets={"": 13},
        inputs=[TensorSpec("x", "float32", [1, 3, 224, 224])],
        outputs=[TensorSpec("y", "float32", [1, 1000])],
        initializer_inputs=[],
        initializer_bytes=1024,
        max_tensor_dim=224,
        max_rank=4,
        conditional=[],
        has_external_data=False,
    )
    defaults.update(kw)
    return Probe(**defaults)


def _classes(result: lmod.LintResult) -> set[str]:
    return {v.failure_class for v in result.errors}


def test_a_clean_conv_graph_passes(table) -> None:
    result = lmod.lint(_probe(), table, POLICY)
    assert result.ok
    assert result.sw_instances == 0


def test_control_flow_is_blocking_and_reports_hidden_nodes(table) -> None:
    """A one-node graph wrapping 156 nodes must not read as a tiny model."""
    result = lmod.lint(
        _probe(
            op_types=Counter({"If": 1, "Identity": 2}),
            subgraph_op_types=Counter({"Conv": 12, "Relu": 10}),
        ),
        table,
        POLICY,
    )
    assert not result.ok
    assert FC.OP_CONTROL_FLOW in _classes(result)
    violation = next(v for v in result.errors if v.rule == "control_flow")
    assert violation.detail["subgraph_nodes"] == 22


def test_unsupported_op_is_blocking_and_names_the_reason(table) -> None:
    result = lmod.lint(_probe(op_types=Counter({"Einsum": 2, "Conv": 4})), table, POLICY)
    assert FC.OP_UNSUPPORTED in _classes(result)
    violation = next(v for v in result.errors if v.rule == "unsupported_ops")
    # A remedy, not just a rejection.
    assert "Slice/MatMul/Concat" in violation.remedy


def test_oversized_interior_dimension_is_blocking(table) -> None:
    """The 65536 limit applies to interior tensors, not only graph I/O — the
    reason a flat layout had to become rank-3 in prior work."""
    result = lmod.lint(_probe(max_tensor_dim=144160), table, POLICY)
    assert FC.SHAPE_DIM_TOO_LARGE in _classes(result)


def test_rank_five_is_blocking(table) -> None:
    result = lmod.lint(_probe(max_rank=5), table, POLICY)
    violation = next(v for v in result.errors if v.rule == "rank")
    assert violation.failure_class == FC.RANK_TOO_HIGH
    assert "segfault" in violation.remedy.lower()


def test_unpinned_named_dim_blocks_but_a_pin_clears_it(table) -> None:
    probe = _probe(inputs=[TensorSpec("x", "float32", ["batch", 3, 224, 224])])
    assert FC.SHAPE_DYNAMIC_UNPINNABLE in _classes(lmod.lint(probe, table, POLICY))
    assert lmod.lint(probe, table, POLICY, pinned={"batch": 1}).ok


def test_anonymous_axis_gets_a_different_remedy_than_a_named_one(table) -> None:
    """`--fix-parametric-shapes` keys on names. Offering it for an unnamed
    axis would send someone after a fix that cannot work."""
    probe = _probe(inputs=[TensorSpec("x", "float32", [None, 512])])
    result = lmod.lint(probe, table, POLICY)
    violation = next(v for v in result.errors if v.rule == "anonymous_dims")
    assert "no command-line remedy" in violation.remedy
    assert violation.patch == "pin_anonymous_axes"


def test_ir_version_violation_suggests_the_patch(table) -> None:
    result = lmod.lint(_probe(ir_version=10), table, POLICY)
    assert FC.IR_TOO_HIGH in _classes(result)
    assert "downgrade_ir" in result.suggested_patches


def test_gated_softmax_is_reported_with_its_flag(table) -> None:
    """Reporting Softmax as free would understate every attention model;
    reporting it as a hard software epoch would overstate the problem. The
    honest answer names the flag."""
    probe = _probe(op_types=Counter({"Softmax": 4, "Conv": 4}))
    default = lmod.lint(probe, table, POLICY)
    assert default.sw_instances == 4
    assert default.sw_instances_unlocked == 0
    note = next(v for v in default.violations if v.rule == "gated_ops")
    assert note.detail["ops"] == {"Softmax": "--expand-softmax"}


def test_dynamic_matmul_is_surfaced_as_the_attention_signature(table) -> None:
    conditional = [
        ConditionalOp("m0", "MatMul", "w", directly_constant=True, foldable=True),
        ConditionalOp("m1", "MatMul", "scores", directly_constant=False, foldable=False),
        ConditionalOp("m2", "MatMul", "probs", directly_constant=False, foldable=False),
    ]
    result = lmod.lint(_probe(conditional=conditional), table, POLICY)
    assert result.dynamic_conditional == 2
    violation = next(v for v in result.violations if v.rule == "dynamic_matmul")
    assert violation.detail["by_type"] == {"MatMul": 2}
    assert "self-attention" in violation.remedy


def test_foldable_operand_recommends_the_simplifier_not_rejection(table) -> None:
    """A weight reshaped before a MatMul is still a weight. Treating it as a
    dynamic operand would make an ordinary model look attention-bound."""
    conditional = [ConditionalOp("m0", "MatMul", "w_reshaped", directly_constant=False, foldable=True)]
    result = lmod.lint(_probe(conditional=conditional), table, POLICY)
    assert result.ok
    assert result.dynamic_conditional == 0
    assert "constant_fold" in result.suggested_patches


def test_software_budget_escalates_from_warning_to_rejection(table) -> None:
    warn = lmod.lint(_probe(op_types=Counter({"Gather": 20, "Conv": 4})), table, POLICY)
    assert warn.ok
    assert any(v.rule == "sw_budget" and v.severity == lmod.WARN for v in warn.violations)

    fail = lmod.lint(_probe(op_types=Counter({"Gather": 50, "Conv": 4})), table, POLICY)
    assert not fail.ok
    assert FC.OP_SW_POLICY_REJECT in _classes(fail)


def test_initializers_declared_as_inputs_are_flagged_with_a_patch(table) -> None:
    result = lmod.lint(_probe(initializer_inputs=[f"w{i}" for i in range(140)]), table, POLICY)
    violation = next(v for v in result.violations if v.rule == "initializers_as_inputs")
    assert violation.detail["count"] == 140
    assert violation.patch == "strip_initializer_inputs"


def test_metrics_are_event_ready(table) -> None:
    result = lmod.lint(_probe(op_types=Counter({"Conv": 8, "Gather": 3})), table, POLICY)
    metrics = result.metrics()
    assert metrics["ops_npu"] == 8
    assert metrics["ops_sw"] == 3
    assert metrics["nodes"] == 11
