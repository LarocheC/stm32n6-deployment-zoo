"""The static screen: reject what cannot work, before any ST tool runs.

This is the cheapest stage in the funnel and it is expected to kill most
candidates. Under a second, no compiler, no board.

Two design commitments shape it.

**Every violation carries a remedy.** "Unsupported operator" is not a useful
result; "Einsum has no Neural-ART mapping — rewrite as Slice/MatMul/Concat,
which does compile" is. Where a mechanical rewrite exists, the violation names
the patch, so the funnel can apply it and retry rather than stopping.

**The rules are known to under-approximate, and the harness says so.** The
prior project's equivalent lint was explicit that it "had never been
cross-checked against an actual compile", and in practice every graph that
passed it still needed patching. So each rule carries a confidence, failures
the lint did not predict are recorded as `LINT_GAP`, and the leaderboard
reports the pass's own precision and recall. A screen that is trusted rather
than measured is how a zoo starts publishing confident nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zoo.faults.taxonomy import FailureClass as FC
from zoo.graph import ops as opsmod
from zoo.graph.probe import Probe

ERROR = "error"
WARN = "warn"
INFO = "info"


@dataclass
class Violation:
    rule: str
    severity: str
    failure_class: str
    message: str
    remedy: str = ""
    #: Name of a graph patch that fixes this mechanically, if one exists.
    patch: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    #: How sure we are the compiler agrees. Rules derived from ST's own
    #: documentation or from a reproduced failure are "high"; rules inferred
    #: from silence are "medium" and must not be the only reason to reject.
    confidence: str = "high"

    @property
    def blocking(self) -> bool:
        return self.severity == ERROR


@dataclass
class LintResult:
    violations: list[Violation] = field(default_factory=list)
    census: dict[str, dict[str, int]] = field(default_factory=dict)
    sw_instances: int = 0
    sw_instances_unlocked: int = 0
    dynamic_conditional: int = 0
    foldable_conditional: int = 0
    node_count: int = 0

    @property
    def ok(self) -> bool:
        return not any(v.blocking for v in self.violations)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == WARN]

    @property
    def failure_classes(self) -> list[str]:
        seen: list[str] = []
        for v in self.errors:
            if v.failure_class not in seen:
                seen.append(v.failure_class)
        return seen

    @property
    def suggested_patches(self) -> list[str]:
        seen: list[str] = []
        for v in self.violations:
            if v.patch and v.patch not in seen:
                seen.append(v.patch)
        return seen

    def metrics(self) -> dict[str, Any]:
        """The subset that belongs in an event record."""
        return {
            "nodes": self.node_count,
            "ops_npu": sum(self.census.get(opsmod.HW, {}).values()),
            "ops_sw": self.sw_instances,
            "ops_sw_unlocked": self.sw_instances_unlocked,
            "ops_frontend_only": sum(self.census.get(opsmod.FRONTEND_ONLY, {}).values()),
            "ops_unsupported": sum(self.census.get(opsmod.UNSUPPORTED, {}).values()),
            "dynamic_conditional": self.dynamic_conditional,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


def lint(
    probe: Probe,
    table: opsmod.OpTable,
    policy: dict,
    *,
    pinned: dict[str, int] | None = None,
    unlocked: bool = False,
) -> LintResult:
    """Screen one graph. `pinned` is the recipe's symbolic-dim assignment."""
    cfg = policy.get("lint", {})
    pinned = pinned or {}
    result = LintResult(node_count=sum(probe.op_types.values()))
    add = result.violations.append

    op_types = dict(probe.op_types)
    result.census = table.census(op_types, unlocked=unlocked)
    result.sw_instances = table.software_op_count(op_types)
    result.sw_instances_unlocked = table.software_op_count(op_types, unlocked=True)
    result.dynamic_conditional = len(probe.dynamic_conditional_ops)
    result.foldable_conditional = sum(1 for c in probe.conditional if c.needs_folding)

    # -- versions ---------------------------------------------------------

    opset = probe.default_opset
    max_opset = int(cfg.get("max_opset", 20))
    preferred = int(cfg.get("preferred_opset", 13))
    if opset is not None:
        if opset > max_opset:
            add(Violation(
                rule="opset", severity=ERROR, failure_class=FC.OPSET_TOO_HIGH,
                message=f"opset {opset} exceeds the ST front end's cap of {max_opset}",
                remedy=f"re-export at opset {preferred} with torch.onnx.export(dynamo=False); "
                       "the dynamo exporter cannot go below 18",
                patch="downgrade_opset", detail={"opset": opset},
            ))
        elif opset > preferred:
            add(Violation(
                rule="opset", severity=WARN, failure_class=FC.OPSET_TOO_HIGH,
                message=f"opset {opset} is above the {preferred} both prior projects pinned to",
                remedy=f"opset {preferred} is the best-tested path; higher usually works but "
                       "is where novel front-end failures show up",
                confidence="medium", detail={"opset": opset},
            ))

    max_ir = int(cfg.get("max_ir_version", 8))
    if probe.ir_version > max_ir:
        add(Violation(
            rule="ir_version", severity=ERROR, failure_class=FC.IR_TOO_HIGH,
            message=f"IR version {probe.ir_version} exceeds {max_ir}",
            remedy="downgrade the IR version in place; the graph itself is usually fine",
            patch="downgrade_ir", detail={"ir_version": probe.ir_version},
        ))

    # -- operators --------------------------------------------------------

    forbidden = set(cfg.get("forbidden_ops", []))
    control = {op: n for op, n in op_types.items() if op in forbidden or op in ("If", "Loop", "Scan")}
    if control:
        hidden = sum(probe.subgraph_op_types.values())
        add(Violation(
            rule="control_flow", severity=ERROR, failure_class=FC.OP_CONTROL_FLOW,
            message=f"control flow at the top level: {control}"
                    + (f" hiding {hidden} nodes in subgraphs" if hidden else ""),
            remedy="a statically scheduled epoch graph cannot branch. Look for a "
                   "single-branch export of the same model — e.g. istupakov/silero-vad-onnx "
                   "publishes the 16 kHz-only graph that onnx-community/silero-vad wraps in an If",
            detail={"ops": control, "subgraph_nodes": hidden},
        ))

    unsupported = result.census.get(opsmod.UNSUPPORTED, {})
    unsupported = {k: v for k, v in unsupported.items() if k not in control}
    if unsupported:
        reasons = {
            op: opsmod.HARD_BLOCKED.get(op, "no Neural-ART mapping and no documented fallback")
            for op in unsupported
        }
        add(Violation(
            rule="unsupported_ops", severity=ERROR, failure_class=FC.OP_UNSUPPORTED,
            message=f"operators with no accelerator mapping: {unsupported}",
            remedy="; ".join(f"{op}: {why}" for op, why in reasons.items()),
            detail={"ops": unsupported},
        ))

    frontend_only = result.census.get(opsmod.FRONTEND_ONLY, {})
    if frontend_only:
        add(Violation(
            rule="frontend_only_ops", severity=WARN, failure_class=FC.OP_UNSUPPORTED,
            message=f"the importer accepts these but ST's mapping table never mentions them: "
                    f"{frontend_only}",
            remedy="behaviour on the NPU is undocumented. Expect a software epoch at best; "
                   "LayerNormalization in particular expands to roughly 33 ReduceMean per "
                   "instance. Verify against the compile report rather than assuming",
            confidence="medium", detail={"ops": frontend_only},
        ))

    # -- software-epoch budget --------------------------------------------

    warn_sw = int(cfg.get("warn_sw_ops", 12))
    max_sw = int(cfg.get("max_sw_ops", 40))
    total_sw = result.sw_instances + result.dynamic_conditional
    if total_sw > max_sw:
        add(Violation(
            rule="sw_budget", severity=ERROR, failure_class=FC.OP_SW_POLICY_REJECT,
            message=f"{total_sw} node instances would run on the Cortex-M55 "
                    f"(limit {max_sw})",
            remedy="latency here is epoch-bound, not MAC-bound: each software epoch costs "
                   "an NPU teardown, a memory round-trip and cache maintenance. Raise "
                   "policy.lint.max_sw_ops deliberately if this model is worth measuring anyway",
            confidence="medium",
            detail={"sw_ops": result.sw_instances, "dynamic_matmul": result.dynamic_conditional},
        ))
    elif total_sw > warn_sw:
        add(Violation(
            rule="sw_budget", severity=WARN, failure_class=FC.OP_SW_POLICY_REJECT,
            message=f"{total_sw} node instances land on the Cortex-M55",
            remedy="expect the epoch count, not the MAC count, to set the latency floor",
            confidence="medium",
            detail={"sw_ops": result.sw_instances, "dynamic_matmul": result.dynamic_conditional},
        ))

    gated = table.gated_ops(op_types)
    if gated and not unlocked:
        add(Violation(
            rule="gated_ops", severity=INFO, failure_class="",
            message=f"hardware mapping available but not enabled: {gated}",
            remedy="compile with the `transformer` profile, which passes these flags. "
                   "None of ST's shipped profiles do",
            detail={"ops": gated},
        ))

    if probe.dynamic_conditional_ops:
        by_type: dict[str, int] = {}
        for cond in probe.dynamic_conditional_ops:
            by_type[cond.op_type] = by_type.get(cond.op_type, 0) + 1
        add(Violation(
            rule="dynamic_matmul", severity=WARN, failure_class="",
            message=f"{len(probe.dynamic_conditional_ops)} of "
                    f"{len(probe.conditional)} MatMul/Gemm/Conv have a non-constant "
                    f"second operand: {by_type}",
            remedy="these fall back to the Cortex-M55. Both operands being activations is "
                   "the signature of self-attention; projections against weights are fine",
            detail={"by_type": by_type},
        ))

    if result.foldable_conditional:
        add(Violation(
            rule="foldable_operand", severity=INFO, failure_class="",
            message=f"{result.foldable_conditional} operand(s) are constant only after folding",
            remedy="run the simplifier (--use-onnx-simplifier / onnxsim) so the compiler "
                   "sees them as constants and maps the op to hardware",
            patch="constant_fold",
        ))

    # -- shapes -----------------------------------------------------------

    max_dim = int(cfg.get("max_tensor_dim", 65535))
    if probe.max_tensor_dim > max_dim:
        add(Violation(
            rule="tensor_dim", severity=ERROR, failure_class=FC.SHAPE_DIM_TOO_LARGE,
            message=f"a tensor dimension is {probe.max_tensor_dim} (limit {max_dim})",
            remedy="the limit applies to interior tensors too, not only graph I/O. "
                   "Reshape into a rank-3 layout — a flat [1, 144160] became [1, 901, 160] "
                   "in prior work for exactly this reason",
            detail={"max_dim": probe.max_tensor_dim},
        ))

    max_rank = int(cfg.get("max_rank", 4))
    if probe.max_rank > max_rank:
        add(Violation(
            rule="rank", severity=ERROR, failure_class=FC.RANK_TOO_HIGH,
            message=f"maximum tensor rank is {probe.max_rank} (limit {max_rank})",
            remedy="rank 5 segfaults the ST front end (signo=11) rather than reporting an "
                   "error. Restructure the offending tensors before compiling",
            detail={"max_rank": probe.max_rank},
        ))

    rank1_inputs = [s.name for s in probe.inputs if s.rank <= 1 and s.name not in probe.initializer_inputs]
    if rank1_inputs:
        add(Violation(
            rule="rank1_input", severity=WARN, failure_class=FC.RANK1_GRAPH_INPUT,
            message=f"rank-0/1 graph inputs: {rank1_inputs}",
            remedy="ST rejects rank-1 graph inputs. Where these are configuration scalars "
                   "(a sample rate, a beam count), folding them into initializers removes "
                   "the input entirely — mark them role='constant' in the recipe",
            patch="fold_const_inputs", detail={"inputs": rank1_inputs},
        ))

    if probe.initializer_inputs:
        add(Violation(
            rule="initializers_as_inputs", severity=WARN,
            failure_class=FC.INITIALIZERS_AS_INPUTS,
            message=f"{len(probe.initializer_inputs)} initializers are also declared as "
                    "graph inputs (old IR-v3 style export)",
            remedy="left alone, stedgeai sees a model with that many inputs. The simplifier "
                   "removes them. Seen in opencv/object_detection_nanodet (~140) and "
                   "opencv/facial_expression_recognition (36)",
            patch="strip_initializer_inputs",
            detail={"count": len(probe.initializer_inputs)},
        ))

    unresolved: dict[str, list[str]] = {}
    anonymous: list[str] = []
    for spec in probe.inputs:
        for axis, name in spec.dynamic_axes:
            if name is None:
                anonymous.append(f"{spec.name}[{axis}]")
            elif name not in pinned:
                unresolved.setdefault(name, []).append(spec.name)

    if unresolved:
        add(Violation(
            rule="unpinned_dims", severity=ERROR,
            failure_class=FC.SHAPE_DYNAMIC_UNPINNABLE,
            message=f"symbolic dimensions with no value: {sorted(unresolved)}",
            remedy="set them under [graph.pin] in the recipe; they become "
                   "--fix-parametric-shapes. The NPU requires static input and output shapes",
            detail={"dims": {k: v for k, v in sorted(unresolved.items())}},
        ))

    if anonymous:
        add(Violation(
            rule="anonymous_dims", severity=ERROR,
            failure_class=FC.SHAPE_DYNAMIC_UNPINNABLE,
            message=f"dynamic axes with no name: {anonymous}",
            remedy="--fix-parametric-shapes addresses dimensions by name, so these have no "
                   "command-line remedy. Re-export with named or fixed dims, or rewrite the "
                   "graph's value_info",
            patch="pin_anonymous_axes", detail={"axes": anonymous},
        ))

    if probe.has_external_data:
        add(Violation(
            rule="external_data", severity=INFO, failure_class="",
            message="weights live in an external .onnx_data file",
            remedy="fetch it alongside the graph; the compiler needs both",
        ))

    return result
