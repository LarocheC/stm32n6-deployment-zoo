"""Stage orchestration: run a graph down the funnel, recording every step.

Stage order is by cost, cheapest first, so most candidates die before anything
expensive runs. But two stages sit earlier than a naive cost ordering would put
them, and both placements are deliberate.

`shape` and `patch` run *before* `lint` renders its verdict. Linting the graph
as downloaded would report problems the harness already knows how to fix — an
IR version to lower, a positional embedding to slice, initializers to strip —
and a screen that rejects models it could have repaired is not a screen, it is
a bottleneck. The pre-repair violations are still recorded, on the `patch`
event, because "this class of export needs X" is the finding.

`budget` runs last of the board-free stages because it is worthless before
shapes are pinned. An analytic peak over a graph with unresolved dimensions is
an under-count that reads exactly like a comfortable fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

from zoo import fetch as fetchmod
from zoo import ids
from zoo.config import RUNS_DIR, Toolchain
from zoo.faults import signatures
from zoo.faults.taxonomy import FailureClass as FC
from zoo.graph import budget as budgetmod
from zoo.graph import lint as lintmod
from zoo.graph import ops as opsmod
from zoo.graph import parity, patches, probe
from zoo.graph.patches import structural
from zoo.recipe import GraphSpec, Recipe
from zoo.store.events import EventLog
from zoo.store.schema import Event, Stage, Status


@dataclass
class Context:
    toolchain: Toolchain
    table: opsmod.OpTable
    policy: dict
    log: EventLog
    run_id: str = field(default_factory=ids.new_run_id)
    zoo_commit: str = ""

    def toolchain_block(self) -> dict[str, Any]:
        return {
            "stedgeai": self.toolchain.stedgeai_expect_version,
            "core": self.toolchain.core_tag,
            # Whether the BOOL-strip shim is installed changes results, so it
            # is recorded rather than inferred afterwards.
            "atonn_shim": self.toolchain.atonn_real.exists(),
        }


@dataclass
class StageOutcome:
    stage: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_class: str = ""
    error: str = ""
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    #: Carried between stages, never written to the log.
    payload: Any = None


def workdir_for(recipe: Recipe, graph: GraphSpec) -> Path:
    return RUNS_DIR / recipe.id / graph.id


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_fetch(recipe: Recipe, graph: GraphSpec) -> StageOutcome:
    started = time.monotonic()
    try:
        remotes = [
            f
            for f in fetchmod.list_hf_onnx(recipe.source, recipe.revision)
            if f.filename == graph.file
        ]
        if not remotes:
            return StageOutcome(
                Stage.FETCH, Status.FAIL,
                failure_class=FC.FETCH_ERROR,
                error=f"{graph.file} is not published by {recipe.source}",
                duration_s=time.monotonic() - started,
            )
        remote = remotes[0]
        if remote.is_prequantised:
            # Vendors' int8 files are dynamic or QOperator quantisations, and
            # ST silently converts those back to float. Using one would produce
            # a model that looks quantised and deploys as fp32.
            return StageOutcome(
                Stage.FETCH, Status.FAIL,
                failure_class=FC.FETCH_ERROR,
                error=f"{graph.file} is a vendor pre-quantisation; fetch the fp32 file "
                      "and let the zoo run its own static QDQ pass",
                duration_s=time.monotonic() - started,
            )
        path = fetchmod.download(remote, recipe.revision)
        return StageOutcome(
            Stage.FETCH, Status.PASS,
            metrics={"bytes": path.stat().st_size, "hf_revision": recipe.revision},
            artifacts=[str(path)],
            duration_s=time.monotonic() - started,
            payload=path,
        )
    except Exception as exc:  # noqa: BLE001
        return StageOutcome(
            Stage.FETCH, Status.FAIL,
            failure_class=FC.FETCH_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - started,
        )


def stage_probe(path: Path) -> StageOutcome:
    started = time.monotonic()
    try:
        result = probe.probe_file(path)
    except Exception as exc:  # noqa: BLE001
        return StageOutcome(
            Stage.PROBE, Status.FAIL,
            failure_class=FC.FETCH_ERROR,
            error=f"could not parse: {type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - started,
        )
    return StageOutcome(
        Stage.PROBE, Status.PASS,
        metrics={
            "nodes": sum(result.op_types.values()),
            "subgraph_nodes": sum(result.subgraph_op_types.values()),
            "opset": result.default_opset,
            "ir_version": result.ir_version,
            "max_tensor_dim": result.max_tensor_dim,
            "max_rank": result.max_rank,
            "weight_bytes_raw": result.initializer_bytes,
        },
        duration_s=time.monotonic() - started,
        payload=result,
    )


def stage_shape(model: Any, graph: GraphSpec) -> StageOutcome:
    """Pin symbolic dimensions and fold recipe-declared constant inputs."""
    started = time.monotonic()
    notes: list[str] = []
    changed = 0

    out = model
    if graph.pin:
        out, result = structural.pin_dims(out, graph.pin)
        notes.append(result.note)
        changed += result.changed

    constants = {
        spec.name: spec.value for spec in graph.inputs if spec.role == "constant"
    }
    if constants:
        out, result = structural.fold_const_inputs(out, constants)
        notes.append(result.note)
        changed += result.changed

    return StageOutcome(
        Stage.SHAPE, Status.PASS,
        metrics={
            "pinned": graph.pin,
            "folded_constants": sorted(constants),
            "changed": changed,
        },
        error="; ".join(n for n in notes if n),
        duration_s=time.monotonic() - started,
        payload=out,
    )


def stage_patch(model: Any, wanted: list[str], *, tolerance: float = 1e-5) -> StageOutcome:
    """Apply patches, each gated on a numerical parity check.

    A patch whose parity check fails is reverted, not reported as applied. A
    patch whose parity could not be evaluated is applied but flagged, because
    refusing every rewrite on an unrunnable graph would block exactly the
    models that need repairing most.
    """
    started = time.monotonic()
    applied: list[str] = []
    rejected: list[dict[str, Any]] = []
    unverified: list[str] = []
    current = model

    for name in wanted:
        try:
            patch = patches.get(name)
        except KeyError:
            continue
        if not patch.applies(current):
            continue

        candidate, result = patch.run(current)
        if not result.applied:
            continue

        if patch.changes_signature:
            # Cannot be parity-checked by feeding identical inputs; the
            # rewrite's whole purpose is to change the interface.
            current = candidate
            applied.append(name)
            unverified.append(name)
            continue

        check = parity.compare(current, candidate, tol=tolerance)
        if check.ok is False:
            rejected.append({"patch": name, "max_abs": check.max_abs, "note": check.note})
            continue
        if check.ok is None:
            unverified.append(name)
        current = candidate
        applied.append(name)

    status = Status.PASS if not rejected else Status.FAIL
    return StageOutcome(
        Stage.PATCH, Status.PASS if status == Status.PASS else Status.FAIL,
        metrics={
            "applied": applied,
            "rejected": rejected,
            "parity_unverified": unverified,
            "requested": wanted,
        },
        failure_class="" if not rejected else FC.UNKNOWN,
        error="" if not rejected
        else f"patch(es) changed the graph's numerics and were reverted: {rejected}",
        duration_s=time.monotonic() - started,
        payload=current,
    )


def stage_lint(
    result: probe.Probe, ctx: Context, graph: GraphSpec, *, unlocked: bool = False
) -> StageOutcome:
    started = time.monotonic()
    report = lintmod.lint(result, ctx.table, ctx.policy, pinned=graph.pin, unlocked=unlocked)
    metrics = report.metrics()
    metrics["suggested_patches"] = report.suggested_patches
    metrics["violations"] = [
        {"rule": v.rule, "severity": v.severity, "message": v.message}
        for v in report.violations
        if v.severity != lintmod.INFO
    ]
    return StageOutcome(
        Stage.LINT,
        Status.PASS if report.ok else Status.FAIL,
        metrics=metrics,
        failure_class=(report.failure_classes[0] if report.errors else ""),
        error="; ".join(f"[{v.rule}] {v.message}" for v in report.errors),
        duration_s=time.monotonic() - started,
        payload=report,
    )


def stage_budget(model: Any, ctx: Context) -> StageOutcome:
    started = time.monotonic()
    try:
        report = budgetmod.analyse(model)
    except Exception as exc:  # noqa: BLE001
        return StageOutcome(
            Stage.BUDGET, Status.FAIL,
            failure_class=FC.UNKNOWN,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - started,
        )

    placement = report.placement(ctx.policy)
    metrics = report.metrics()
    metrics["placement"] = placement

    if placement == "does-not-fit":
        return StageOutcome(
            Stage.BUDGET, Status.FAIL,
            metrics=metrics,
            failure_class=FC.ALLOC_FAILED_ALL,
            error=f"peak activation {report.peak_activation_fused / 1e6:.2f} MB exceeds "
                  "every available pool",
            duration_s=time.monotonic() - started,
            payload=report,
        )
    return StageOutcome(
        Stage.BUDGET, Status.PASS,
        metrics=metrics,
        error="; ".join(report.notes),
        duration_s=time.monotonic() - started,
        payload=report,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: Board-free stages, in the order they run.
SCREEN_STAGES = (Stage.FETCH, Stage.PROBE, Stage.SHAPE, Stage.PATCH, Stage.LINT, Stage.BUDGET)


def _emit(ctx: Context, recipe: Recipe, graph: GraphSpec, outcome: StageOutcome,
          *, variant: str = "") -> Event:
    classification = None
    if outcome.status == Status.FAIL and outcome.error:
        classification = signatures.classify(outcome.error, default=outcome.failure_class or FC.UNKNOWN)

    event = Event(
        stage=outcome.stage,
        status=outcome.status,
        model_id=recipe.id,
        graph_id=graph.id,
        variant_id=variant,
        run_id=ctx.run_id,
        zoo_commit=ctx.zoo_commit,
        toolchain=ctx.toolchain_block(),
        duration_s=round(outcome.duration_s, 3),
        metrics=outcome.metrics,
        failure_class=(classification.failure_class if classification else outcome.failure_class),
        signature=(classification.signature if classification else ""),
        known_issue=(classification.known_issue if classification else ""),
        is_infra=(classification.is_infra if classification else False),
        error=outcome.error[:4000],
        artifacts=outcome.artifacts,
    )
    ctx.log.append(event)
    return event


def screen_graph(
    recipe: Recipe, graph: GraphSpec, ctx: Context, *, unlocked: bool = False
) -> list[Event]:
    """Run one graph through every board-free stage."""
    events: list[Event] = []

    if not graph.enabled:
        events.append(
            _emit(
                ctx, recipe, graph,
                StageOutcome(
                    Stage.FETCH, Status.SKIP,
                    failure_class="SKIPPED",
                    error=graph.skip_reason or "disabled with no reason given",
                ),
            )
        )
        return events

    fetched = stage_fetch(recipe, graph)
    events.append(_emit(ctx, recipe, graph, fetched))
    if fetched.status != Status.PASS:
        return events

    probed = stage_probe(fetched.payload)
    events.append(_emit(ctx, recipe, graph, probed))
    if probed.status != Status.PASS:
        return events

    # Weights are resolved here, unlike in `probe`, which deliberately avoids
    # loading them just to count operators. The patch and parity stages need
    # real tensor data: serialising a model whose initializers are still
    # external references gives ONNX Runtime nothing to load.
    model = onnx.load(str(fetched.payload))

    shaped = stage_shape(model, graph)
    events.append(_emit(ctx, recipe, graph, shaped))
    model = shaped.payload

    # Which patches to try: whatever a first lint pass over the shaped graph
    # says would help. This is the repair-then-judge order — screening a graph
    # for problems the harness can fix would make the funnel a bottleneck.
    scratch = workdir_for(recipe, graph)
    scratch.mkdir(parents=True, exist_ok=True)
    shaped_path = scratch / "shaped.onnx"
    onnx.save(model, str(shaped_path))
    pre = stage_lint(probe.probe_file(shaped_path), ctx, graph, unlocked=unlocked)
    wanted = list(pre.metrics.get("suggested_patches", []))

    patched = stage_patch(model, wanted)
    patched.metrics["pre_patch_violations"] = pre.metrics.get("violations", [])
    events.append(_emit(ctx, recipe, graph, patched))
    model = patched.payload

    final_path = scratch / "prepared.onnx"
    onnx.save(model, str(final_path))

    linted = stage_lint(probe.probe_file(final_path), ctx, graph, unlocked=unlocked)
    linted.artifacts = [str(final_path)]
    events.append(_emit(ctx, recipe, graph, linted))
    if linted.status != Status.PASS:
        return events

    budgeted = stage_budget(model, ctx)
    events.append(_emit(ctx, recipe, graph, budgeted))
    return events


def screen(recipe: Recipe, ctx: Context, *, unlocked: bool = False) -> list[Event]:
    events: list[Event] = []
    for graph in recipe.graphs:
        events += screen_graph(recipe, graph, ctx, unlocked=unlocked)
    return events
