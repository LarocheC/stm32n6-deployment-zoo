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
    """Pin symbolic dimensions, re-resolve the input, fold constant inputs."""
    started = time.monotonic()
    notes: list[str] = []
    changed = 0
    failed: list[str] = []

    out = model
    if graph.resolution:
        # Before pinning: re-resolving clears every derived shape, so anything
        # pinned first would be discarded by the re-inference anyway.
        out, result = structural.retarget_input_resolution(out, graph.resolution)
        notes.append(result.note)
        changed += result.changed
        if not result.applied:
            failed.append(f"resolution {graph.resolution}: {result.note}")

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

    # A requested re-resolution that did not happen must fail the stage. It is
    # the difference between measuring yunet at 320 and measuring yunet at 640
    # under a row labelled 320.
    return StageOutcome(
        Stage.SHAPE,
        Status.FAIL if failed else Status.PASS,
        metrics={
            "pinned": graph.pin,
            "resolution": graph.resolution,
            "folded_constants": sorted(constants),
            "changed": changed,
        },
        failure_class=FC.SHAPE_DYNAMIC_UNPINNABLE if failed else "",
        error=("; ".join(failed) if failed else "; ".join(n for n in notes if n)),
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
    if shaped.status != Status.PASS:
        return events
    model = shaped.payload

    # Which patches to try: whatever a first lint pass over the shaped graph
    # says would help. This is the repair-then-judge order — screening a graph
    # for problems the harness can fix would make the funnel a bottleneck.
    scratch = workdir_for(recipe, graph)
    scratch.mkdir(parents=True, exist_ok=True)
    shaped_path = scratch / "shaped.onnx"
    onnx.save(model, str(shaped_path))
    pre = stage_lint(probe.probe_file(shaped_path), ctx, graph, unlocked=unlocked)
    # Recipe-declared patches run first: they exist because of a decision the
    # recipe made (a shorter window, a different resolution), so the graph lint
    # sees afterwards is the one that will actually be compiled.
    wanted = graph.patches + [
        p for p in pre.metrics.get("suggested_patches", []) if p not in graph.patches
    ]

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


# ---------------------------------------------------------------------------
# Board-attached stages
# ---------------------------------------------------------------------------


def stage_quantize(
    prepared: Path,
    out: Path,
    recipe: Recipe,
    graph: GraphSpec,
    *,
    policy: dict | None = None,
) -> StageOutcome:
    import time as _time

    from zoo.quant import qdq
    from zoo.quant.calib import CalibrationSpec

    started = _time.monotonic()
    spec = CalibrationSpec(
        provider=recipe.calibration.provider,
        n=recipe.calibration.n,
        seed=recipe.calibration.seed,
        source=recipe.calibration.dataset,
        preprocessor=recipe.calibration.preprocessor,
        options=recipe.calibration.options,
    )
    roles = {s.name: s.role for s in graph.inputs}
    result = qdq.quantize(prepared, out, calibration=spec, roles=roles)

    metrics = result.metrics()
    if result.ok and result.path and result.path.is_file():
        # Re-run the memory accounting on the int8 graph. This is the number
        # that decides placement — the fp32 budget taken at screen time is a
        # projection, and "int8 will be about four times smaller" is an
        # expectation that has to be replaced by a measurement of the actual
        # artifact before it can be quoted.
        try:
            after = budgetmod.analyse(result.path)
            metrics["int8_peak_activation_fused"] = after.peak_activation_fused
            metrics["int8_quantised_weight_bytes"] = after.quantised_weight_bytes
            metrics["int8_unresolved_tensors"] = after.unresolved_tensors
            if policy is not None:
                metrics["int8_placement"] = after.placement(policy)
        except Exception as exc:  # noqa: BLE001 - accounting must not fail a good artifact
            metrics["int8_budget_error"] = f"{type(exc).__name__}: {exc}"

    return StageOutcome(
        Stage.QUANTIZE,
        Status.PASS if result.ok else Status.FAIL,
        metrics=metrics,
        failure_class="" if result.ok else FC.QUANT_FAILED,
        error=result.error or "; ".join(result.audit_failures),
        artifacts=[str(out)] if result.path else [],
        duration_s=_time.monotonic() - started,
        payload=result,
    )


def stage_generate(tc: Toolchain, model: Path, base: Path, graph: GraphSpec) -> StageOutcome:
    from zoo.st import generate as gmod

    best, attempts = gmod.generate_ladder(
        tc, model, base, fix_shapes=graph.fix_parametric_shapes()
    )
    metrics = best.metrics()
    metrics["ladder"] = [{"profile": a.profile, "ok": a.ok} for a in attempts]
    return StageOutcome(
        Stage.GENERATE,
        Status.PASS if best.ok else Status.FAIL,
        metrics=metrics,
        failure_class="" if best.ok else FC.CODEGEN_ERROR,
        error="" if best.ok else best.error,
        artifacts=[str(best.out_dir)],
        duration_s=sum(a.duration_s for a in attempts),
        payload=best,
    )


def stage_board(
    tc: Toolchain,
    compiled,
    model: Path,
    graph: GraphSpec,
    *,
    policy: dict | None = None,
    canary: Any = None,
    val_input: list[Path] | None = None,
    on_reading=None,  # noqa: ANN001
) -> StageOutcome:
    """Reload and re-measure until the policy's evidence bar is met.

    The canary is read *before* the bracket rather than after, so a bench that
    was already wrong is caught before spending three loads on a model. A row
    whose canary has drifted is still measured — the numbers are recorded, and
    quarantined — because a quarantined measurement is evidence about the bench
    and throwing it away would lose that.
    """
    import time as _time

    from zoo.board import bracket as bmod
    from zoo.board import link

    started = _time.monotonic()
    cfg = (policy or {}).get("measure", {})
    loads = int(cfg.get("min_loads", 3))
    invokes = int(cfg.get("invokes_per_load", 10))
    unstable_cv = float(cfg.get("unstable_cv", 0.02))

    try:
        link.preflight(tc)
    except Exception as exc:  # noqa: BLE001
        return StageOutcome(
            Stage.BOARD, Status.FAIL, failure_class=FC.BOARD_NOT_ATTACHED,
            error=f"{type(exc).__name__}: {exc}", duration_s=_time.monotonic() - started,
        )

    reading = canary.read() if canary is not None else None

    result = bmod.run(
        tc,
        network_c=compiled.out_dir / "network.c",
        model=model,
        profile=compiled.profile,
        out_dir=compiled.out_dir / "bracket",
        fix_shapes=graph.fix_parametric_shapes(),
        loads=loads,
        invokes=invokes,
        unstable_cv=unstable_cv,
        loader_retries=int((policy or {}).get("board", {}).get("loader_retries", 3)),
        val_input=val_input,
        on_reading=on_reading,
    )
    if canary is not None and reading is not None:
        canary.apply(result, reading)

    metrics = result.metrics()
    metrics["profile_used"] = compiled.profile
    # What the on-target cosine was measured against. Without this the column
    # is not comparable between rows: `validate`'s default is uniform noise in
    # [0, 1], which flatters a model calibrated on noise and punishes one
    # calibrated on real pixels.
    metrics["ontarget_input_source"] = (
        "calibration-corpus" if val_input else "random-uniform-0-1"
    )
    median = result.median_ms
    if compiled.info and median:
        predicted = compiled.info.predicted_ms()
        if predicted:
            metrics["predicted_ms"] = predicted
            metrics["predicted_vs_measured"] = median / predicted

    # A bracket that produced no measurement at all failed; one that produced
    # measurements the gate distrusts did not. The distinction is the whole
    # point of recording the gate rather than silently dropping the row.
    if not result.good:
        first = next((ld for ld in result.loads if ld.error), None)
        return StageOutcome(
            Stage.BOARD, Status.FAIL,
            metrics=metrics,
            failure_class=(
                FC.LOADER_NO_SUCCESS_MARKER
                if first and first.infra
                else FC.TARGET_HANG
            ),
            error=(first.error if first else "no load produced a measurement"),
            duration_s=_time.monotonic() - started,
            payload=result,
        )

    return StageOutcome(
        Stage.BOARD, Status.PASS,
        metrics=metrics,
        error="" if result.trusted else f"[{result.gate}] {result.reason}",
        duration_s=_time.monotonic() - started,
        payload=result,
    )


def write_val_inputs(
    recipe: Recipe, model: Path, out_dir: Path, *, samples: int = 8
) -> list[Path] | None:
    """Real inputs for `validate -vi`, drawn from the recipe's own corpus.

    Returns None for a synthetically calibrated recipe: there is no real data
    to offer, and fabricating some here would only relocate the problem.

    The seed is offset from the calibration seed for the same reason the host
    fidelity check offsets it — a model should not be graded on the samples
    that set its scales.
    """
    if recipe.calibration.is_synthetic:
        return None

    import numpy as np

    from zoo.quant.calib import CalibrationSpec, get_provider, specs_from_model

    spec = CalibrationSpec(
        provider=recipe.calibration.provider,
        n=samples,
        seed=recipe.calibration.seed + 1,
        source=recipe.calibration.dataset,
        preprocessor=recipe.calibration.preprocessor,
        options=recipe.calibration.options,
    )
    specs = specs_from_model(onnx.load(str(model)))
    try:
        feeds = list(get_provider(spec.provider).batches(specs, spec))
    except Exception:  # noqa: BLE001 - falling back to random data is not fatal
        return None
    if not feeds:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # One file per input, in the graph's own input order, which is the order
    # `-vi` expects them in.
    for item in specs:
        stacked = np.concatenate([f[item.name] for f in feeds], axis=0)
        path = out_dir / f"valinput_{len(written)}.npy"
        np.save(path, stacked)
        written.append(path)
    return written


def measure_graph(
    recipe: Recipe,
    graph: GraphSpec,
    ctx: Context,
    *,
    canary: Any = None,
    board: bool = True,
    on_reading=None,  # noqa: ANN001
) -> list[Event]:
    """Quantise, compile and measure a graph that has already screened clean.

    `board=False` stops after the compile. Everything up to and including
    `network.c` is board-free, so a wedged probe should cost the compile
    evidence too only if someone asks it to.
    """
    events: list[Event] = []
    scratch = workdir_for(recipe, graph)
    prepared = scratch / "prepared.onnx"
    if not prepared.is_file():
        events += screen_graph(recipe, graph, ctx)
        if not prepared.is_file():
            return events

    # `prepared.onnx` is written before lint renders its verdict, so its mere
    # existence proves the graph was *repaired*, not that it was accepted. A
    # cached file from a previous rejected screen would otherwise walk straight
    # past the rejection into the quantiser — which is how silero-vad, whose
    # entire network sits inside two `If` branches, reached the quantize stage
    # after lint had already refused it.
    screened = stage_lint(probe.probe_file(prepared), ctx, graph)
    if screened.status != Status.PASS:
        events.append(_emit(ctx, recipe, graph, screened))
        return events

    quantised = scratch / "int8.onnx"
    q = stage_quantize(prepared, quantised, recipe, graph, policy=ctx.policy)
    events.append(_emit(ctx, recipe, graph, q))
    if q.status != Status.PASS:
        return events

    g = stage_generate(ctx.toolchain, quantised, scratch / "compile", graph)
    variant = ids.variant_id(profile=g.metrics.get("profile_used", "?"))
    events.append(_emit(ctx, recipe, graph, g, variant=variant))
    if g.status != Status.PASS or not board:
        return events

    b = stage_board(
        ctx.toolchain, g.payload, quantised, graph,
        policy=ctx.policy, canary=canary, on_reading=on_reading,
        val_input=write_val_inputs(recipe, quantised, scratch / "valinput"),
    )
    events.append(_emit(ctx, recipe, graph, b, variant=variant))
    return events


def build_canary(ctx: Context, recipes: list[Recipe]) -> Any:
    """The session's canary, compiled if it is not already lying around.

    Returns None when the policy names no canary, or when the named recipe is
    not in the zoo — a missing canary weakens the evidence and is reported as
    such, but it must not stop a measurement session outright.
    """
    from zoo.board import bracket as bmod

    cfg = ctx.policy.get("measure", {})
    wanted = cfg.get("canary_model")
    if not wanted:
        return None

    match = next((r for r in recipes if r.id == wanted), None)
    if match is None or not match.enabled_graphs:
        return None
    graph = match.enabled_graphs[0]

    scratch = workdir_for(match, graph)
    quantised = scratch / "int8.onnx"
    compiled = scratch / "compile"

    network_c = None
    profile = ""
    for candidate in sorted(compiled.glob("*/network.c")):
        network_c = candidate
        profile = candidate.parent.name
        break

    if network_c is None or not quantised.is_file():
        # Build it. The canary is only useful if it is the same artifact every
        # time, so this happens once and is then reused from disk.
        events = measure_graph(match, graph, ctx, board=False)
        if not events or events[-1].status != Status.PASS:
            return None
        for candidate in sorted(compiled.glob("*/network.c")):
            network_c = candidate
            profile = candidate.parent.name
            break
        if network_c is None:
            return None

    return bmod.Canary(
        ctx.toolchain,
        network_c=network_c,
        model=quantised,
        profile=profile,
        out_dir=scratch / "canary",
        fix_shapes=graph.fix_parametric_shapes(),
        drift_limit=float(cfg.get("canary_drift_frac", 0.10)),
        invokes=int(cfg.get("invokes_per_load", 10)),
    )
