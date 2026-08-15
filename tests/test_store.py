"""The event log and the fold.

The fold encodes two editorial rules that decide whether the leaderboard is
honest, so both get a test that would fail loudly if someone "simplified" them.
"""

from __future__ import annotations

from pathlib import Path

from zoo.store import snapshot as smod
from zoo.store.events import EventLog
from zoo.store.schema import Event, Stage, Status, Verdict


def _ev(stage: str, status: str, **kw) -> Event:
    kw.setdefault("model_id", "demo")
    kw.setdefault("graph_id", "main")
    kw.setdefault("variant_id", "onchip/int8/in-f32/npu")
    return Event(stage=stage, status=status, **kw)


def test_log_round_trips(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.LINT, Status.PASS, metrics={"ops_npu": 100}))
    log.append(_ev(Stage.GENERATE, Status.PASS, metrics={"epochs_total": 40}))
    events = log.read_all()
    assert [e.stage for e in events] == [Stage.LINT, Stage.GENERATE]
    assert events[1].metrics["epochs_total"] == 40


def test_log_survives_a_torn_final_line(tmp_path: Path) -> None:
    """A process killed mid-write must not make the whole history unreadable."""
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(_ev(Stage.LINT, Status.PASS))
    with path.open("a") as fh:
        fh.write('{"stage": "generate", "sta')
    assert len(log.read_all()) == 1


def test_infra_failure_is_never_a_model_verdict(tmp_path: Path) -> None:
    """The rule that keeps a broken bench from becoming a modelling result.

    A wedged ST-LINK during the board stage must leave the model's verdict at
    what it actually earned — COMPILES — not drag it to REJECTED.
    """
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.GENERATE, Status.PASS, ts="2026-01-01T00:00:00Z"))
    log.append(
        _ev(
            Stage.BOARD,
            Status.FAIL,
            ts="2026-01-01T00:01:00Z",
            is_infra=True,
            failure_class="STLINK_WEDGED",
            signature="deadbeef",
        )
    )
    snap = smod.fold(log)
    graph = snap.graphs[("demo", "main")]
    assert graph.verdict == Verdict.COMPILES
    assert graph.infra_events == 1
    assert snap.infra_events == 1
    # An infra failure must not enter the atlas as a model constraint either.
    assert snap.signatures == {}


def test_failure_after_success_keeps_the_earned_verdict(tmp_path: Path) -> None:
    """A model that compiles and then fails on the board has still compiled.

    Recording that as REJECTED would erase the most useful thing known about
    it. Where it stopped belongs in `blocked_by`, not in the verdict.
    """
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.ANALYZE, Status.PASS, ts="2026-01-01T00:00:00Z"))
    log.append(_ev(Stage.QUANTIZE, Status.PASS, ts="2026-01-01T00:01:00Z"))
    log.append(_ev(Stage.GENERATE, Status.PASS, ts="2026-01-01T00:02:00Z"))
    log.append(
        _ev(
            Stage.BOARD,
            Status.FAIL,
            ts="2026-01-01T00:03:00Z",
            failure_class="TARGET_HANG",
            signature="abc123",
        )
    )
    snap = smod.fold(log)
    state = snap.graphs[("demo", "main")].best
    assert state.verdict == Verdict.COMPILES
    assert state.blocked_by == Stage.BOARD
    assert state.failure_class == "TARGET_HANG"


def test_rejected_when_nothing_was_earned(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.LINT, Status.FAIL, failure_class="OP_UNSUPPORTED", signature="s1"))
    snap = smod.fold(log)
    assert snap.graphs[("demo", "main")].verdict == Verdict.REJECTED


def test_skipped_graph_is_distinguishable_from_rejected(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.FETCH, Status.SKIP, failure_class="SKIPPED"))
    snap = smod.fold(log)
    assert snap.graphs[("demo", "main")].verdict == Verdict.SKIPPED


def test_signatures_collapse_across_models(tmp_path: Path) -> None:
    """One constraint, six models — the atlas must say so.

    This is the mechanism that turns thirty heterogeneous failures into ten
    real constraints.
    """
    log = EventLog(tmp_path / "events.jsonl")
    for model in ("a", "b", "c"):
        log.append(
            _ev(
                Stage.GENERATE,
                Status.FAIL,
                model_id=model,
                failure_class="FRONTEND_SHAPE_ERROR",
                signature="same-sig",
            )
        )
    snap = smod.fold(log)
    assert list(snap.signatures) == ["same-sig"]
    assert snap.signatures["same-sig"]["count"] == 3
    assert snap.signatures["same-sig"]["models"] == ["a", "b", "c"]


def test_best_variant_wins_the_row(tmp_path: Path) -> None:
    """The leaderboard shows the best rung the profile ladder reached."""
    log = EventLog(tmp_path / "events.jsonl")
    log.append(
        _ev(Stage.GENERATE, Status.FAIL, variant_id="onchip/int8/in-f32/npu",
            failure_class="ALLOC_FAILED_ONCHIP", signature="s")
    )
    log.append(
        _ev(Stage.GENERATE, Status.PASS, variant_id="extflash/int8/in-f32/npu",
            metrics={"weights_bytes": 1024})
    )
    snap = smod.fold(log)
    graph = snap.graphs[("demo", "main")]
    assert graph.verdict == Verdict.COMPILES
    assert graph.best.variant_id == "extflash/int8/in-f32/npu"


def test_report_renders_without_a_board(tmp_path: Path) -> None:
    from zoo.store import report as rmod

    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.GENERATE, Status.PASS,
                   metrics={"weights_bytes": 185425, "activations_bytes": 571392,
                            "epochs_total": 40, "epochs_hw": 12, "epochs_sw": 23}))
    text = rmod.render(smod.fold(log))
    assert "## Leaderboard" in text
    assert "## Failure atlas" in text
    assert "40 (12H/0y/23S)" in text


def test_a_latency_measured_before_a_requantise_is_marked_stale(tmp_path: Path) -> None:
    """Re-quantising replaces the artifact; the old latency describes nothing.

    The fold merges every stage's metrics into one row, which is what lets a
    fresh calibration provenance sit beside a stale board number as though the
    two came from the same model. They did not.
    """
    from zoo.store import report as rmod

    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.QUANTIZE, Status.PASS, ts="2026-08-15T09:00:00Z",
                   metrics={"int8_cos": 0.94, "calibration_synthetic": True}))
    log.append(_ev(Stage.GENERATE, Status.PASS, ts="2026-08-15T09:05:00Z",
                   metrics={"epochs_total": 40}))
    log.append(_ev(Stage.BOARD, Status.PASS, ts="2026-08-15T10:00:00Z",
                   metrics={"latency_ms_median": 10.15, "loads_ok": 3,
                            "invokes_per_load": 10, "determinism_gate": "trusted",
                            "latency_ms_cv": 0.004}))
    def _row(snapshot) -> str:  # noqa: ANN001 - the leaderboard line, not the legend
        return next(
            line for line in rmod.render(snapshot).splitlines()
            if line.startswith("| demo |")
        )

    assert "⧖ stale" not in _row(smod.fold(log))

    # Same graph, re-quantised afterwards with real data.
    log.append(_ev(Stage.QUANTIZE, Status.PASS, ts="2026-08-15T14:00:00Z",
                   metrics={"int8_cos": 0.97, "calibration_synthetic": False}))
    row = _row(smod.fold(log))
    assert "⧖ stale" in row
    # The number itself survives the annotation.
    assert "10.15" in row


def test_evidence_column_distinguishes_a_gated_row_from_a_single_shot(tmp_path: Path) -> None:
    from zoo.store import report as rmod

    log = EventLog(tmp_path / "events.jsonl")
    log.append(_ev(Stage.BOARD, Status.PASS, graph_id="gated",
                   metrics={"latency_ms_median": 1.0, "loads_ok": 3,
                            "invokes_per_load": 10, "determinism_gate": "trusted",
                            "latency_ms_cv": 0.004}))
    log.append(_ev(Stage.BOARD, Status.PASS, graph_id="ungated",
                   metrics={"latency_ms_median": 1.0}))
    text = rmod.render(smod.fold(log))
    assert "3x10 ✓ 0.4%" in text
    assert "1x? ?" in text
