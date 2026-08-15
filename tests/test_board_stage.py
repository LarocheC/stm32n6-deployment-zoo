"""The board stage's wiring, with the board faked.

These exist because the gate cannot be exercised on hardware on demand — the
ST-LINK wedges, and a wedge is cleared by a human with a USB cable. A wiring
mistake between `stage_board`, the bracket and the canary would otherwise stay
invisible until the next session with a working probe, which is exactly when it
costs the most.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from zoo import funnel
from zoo.board import bracket as bmod
from zoo.board import measure as mmod
from zoo.recipe import GraphSpec
from zoo.store.schema import Status

POLICY = {
    "measure": {"min_loads": 3, "invokes_per_load": 10, "unstable_cv": 0.02},
    "board": {"loader_retries": 3},
}


@dataclass
class _Compiled:
    out_dir: Path
    profile: str = "onchip"
    info: object = None


class _Bench:
    """A pretend board: hands out latencies in order, records what was asked."""

    def __init__(self, latencies, *, load_ok=True):
        self.latencies = list(latencies)
        self.load_ok = load_ok
        self.loads = 0
        self.batches: list[int] = []

    def install(self, monkeypatch):
        def _load(tc, network_c, **kw):  # noqa: ANN001
            self.loads += 1
            return mmod.LoadResult(
                ok=self.load_ok, duration_s=1.0,
                log=mmod.SUCCESS_MARKER if self.load_ok else "",
                error="" if self.load_ok else "no success marker",
            )

        def _validate(tc, model, *, profile, out_dir, fix_shapes=None, batches=4, **kw):  # noqa: ANN001
            self.batches.append(batches)
            ms = self.latencies.pop(0) if self.latencies else None
            return mmod.ValidateResult(
                ok=ms is not None, latency_ms=ms, cosine=0.999,
                latency_min_ms=ms, latency_max_ms=ms, latency_std_ms=0.0,
                samples=batches,
            )

        monkeypatch.setattr(mmod, "load_network", _load)
        monkeypatch.setattr(mmod, "validate", _validate)
        monkeypatch.setattr(funnel, "Toolchain", object, raising=False)
        from zoo.board import link

        monkeypatch.setattr(link, "preflight", lambda tc: None)
        return self


def _graph() -> GraphSpec:
    return GraphSpec(id="g", file="m.onnx")


def test_the_policy_decides_how_many_loads_and_invokes(tmp_path, monkeypatch):
    bench = _Bench([10.0, 10.0, 10.0]).install(monkeypatch)
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(), policy=POLICY
    )
    assert out.status == Status.PASS
    assert bench.loads == 3
    assert bench.batches == [10, 10, 10]
    assert out.metrics["determinism_gate"] == bmod.TRUSTED
    assert out.metrics["latency_ms_median"] == pytest.approx(10.0)


def test_disagreeing_reloads_still_pass_the_stage_but_not_the_gate(tmp_path, monkeypatch):
    """An unstable measurement is a result, not a stage failure."""
    _Bench([10.0, 13.0, 10.0]).install(monkeypatch)
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(), policy=POLICY
    )
    assert out.status == Status.PASS
    assert out.metrics["determinism_gate"] == bmod.UNSTABLE
    assert "[unstable]" in out.error


def test_a_board_that_never_loads_fails_the_stage_as_infrastructure(tmp_path, monkeypatch):
    _Bench([], load_ok=False).install(monkeypatch)
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(), policy=POLICY
    )
    assert out.status == Status.FAIL
    assert out.failure_class == "LOADER_NO_SUCCESS_MARKER"


def test_the_canary_is_read_before_the_bracket_and_can_quarantine_it(tmp_path, monkeypatch):
    """A moved bench outranks a model that looks perfectly reproducible."""
    # First reading is the canary's, then three identical model loads.
    bench = _Bench([0.100, 10.0, 10.0, 10.0]).install(monkeypatch)
    canary = bmod.Canary(
        None, network_c=tmp_path / "c.c", model=tmp_path / "c.onnx",
        profile="onchip", out_dir=tmp_path / "canary", drift_limit=0.10,
    )
    canary.reference = bmod.CanaryReading(ok=True, latency_ms=0.080)  # 25% away

    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(),
        policy=POLICY, canary=canary,
    )
    assert bench.loads == 4  # canary first, then three reloads
    assert out.status == Status.PASS
    assert out.metrics["determinism_gate"] == bmod.QUARANTINED
    assert out.metrics["canary_ms"] == pytest.approx(0.100)
    # Quarantined, and the measurements are still there.
    assert out.metrics["latency_ms_median"] == pytest.approx(10.0)


def test_a_steady_canary_leaves_a_good_bracket_trusted(tmp_path, monkeypatch):
    _Bench([0.101, 10.0, 10.0, 10.0]).install(monkeypatch)
    canary = bmod.Canary(
        None, network_c=tmp_path / "c.c", model=tmp_path / "c.onnx",
        profile="onchip", out_dir=tmp_path / "canary", drift_limit=0.10,
    )
    canary.reference = bmod.CanaryReading(ok=True, latency_ms=0.100)

    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(),
        policy=POLICY, canary=canary,
    )
    assert out.metrics["determinism_gate"] == bmod.TRUSTED
    assert out.metrics["canary_drift"] == pytest.approx(0.01, abs=1e-9)


def test_the_first_canary_reading_of_a_session_becomes_the_reference(tmp_path, monkeypatch):
    _Bench([0.100, 10.0, 10.0, 10.0]).install(monkeypatch)
    canary = bmod.Canary(
        None, network_c=tmp_path / "c.c", model=tmp_path / "c.onnx",
        profile="onchip", out_dir=tmp_path / "canary",
    )
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(),
        policy=POLICY, canary=canary,
    )
    assert canary.reference is not None
    # Nothing to drift against yet, so the gate judges on the loads alone.
    assert out.metrics["canary_drift"] is None
    assert out.metrics["determinism_gate"] == bmod.TRUSTED


def test_a_board_free_run_stops_after_the_compile(tmp_path, monkeypatch):
    """--no-board must not cost the compile evidence when the probe is wedged."""
    bench = _Bench([10.0, 10.0, 10.0]).install(monkeypatch)
    calls = {"board": 0}
    monkeypatch.setattr(
        funnel, "stage_board",
        lambda *a, **k: calls.__setitem__("board", calls["board"] + 1),
    )
    assert bench.loads == 0
    assert calls["board"] == 0


def test_the_on_target_input_source_is_recorded(tmp_path, monkeypatch):
    """The accuracy column means different things with and without real data.

    `validate` defaults to uniform noise in [0, 1]. A row measured that way and
    a row measured on the model's own corpus are not comparable, so which one
    happened travels with the number.
    """
    _Bench([10.0, 10.0, 10.0]).install(monkeypatch)
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(), policy=POLICY
    )
    assert out.metrics["ontarget_input_source"] == "random-uniform-0-1"

    _Bench([10.0, 10.0, 10.0]).install(monkeypatch)
    out = funnel.stage_board(
        None, _Compiled(tmp_path), tmp_path / "m.onnx", _graph(), policy=POLICY,
        val_input=[tmp_path / "valinput_0.npy"],
    )
    assert out.metrics["ontarget_input_source"] == "calibration-corpus"


def test_real_validation_inputs_come_from_the_recipe_corpus(tmp_path):
    """And a synthetic recipe offers none, rather than fabricating some."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    from PIL import Image

    from zoo.recipe import Calibration, Recipe

    model_path = tmp_path / "m.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
              str(model_path))

    corpus = tmp_path / "img"
    corpus.mkdir()
    for i in range(4):
        Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(corpus / f"{i}.png")

    real = Recipe(id="r", source="s", path=tmp_path / "r.toml",
                  calibration=Calibration(provider="image_folder", dataset=str(corpus)))
    written = funnel.write_val_inputs(real, model_path, tmp_path / "vi", samples=3)
    assert written and written[0].suffix == ".npy"
    assert np.load(written[0]).shape == (3, 3, 8, 8)

    synthetic = Recipe(id="r", source="s", path=tmp_path / "r.toml")
    assert funnel.write_val_inputs(synthetic, model_path, tmp_path / "vi2") is None
