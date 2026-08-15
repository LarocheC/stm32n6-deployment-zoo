"""The determinism gate: what makes a row trustworthy, and what revokes it."""

from __future__ import annotations

import pytest

from zoo.board import bracket as bmod
from zoo.board.measure import ValidateResult


def _bracket(latencies, *, required=3, invokes=10, cv=0.02, stds=None):
    b = bmod.Bracket(required_loads=required, invokes_per_load=invokes, unstable_cv=cv)
    for i, ms in enumerate(latencies, start=1):
        b.loads.append(
            bmod.LoadReading(
                index=i, ok=ms is not None, latency_ms=ms, cosine=0.999,
                std_ms=(stds[i - 1] if stds else None),
                error="" if ms is not None else "loader did not report success",
                infra=ms is None,
            )
        )
    return b


def test_three_agreeing_loads_are_trusted():
    b = _bracket([10.00, 10.02, 9.99])
    assert b.judge() == bmod.TRUSTED
    assert b.median_ms == pytest.approx(10.00)
    assert b.cv < 0.01


def test_too_few_loads_is_insufficient_not_unstable():
    """One load cannot be unstable; it can only be unsupported.

    Reporting a single measurement as stable because it has no spread is the
    exact failure the gate exists to prevent.
    """
    b = _bracket([10.0])
    assert b.judge() == bmod.INSUFFICIENT
    assert b.cv is None


def test_disagreeing_loads_are_unstable_and_keep_their_median():
    b = _bracket([10.0, 14.0, 10.1])
    assert b.judge() == bmod.UNSTABLE
    # The number survives the verdict: an unstable row is still evidence, and
    # the gate annotates rather than deletes.
    assert b.median_ms == pytest.approx(10.1)
    assert "CV" in b.reason


def test_failed_loads_do_not_count_toward_the_requirement():
    b = _bracket([10.0, None, 10.0])
    assert b.judge() == bmod.INSUFFICIENT
    assert len(b.good) == 2


def test_canary_drift_quarantines_even_a_tight_bracket():
    """A bench that moved outranks a model that looks reproducible."""
    b = _bracket([10.0, 10.0, 10.0])
    assert b.judge() == bmod.TRUSTED

    canary = bmod.Canary.__new__(bmod.Canary)
    canary.reference = bmod.CanaryReading(ok=True, latency_ms=0.100)
    canary.drift_limit = 0.10
    reading = bmod.CanaryReading(ok=True, latency_ms=0.130)  # +30%
    bmod.Canary.apply(canary, b, reading)

    assert b.gate == bmod.QUARANTINED
    assert b.canary_drift == pytest.approx(0.30)
    assert b.median_ms == pytest.approx(10.0)


def test_canary_within_tolerance_leaves_the_verdict_alone():
    b = _bracket([10.0, 10.0, 10.0])
    canary = bmod.Canary.__new__(bmod.Canary)
    canary.reference = bmod.CanaryReading(ok=True, latency_ms=0.100)
    canary.drift_limit = 0.10
    bmod.Canary.apply(canary, b, bmod.CanaryReading(ok=True, latency_ms=0.105))
    assert b.gate == bmod.TRUSTED


def test_within_load_spread_is_reported_separately_from_across_load():
    """Ten invokes agreeing says nothing about reloading being reproducible."""
    b = _bracket([10.0, 12.0, 10.0], stds=[0.001, 0.001, 0.001])
    b.judge()
    assert b.gate == bmod.UNSTABLE
    assert b.within_load_cv < 0.001
    assert b.cv > 0.05


def test_metrics_carry_the_gate_and_the_evidence():
    b = _bracket([10.0, 10.1, 10.05])
    b.judge()
    m = b.metrics()
    assert m["determinism_gate"] == bmod.TRUSTED
    assert m["loads_ok"] == 3
    assert m["invokes_per_load"] == 10
    assert m["latency_ms_per_load"] == [10.0, 10.1, 10.05]
    assert m["latency_ms_median"] == pytest.approx(10.05)


# ---------------------------------------------------------------------------


VALIDATE_TAIL = """
  nb sample(s)                   :   10
  duration                       :   0.106 ms by sample (0.103/0.115/0.005)
  acc=n.a. rmse=0.041 mae=0.017 cos=0.999978
"""


def test_validate_parses_the_free_within_run_spread(monkeypatch):
    """`validate` already prints (min/max/std); throwing it away costs evidence."""
    from zoo.board import measure as mmod

    class _Result:
        ok = True
        combined = VALIDATE_TAIL
        duration_s = 4.2
        argv: list[str] = []

    monkeypatch.setattr(mmod, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(
        mmod.profiles, "get", lambda *a, **k: type("P", (), {"selector": "x"})()
    )

    out = mmod.validate(
        type("TC", (), {"stedgeai": "x", "serial_port": "p", "serial_baud": 1,
                        "core_tag": "4.0"})(),
        __import__("pathlib").Path("model.onnx"),
        profile="onchip",
        out_dir=__import__("pathlib").Path("."),
    )
    assert isinstance(out, ValidateResult)
    assert out.latency_ms == pytest.approx(0.106)
    assert out.latency_min_ms == pytest.approx(0.103)
    assert out.latency_max_ms == pytest.approx(0.115)
    assert out.latency_std_ms == pytest.approx(0.005)
    assert out.samples == 10
    assert out.cosine == pytest.approx(0.999978)


def test_a_wedge_mid_bracket_abandons_the_remaining_loads(tmp_path, monkeypatch):
    """Retrying a wedged probe cannot succeed; it only costs minutes.

    The loader burns `loader_retries` attempts per load, so a wedge arriving at
    load 2 of 3 costs six futile attempts before the row is written off. One
    health check converts that into a single honest diagnosis, and leaves the
    unattempted loads visible as unattempted rather than as failures.
    """
    from zoo.board import link
    from zoo.board import measure as mmod

    calls = {"loads": 0}

    def _load(tc, network_c, **kw):  # noqa: ANN001
        calls["loads"] += 1
        ok = calls["loads"] == 1
        return mmod.LoadResult(
            ok=ok, log=mmod.SUCCESS_MARKER if ok else "", error="" if ok else "no marker"
        )

    def _validate(tc, model, *, profile, out_dir, batches=4, **kw):  # noqa: ANN001
        return mmod.ValidateResult(ok=True, latency_ms=5.0, cosine=0.99, samples=batches)

    def _wedged(tc):  # noqa: ANN001
        raise link.BoardError("ST-LINK is wedged (DEV_USB_COMM_ERR)")

    monkeypatch.setattr(mmod, "load_network", _load)
    monkeypatch.setattr(mmod, "validate", _validate)
    monkeypatch.setattr(link, "assert_probe_healthy", _wedged)

    result = bmod.run(
        None, network_c=tmp_path / "n.c", model=tmp_path / "m.onnx",
        profile="onchip", out_dir=tmp_path / "b", loads=3, invokes=10,
    )
    # Load 1 succeeded, load 2 failed and found the wedge; load 3 never ran.
    assert calls["loads"] == 2
    assert len(result.loads) == 2
    assert result.gate == bmod.INSUFFICIENT
    assert "wedged" in result.reason
    assert result.median_ms == pytest.approx(5.0)
