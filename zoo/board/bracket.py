"""Repeat measurement, and the gate that decides whether a row is trustworthy.

A single load and a single validate produce a number. They do not produce
evidence, and the difference matters here more than it does on a workstation,
because the ways this bench lies are specific:

**The firmware may not be the one you think.** A load that fails quietly leaves
the previous network resident, and the next `validate` times *that* — happily,
plausibly, and wrongly. `load_network` already refuses to proceed without the
success marker; what a single load cannot tell you is whether the number is
reproducible across the load path at all. So the unit of repetition is a full
reload, not a repeated invoke: repeating invokes inside one load re-measures
the same possibly-wrong firmware more precisely.

**The bench drifts.** A wedged probe, a thermally throttled board, a stale
gdbserver holding the port — none of these announce themselves in a latency
figure, and all of them move it. A canary answers this: a graph whose latency
is already known, measured in the same session, immediately before the row.
If the canary moved, the bench moved, and every number taken beside it is
suspect regardless of how tight its own spread looks.

**Precision is not accuracy.** Ten invokes inside one load can agree to three
decimal places and still be wrong by 3x. That is why the gate reports the
across-load coefficient of variation rather than the within-load one, and why
the within-load spread is recorded separately instead of being folded in.

The gate never edits a number. It attaches a verdict — trusted, unstable,
insufficient, quarantined — and the leaderboard renders that verdict alongside
the latency, so an untrusted row is visible as untrusted rather than absent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zoo.board import measure
from zoo.config import Toolchain

#: Gate verdicts, worst to best.
QUARANTINED = "quarantined"   # the canary moved; the bench, not the model
INSUFFICIENT = "insufficient"  # fewer successful loads than policy demands
UNSTABLE = "unstable"          # enough loads, but they disagree
TRUSTED = "trusted"


@dataclass
class LoadReading:
    """One full reload, plus the invokes measured against it."""

    index: int
    ok: bool
    latency_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None
    cosine: float | None = None
    invokes: int = 0
    load_s: float = 0.0
    validate_s: float = 0.0
    error: str = ""
    #: True when the failure was the bench rather than the model.
    infra: bool = False


@dataclass
class Bracket:
    loads: list[LoadReading] = field(default_factory=list)
    required_loads: int = 3
    invokes_per_load: int = 10
    unstable_cv: float = 0.02
    gate: str = INSUFFICIENT
    reason: str = ""
    canary_ms: float | None = None
    canary_reference_ms: float | None = None
    canary_drift: float | None = None
    #: Set when the bracket stopped early because the probe wedged. Kept
    #: separate from `reason` so re-judging cannot lose it.
    abandoned: str = ""
    #: Set from policy when a canary is attached; see `Canary.apply`.
    _drift_limit: float = 0.10

    @property
    def good(self) -> list[LoadReading]:
        return [ld for ld in self.loads if ld.ok and ld.latency_ms is not None]

    @property
    def latencies(self) -> list[float]:
        return [ld.latency_ms for ld in self.good]

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def cv(self) -> float | None:
        """Coefficient of variation *across loads* — the number the gate uses.

        Undefined for a single load, and reported as such rather than as zero:
        one sample has no spread, which is not the same as having no variation.
        """
        values = self.latencies
        if len(values) < 2:
            return None
        mean = statistics.fmean(values)
        if mean <= 0:
            return None
        return statistics.stdev(values) / mean

    @property
    def within_load_cv(self) -> float | None:
        """Mean within-load spread, for contrast with the across-load figure."""
        pairs = [
            (ld.std_ms, ld.latency_ms)
            for ld in self.good
            if ld.std_ms is not None and ld.latency_ms
        ]
        if not pairs:
            return None
        return statistics.fmean(std / mean for std, mean in pairs)

    @property
    def cosine(self) -> float | None:
        values = [ld.cosine for ld in self.good if ld.cosine is not None]
        return min(values) if values else None

    @property
    def trusted(self) -> bool:
        return self.gate == TRUSTED

    def judge(self) -> str:
        """Apply the policy. Order matters: a moved bench outranks everything."""
        if self.canary_drift is not None and abs(self.canary_drift) > self._drift_limit:
            self.gate = QUARANTINED
            self.reason = (
                f"canary moved {self.canary_drift * 100:+.1f}% against its reference "
                f"({self.canary_reference_ms:.3f} → {self.canary_ms:.3f} ms). The bench "
                "changed during this session, so nothing measured beside it is evidence "
                "about a model."
            )
            return self.gate

        if len(self.good) < self.required_loads:
            self.gate = INSUFFICIENT
            self.reason = (
                f"{len(self.good)} of {self.required_loads} required load(s) produced a "
                "measurement"
            )
            if self.abandoned:
                self.reason += f"; {self.abandoned}"
            return self.gate

        spread = self.cv
        if spread is not None and spread > self.unstable_cv:
            self.gate = UNSTABLE
            self.reason = (
                f"across-load CV {spread * 100:.1f}% exceeds the {self.unstable_cv * 100:.0f}% "
                f"policy limit over {len(self.good)} loads "
                f"({', '.join(f'{v:.3f}' for v in self.latencies)} ms)"
            )
            return self.gate

        self.gate = TRUSTED
        self.reason = (
            f"{len(self.good)} loads x {self.invokes_per_load} invokes, across-load CV "
            f"{(spread or 0) * 100:.2f}%"
        )
        return self.gate

    def metrics(self) -> dict[str, Any]:
        return {
            # The leaderboard's latency column reads this key, so what it shows
            # is the median across reloads rather than any single run.
            "latency_ms_median": self.median_ms,
            "latency_ms_per_load": self.latencies,
            "latency_ms_cv": self.cv,
            "latency_ms_within_load_cv": self.within_load_cv,
            "loads_ok": len(self.good),
            "loads_attempted": len(self.loads),
            "loads_required": self.required_loads,
            "invokes_per_load": self.invokes_per_load,
            "ontarget_cos": self.cosine,
            "determinism_gate": self.gate,
            "determinism_reason": self.reason,
            "canary_ms": self.canary_ms,
            "canary_reference_ms": self.canary_reference_ms,
            "canary_drift": self.canary_drift,
        }


def run(
    tc: Toolchain,
    *,
    network_c: Path,
    model: Path,
    profile: str,
    out_dir: Path,
    fix_shapes: str | None = None,
    loads: int = 3,
    invokes: int = 10,
    unstable_cv: float = 0.02,
    loader_retries: int = 3,
    val_input: list[Path] | None = None,
    on_reading=None,  # noqa: ANN001 - progress callback, (LoadReading) -> None
) -> Bracket:
    """Reload and re-measure `loads` times, `invokes` samples each time."""
    bracket = Bracket(
        required_loads=loads, invokes_per_load=invokes, unstable_cv=unstable_cv
    )

    for index in range(1, loads + 1):
        cycle_dir = out_dir / f"load{index}"
        loaded = measure.load_network(
            tc, network_c, retries=loader_retries, log_dir=cycle_dir
        )
        if not loaded.ok:
            reading = LoadReading(
                index=index, ok=False, load_s=loaded.duration_s,
                error=loaded.error, infra=True,
            )
            bracket.loads.append(reading)
            if on_reading:
                on_reading(reading)

            # A wedge that arrives mid-bracket costs `loader_retries` attempts
            # per remaining load, each of which cannot succeed: the probe is in
            # a state only a physical replug clears. Re-checking it here turns
            # minutes of futile retrying into one honest diagnosis, and leaves
            # the loads that were never attempted visible as never attempted.
            from zoo.board import link

            try:
                link.assert_probe_healthy(tc)
            except Exception as exc:  # noqa: BLE001
                bracket.abandoned = f"abandoned after load {index}: {exc}"
                break
            continue

        result = measure.validate(
            tc, model, profile=profile, out_dir=cycle_dir / "val",
            fix_shapes=fix_shapes, batches=invokes, val_input=val_input,
        )
        reading = LoadReading(
            index=index,
            ok=result.ok,
            latency_ms=result.latency_ms,
            min_ms=result.latency_min_ms,
            max_ms=result.latency_max_ms,
            std_ms=result.latency_std_ms,
            cosine=result.cosine,
            invokes=result.samples or invokes,
            load_s=loaded.duration_s,
            validate_s=result.duration_s,
            error=result.error,
        )
        bracket.loads.append(reading)
        if on_reading:
            on_reading(reading)

    bracket.judge()
    return bracket


# ---------------------------------------------------------------------------


@dataclass
class CanaryReading:
    ok: bool
    latency_ms: float | None = None
    cosine: float | None = None
    error: str = ""


class Canary:
    """A known graph, re-measured in-session to detect the bench moving.

    Deliberately the smallest model in the zoo. The canary's purpose is to be
    boring and cheap — one load, one validate — so that running it before every
    row costs minutes rather than hours, and so that any change in its number
    is about the bench rather than about the model being complicated.

    The first reading of a session becomes the reference; every later one is
    compared against it. Comparing against a figure recorded on a previous day
    would be worse than useless: it would quarantine a whole session over a
    board that simply came up slightly differently.
    """

    def __init__(
        self,
        tc: Toolchain,
        *,
        network_c: Path,
        model: Path,
        profile: str,
        out_dir: Path,
        fix_shapes: str | None = None,
        drift_limit: float = 0.10,
        invokes: int = 10,
    ) -> None:
        self.tc = tc
        self.network_c = network_c
        self.model = model
        self.profile = profile
        self.out_dir = out_dir
        self.fix_shapes = fix_shapes
        self.drift_limit = drift_limit
        self.invokes = invokes
        self.reference: CanaryReading | None = None
        self.readings: list[CanaryReading] = []

    def read(self, label: str = "") -> CanaryReading:
        cycle = self.out_dir / (label or f"reading{len(self.readings) + 1}")
        loaded = measure.load_network(self.tc, self.network_c, log_dir=cycle)
        if not loaded.ok:
            reading = CanaryReading(ok=False, error=loaded.error)
        else:
            result = measure.validate(
                self.tc, self.model, profile=self.profile, out_dir=cycle / "val",
                fix_shapes=self.fix_shapes, batches=self.invokes,
            )
            reading = CanaryReading(
                ok=result.ok, latency_ms=result.latency_ms,
                cosine=result.cosine, error=result.error,
            )
        self.readings.append(reading)
        if self.reference is None and reading.ok:
            self.reference = reading
        return reading

    def drift(self, reading: CanaryReading) -> float | None:
        """Fractional change against the session reference.

        None for the reading that *established* the reference. Returning 0.0
        there would be a drift of zero measured against nothing, and would read
        on the leaderboard as a bench verified steady rather than as the first
        row of the session.
        """
        if (
            self.reference is None
            or reading is self.reference
            or not reading.ok
            or not reading.latency_ms
            or not self.reference.latency_ms
        ):
            return None
        return reading.latency_ms / self.reference.latency_ms - 1.0

    def apply(self, bracket: Bracket, reading: CanaryReading) -> Bracket:
        """Attach this canary reading to a bracket and re-judge it."""
        bracket.canary_ms = reading.latency_ms
        bracket.canary_reference_ms = (
            self.reference.latency_ms if self.reference else None
        )
        bracket.canary_drift = self.drift(reading)
        bracket._drift_limit = self.drift_limit
        bracket.judge()
        return bracket
