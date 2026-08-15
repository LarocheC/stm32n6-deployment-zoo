"""Drive `stedgeai` — analyze for the front-end gate, generate for the compile.

Every invocation runs with an explicit `cwd` set to its own output directory
and an absolute profile path, so the compiler's scratch directories land where
they belong and nothing has to `cd` into the vendor install first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zoo.config import Toolchain
from zoo.st import cinfo, profiles
from zoo.st.run import RunResult, run


@dataclass
class CompileOutcome:
    profile: str
    ok: bool
    out_dir: Path
    info: cinfo.CompileInfo | None = None
    error: str = ""
    duration_s: float = 0.0
    argv: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        base = {"profile_used": self.profile}
        if self.info:
            base.update(self.info.metrics())
        return base


def _common(tc: Toolchain, model: Path, out_dir: Path, *, name: str = "network") -> list[str]:
    # Absolute, always. The runner sets cwd to the per-run output directory so
    # the compiler's scratch lands there, which means any relative path handed
    # to stedgeai would be resolved against that directory rather than the
    # caller's.
    return [
        str(tc.stedgeai),
        "generate",
        "-m", str(model.resolve()),
        "--target", "stm32n6",
        "-n", name,
        "-o", str(out_dir.resolve()),
        "--workspace", str((out_dir / "ws").resolve()),
        "--no-report",
    ]


def analyze(
    tc: Toolchain,
    model: Path,
    out_dir: Path,
    *,
    fix_shapes: str | None = None,
    timeout_s: float = 900.0,
) -> RunResult:
    """Front-end import gate, and the Cortex-M55 baseline in one run.

    Omitting `--st-neural-art` selects ST's classical STM32 flow — the
    Cortex-M55 alone — which their FAQ states explicitly. So this both proves
    the ONNX front end accepted the graph and yields the CPU-only comparison
    for free, before a single NPU-specific option is involved.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(tc.stedgeai), "analyze",
        "-m", str(model.resolve()),
        "--target", "stm32n6",
        "--workspace", str((out_dir / "ws").resolve()),
        "--no-report",
    ]
    if fix_shapes:
        argv += ["--fix-parametric-shapes", fix_shapes]
    return run(argv, cwd=out_dir, timeout_s=timeout_s, log_dir=out_dir, log_name="analyze")


def generate(
    tc: Toolchain,
    model: Path,
    out_dir: Path,
    *,
    profile: str,
    fix_shapes: str | None = None,
    extra: list[str] | None = None,
    timeout_s: float = 1800.0,
) -> CompileOutcome:
    """Compile for the Neural-ART NPU with one named profile."""
    prof = profiles.get(profile, tc.core_tag)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = _common(tc, model, out_dir)
    argv += ["--st-neural-art", prof.selector]
    if fix_shapes:
        argv += ["--fix-parametric-shapes", fix_shapes]
    argv += extra or []

    result = run(argv, cwd=out_dir, timeout_s=timeout_s, log_dir=out_dir, log_name=f"gen-{profile}")

    info = None
    info_path = cinfo.find(out_dir)
    if info_path:
        try:
            info = cinfo.parse(info_path)
        except Exception as exc:  # noqa: BLE001
            return CompileOutcome(
                profile, False, out_dir,
                error=f"{result.combined[-2000:]}\n[zoo] c_info parse failed: {exc}",
                duration_s=result.duration_s, argv=result.argv,
            )

    # The compiler can exit non-zero after emitting a usable c_info, and can
    # exit zero having emitted nothing. Trust the artifact, not the code.
    ok = bool(info) and result.ok
    return CompileOutcome(
        profile=profile,
        ok=ok,
        out_dir=out_dir,
        info=info,
        error="" if ok else result.combined[-4000:],
        duration_s=result.duration_s,
        argv=result.argv,
    )


def generate_ladder(
    tc: Toolchain,
    model: Path,
    base_dir: Path,
    *,
    ladder: tuple[str, ...] = profiles.SCREENING_LADDER,
    fix_shapes: str | None = None,
) -> tuple[CompileOutcome, list[CompileOutcome]]:
    """Walk the profile ladder, stopping at the first rung that compiles.

    Which rung succeeds is itself the result. `onchip` means weights and
    activations both fit internal memory; falling through to `extflash` means
    weights stream from external flash at roughly a 1.6x latency cost; reaching
    `extram` means activations spilled, which measured 5-20x on ST's own
    numbers. Recording the whole walk keeps that distinction visible instead of
    collapsing it into a single "it compiled".
    """
    attempts: list[CompileOutcome] = []
    for profile in ladder:
        outcome = generate(
            tc, model, base_dir / profile, profile=profile, fix_shapes=fix_shapes
        )
        attempts.append(outcome)
        if outcome.ok:
            return outcome, attempts
    return attempts[-1], attempts
