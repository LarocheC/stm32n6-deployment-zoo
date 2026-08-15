"""Load a compiled network onto the board and measure it.

Two invariant gates stand between a run and a recorded number, and both exist
because of a specific documented failure rather than caution in general.

**The loader must say it succeeded.** `n6_loader.py` builds ST's
NPU_Validation application around the generated `network.c` and loads it into
AXISRAM over gdb. If that fails quietly, the *previous* firmware is still
resident and the subsequent measurement runs happily against the wrong model,
producing a number that looks entirely reasonable. The success marker is
checked, and a missing marker aborts before `validate` is ever invoked.

**Everything that fails here is infra until proven otherwise.** A wedged
probe, a serial timeout, a stale gdbserver: those are facts about the bench.
They are recorded so board reliability can be measured, and they never become
a model's verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zoo.config import Toolchain
from zoo.st import profiles
from zoo.st.run import gcc_env, run

#: Printed by NPU_Validation once the network is installed and running.
SUCCESS_MARKER = "Start operation achieved successfully"


@dataclass
class LoadResult:
    ok: bool
    duration_s: float = 0.0
    error: str = ""
    log: str = ""

    @property
    def saw_marker(self) -> bool:
        return SUCCESS_MARKER.lower() in self.log.lower()


@dataclass
class ValidateResult:
    ok: bool
    latency_ms: float | None = None
    cosine: float | None = None
    duration_s: float = 0.0
    error: str = ""
    raw: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        out = {"latency_ms": self.latency_ms, "ontarget_cos": self.cosine}
        out.update(self.extra)
        return {k: v for k, v in out.items() if v is not None}


def load_network(
    tc: Toolchain,
    network_c: Path,
    *,
    retries: int = 3,
    timeout_s: float = 900.0,
    log_dir: Path | None = None,
) -> LoadResult:
    """Build and load the validation firmware around a generated network.

    `GCC_PATH` is exported rather than merely set: ST's own
    `NPU_Validation/armgcc/Makefile` opens with a hardcoded Windows path, so a
    link step without it dies with a bare `Error 127` that names nothing.

    Runs with cwd inside ST's `N6_scripts`, because `n6_loader.py` resolves its
    bundled application and build directories relative to itself. This is the
    one place the zoo works inside the vendor tree, and it is unavoidable:
    the loader copies `network.c` into ST's application and builds it there.
    """
    last = LoadResult(ok=False, error="not attempted")
    for attempt in range(1, retries + 1):
        result = run(
            [
                "python3", str(tc.n6_loader),
                "--config", str(tc.n6_scripts / "config.json"),
                "-nf", str(network_c.resolve()),
                "-bc", tc.build_config,
            ],
            cwd=tc.n6_scripts,
            env_extra=gcc_env(Path(tc.raw["gcc"]["path"]).expanduser()),
            timeout_s=timeout_s,
            log_dir=log_dir,
            log_name=f"load-attempt{attempt}",
        )
        loaded = LoadResult(
            ok=False, duration_s=result.duration_s, log=result.combined
        )
        if loaded.saw_marker:
            loaded.ok = True
            return loaded
        loaded.error = (
            f"attempt {attempt}/{retries}: loader did not report "
            f"{SUCCESS_MARKER!r}. Measuring now would time whatever firmware "
            f"is still resident, not this model.\n{result.combined[-1500:]}"
        )
        last = loaded
    return last


#: `validate` prints e.g. "duration     : 2.791 ms by sample" and a per-output
#: cosine line. Both spellings have moved between core releases, so match
#: loosely and record the raw text either way.
_MS = re.compile(r"duration[^\n:]*:\s*([0-9.]+)\s*ms", re.I)
_MS_ALT = re.compile(r"([0-9.]+)\s*ms\s+by\s+sample", re.I)
_COS = re.compile(r"cos(?:ine)?[^\n=:]*[=:]\s*([0-9.]+)", re.I)


def validate(
    tc: Toolchain,
    model: Path,
    *,
    profile: str,
    out_dir: Path,
    fix_shapes: str | None = None,
    batches: int = 4,
    timeout_s: float = 900.0,
) -> ValidateResult:
    """Run the on-target validation: latency, and accuracy against the host."""
    prof = profiles.get(profile, tc.core_tag)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        str(tc.stedgeai), "validate",
        "-m", str(model.resolve()),
        "--target", "stm32n6",
        "--st-neural-art", prof.selector,
        "--mode", "target",
        "-d", f"serial:{tc.serial_port}:{tc.serial_baud}",
        "-b", str(batches),
        "--workspace", str((out_dir / "ws").resolve()),
        "--no-report",
    ]
    if fix_shapes:
        argv += ["--fix-parametric-shapes", fix_shapes]

    result = run(argv, cwd=out_dir, timeout_s=timeout_s, log_dir=out_dir, log_name="validate")
    text = result.combined

    latency = None
    for pattern in (_MS_ALT, _MS):
        match = pattern.search(text)
        if match:
            latency = float(match.group(1))
            break

    cosines = [float(m) for m in _COS.findall(text)]
    cosine = min(cosines) if cosines else None

    return ValidateResult(
        ok=result.ok and latency is not None,
        latency_ms=latency,
        cosine=cosine,
        duration_s=result.duration_s,
        error="" if result.ok else text[-3000:],
        raw=text[-6000:],
    )
