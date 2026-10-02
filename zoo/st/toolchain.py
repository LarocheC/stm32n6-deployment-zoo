"""Toolchain discovery, version pinning, and the shim status check.

`zoo doctor` is built on this. The structure follows
`eco8-neaixt:deploy/stm32n6/scripts/doctor.sh`, which got the important
thing right: it is not enough to check that a tool *exists*, because every
tool in this chain has a version whose semantics matter.

  - ST Edge AI Core: the operator mapping, the emitted JSON schema and the
    set of accepted graphs all move between releases. A result is meaningless
    without the build string that produced it, so the version is pinned and a
    mismatch is a hard failure rather than a warning.
  - Arm GNU: ST validates 13.3.Rel1 for `-mcpu=cortex-m55 -mcmse`. Distro
    10.x predates usable M55 support and produces subtly wrong code.
  - STM32CubeProgrammer: 2.21 introduced the mandatory `-align` on signing;
    older versions silently produce an image the ROM will not accept.
"""

from __future__ import annotations

import importlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from zoo.config import Toolchain
from zoo.st import profiles
from zoo.st.run import run

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _exists(name: str, path: Path, *, executable: bool = True) -> Check:
    if not path.exists():
        return Check(name, FAIL, f"not found: {path}")
    if executable and not (path.is_file() and shutil.which(str(path))):
        # shutil.which honours the executable bit for absolute paths.
        return Check(name, FAIL, f"not executable: {path}")
    return Check(name, OK, str(path))


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def check_stedgeai(tc: Toolchain, workdir: Path) -> list[Check]:
    checks = [_exists("stedgeai", tc.stedgeai)]
    if checks[0].failed:
        return checks

    res = run([tc.stedgeai, "--version"], cwd=workdir, timeout_s=120)
    banner = res.stdout.strip().splitlines()[0] if res.stdout.strip() else res.stderr.strip()
    if not res.ok:
        checks.append(Check("stedgeai --version", FAIL, banner or f"exit {res.returncode}"))
        return checks

    if tc.stedgeai_expect_version in banner:
        checks.append(Check("stedgeai version", OK, banner))
    else:
        checks.append(
            Check(
                "stedgeai version",
                FAIL,
                f"expected {tc.stedgeai_expect_version!r}, got {banner!r}. "
                "Results are only comparable within one pinned build; update "
                "config/toolchain.toml deliberately, not incidentally.",
            )
        )
    return checks


def check_atonn(tc: Toolchain) -> list[Check]:
    """atonn presence, plus the BOOL-strip shim's install state.

    The shim matters far beyond convenience. When the quantisation JSON that
    ST's own front end emits contains a BOOL tensor, atonn rejects the file
    and then proceeds *as if no quantisation had been requested at all* --
    producing a model that compiles, runs, and is silently float. The shim
    strips those entries. Whether it is installed therefore changes results,
    so its state is reported here and stamped into every run record.
    """
    checks = [_exists("atonn", tc.atonn)]
    if checks[0].failed:
        return checks

    if tc.atonn_real.exists():
        checks.append(
            Check("atonn shim", OK, f"installed (real binary at {tc.atonn_real.name})")
        )
    else:
        checks.append(
            Check(
                "atonn shim",
                WARN,
                "not installed. Quantisation is silently discarded for graphs "
                "whose *_Q.json contains BOOL tensors (from Equal/Cast). "
                "Install with `zoo shim --install` if a model reports "
                "QUANT_SILENTLY_DISCARDED.",
            )
        )
    return checks


def check_profiles(tc: Toolchain) -> list[Check]:
    checks: list[Check] = []
    try:
        loaded = profiles.load(tc.core_tag)
    except Exception as exc:  # noqa: BLE001 - doctor must report, not raise
        return [Check("profiles", FAIL, f"{type(exc).__name__}: {exc}")]

    missing = [p.name for p in loaded.values() if not p.mpool.is_file()]
    if missing:
        checks.append(Check("profiles", FAIL, f"mpool missing for: {', '.join(missing)}"))
        return checks

    checks.append(
        Check("profiles", OK, f"{len(loaded)} profiles, absolute mpool paths, no cd required")
    )

    # Report each distinct pool's budget. There is no single "on-chip budget"
    # for this part, and confusing three of them is a documented failure mode.
    seen: set[Path] = set()
    for prof in loaded.values():
        if prof.mpool in seen:
            continue
        seen.add(prof.mpool)
        try:
            mm = profiles.read_mpool(prof.mpool)
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"mpool {prof.mpool.name}", FAIL, str(exc)))
            continue
        flash = mm.octoflash
        ram = mm.hyperram
        checks.append(
            Check(
                f"mpool {prof.mpool.name}",
                OK,
                f"on-chip {mm.onchip_bytes // 1024} KB · "
                f"octoFlash {(flash.size_bytes // 1024 // 1024) if flash else 0} MB"
                f"{f' @ 0x{flash.offset:08X}' if flash and flash.size_bytes else ''} · "
                f"hyperRAM {(ram.size_bytes // 1024 // 1024) if ram else 0} MB",
            )
        )
    return checks


def check_gcc(tc: Toolchain, workdir: Path) -> list[Check]:
    checks = [_exists("arm-none-eabi-gcc", tc.arm_gcc)]
    if checks[0].failed:
        return checks
    res = run([tc.arm_gcc, "-dumpversion"], cwd=workdir, timeout_s=60)
    ver = res.stdout.strip()
    if not res.ok or not ver:
        return checks + [Check("gcc version", FAIL, res.combined.strip()[:200])]
    if _version_tuple(ver) >= _version_tuple(tc.gcc_min_version):
        return checks + [Check("gcc version", OK, ver)]
    return checks + [
        Check(
            "gcc version",
            FAIL,
            f"{ver} < {tc.gcc_min_version}; ST validates 13.3.Rel1 for Cortex-M55 "
            "(-mcpu=cortex-m55 -mcmse). Older toolchains miscompile.",
        )
    ]


def check_cube(tc: Toolchain) -> list[Check]:
    return [
        _exists("ST-LINK_gdbserver", tc.gdbserver),
        _exists("STM32_Programmer_CLI", tc.programmer_cli),
        _exists("STM32_SigningTool_CLI", tc.signing_cli),
        _exists("external loader (.stldr)", tc.external_loader, executable=False),
    ]


def check_st_scripts(tc: Toolchain) -> list[Check]:
    return [
        _exists("n6_loader.py", tc.n6_loader, executable=False),
        _exists("npu_profiler.py", tc.npu_profiler, executable=False),
        _exists("operator_support.html", tc.operator_support_html, executable=False),
    ]


def check_python() -> list[Check]:
    checks: list[Check] = []
    for mod in ("onnx", "onnxruntime", "numpy", "huggingface_hub"):
        try:
            m = importlib.import_module(mod)
            checks.append(Check(f"python: {mod}", OK, getattr(m, "__version__", "?")))
        except ImportError:
            checks.append(Check(f"python: {mod}", FAIL, "not importable; run `uv sync`"))
    return checks


def check_network() -> list[Check]:
    try:
        import requests

        r = requests.head("https://huggingface.co", timeout=10)
        status = OK if r.status_code < 500 else WARN
        return [Check("huggingface.co", status, f"HTTP {r.status_code}")]
    except Exception as exc:  # noqa: BLE001
        return [Check("huggingface.co", WARN, f"unreachable: {exc}")]


# ---------------------------------------------------------------------------


def probe_tools(tc: Toolchain, workdir: Path) -> list[Check]:
    """Every board-free check, in report order."""
    checks: list[Check] = []
    checks += check_stedgeai(tc, workdir)
    checks += check_atonn(tc)
    checks += check_profiles(tc)
    checks += check_st_scripts(tc)
    checks += check_gcc(tc, workdir)
    checks += check_cube(tc)
    checks += check_python()
    checks += check_network()
    return checks
