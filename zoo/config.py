"""Repo layout, machine-local tool paths, and funnel thresholds.

Two config files, deliberately separate:

  config/toolchain.toml  -- WHERE the tools are. Machine-local, gitignored.
  config/policy.toml     -- WHAT counts as a pass. Committed, and its git sha
                            is part of every cache key, because changing a
                            threshold changes verdicts.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unusable."""


def repo_root() -> Path:
    """Directory holding pyproject.toml, found by walking up from this file."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"no pyproject.toml above {here}")


ROOT = repo_root()
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


def _env_override(section: str, key: str) -> str | None:
    """ZOO_<SECTION>_<KEY>, upper snake case."""
    return os.environ.get(f"ZOO_{section.upper()}_{key.upper()}")


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


@dataclass(frozen=True)
class Toolchain:
    """Resolved absolute paths to every external tool the zoo drives.

    Nothing here is validated at construction time -- `zoo doctor` is what
    reports on it, and it must be able to report on a *broken* config rather
    than refusing to load one.
    """

    # ST Edge AI Core
    stedgeai_root: Path
    stedgeai_expect_version: str
    n6_scripts: Path
    ai_runner: Path
    operator_support_html: Path

    # Arm GNU toolchain
    gcc_path: Path
    gcc_min_version: str

    # STM32Cube tooling
    cubeclt_root: Path
    cubeprog_root: Path
    external_loader_rel: str

    # Board
    serial_port: str
    serial_baud: int
    build_config: str
    usbipd: str
    usbipd_busid: str

    raw: dict = field(repr=False, default_factory=dict)

    # -- derived paths -----------------------------------------------------

    @property
    def stedgeai(self) -> Path:
        return self.stedgeai_root / "Utilities" / "linux" / "stedgeai"

    @property
    def atonn(self) -> Path:
        return self.stedgeai_root / "Utilities" / "linux" / "atonn"

    @property
    def atonn_real(self) -> Path:
        """Where the BOOL-strip shim moves the genuine binary."""
        return self.stedgeai_root / "Utilities" / "linux" / "atonn.real"

    @property
    def n6_loader(self) -> Path:
        return self.n6_scripts / "n6_loader.py"

    @property
    def npu_profiler(self) -> Path:
        return self.ai_runner / "examples" / "npu_profiler.py"

    @property
    def gdbserver(self) -> Path:
        return self.cubeclt_root / "STLink-gdb-server" / "bin" / "ST-LINK_gdbserver"

    @property
    def programmer_cli(self) -> Path:
        return self.cubeprog_root / "bin" / "STM32_Programmer_CLI"

    @property
    def signing_cli(self) -> Path:
        return self.cubeprog_root / "bin" / "STM32_SigningTool_CLI"

    @property
    def external_loader(self) -> Path:
        return self.cubeprog_root / "bin" / self.external_loader_rel

    @property
    def arm_gcc(self) -> Path:
        return self.gcc_path / "arm-none-eabi-gcc"

    @property
    def core_tag(self) -> str:
        """Short label for the pinned core, used in cache and run paths."""
        return self.stedgeai_root.name  # e.g. "4.0"


def load_toolchain(path: Path | None = None) -> Toolchain:
    path = path or (CONFIG_DIR / "toolchain.toml")
    if not path.is_file():
        example = CONFIG_DIR / "toolchain.example.toml"
        raise ConfigError(
            f"{path} not found.\n"
            f"  cp {example.relative_to(ROOT)} {path.relative_to(ROOT)}\n"
            f"  then edit it for this machine and run: zoo doctor"
        )
    raw = _read_toml(path)

    def get(section: str, key: str, default=None):
        env = _env_override(section, key)
        if env is not None:
            return env
        try:
            return raw[section][key]
        except KeyError:
            if default is None:
                raise ConfigError(f"{path}: missing [{section}] {key}") from None
            return default

    return Toolchain(
        stedgeai_root=Path(get("stedgeai", "root")).expanduser(),
        stedgeai_expect_version=str(get("stedgeai", "expect_version")),
        n6_scripts=Path(get("stedgeai", "n6_scripts")).expanduser(),
        ai_runner=Path(get("stedgeai", "ai_runner")).expanduser(),
        operator_support_html=Path(get("stedgeai", "operator_support_html")).expanduser(),
        gcc_path=Path(get("gcc", "path")).expanduser(),
        gcc_min_version=str(get("gcc", "min_version", "13.0")),
        cubeclt_root=Path(get("cubeclt", "root")).expanduser(),
        cubeprog_root=Path(get("cubeprog", "root")).expanduser(),
        external_loader_rel=str(get("cubeprog", "external_loader")),
        serial_port=str(get("board", "serial_port", "/dev/ttyACM0")),
        serial_baud=int(get("board", "serial_baud", 921600)),
        build_config=str(get("board", "build_config", "N6-DK")),
        usbipd=str(get("board", "usbipd", "")),
        usbipd_busid=str(get("board", "usbipd_busid", "")),
        raw=raw,
    )


def load_policy(path: Path | None = None) -> dict:
    return _read_toml(path or (CONFIG_DIR / "policy.toml"))
