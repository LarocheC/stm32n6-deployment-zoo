"""Materialise the zoo's Neural-ART compilation profiles with absolute paths.

Why this module exists at all:

ST ships `user_neuralart.json` with *relative* memory-pool paths
(`"./my_mpools/stm32n6.mpool"`). Because atonn resolves them relative to the
profile file, every `stedgeai --st-neural-art` invocation in both prior
projects on this machine had to begin with

    cd ~/stedgeai/install/4.0/scripts/N6_scripts

which in turn meant the compiler's scratch directories landed inside the
vendor install, the profiles could not be version-controlled, and adding a
debug profile meant editing ST's tree in place.

ST's own header states that the `memory_pool` value may be absolute. So the
zoo keeps `config/profiles/zoo_neuralart.json.in` as a template, substitutes
an absolute `{{MPOOL_DIR}}` at materialisation time, and writes the result
under `runs/_toolchain/<core>/`. After that, `--st-neural-art <name>@<abs>` is
callable from any working directory and the vendor install stays pristine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from zoo.config import CONFIG_DIR, RUNS_DIR

TEMPLATE = CONFIG_DIR / "profiles" / "zoo_neuralart.json.in"
MPOOL_DIR = CONFIG_DIR / "profiles" / "mpools"

# The profile ladder tried in order by the compile stage. Which rung a model
# lands on IS a result -- "needs external memory" is a finding, not a failure.
SCREENING_LADDER = ("onchip", "extflash", "extram", "allmems")


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    mpool: Path
    options: str
    json_path: Path

    @property
    def selector(self) -> str:
        """The literal `--st-neural-art` argument."""
        return f"{self.name}@{self.json_path}"


def _strip_json_comments(text: str) -> str:
    """ST's profile and mpool files allow `//` comments. Python's json does not.

    Only line comments are used by ST, and no string literal in these files
    contains `//` (all paths are relative or absolute POSIX paths without a
    scheme), so a line-oriented strip is safe here.
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        # Trailing comment after a value.
        idx = line.find("//")
        if idx >= 0 and line.count('"', 0, idx) % 2 == 0:
            line = line[:idx].rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)


def load_json_with_comments(path: Path) -> dict:
    import json

    return json.loads(_strip_json_comments(path.read_text()))


def materialise(core_tag: str = "4.0", force: bool = False) -> Path:
    """Write the absolute-path profile JSON and return its path."""
    if not TEMPLATE.is_file():
        raise ProfileError(f"missing profile template: {TEMPLATE}")

    out_dir = RUNS_DIR / "_toolchain" / core_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zoo_neuralart.json"

    rendered = TEMPLATE.read_text().replace("{{MPOOL_DIR}}", str(MPOOL_DIR.resolve()))
    if "{{" in rendered:
        leftover = set(re.findall(r"\{\{(\w+)\}\}", rendered))
        raise ProfileError(f"unsubstituted placeholders in template: {sorted(leftover)}")

    if force or not out_path.is_file() or out_path.read_text() != rendered:
        out_path.write_text(rendered)
    return out_path


def load(core_tag: str = "4.0") -> dict[str, Profile]:
    """Materialise if needed, then parse into Profile objects."""
    json_path = materialise(core_tag)
    doc = load_json_with_comments(json_path)
    profiles: dict[str, Profile] = {}
    for name, spec in doc.get("Profiles", {}).items():
        profiles[name] = Profile(
            name=name,
            mpool=Path(spec["memory_pool"]),
            options=spec.get("options", ""),
            json_path=json_path,
        )
    if not profiles:
        raise ProfileError(f"{json_path} defines no profiles")
    return profiles


def get(name: str, core_tag: str = "4.0") -> Profile:
    profiles = load(core_tag)
    try:
        return profiles[name]
    except KeyError:
        raise ProfileError(
            f"unknown profile {name!r}; available: {', '.join(sorted(profiles))}"
        ) from None


# ---------------------------------------------------------------------------
# Memory pools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemPool:
    name: str
    offset: int
    size_bytes: int
    read_only: bool


@dataclass(frozen=True)
class MemMap:
    """Parsed .mpool file.

    The reason this is parsed rather than hardcoded: there is no single
    "on-chip budget" for this part. The screening harness pool exposes
    ~2816 KB; ST's vision application pool exposes the same but puts octoFlash
    at 0x70380000; ST's audio application pool exposes only 1472 KB and puts
    octoFlash at 0x70180000. A prior project generated against one pool and
    flashed using another's base address, which looked for months like a bug
    in the makefile. Reading the address out of the pool file that was
    actually used makes that class of mistake unrepresentable.
    """

    path: Path
    pools: tuple[MemPool, ...]

    def by_name(self, name: str) -> MemPool | None:
        for p in self.pools:
            if p.name == name:
                return p
        return None

    @property
    def onchip_bytes(self) -> int:
        """Sum of the internal AXISRAM pools the NPU compiler may use."""
        return sum(
            p.size_bytes
            for p in self.pools
            if p.name.startswith(("cpuRAM", "npuRAM", "flexMEM"))
        )

    @property
    def octoflash(self) -> MemPool | None:
        return self.by_name("octoFlash")

    @property
    def hyperram(self) -> MemPool | None:
        return self.by_name("hyperRAM")

    @property
    def weights_base(self) -> int | None:
        """Base address for the weight blob -- the objcopy/programmer offset.

        Read from the pool that was actually compiled against. Never typed by
        hand, never inherited from another target.
        """
        pool = self.octoflash
        return pool.offset if pool and pool.size_bytes else None


_MAGNITUDE = {
    "BYTES": 1,
    "KBYTES": 1024,
    "MBYTES": 1024 * 1024,
    "GBYTES": 1024 * 1024 * 1024,
}


def read_mpool(path: Path) -> MemMap:
    doc = load_json_with_comments(path)
    pools: list[MemPool] = []
    for entry in doc.get("memory", {}).get("mempools", []):
        size = entry.get("size", {})
        magnitude = _MAGNITUDE.get(str(size.get("magnitude", "BYTES")).upper(), 1)
        rights = str(entry.get("prop", {}).get("rights", ""))
        pools.append(
            MemPool(
                name=entry.get("name", entry.get("fname", "?")),
                offset=int(str(entry.get("offset", {}).get("value", "0")), 0),
                size_bytes=int(str(size.get("value", "0")), 0) * magnitude,
                read_only="WRITE" not in rights.upper(),
            )
        )
    return MemMap(path=path, pools=tuple(pools))
