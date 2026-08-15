"""Identity and cache keys.

Getting these wrong is the quietest way to poison a leaderboard: a stale
cached result looks exactly like a fresh one. So a cache key covers not only
the inputs but the *code that produced the output* — editing `budget.py` must
invalidate every budget result, without anyone remembering to say so.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford: no I, L, O, U


def new_run_id() -> str:
    """Sortable, collision-resistant id: millisecond timestamp + randomness.

    Lexicographic order matches chronological order, which makes an
    append-only log readable without parsing timestamps.
    """
    ms = int(time.time() * 1000)
    head = ""
    for _ in range(10):
        head = _B32[ms & 0x1F] + head
        ms >>= 5
    tail = "".join(_B32[b & 0x1F] for b in os.urandom(10))
    return head + tail


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(digest: str, n: int = 12) -> str:
    return digest[:n]


def variant_id(
    *,
    profile: str,
    precision: str = "int8",
    input_data_type: str = "float32",
    backend: str = "npu",
) -> str:
    """Stable label for one point in the variant sweep.

    Slash-separated rather than JSON so it reads in a table and sorts
    sensibly: `onchip/int8/in-f32/npu`.
    """
    dt = {"float32": "in-f32", "int8": "in-i8", "uint8": "in-u8"}.get(
        input_data_type, f"in-{input_data_type}"
    )
    return f"{profile}/{precision}/{dt}/{backend}"


def module_fingerprint(*modules: Any) -> str:
    """Hash the source of the modules a stage's output depends on.

    Deliberately hashes file *contents* rather than a git sha: an uncommitted
    edit to a stage must invalidate that stage's cache too, otherwise the
    fastest way to get a wrong leaderboard is to tweak a threshold and re-run.
    """
    digest = hashlib.sha256()
    for module in sorted(modules, key=lambda m: getattr(m, "__name__", str(m))):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source)
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_key(
    *,
    stage: str,
    input_sha: str,
    params: dict[str, Any],
    tool_versions: dict[str, str],
    code_fingerprint: str,
) -> str:
    """One key covering inputs, parameters, tool versions and code."""
    import json

    payload = json.dumps(
        {
            "stage": stage,
            "input": input_sha,
            "params": params,
            "tools": tool_versions,
            "code": code_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)
