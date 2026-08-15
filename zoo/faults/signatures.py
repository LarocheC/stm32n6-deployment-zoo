"""Normalise tool output into a stable signature, then classify it.

Normalisation is what makes counting possible. Two runs of the same failure
differ in paths, addresses, temporary directory names, tensor indices and
timings; strip those and the same underlying problem hashes to the same
signature every time, across models and across months.

The matching table is ordered and the order matters: infra patterns are tested
first, so that a serial timeout during a compile-then-measure sequence is never
mistaken for a property of the graph.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from zoo.config import ROOT
from zoo.faults.taxonomy import FailureClass, is_infra

KNOWN_ISSUES_PATH = Path(__file__).with_name("known_issues.toml")

# -- normalisation -----------------------------------------------------------

_SCRUB = (
    (re.compile(r"/[^\s'\"]*/"), "<path>/"),                    # absolute paths
    # Raw string: a plain "<path>\\" leaves re.sub a trailing backslash, which
    # it reads as the start of an escape and rejects.
    (re.compile(r"\b[A-Za-z]:\\[^\s'\"]*\\"), r"<path>\\"),      # windows paths
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b"), "<sha>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ][\d:.]+\b"), "<ts>"),
    (re.compile(r"\b\d+\.\d+\s*(ms|s|MB|KB|kB|GB)\b"), "<measure>"),
    (re.compile(r"\bline \d+\b"), "line <n>"),
    # Node-index suffixes: `Slice_17`, `Conv3`, `epoch_65`. `\b\d+\b` misses
    # these because `_` is a word character, and they are exactly the part
    # that varies between two runs of the same underlying failure.
    (re.compile(r"(?<=[A-Za-z_])\d+(?![A-Za-z])"), "<n>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"\s+"), " "),
)

#: Lines that carry no diagnostic weight and only add variance.
_NOISE = re.compile(
    r"^\s*(ST Edge AI Core|elapsed time|Exporting|Importing|Setting|\[zoo\]|#|\$)",
    re.I,
)


def normalise(text: str, *, max_lines: int = 12) -> str:
    """Reduce a log to the part that identifies the failure."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not _NOISE.match(ln)]

    # Prefer lines that look like diagnostics; fall back to the tail, which is
    # where a crash leaves its last words.
    diag = [
        ln
        for ln in lines
        if re.search(r"\b(error|failed|failure|unsupported|invalid|cannot|unable|"
                     r"not implemented|internal|segmentation|signo|traceback|"
                     r"unallocatable|mismatch)\b", ln, re.I)
    ]
    chosen = (diag or lines)[-max_lines:]

    out = " | ".join(chosen)
    for pattern, replacement in _SCRUB:
        out = pattern.sub(replacement, out)
    return out.strip()[:1200]


def fingerprint(text: str) -> str:
    """Stable short id for a normalised failure."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:12]


# -- classification ----------------------------------------------------------

FC = FailureClass

#: Ordered (pattern, class) table. Infra first, deliberately: a board problem
#: seen during a model's run must not be attributed to the model.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # ---- infrastructure -------------------------------------------------
    (re.compile(r"DEV_USB_COMM_ERR|Loading memories failed", re.I), FC.STLINK_WEDGED),
    (re.compile(r"No STM32 target found|ST-LINK error|Error: No debug probe", re.I), FC.BOARD_NOT_ATTACHED),
    (re.compile(r"E801\(HwIOError\)|Invalid firmware", re.I), FC.LOAD_FAILED),
    (re.compile(r"Unable to bind the ST\.AI runtime", re.I), FC.LOAD_FAILED),
    (re.compile(r"\btimed out\b|TimeoutError|serial.*timeout", re.I), FC.SERIAL_TIMEOUT),
    (re.compile(r"no such file or directory:.*(stedgeai|atonn|arm-none-eabi)", re.I), FC.TOOLCHAIN_MISCONFIGURED),

    # ---- compiler crashes ----------------------------------------------
    (re.compile(r"signo=11|Segmentation fault|SIGSEGV", re.I), FC.FRONTEND_CRASH),

    # ---- front end ------------------------------------------------------
    (re.compile(r"Error in computation of shapes", re.I), FC.FRONTEND_SHAPE_ERROR),
    (re.compile(r"Mismatch in channel position", re.I), FC.FRONTEND_CHANNEL_POSITION),
    (re.compile(r"missing shape or size for value", re.I), FC.FRONTEND_SHAPE_ERROR),
    (re.compile(r"not implemented shape len for Conversion", re.I), FC.FRONTEND_IMPORT_ERROR),
    (re.compile(r"input of this model is not quantized", re.I), FC.QUANT_FAILED),
    (re.compile(r"quantized unsigned integer format is not supported", re.I), FC.QUANT_FAILED),
    (re.compile(r"Ignore Quantization JSON file", re.I), FC.QUANT_SILENTLY_DISCARDED),
    (re.compile(r"invalid value \"BOOL\"", re.I), FC.QUANT_SILENTLY_DISCARDED),
    (re.compile(r"INTERNAL ERROR", re.I), FC.CODEGEN_ERROR),
    (re.compile(r"TOOL ERROR", re.I), FC.FRONTEND_IMPORT_ERROR),

    # ---- allocation -----------------------------------------------------
    (re.compile(r"unallocatable|cannot allocate|out of memory", re.I), FC.ALLOC_FAILED_ALL),

    # ---- generic --------------------------------------------------------
    (re.compile(r"\bunsupported\b.*\boperator\b|operator .* not supported", re.I), FC.OP_UNSUPPORTED),
)


@dataclass
class KnownIssue:
    id: str
    failure_class: str
    title: str = ""
    cause: str = ""
    workaround: str = ""
    patch: str = ""
    zoo_action: str = ""
    silent: bool = False
    sources: tuple[str, ...] = ()


_KNOWN: dict[str, KnownIssue] | None = None


def known_issues(path: Path | None = None, *, refresh: bool = False) -> dict[str, KnownIssue]:
    """Catalogue keyed by failure class, loaded from `known_issues.toml`."""
    global _KNOWN
    if _KNOWN is not None and not refresh:
        return _KNOWN
    target = path or KNOWN_ISSUES_PATH
    issues: dict[str, KnownIssue] = {}
    if target.is_file():
        with target.open("rb") as fh:
            doc = tomllib.load(fh)
        for entry in doc.get("issue", []):
            issue = KnownIssue(
                id=entry.get("id", ""),
                failure_class=entry.get("failure_class", FC.UNKNOWN),
                title=entry.get("title", ""),
                cause=entry.get("cause", ""),
                workaround=entry.get("workaround", ""),
                patch=entry.get("patch", ""),
                zoo_action=entry.get("zoo_action", ""),
                silent=bool(entry.get("silent", False)),
                sources=tuple(entry.get("sources", [])),
            )
            issues[issue.failure_class] = issue
    _KNOWN = issues
    return issues


@dataclass
class Classification:
    failure_class: str
    signature: str
    normalised: str
    is_infra: bool
    known_issue: str = ""
    workaround: str = ""

    @property
    def is_new(self) -> bool:
        """No catalogue entry — this is the discovery queue."""
        return not self.known_issue


def classify(text: str, *, default: str = FC.UNKNOWN) -> Classification:
    """Fingerprint a failure and name it."""
    haystack = text or ""
    failure_class = default
    for pattern, klass in RULES:
        if pattern.search(haystack):
            failure_class = klass
            break

    issue = known_issues().get(failure_class)
    return Classification(
        failure_class=failure_class,
        signature=fingerprint(haystack),
        normalised=normalise(haystack),
        is_infra=is_infra(failure_class),
        known_issue=issue.id if issue else "",
        workaround=issue.workaround if issue else "",
    )


def relative_source(path: str) -> str:
    """Render a source path relative to the repo when it is inside it."""
    try:
        return str(Path(path).relative_to(ROOT))
    except (ValueError, TypeError):
        return str(path)
