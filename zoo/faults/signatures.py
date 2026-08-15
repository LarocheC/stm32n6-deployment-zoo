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
    symptom: str = ""
    cause: str = ""
    workaround: str = ""
    patch: str = ""
    zoo_action: str = ""
    stage: str = ""
    error_signature: str = ""
    silent: bool = False
    is_infra: bool = False
    detectable_statically: bool = False
    sources: tuple[str, ...] = ()


_KNOWN: list[KnownIssue] | None = None


def known_issues(path: Path | None = None, *, refresh: bool = False) -> list[KnownIssue]:
    """The catalogue, in file order (lint rules first, document-only last).

    A list rather than a dict keyed by failure class: several distinct
    constraints legitimately share a class, and collapsing them would throw
    away the one that happened to be loaded second.
    """
    global _KNOWN
    if _KNOWN is not None and not refresh:
        return _KNOWN
    target = path or KNOWN_ISSUES_PATH
    issues: list[KnownIssue] = []
    if target.is_file():
        with target.open("rb") as fh:
            doc = tomllib.load(fh)
        for entry in doc.get("issue", []):
            issues.append(
                KnownIssue(
                    id=entry.get("id", ""),
                    failure_class=entry.get("failure_class", FC.UNKNOWN),
                    title=" ".join(entry.get("title", "").split()),
                    symptom=" ".join(entry.get("symptom", "").split()),
                    cause=" ".join(entry.get("cause", "").split()),
                    workaround=" ".join(entry.get("workaround", "").split()),
                    patch=entry.get("patch", ""),
                    zoo_action=entry.get("zoo_action", ""),
                    stage=entry.get("stage", ""),
                    error_signature=entry.get("error_signature", ""),
                    silent=bool(entry.get("silent", False)),
                    is_infra=bool(entry.get("is_infra", False)),
                    detectable_statically=bool(entry.get("detectable_statically", False)),
                    sources=tuple(entry.get("sources", [])),
                )
            )
    _KNOWN = issues
    return issues


def by_failure_class(name: str) -> KnownIssue | None:
    for issue in known_issues():
        if issue.failure_class == name:
            return issue
    return None


def match_catalogue(text: str) -> KnownIssue | None:
    """Find the catalogue entry whose recorded error signature appears in `text`.

    Substring, case-insensitive, longest signature first — a longer signature
    is the more specific claim, and a generic one must not shadow it. These
    strings were transcribed from real tool output and verified against their
    sources, so a match here is worth far more than the hand-written regex
    fallback below.
    """
    if not text:
        return None
    candidates = [i for i in known_issues() if i.error_signature]
    for issue in sorted(candidates, key=lambda i: -len(i.error_signature)):
        if _signature_pattern(issue.error_signature).search(text):
            return issue
    return None


_SIG_CACHE: dict[str, re.Pattern[str]] = {}


def _signature_pattern(signature: str) -> re.Pattern[str]:
    """Compile a catalogue signature, treating `*` as a wildcard.

    Several signatures were transcribed with a `*` standing in for the part
    that varies between runs — `value=Pad_*_constant_value` names a node whose
    index differs every time. Matching them literally would mean those entries
    never fire, which is worse than not having them: the atlas would show the
    failure as unexplained while the explanation sat in the file.
    """
    cached = _SIG_CACHE.get(signature)
    if cached is None:
        # Whitespace-insensitive. Signatures were stored with newlines
        # collapsed to spaces, while real tool output spans lines — matching
        # literally meant a two-line signature could never fire, which is how
        # a documented fault reached the board as an unexplained one.
        parts = [re.escape(tok).replace(r"\*", r".*?") for tok in signature.split()]
        cached = re.compile(r"\s+".join(parts), re.I)
        _SIG_CACHE[signature] = cached
    return cached


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
    """Fingerprint a failure and name it.

    The catalogue is consulted first. Its signatures were transcribed from
    real tool output and verified against their sources, so a hit there is
    both more specific and better evidenced than the hand-written regex table,
    and it carries a workaround.
    """
    haystack = text or ""

    issue = match_catalogue(haystack)
    if issue is not None:
        return Classification(
            failure_class=issue.failure_class,
            signature=fingerprint(haystack),
            normalised=normalise(haystack),
            # The catalogue's own judgement about whether this is the bench or
            # the model, rather than a guess from the class name.
            is_infra=issue.is_infra,
            known_issue=issue.id,
            workaround=issue.workaround,
        )

    failure_class = default
    for pattern, klass in RULES:
        if pattern.search(haystack):
            failure_class = klass
            break

    fallback = by_failure_class(failure_class)
    return Classification(
        failure_class=failure_class,
        signature=fingerprint(haystack),
        normalised=normalise(haystack),
        is_infra=is_infra(failure_class) or bool(fallback and fallback.is_infra),
        known_issue=fallback.id if fallback else "",
        workaround=fallback.workaround if fallback else "",
    )


def relative_source(path: str) -> str:
    """Render a source path relative to the repo when it is inside it."""
    try:
        return str(Path(path).relative_to(ROOT))
    except (ValueError, TypeError):
        return str(path)
