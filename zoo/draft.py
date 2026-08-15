"""`zoo init` — draft a recipe from a Hugging Face id.

This exists because recipe authoring, not compilation, is the real bottleneck.
Hand-specifying per-input names, shapes, dtypes and roles for thirty
heterogeneous models costs more wall-clock than every compile combined, and it
is exactly the kind of transcription that produces silent errors. The drafter
reads the graph and fills in everything mechanical, leaving the human about
five judgement calls per model: the real-time window, the tier, the
calibration provider, any pin value, and any role it guessed wrong.

Every guess is marked. A `TODO` in the output is a question the tool could not
answer, not a placeholder to be ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from zoo import fetch
from zoo.graph import probe

#: Inputs whose name looks like recurrent state carried across invocations.
_STATE_RE = re.compile(r"(^|_)(state|past|h0|c0|hidden|cache|fifo|mem)(_|$|\d)", re.I)
_MASK_RE = re.compile(r"mask", re.I)
_TOKEN_RE = re.compile(r"(input_ids|token|position_ids)", re.I)

#: HF pipeline tags mapped to the zoo's coarse domains.
_DOMAIN = {
    "audio-classification": "audio",
    "automatic-speech-recognition": "audio",
    "voice-activity-detection": "audio",
    "audio-to-audio": "audio",
    "text-to-speech": "audio",
    "image-classification": "vision",
    "object-detection": "vision",
    "image-segmentation": "vision",
    "depth-estimation": "vision",
    "keypoint-detection": "vision",
    "image-feature-extraction": "vision",
    "feature-extraction": "embeddings",
}


@dataclass
class DraftedGraph:
    file: str
    graph_id: str
    size: int | None
    probe: probe.Probe | None
    scan: probe.ScanResult | None
    skip_reason: str | None


def _infer_role(spec: probe.TensorSpec, all_inputs: list[probe.TensorSpec]) -> str:
    if spec.dtype in ("int64", "int32") and spec.rank <= 1:
        # A rank-0/1 integer input is a configuration scalar, not data. Folding
        # it into an initializer is also what removes the rank-1-graph-input
        # rejection that ST's front end raises on a large class of exports.
        return "constant"
    if _MASK_RE.search(spec.name):
        return "mask"
    if _TOKEN_RE.search(spec.name):
        return "token"
    if _STATE_RE.search(spec.name):
        return "state"
    float_inputs = [s for s in all_inputs if s.dtype.startswith(("float", "bfloat"))]
    if float_inputs:
        largest = max(float_inputs, key=lambda s: (s.numel() or 0, s.rank))
        if spec.name == largest.name:
            return "feature"
    return "feature"


def _guess_feeds_from(
    state: probe.TensorSpec, outputs: list[probe.TensorSpec], claimed: set[str]
) -> str | None:
    """Pair a state input with the output that carries its next value.

    Tried in order of how much evidence each rule carries: an explicit naming
    convention beats a shape match, because two unrelated states often share a
    shape and pairing them wrongly produces a model that runs and is wrong.
    """
    free = [o for o in outputs if o.name not in claimed]
    base = re.sub(r"(_in|_input|In)$", "", state.name)
    for suffix in ("N", "_out", "_output", "Out", ""):
        for out in free:
            if out.name == f"{base}{suffix}":
                return out.name
    for out in free:
        if base.lower() in out.name.lower():
            return out.name
    same_shape = [o for o in free if o.shape == state.shape and o.dtype == state.dtype]
    if len(same_shape) == 1:
        return same_shape[0].name
    return None


def _toml_shape(shape: list[int | str | None]) -> str:
    parts = []
    for dim in shape:
        parts.append(str(dim) if isinstance(dim, int) else f'"{dim if dim else "?"}"')
    return "[" + ", ".join(parts) + "]"


def _fetch_metadata(repo: str) -> dict:
    import requests

    try:
        resp = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
        return resp.json() if resp.ok else {}
    except Exception:  # noqa: BLE001 - metadata is a nicety, not a requirement
        return {}


def collect(
    repo: str, *, revision: str = "main", max_mb: float = 400.0
) -> list[DraftedGraph]:
    """Probe every fp32 ONNX in a repo, downloading only what is small enough."""
    remotes = [
        f
        for f in fetch.list_hf_onnx(repo, revision)
        if not f.is_prequantised and not f.is_external_data
    ]
    out: list[DraftedGraph] = []
    for remote in remotes:
        size_mb = (remote.size or 0) / 1e6
        graph_id = re.sub(r"[^a-z0-9]+", "-", remote.stem.lower()).strip("-") or "graph"
        if remote.size is not None and size_mb > max_mb:
            # Too big to pull just to draft a recipe, but a head scan still
            # yields the operator census — which is usually enough to know
            # whether it was ever worth pulling.
            scan = probe.probe_url(remote.url)
            out.append(
                DraftedGraph(
                    file=remote.filename,
                    graph_id=graph_id,
                    size=remote.size,
                    probe=None,
                    scan=scan,
                    skip_reason=(
                        f"{size_mb:.0f} MB exceeds the --max-mb draft limit; "
                        "I/O signature not read. Op census came from a head scan."
                    ),
                )
            )
            continue

        local = fetch.download(remote, revision)
        try:
            p = probe.probe_file(local)
        except Exception as exc:  # noqa: BLE001
            out.append(
                DraftedGraph(
                    file=remote.filename,
                    graph_id=graph_id,
                    size=remote.size,
                    probe=None,
                    scan=None,
                    skip_reason=f"probe failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        out.append(
            DraftedGraph(
                file=remote.filename,
                graph_id=graph_id,
                size=remote.size,
                probe=p,
                scan=None,
                skip_reason=None,
            )
        )
    return out


def render(repo: str, drafted: list[DraftedGraph], *, revision: str = "main") -> tuple[str, str]:
    """Return `(toml_text, domain)`."""
    meta = _fetch_metadata(repo)
    pipeline = meta.get("pipeline_tag") or ""
    domain = _DOMAIN.get(pipeline, "misc")
    slug = repo.split("/")[-1].lower()
    licence = (meta.get("cardData") or {}).get("license") or meta.get("license") or "TODO"

    lines: list[str] = [
        "# Drafted by `zoo init`. Every TODO is a question the drafter could not",
        "# answer from the graph — review them before running the funnel.",
        "schema = 1",
        f'id      = "{slug}"',
        f'source  = "{repo}"',
        f'domain  = "{domain}"',
        f'task    = "{pipeline or "TODO"}"',
        "tier    = 1        # 1=screen  2=measure on board  3=firmware demo",
        f'license = "{licence}"',
        f'notes   = "lab/{slug}.md"',
        "",
        "[fetch]",
        'provider = "hf_onnx"',
        f'revision = "{revision}"   # TODO: pin to a commit sha once it screens clean',
        "",
    ]

    for item in drafted:
        lines.append('[[graph]]')
        lines.append(f'id   = "{item.graph_id}"')
        lines.append(f'file = "{item.file}"')
        if item.size:
            lines.append(f"# fp32 on disk: {item.size / 1e6:.1f} MB")

        if item.probe is None:
            lines.append("enabled     = false")
            lines.append(f'skip_reason = "{item.skip_reason}"')
            if item.scan and item.scan.op_types:
                top = ", ".join(f"{k}:{v}" for k, v in item.scan.op_types.most_common(10))
                lines.append(f"# head-scan ops: {top}")
                if item.scan.control_flow_ops:
                    lines.append(
                        f"# CONTROL FLOW at top level: {dict(item.scan.control_flow_ops)} "
                        "— the real graph is inside subgraphs and cannot be compiled."
                    )
            lines.append("")
            continue

        p = item.probe
        lines.append("realtime_ms = 0.0   # TODO: wall-clock covered by one invocation (for RTF)")

        unpinned = sorted({d for s in p.inputs for d in s.symbolic_dims})
        anon = [(s.name, ax) for s in p.inputs for ax, nm in s.dynamic_axes if nm is None]
        if unpinned:
            lines.append("")
            lines.append("[graph.pin]   # TODO: every symbolic dim must be fixed before compiling")
            for dim in unpinned:
                default = 1 if dim.lower() in ("batch", "batch_size", "b", "n") else 0
                suffix = "" if default else "   # TODO"
                lines.append(f'"{dim}" = {default}{suffix}')
        if anon:
            lines.append(
                "# WARNING: anonymous dynamic axes "
                + ", ".join(f"{n}[{a}]" for n, a in anon)
                + " — --fix-parametric-shapes keys on names, so these need a re-export."
            )

        claimed: set[str] = set()
        for spec in p.inputs:
            role = _infer_role(spec, p.inputs)
            lines.append("")
            lines.append("  [[graph.input]]")
            lines.append(f'  name  = "{spec.name}"')
            lines.append(f'  role  = "{role}"')
            lines.append(f"  shape = {_toml_shape(spec.shape)}")
            lines.append(f'  dtype = "{spec.dtype}"')
            if role == "state":
                pair = _guess_feeds_from(spec, p.outputs, claimed)
                if pair:
                    claimed.add(pair)
                    lines.append(f'  feeds_from = "{pair}"')
                else:
                    lines.append('  feeds_from = "TODO"   # no matching output found')
                lines.append('  init = "zeros"')
            elif role == "constant":
                lines.append("  value = 0   # TODO: the scalar to fold in")
            elif role == "feature":
                lines.append('  calib = "TODO"   # calibration provider key')

        for spec in p.outputs:
            lines.append("")
            lines.append("  [[graph.output]]")
            lines.append(f'  name = "{spec.name}"')
            lines.append(f"  # shape {_toml_shape(spec.shape)}")

        lines.append("")

    lines += [
        "[variants]",
        'profile         = ["onchip", "extflash", "allmems"]',
        'precision       = ["int8"]',
        'backend         = ["npu", "m55"]   # m55 gives the NPU-vs-CPU speedup',
        'policy          = "first_success"',
        "",
        "[calibration]",
        'provider = "synthetic"   # TODO: a real provider is required for tier 2',
        "n        = 128",
        "seed     = 0",
        "",
    ]
    return "\n".join(lines), domain


@dataclass
class DraftResult:
    repo: str
    text: str
    domain: str
    graphs: list[DraftedGraph]

    @property
    def todo_count(self) -> int:
        return self.text.count("TODO")

    def path_in(self, models_dir: Path) -> Path:
        slug = self.repo.split("/")[-1].lower()
        return models_dir / self.domain / f"{slug}.toml"


def draft(repo: str, *, revision: str = "main", max_mb: float = 400.0) -> DraftResult:
    graphs = collect(repo, revision=revision, max_mb=max_mb)
    text, domain = render(repo, graphs, revision=revision)
    return DraftResult(repo=repo, text=text, domain=domain, graphs=graphs)
