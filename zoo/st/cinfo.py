"""Parse `network_c_info.json` — the compiler's own account of what it built.

Both prior projects on this machine read `network_generate_report.txt` by eye.
The compiler also emits a typed JSON next to it, and everything the funnel
needs is in there: per-epoch hardware/software mapping, per-pool occupancy,
MACs, and — less obviously — per-node cycle estimates, which give a **predicted
latency without a board**.

That last one changes the shape of the project. It means the compile stage
produces a latency column on its own, and the board stage becomes a check on
the prediction rather than the only source of numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: `graphs[].nodes[].mapping` values.
NODE_HW = "NODE_HW"
NODE_SW = "NODE_SW"
NODE_NO_X = "NODE_NO_X"  # no execution: bookkeeping epochs


@dataclass
class Pool:
    name: str
    size_bytes: int
    used_bytes: int
    address: int = 0

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.size_bytes if self.size_bytes else 0.0


@dataclass
class Epoch:
    id: int
    name: str
    mapping: str
    macc: int = 0
    #: The ops behind a software epoch — what to attack to remove it.
    ops: tuple[str, ...] = ()
    compute_cycles: int = 0
    max_cycles: int = 0

    @property
    def is_software(self) -> bool:
        return self.mapping == NODE_SW


@dataclass
class CompileInfo:
    path: Path
    #: `memory_footprint.weights`. Read the docstring on `param_bytes` before
    #: quoting this as a model's weight size.
    weights_bytes: int = 0
    #: Summed size of the buffers the compiler flags `is_param` — the actual
    #: learned parameters. These two figures are not the same number, and the
    #: gap is not small: face_detection_yunet has 186 parameter buffers at both
    #: 320x320 and 640x640 — 77 KB and 80 KB — while `memory_footprint.weights`
    #: reads 78 KB at 320 and 1,681 KB at 640, for the same 53,104 parameters.
    #: So the footprint figure includes something that scales with input
    #: resolution, which learned weights cannot. Where the two agree — as they
    #: do for mobilenet_v2, which is what matched ST's published figure — the
    #: distinction does not matter; where they disagree, `param_bytes` is the
    #: one that means "how big is this model".
    param_bytes: int = 0
    param_buffers: int = 0
    activations_bytes: int = 0
    io_bytes: int = 0
    pools: list[Pool] = field(default_factory=list)
    epochs: list[Epoch] = field(default_factory=list)
    macc_total: int = 0
    total_cycles: int = 0
    ec_blobs: int = 0

    # -- derived ----------------------------------------------------------

    @property
    def epochs_hw(self) -> int:
        return sum(1 for e in self.epochs if e.mapping == NODE_HW)

    @property
    def epochs_sw(self) -> int:
        return sum(1 for e in self.epochs if e.mapping == NODE_SW)

    @property
    def epochs_nox(self) -> int:
        return sum(1 for e in self.epochs if e.mapping == NODE_NO_X)

    @property
    def sw_op_histogram(self) -> dict[str, int]:
        """Which operators are costing software epochs, and how many each."""
        out: dict[str, int] = {}
        for epoch in self.epochs:
            if not epoch.is_software:
                continue
            for op in epoch.ops:
                out[op] = out.get(op, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def used_pools(self) -> dict[str, int]:
        return {p.name: p.used_bytes for p in self.pools if p.used_bytes}

    @property
    def placement(self) -> str:
        """Where the compiler actually put things — the authoritative answer."""
        used = self.used_pools
        external = {n: b for n, b in used.items() if n in ("octoFlash", "hyperRAM")}
        if not external:
            return "all-on-chip"
        if "hyperRAM" in external:
            return "activations-in-psram"
        return "weights-in-flash"

    def predicted_ms(self, npu_hz: float = 1_000_000_000.0) -> float | None:
        """Latency implied by the compiler's own per-node cycle estimates.

        A prediction, not a measurement — it accounts for compute but not for
        every stall, and the software epochs run on a different clock. It is
        recorded next to the measured figure precisely so the gap between them
        can be watched.
        """
        if not self.total_cycles:
            return None
        return self.total_cycles / npu_hz * 1000.0

    def metrics(self) -> dict[str, Any]:
        return {
            "weights_bytes": self.weights_bytes,
            "param_bytes": self.param_bytes,
            "param_buffers": self.param_buffers,
            "activations_bytes": self.activations_bytes,
            "io_bytes": self.io_bytes,
            "pool_placement": self.placement,
            "pools": self.used_pools,
            "epochs_total": len(self.epochs),
            "epochs_hw": self.epochs_hw,
            "epochs_sw": self.epochs_sw,
            "epochs_nox": self.epochs_nox,
            "sw_epoch_ops": self.sw_op_histogram,
            "macc_total": self.macc_total,
            "predicted_ms": self.predicted_ms(),
            "ec_blobs": self.ec_blobs,
        }


def parse(path: Path) -> CompileInfo:
    doc = json.loads(Path(path).read_text())
    info = CompileInfo(path=Path(path))

    footprint = doc.get("memory_footprint") or {}
    info.weights_bytes = int(footprint.get("weights") or 0)
    info.activations_bytes = int(footprint.get("activations") or 0)
    io = footprint.get("io") or []
    info.io_bytes = sum(int(v or 0) for v in io) if isinstance(io, list) else 0

    params = [b for b in (doc.get("buffers") or []) if b.get("is_param")]
    info.param_bytes = sum(int(b.get("size_bytes") or 0) for b in params)
    info.param_buffers = len(params)

    for entry in doc.get("memory_pools") or []:
        info.pools.append(
            Pool(
                name=entry.get("name", "?"),
                size_bytes=int(entry.get("size_bytes") or 0),
                used_bytes=int(entry.get("used_size_bytes") or 0),
                address=int(entry.get("address") or 0),
            )
        )

    cycles_by_node: dict[int, tuple[int, int]] = {}
    for entry in doc.get("power_estimates") or []:
        node_id = int(entry.get("node_id") or 0)
        cycles_by_node[node_id] = (
            int(entry.get("compute_cycles") or 0),
            int(entry.get("max_cycles") or 0),
        )

    graphs = doc.get("graphs") or []
    if graphs:
        for node in graphs[0].get("nodes") or []:
            node_id = int(node.get("id") or 0)
            compute, longest = cycles_by_node.get(node_id, (0, 0))
            ops = tuple(
                sub.get("type", "")
                for sub in (node.get("subgraph_nodes") or [])
                # `Param` entries are weights, not operators.
                if sub.get("type") and sub.get("type") != "Param"
            )
            info.epochs.append(
                Epoch(
                    id=node_id,
                    name=node.get("name", ""),
                    mapping=node.get("mapping", ""),
                    macc=int(node.get("macc") or 0),
                    ops=ops,
                    compute_cycles=compute,
                    max_cycles=longest,
                )
            )

    info.macc_total = sum(e.macc for e in info.epochs)
    # `max_cycles` is the epoch's critical path; summing them is the closest
    # thing to a serial-execution estimate the file supports.
    info.total_cycles = sum(e.max_cycles for e in info.epochs)
    info.ec_blobs = len(doc.get("ec_blobs_info") or [])
    return info


def find(output_dir: Path) -> Path | None:
    """Locate the emitted c_info file, whatever the network was named."""
    matches = sorted(Path(output_dir).glob("*_c_info.json"))
    return matches[0] if matches else None
