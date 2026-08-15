"""Numerical parity: did that rewrite change what the graph computes?

Every patch in the zoo claims to be semantics-preserving. This is where the
claim is checked, because the failure mode of an unchecked patch is the worst
one available: a model that compiles, runs, and is quietly wrong, with every
downstream signal reading as success.

The check is deliberately blunt — same inputs, both graphs, compare outputs —
and deliberately honest about when it cannot run. A graph with unpinned
dimensions or an op ONNX Runtime declines to execute yields "unknown", not
"fine".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


@dataclass
class Parity:
    ok: bool | None
    max_abs: float | None = None
    max_rel: float | None = None
    note: str = ""

    @property
    def evaluated(self) -> bool:
        return self.ok is not None


def _session(model_bytes: bytes):  # noqa: ANN202
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    # Single-threaded on purpose. Int8 graphs with max-pool ties route
    # non-deterministically under ORT's thread pool — a documented case saw
    # 5-50% of executions diverge — and a parity check that is itself
    # non-deterministic cannot distinguish a bad patch from a bad run.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _random_inputs(session, *, seed: int = 0) -> dict[str, Any] | None:
    rng = np.random.default_rng(seed)
    feed: dict[str, Any] = {}
    for meta in session.get_inputs():
        dtype = _NP_DTYPE.get(meta.type)
        if dtype is None:
            return None
        shape = []
        for dim in meta.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                return None  # unpinned: cannot fabricate a meaningful input
        if np.issubdtype(dtype, np.floating):
            feed[meta.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype is np.bool_:
            feed[meta.name] = rng.integers(0, 2, shape).astype(bool)
        else:
            # Small non-negative integers: these are indices, ids and flags,
            # and large random values would index out of bounds.
            feed[meta.name] = rng.integers(0, 2, shape).astype(dtype)
    return feed


def compare(
    before: Any,
    after: Any,
    *,
    tol: float = 1e-5,
    seed: int = 0,
    trials: int = 2,
) -> Parity:
    """Run both models on identical random inputs and compare every output."""
    try:
        sess_before = _session(before.SerializeToString())
        sess_after = _session(after.SerializeToString())
    except Exception as exc:  # noqa: BLE001
        return Parity(None, note=f"could not load into onnxruntime: {type(exc).__name__}: {exc}")

    names_before = [o.name for o in sess_before.get_outputs()]
    names_after = [o.name for o in sess_after.get_outputs()]
    if names_before != names_after:
        return Parity(
            False,
            note=f"output signature changed: {names_before} -> {names_after}",
        )

    worst_abs = 0.0
    worst_rel = 0.0
    for trial in range(trials):
        feed = _random_inputs(sess_before, seed=seed + trial)
        if feed is None:
            return Parity(None, note="inputs are not fully static; cannot fabricate a feed")
        try:
            out_before = sess_before.run(None, feed)
            out_after = sess_after.run(None, feed)
        except Exception as exc:  # noqa: BLE001
            return Parity(None, note=f"execution failed: {type(exc).__name__}: {exc}")

        for a, b in zip(out_before, out_after, strict=True):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if a.shape != b.shape:
                return Parity(False, note=f"output shape changed: {a.shape} -> {b.shape}")
            diff = np.abs(a - b)
            worst_abs = max(worst_abs, float(diff.max()) if diff.size else 0.0)
            denom = np.maximum(np.abs(a), 1e-12)
            worst_rel = max(worst_rel, float((diff / denom).max()) if diff.size else 0.0)

    return Parity(
        ok=worst_abs <= tol,
        max_abs=worst_abs,
        max_rel=worst_rel,
        note="" if worst_abs <= tol else f"max |delta| {worst_abs:.3e} exceeds {tol:.1e}",
    )
