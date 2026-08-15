"""Static QDQ int8 quantisation, plus the audit that says whether it worked.

The quantisation call itself is short. Everything around it is the part that
matters, because ONNX Runtime will happily produce an artifact that looks
quantised and is not deployable — and the ways it does that are specific,
documented, and each cost somebody a week once.

Three traps are guarded here:

**Node exclusion names come from the preprocessed graph, not the original.**
`quant_pre_process` renames nodes. Exclusion lists gathered before it run
therefore match nothing, and the quantiser silently quantises everything the
caller meant to protect. Nothing errors; the accuracy just drops.

**The audit is not optional.** A quantised graph that still contains opaque
`GRU`/`LSTM` nodes, or unfolded `BatchNormalization`, or — worst — no
`QuantizeLinear` at all, is not a quantised graph. Each of those is a silent
pass through ORT and a loud failure, much later, in the compiler or on the
board.

**Fidelity is reported, never gated.** A model that quantises badly is a valid
and interesting zoo entry. Suppressing it would hide exactly the finding the
zoo exists to record. What must not happen is a bad number being presented as
a good one, so the provenance of the calibration data travels with it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from zoo.quant.calib import (
    CalibrationSpec,
    InputSpec,
    Reader,
    get_provider,
    specs_from_model,
)

#: Ops that must not survive quantisation. Each is a silent ORT pass and a
#: later failure elsewhere.
_MUST_NOT_SURVIVE = ("GRU", "LSTM", "RNN", "BatchNormalization")


@dataclass
class QuantResult:
    path: Path | None = None
    ok: bool = False
    error: str = ""
    calibration_samples: int = 0
    calibration_provider: str = "synthetic"
    calibration_is_real: bool = False
    #: Audit findings. Non-empty means the artifact is not deployable.
    audit_failures: list[str] = field(default_factory=list)
    op_counts: dict[str, int] = field(default_factory=dict)
    weight_bytes_before: int = 0
    weight_bytes_after: int = 0
    #: Fidelity against the fp32 graph. None when it could not be evaluated.
    cosine: float | None = None
    mae: float | None = None
    max_abs: float | None = None
    #: ST's scheme is per-channel symmetric weights. False means a documented
    #: fallback was taken and accuracy is expected to be worse.
    per_channel: bool = True
    upgraded_opset_from: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def compression(self) -> float | None:
        if not self.weight_bytes_before or not self.weight_bytes_after:
            return None
        return self.weight_bytes_before / self.weight_bytes_after

    def metrics(self) -> dict[str, Any]:
        return {
            "calibration_provider": self.calibration_provider,
            "calibration_synthetic": not self.calibration_is_real,
            "calibration_samples": self.calibration_samples,
            "int8_cos": self.cosine,
            "int8_mae": self.mae,
            "int8_max_abs": self.max_abs,
            "weight_bytes_fp32": self.weight_bytes_before,
            "weight_bytes_int8": self.weight_bytes_after,
            "weight_compression": self.compression,
            "quantize_ops": self.op_counts.get("QuantizeLinear", 0),
            "audit_failures": self.audit_failures,
            "per_channel": self.per_channel,
            "upgraded_opset_from": self.upgraded_opset_from,
            "quant_notes": self.notes,
        }


def _weight_bytes(model: Any) -> int:
    from onnx import numpy_helper

    total = 0
    for init in model.graph.initializer:
        try:
            total += int(numpy_helper.to_array(init).nbytes)
        except Exception:  # noqa: BLE001
            continue
    return total


def audit(model: Any, path: Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Check a quantised graph is actually quantised and actually deployable.

    The loadability check is not paranoia. ONNX Runtime's own quantiser will
    emit a graph ONNX Runtime cannot load — per-channel DequantizeLinear
    carries an `axis` attribute that did not exist before opset 13, and
    quantising an opset-11 model per-channel produces exactly that. Nothing in
    the quantisation call fails. Without opening the result, the audit would
    report a broken artifact as ready to compile.
    """
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    failures: list[str] = []
    if not counts.get("QuantizeLinear") and not counts.get("QLinearConv"):
        failures.append(
            "no QuantizeLinear nodes — the graph passed through unquantised. "
            "Usually means every op was excluded, or none was of a quantisable type"
        )
    for op in _MUST_NOT_SURVIVE:
        if counts.get(op):
            failures.append(
                f"{counts[op]} {op} node(s) survived. These are opaque to the quantiser "
                "and have no Neural-ART mapping; fold or decompose them before quantising"
            )

    if path is not None and path.is_file():
        try:
            import onnxruntime as ort

            from zoo.graph.parity import silence_ort

            silence_ort()
            options = ort.SessionOptions()
            options.log_severity_level = 4
            ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"the quantised graph does not load: {type(exc).__name__}: "
                f"{str(exc)[:300]}"
            )
    return failures, counts


def _fidelity(
    fp32_path: Path, int8_path: Path, specs: list[InputSpec], samples: int = 8, seed: int = 0
) -> tuple[float | None, float | None, float | None, str]:
    """Cosine, MAE and max-abs of int8 against fp32 on the same inputs."""
    try:
        import onnxruntime as ort

        from zoo.graph.parity import silence_ort

        silence_ort()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        a = ort.InferenceSession(str(fp32_path), options, providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(str(int8_path), options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"could not open both graphs: {type(exc).__name__}: {exc}"

    rng = np.random.default_rng(seed)
    dots = norms_a = norms_b = 0.0
    abs_err = 0.0
    count = 0
    worst = 0.0

    for _ in range(samples):
        feed = {}
        for item in specs:
            if np.issubdtype(item.np_dtype, np.floating):
                feed[item.name] = rng.standard_normal(item.shape).astype(item.np_dtype)
            else:
                feed[item.name] = rng.integers(0, 2, item.shape).astype(item.np_dtype)
        try:
            out_a = a.run(None, feed)
            out_b = b.run(None, feed)
        except Exception as exc:  # noqa: BLE001
            return None, None, None, f"execution failed: {type(exc).__name__}: {exc}"

        for x, y in zip(out_a, out_b, strict=False):
            x = np.asarray(x, dtype=np.float64).ravel()
            y = np.asarray(y, dtype=np.float64).ravel()
            if x.shape != y.shape or x.size == 0:
                continue
            dots += float(x @ y)
            norms_a += float(x @ x)
            norms_b += float(y @ y)
            abs_err += float(np.abs(x - y).sum())
            worst = max(worst, float(np.abs(x - y).max()))
            count += x.size

    if not count or norms_a <= 0 or norms_b <= 0:
        return None, None, None, "no comparable outputs"
    cosine = dots / (np.sqrt(norms_a) * np.sqrt(norms_b))
    return float(cosine), abs_err / count, worst, ""


def quantize(
    fp32_path: Path,
    out_path: Path,
    *,
    calibration: CalibrationSpec,
    roles: dict[str, str] | None = None,
    exclude_rule: str | None = None,
    per_channel: bool = True,
) -> QuantResult:
    """Static QDQ int8, to ST's scheme, with the audit run afterwards."""
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    result = QuantResult(
        calibration_provider=calibration.provider,
    )

    fp32 = onnx.load(str(fp32_path))
    result.weight_bytes_before = _weight_bytes(fp32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path = out_path.with_name(out_path.stem + ".preprocessed.onnx")

    # Per-channel quantisation emits DequantizeLinear with an `axis` attribute,
    # which does not exist before opset 13. ONNX Runtime does not check: it
    # writes the graph anyway, and the result is a model ORT itself refuses to
    # load — "Unrecognized attribute: axis for operator DequantizeLinear".
    # Nothing in the quantisation call fails, so without this the artifact
    # looks fine until something tries to open it.
    #
    # Upgrading is the right repair rather than dropping to per-tensor: opset
    # 13 is what ST recommends, and per-channel symmetric weights are the
    # scheme the NPU wants.
    source_opset = next(
        (op.version for op in fp32.opset_import if op.domain in ("", "ai.onnx")), None
    )
    if per_channel and source_opset is not None and source_opset < 13:
        try:
            from onnx import version_converter

            fp32 = version_converter.convert_version(fp32, 13)
            fp32_path = out_path.with_name(out_path.stem + ".opset13.onnx")
            onnx.save(fp32, str(fp32_path))
            result.notes.append(
                f"upgraded opset {source_opset} -> 13 before quantising; per-channel "
                "DequantizeLinear needs the axis attribute, which opset 13 introduced"
            )
            result.upgraded_opset_from = source_opset
        except Exception as exc:  # noqa: BLE001
            result.notes.append(
                f"opset {source_opset} < 13 and the upgrade failed ({type(exc).__name__}); "
                "falling back to per-tensor weights, which needs no axis attribute"
            )
            per_channel = False

    try:
        # `skip_symbolic_shape` because ORT's symbolic inference asserts on
        # graphs carrying symbolic batch dims from a dynamo export.
        quant_pre_process(
            str(fp32_path), str(pre_path), skip_symbolic_shape=True, auto_merge=True
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"quant_pre_process failed: {type(exc).__name__}: {exc}"
        return result

    # Exclusions must be resolved against the PREPROCESSED graph: preprocessing
    # renames nodes, so a list gathered from the original silently excludes
    # nothing at all.
    pre = onnx.load(str(pre_path))
    exclude = _resolve_exclusions(pre, exclude_rule)

    provider = get_provider(calibration.provider)
    result.calibration_is_real = bool(getattr(provider, "real_data", False))
    specs = specs_from_model(pre, roles)
    reader = Reader(provider, specs, calibration)

    def _run(per_ch: bool) -> None:
        quantize_static(
            str(pre_path),
            str(out_path),
            Reader(provider, specs, calibration),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_ch,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )

    try:
        _run(per_channel)
        result.per_channel = per_channel
    except ValueError as exc:
        # ORT's per-channel int32-bias scale adjustment throws
        # "The truth value of an array with more than one element is ambiguous"
        # on some bias shapes, inside _adjust_weight_scale_for_int32_bias. Prior
        # work hit the same wall and fell back to per-tensor weights.
        #
        # This is a real compromise, not a workaround: ST's scheme is
        # per-channel symmetric weights, and per-tensor costs accuracy on
        # depthwise convolutions especially. So it is taken, and recorded, and
        # surfaces in the metrics — never applied silently.
        if not per_channel or "truth value of an array" not in str(exc):
            result.error = f"quantize_static failed: {type(exc).__name__}: {exc}"
            return result
        try:
            _run(False)
        except Exception as inner:  # noqa: BLE001
            result.error = (
                f"quantize_static failed per-channel ({exc}) and per-tensor "
                f"({type(inner).__name__}: {inner})"
            )
            return result
        result.per_channel = False
        result.notes.append(
            "fell back to per-tensor weights: ORT's per-channel int32-bias "
            "adjustment cannot handle this graph's bias shapes. ST's scheme is "
            "per-channel, so expect worse accuracy than the same model would "
            "reach elsewhere — especially on depthwise convolutions"
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"quantize_static failed: {type(exc).__name__}: {exc}"
        return result
    finally:
        result.calibration_samples = reader.count

    quantised = onnx.load(str(out_path))
    result.path = out_path
    result.weight_bytes_after = _weight_bytes(quantised)
    result.audit_failures, result.op_counts = audit(quantised, out_path)

    cosine, mae, worst, note = _fidelity(fp32_path, out_path, specs, seed=calibration.seed)
    result.cosine, result.mae, result.max_abs = cosine, mae, worst
    if note:
        result.error = note

    result.ok = not result.audit_failures
    return result


def _resolve_exclusions(model: Any, rule: str | None) -> list[str]:
    """Node names to leave in float, resolved against the given graph.

    `prologue_until_first_conv` is the rule that mattered in prior work: a
    feature-compression prologue — `(|stft| + eps) ** 0.3` and similar — must
    stay float, because quantising it drags the raw magnitude onto a coarse
    int8 grid and destroys low-energy detail before the network sees it.
    """
    if not rule:
        return []
    if rule != "prologue_until_first_conv":
        raise ValueError(f"unknown exclusion rule {rule!r}")

    producers = {out: node for node in model.graph.node for out in node.output}
    inputs = {vi.name for vi in model.graph.input}
    excluded: list[str] = []

    def walk(name: str, seen: set[str]) -> None:
        node = producers.get(name)
        if node is None or node.name in seen:
            return
        seen.add(node.name)
        if node.op_type in ("Conv", "Gemm", "MatMul"):
            return
        excluded.append(node.name)
        for inp in node.input:
            if inp and inp not in inputs:
                walk(inp, seen)

    for node in model.graph.node:
        if node.op_type in ("Conv", "Gemm", "MatMul"):
            for inp in node.input[:1]:
                walk(inp, set())
    return [n for n in dict.fromkeys(excluded) if n]


def op_histogram(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def summarise(results: Iterable[QuantResult]) -> str:
    lines = []
    for r in results:
        mark = "✓" if r.ok else "✗"
        cos = f"{r.cosine:.4f}" if r.cosine is not None else "—"
        tag = "" if r.calibration_is_real else " ⚠ synthetic"
        lines.append(f"  {mark} cos={cos}{tag}  {r.error or ''}")
    return "\n".join(lines)
