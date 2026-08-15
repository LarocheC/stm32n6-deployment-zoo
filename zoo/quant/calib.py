"""Calibration data providers.

Calibration decides the activation ranges, and therefore the accuracy of every
quantised model in the zoo. It is also the easiest thing to fake, which is why
the synthetic provider is treated as a first-class citizen *and* permanently
marked as one: a fidelity number derived from Gaussian noise is not an accuracy
result, it is a smoke test that the graph survives quantisation at all.

So `synthetic` is available, is the default, and taints every metric it
produces. Tier-2 promotion requires a real provider. The alternative — refusing
to quantise without real data — would mean no model gets screened until someone
has curated a dataset for its exact input signature, which is precisely the
bottleneck the zoo exists to avoid.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


@dataclass
class InputSpec:
    """What one graph input needs, from the recipe and the graph together."""

    name: str
    shape: list[int]
    dtype: str = "float32"
    role: str = "feature"

    @property
    def np_dtype(self):  # noqa: ANN201
        return _NP_DTYPE.get(self.dtype, np.float32)


@dataclass
class CalibrationSpec:
    provider: str = "synthetic"
    n: int = 128
    seed: int = 0
    #: Provider-specific: a directory, an HF dataset id, a sample rate.
    source: str | None = None
    options: dict[str, Any] | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.provider == "synthetic"


class Provider(Protocol):
    #: False for anything that fabricates its data. Drives the ⚠ marker.
    real_data: bool

    def batches(self, specs: list[InputSpec], spec: CalibrationSpec) -> Iterator[dict]:
        ...


PROVIDERS: dict[str, type] = {}


def register(name: str):  # noqa: ANN201
    def decorator(cls):  # noqa: ANN001, ANN202
        PROVIDERS[name] = cls
        return cls

    return decorator


def providers() -> list[str]:
    return sorted(PROVIDERS)


def get_provider(name: str):  # noqa: ANN201
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise KeyError(
            f"unknown calibration provider {name!r}; available: {', '.join(providers())}"
        ) from None


# ---------------------------------------------------------------------------


@register("synthetic")
class SyntheticProvider:
    """Deterministic pseudo-random data matching each input's signature.

    Good for: proving a graph survives the quantisation pass, and comparing
    two compilations of the same graph. Useless for: any statement about
    accuracy. The ranges it produces bear no relation to real activations, so
    the resulting scales are arbitrary — and a model whose scales are arbitrary
    can still score a respectable cosine against *itself*, which is exactly the
    trap.
    """

    real_data = False

    def batches(self, specs: list[InputSpec], spec: CalibrationSpec) -> Iterator[dict]:
        rng = np.random.default_rng(spec.seed)
        for _ in range(spec.n):
            feed = {}
            for item in specs:
                if np.issubdtype(item.np_dtype, np.floating):
                    feed[item.name] = rng.standard_normal(item.shape).astype(item.np_dtype)
                elif item.np_dtype is np.bool_:
                    feed[item.name] = rng.integers(0, 2, item.shape).astype(bool)
                else:
                    feed[item.name] = rng.integers(0, 2, item.shape).astype(item.np_dtype)
            yield feed


@register("image_folder")
class ImageFolderProvider:
    """Real images from a directory tree, resized to the input signature.

    Assumes NCHW with 1 or 3 channels and normalises to [0, 1]; models wanting
    a different normalisation should carry it inside the graph, which is where
    ST wants it anyway so the camera pipe can hand over raw pixels.
    """

    real_data = True

    _SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def batches(self, specs: list[InputSpec], spec: CalibrationSpec) -> Iterator[dict]:
        from PIL import Image

        if not spec.source:
            raise ValueError("image_folder calibration needs `source` set to a directory")
        root = Path(spec.source).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"calibration image directory not found: {root}")

        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in self._SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images under {root}")
        random.Random(spec.seed).shuffle(files)

        feature = next((s for s in specs if s.role == "feature"), specs[0])
        if len(feature.shape) != 4:
            raise ValueError(
                f"image_folder expects a rank-4 input; {feature.name} is {feature.shape}"
            )
        _, c, h, w = feature.shape
        if c not in (1, 3):
            # NHWC: the channel is last instead.
            _, h, w, c = feature.shape
            channels_last = True
        else:
            channels_last = False

        for path in files[: spec.n]:
            with Image.open(path) as img:
                img = img.convert("L" if c == 1 else "RGB").resize((w, h))
                array = np.asarray(img, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[..., None]
            array = array[None] if channels_last else array.transpose(2, 0, 1)[None]

            feed = {feature.name: array.astype(feature.np_dtype)}
            # Any other inputs still need a value; zeros are the neutral choice
            # for state, and a state input's real distribution is a separate
            # problem that only a streaming provider can solve.
            for item in specs:
                if item.name not in feed:
                    feed[item.name] = np.zeros(item.shape, dtype=item.np_dtype)
            yield feed


@register("audio_folder")
class AudioFolderProvider:
    """Real audio, framed to the input signature.

    Deliberately simple: it fills the feature input with contiguous windows of
    real waveform. A model whose input is a spectrogram rather than a waveform
    needs its own front end run first, which belongs in a model-specific
    provider rather than here.
    """

    real_data = True

    _SUFFIXES = (".wav", ".flac", ".ogg")

    def batches(self, specs: list[InputSpec], spec: CalibrationSpec) -> Iterator[dict]:
        import soundfile as sf

        if not spec.source:
            raise ValueError("audio_folder calibration needs `source` set to a directory")
        root = Path(spec.source).expanduser()
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in self._SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no audio under {root}")
        random.Random(spec.seed).shuffle(files)

        feature = next((s for s in specs if s.role == "feature"), specs[0])
        samples = int(np.prod(feature.shape))

        produced = 0
        for path in files:
            if produced >= spec.n:
                break
            audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            for start in range(0, max(len(audio) - samples, 0) + 1, samples):
                if produced >= spec.n:
                    break
                window = audio[start : start + samples]
                if len(window) < samples:
                    break
                feed = {feature.name: window.reshape(feature.shape).astype(feature.np_dtype)}
                for item in specs:
                    if item.name not in feed:
                        feed[item.name] = np.zeros(item.shape, dtype=item.np_dtype)
                yield feed
                produced += 1


# ---------------------------------------------------------------------------


def specs_from_model(model: Any, roles: dict[str, str] | None = None) -> list[InputSpec]:
    """Input signatures from a graph, annotated with recipe roles."""
    from zoo.graph.probe import _ELEM_TYPE  # noqa: PLC0415

    roles = roles or {}
    out: list[InputSpec] = []
    for vi in model.graph.input:
        tt = vi.type.tensor_type
        shape = []
        for dim in tt.shape.dim:
            shape.append(int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value else 1)
        out.append(
            InputSpec(
                name=vi.name,
                shape=shape,
                dtype=_ELEM_TYPE.get(tt.elem_type, "float32"),
                role=roles.get(vi.name, "feature"),
            )
        )
    return out


class Reader:
    """Adapter onto ONNX Runtime's `CalibrationDataReader` interface."""

    def __init__(self, provider, specs: list[InputSpec], spec: CalibrationSpec) -> None:  # noqa: ANN001
        self._iter = iter(provider.batches(specs, spec))
        self.count = 0

    def get_next(self):  # noqa: ANN201
        try:
            feed = next(self._iter)
        except StopIteration:
            return None
        self.count += 1
        return feed

    def rewind(self) -> None:  # pragma: no cover - ORT calls this on some paths
        raise NotImplementedError("calibration readers in the zoo are single-pass")
