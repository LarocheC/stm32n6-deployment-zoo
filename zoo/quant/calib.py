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
    #: Named transform between the file on disk and the graph's input tensor.
    #: A model whose input is a spectrogram cannot be fed a waveform, and the
    #: front end that bridges them is model-specific, not provider-specific.
    preprocessor: str | None = None
    options: dict[str, Any] | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.provider == "synthetic"

    def opt(self, key: str, default: Any = None) -> Any:
        return (self.options or {}).get(key, default)

    def derive(self, **changes: Any) -> CalibrationSpec:
        """A copy with fields replaced — used to draw held-out evaluation data."""
        from dataclasses import replace

        return replace(self, **changes)


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

    Assumes NCHW with 1 or 3 channels and normalises to [0, 1] by default.

    The default is only right for models that expect [0, 1] RGB, and getting
    this wrong is not cosmetic: MinMax calibration derives the activation scale
    from the input range, so feeding [0, 1] to a network exported for 0–255
    inputs sets every downstream scale about 255× too small. The four options
    below cover the conventions actually met in this zoo, and each recipe
    records which one its source repository documents:

        scale   multiply the [0, 1] pixels by this (255 for raw-pixel models)
        mean    subtract, per channel or scalar, after `scale`
        std     divide by, per channel or scalar, after `mean`
        bgr     reverse the channel order (OpenCV-trained detectors)
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

        scale = float(spec.opt("scale", 1.0))
        mean = np.asarray(spec.opt("mean", 0.0), dtype=np.float32).reshape(-1)
        std = np.asarray(spec.opt("std", 1.0), dtype=np.float32).reshape(-1)
        bgr = bool(spec.opt("bgr", False))

        for path in files[: spec.n]:
            with Image.open(path) as img:
                img = img.convert("L" if c == 1 else "RGB").resize((w, h))
                array = np.asarray(img, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[..., None]
            if bgr and array.shape[-1] == 3:
                array = array[..., ::-1]
            array = array * scale
            # Broadcast over the channel axis, which is still last at this point.
            array = (array - mean) / std
            array = array[None] if channels_last else array.transpose(2, 0, 1)[None]

            feed = {feature.name: np.ascontiguousarray(array, dtype=feature.np_dtype)}
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

    By default it fills the feature input with contiguous windows of real
    waveform. A model whose input is a spectrogram instead gets there through a
    named `preprocessor`: the front end is a property of the *model*, not of the
    corpus, so it is selected per recipe rather than by having a second provider
    per feature type. `audio_folder` + `log_mel_whisper` reads the same WAV
    files as `audio_folder` alone.
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
        front_end = get_preprocessor(spec.preprocessor) if spec.preprocessor else None
        # How much waveform one input tensor consumes. With a front end that is
        # a question only the front end can answer: a [1, 80, 500] mel input is
        # 80,000 samples, and reading 40,000 of them because that is the tensor
        # size would silently calibrate on half-length audio.
        samples = (
            front_end.samples_needed(feature.shape, spec)
            if front_end
            else int(np.prod(feature.shape))
        )
        rate = int(spec.opt("sample_rate", front_end.sample_rate if front_end else 0))

        # Whisper pads or trims every utterance to its fixed window before the
        # mel front end, so for that family a clip shorter than the window is
        # not a gap in the data — it is what the model is actually fed. Opt in
        # per recipe: for a model that consumes a continuous stream, padding
        # would fabricate silence the network would never see there.
        pad_short = bool(spec.opt("pad_short", False))

        produced = 0
        for path in files:
            if produced >= spec.n:
                break
            audio, file_rate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate and file_rate != rate:
                raise ValueError(
                    f"{path.name} is {file_rate} Hz but this model's front end needs "
                    f"{rate} Hz. Resample the corpus rather than letting the mismatch "
                    "through — it would shift every filterbank band."
                )
            if pad_short and len(audio) < samples:
                audio = np.pad(audio, (0, samples - len(audio)))
            for start in range(0, max(len(audio) - samples, 0) + 1, samples):
                if produced >= spec.n:
                    break
                window = audio[start : start + samples]
                if len(window) < samples:
                    break
                value = (
                    front_end(window, feature.shape, spec)
                    if front_end
                    else window.reshape(feature.shape)
                )
                feed = {feature.name: np.ascontiguousarray(value, dtype=feature.np_dtype)}
                for item in specs:
                    if item.name not in feed:
                        feed[item.name] = np.zeros(item.shape, dtype=item.np_dtype)
                yield feed
                produced += 1


# ---------------------------------------------------------------------------
# Front ends
#
# The mel spectrogram a model was trained on is part of the model, not part of
# the dataset. Getting it wrong produces a graph calibrated on a distribution
# it will never see on the board — which is the same failure as synthetic
# calibration, only harder to notice because the data was real.


PREPROCESSORS: dict[str, type] = {}


def register_preprocessor(name: str):  # noqa: ANN201
    def decorator(cls):  # noqa: ANN001, ANN202
        PREPROCESSORS[name] = cls
        return cls

    return decorator


def preprocessors() -> list[str]:
    return sorted(PREPROCESSORS)


def get_preprocessor(name: str):  # noqa: ANN201
    try:
        return PREPROCESSORS[name]()
    except KeyError:
        raise KeyError(
            f"unknown calibration preprocessor {name!r}; available: "
            f"{', '.join(preprocessors()) or '(none)'}"
        ) from None


@register_preprocessor("log_mel_whisper")
class WhisperLogMel:
    """Whisper's own log-mel front end, transcribed from `whisper/audio.py`.

    Every constant here is load-bearing and none is a default: 400-sample FFT,
    160-sample hop, 80 mel bands on a Slaney-normalised filterbank, log10, a
    dynamic-range floor 8 decades below the clip maximum, then `(x + 4) / 4`.
    That last affine step is why the encoder's input lives in roughly [-1, 1]
    rather than in decibels, and it is the difference between calibrating on
    Whisper's actual input distribution and calibrating on something that
    merely looks like a spectrogram.

    `stft[..., :-1]` in the original drops the final frame, so `T` frames need
    exactly `T * hop` samples of audio.
    """

    sample_rate = 16_000
    n_fft = 400
    hop = 160

    def _frames(self, shape: list[int]) -> tuple[int, int]:
        if len(shape) < 2:
            raise ValueError(f"log_mel_whisper needs a (…, mels, frames) input; got {shape}")
        return int(shape[-2]), int(shape[-1])

    def samples_needed(self, shape: list[int], spec: CalibrationSpec) -> int:
        _, frames = self._frames(shape)
        return frames * self.hop

    def __call__(self, audio: np.ndarray, shape: list[int], spec: CalibrationSpec) -> np.ndarray:
        import librosa

        mels, frames = self._frames(shape)
        window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        stft = librosa.stft(
            audio.astype(np.float32),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=window,
            center=True,
        )
        magnitudes = np.abs(stft[:, :-1]) ** 2
        filters = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=mels)
        mel_spec = filters @ magnitudes

        log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if log_spec.shape[1] < frames:
            log_spec = np.pad(log_spec, ((0, 0), (0, frames - log_spec.shape[1])))
        return log_spec[:, :frames].reshape(shape)


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
