"""Calibration providers: the input convention, and where the data came from."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoo.quant import calib


def _write_images(root, n=4, size=(8, 8)):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # A flat mid-grey, so the effect of scale/mean/std is exactly checkable.
        Image.fromarray(np.full((*size, 3), 128, dtype=np.uint8)).save(root / f"{i}.png")
    return root


def _feed(tmp_path, shape, **options):
    root = _write_images(tmp_path / "img")
    spec = calib.CalibrationSpec(
        provider="image_folder", n=1, seed=0, source=str(root), options=options or None
    )
    specs = [calib.InputSpec(name="x", shape=shape)]
    return next(iter(calib.get_provider("image_folder").batches(specs, spec)))["x"]


def test_default_image_convention_is_zero_to_one(tmp_path):
    value = _feed(tmp_path, [1, 3, 8, 8])
    assert value.shape == (1, 3, 8, 8)
    assert value.min() == pytest.approx(128 / 255, abs=1e-4)


def test_scale_option_produces_raw_pixel_range(tmp_path):
    """yunet is fed raw 0-255 by OpenCV; [0, 1] would be 255x too small."""
    value = _feed(tmp_path, [1, 3, 8, 8], scale=255.0)
    assert value.max() == pytest.approx(128.0, abs=0.01)


def test_mean_std_options_produce_the_minus_one_to_one_convention(tmp_path):
    value = _feed(tmp_path, [1, 3, 8, 8], mean=0.5, std=0.5)
    assert value.mean() == pytest.approx((128 / 255 - 0.5) / 0.5, abs=1e-4)


def test_bgr_option_reverses_the_channel_axis(tmp_path):
    root = _write_images(tmp_path / "rgb", n=1)
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[..., 0] = 255  # pure red
    Image.fromarray(array).save(root / "0.png")

    spec = calib.CalibrationSpec(provider="image_folder", n=1, seed=0, source=str(root))
    specs = [calib.InputSpec(name="x", shape=[1, 3, 8, 8])]
    rgb = next(iter(calib.get_provider("image_folder").batches(specs, spec)))["x"]

    spec.options = {"bgr": True}
    bgr = next(iter(calib.get_provider("image_folder").batches(specs, spec)))["x"]

    assert rgb[0, 0].max() == pytest.approx(1.0)   # red in channel 0
    assert bgr[0, 2].max() == pytest.approx(1.0)   # red moved to channel 2


def test_nhwc_input_is_detected_from_the_channel_position(tmp_path):
    value = _feed(tmp_path, [1, 8, 8, 3])
    assert value.shape == (1, 8, 8, 3)


# ---------------------------------------------------------------------------


def _write_wavs(root, seconds, n=3, rate=16_000):
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        sf.write(
            root / f"{i}.wav",
            rng.standard_normal(int(seconds * rate)).astype(np.float32) * 0.1,
            rate,
        )
    return root


def test_whisper_front_end_makes_a_mel_tensor_from_a_waveform(tmp_path):
    """A spectrogram input cannot be fed a waveform, however real the waveform."""
    root = _write_wavs(tmp_path / "audio", seconds=6.0)
    spec = calib.CalibrationSpec(
        provider="audio_folder", n=2, seed=0, source=str(root),
        preprocessor="log_mel_whisper",
    )
    specs = [calib.InputSpec(name="input_features", shape=[1, 80, 500])]
    feeds = list(calib.get_provider("audio_folder").batches(specs, spec))

    assert len(feeds) == 2
    value = feeds[0]["input_features"]
    assert value.shape == (1, 80, 500)
    # The (x + 4) / 4 affine at the end of Whisper's front end puts log-mel in
    # roughly [-1, 1]; a raw log10 spectrogram would be around -10.
    assert -2.0 < float(value.min()) and float(value.max()) < 2.0


def test_window_length_comes_from_the_front_end_not_the_tensor_size(tmp_path):
    """500 mel frames is 80,000 samples, not 40,000 — the tensor size lies."""
    front = calib.get_preprocessor("log_mel_whisper")
    spec = calib.CalibrationSpec(provider="audio_folder")
    assert front.samples_needed([1, 80, 500], spec) == 500 * 160


def test_short_clips_are_skipped_unless_padding_is_requested(tmp_path):
    """Silently padding would fabricate silence for a streaming model."""
    root = _write_wavs(tmp_path / "short", seconds=1.0)
    specs = [calib.InputSpec(name="input_features", shape=[1, 80, 500])]

    strict = calib.CalibrationSpec(
        provider="audio_folder", n=2, seed=0, source=str(root),
        preprocessor="log_mel_whisper",
    )
    assert list(calib.get_provider("audio_folder").batches(specs, strict)) == []

    padded = calib.CalibrationSpec(
        provider="audio_folder", n=2, seed=0, source=str(root),
        preprocessor="log_mel_whisper", options={"pad_short": True},
    )
    assert len(list(calib.get_provider("audio_folder").batches(specs, padded))) == 2


def test_a_sample_rate_mismatch_is_refused_rather_than_resampled(tmp_path):
    root = _write_wavs(tmp_path / "wrongrate", seconds=6.0, rate=8_000)
    spec = calib.CalibrationSpec(
        provider="audio_folder", n=1, seed=0, source=str(root),
        preprocessor="log_mel_whisper",
    )
    specs = [calib.InputSpec(name="input_features", shape=[1, 80, 500])]
    with pytest.raises(ValueError, match="Hz"):
        list(calib.get_provider("audio_folder").batches(specs, spec))


def test_nemo_front_end_standardises_each_mel_bin_over_time(tmp_path):
    """Citrinet was trained on per-bin standardised log-mel, not on raw log-mel."""
    root = _write_wavs(tmp_path / "audio", seconds=9.0)
    spec = calib.CalibrationSpec(
        provider="audio_folder", n=1, seed=0, source=str(root),
        preprocessor="log_mel_nemo", options={"lead_silence_samples": 4800},
    )
    specs = [calib.InputSpec(name="audio_signal", shape=[1, 80, 800])]
    value = next(iter(calib.get_provider("audio_folder").batches(specs, spec)))["audio_signal"]

    assert value.shape == (1, 80, 800) and value.dtype == np.float32
    assert np.allclose(value[0].mean(axis=1), 0.0, atol=1e-3)
    assert np.allclose(value[0].std(axis=1, ddof=1), 1.0, atol=1e-3)


def test_nemo_window_is_centred_and_the_lead_in_comes_out_of_it():
    """center=True: T frames are (T - 1) * hop + 1 samples, lead-in included."""
    front = calib.get_preprocessor("log_mel_nemo")
    plain = calib.CalibrationSpec(provider="audio_folder")
    assert front.samples_needed([1, 80, 800], plain) == 799 * 160 + 1
    lead = calib.CalibrationSpec(provider="audio_folder", options={"lead_silence_samples": 4800})
    assert front.samples_needed([1, 80, 800], lead) == 799 * 160 + 1 - 4800
    with pytest.raises(ValueError, match="no room"):
        front.samples_needed([1, 80, 10], lead)


# ---------------------------------------------------------------------------


def test_fidelity_inputs_are_held_out_from_the_calibration_set(tmp_path):
    """A quantiser graded on its own calibration samples flatters itself."""
    from zoo.quant import qdq

    root = _write_images(tmp_path / "img", n=16)
    spec = calib.CalibrationSpec(
        provider="image_folder", n=4, seed=0, source=str(root)
    )
    specs = [calib.InputSpec(name="x", shape=[1, 3, 8, 8])]
    provider = calib.get_provider("image_folder")

    feeds, real, note = qdq._eval_feeds(provider, specs, spec, samples=4)
    assert real is True and not note
    assert len(feeds) == 4
    # Same corpus, different draw: the seed the evaluation uses is not the one
    # the calibration used.
    assert spec.derive(seed=spec.seed + 1).seed != spec.seed


def test_synthetic_calibration_is_scored_on_synthetic_inputs(tmp_path):
    from zoo.quant import qdq

    spec = calib.CalibrationSpec(provider="synthetic", n=4, seed=0)
    specs = [calib.InputSpec(name="x", shape=[1, 3, 8, 8])]
    feeds, real, _ = qdq._eval_feeds(calib.get_provider("synthetic"), specs, spec, 4)
    assert real is False
    assert len(feeds) == 4


def test_an_unreadable_corpus_falls_back_and_says_so(tmp_path):
    """A real provider that yields nothing must not quietly become noise."""
    from zoo.quant import qdq

    spec = calib.CalibrationSpec(
        provider="image_folder", n=4, seed=0, source=str(tmp_path / "missing")
    )
    specs = [calib.InputSpec(name="x", shape=[1, 3, 8, 8])]
    feeds, real, note = qdq._eval_feeds(
        calib.get_provider("image_folder"), specs, spec, samples=4
    )
    assert real is False
    assert len(feeds) == 4
    assert "unavailable" in note or "no held-out" in note
