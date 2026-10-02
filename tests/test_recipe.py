"""Recipe schema behaviour, especially the parts that must refuse bad input."""

from __future__ import annotations

from pathlib import Path

import pytest

from zoo import recipe

MINIMAL = """
schema = 1
id = "demo"
source = "onnx-community/demo"
domain = "audio"

[[graph]]
id = "main"
file = "onnx/model.onnx"
realtime_ms = 32.0

[graph.pin]
batch_size = 1

  [[graph.input]]
  name = "audio"
  role = "feature"
  shape = ["batch_size", 512]
  dtype = "float32"

  [[graph.input]]
  name = "state"
  role = "state"
  shape = [2, 1, 128]
  dtype = "float32"
  feeds_from = "stateN"

  [[graph.input]]
  name = "sr"
  role = "constant"
  shape = []
  dtype = "int64"
  value = 16000

  [[graph.output]]
  name = "prob"

  [[graph.output]]
  name = "stateN"
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "demo.toml"
    path.write_text(text)
    return path


def test_minimal_recipe_loads(tmp_path: Path) -> None:
    r = recipe.load(_write(tmp_path, MINIMAL))
    assert r.id == "demo"
    graph = r.graphs[0]
    assert graph.state_pairs == [("state", "stateN")]
    assert graph.unpinned == []
    assert graph.is_compilable
    assert graph.fix_parametric_shapes() == "{'batch_size':1}"


def test_state_input_without_a_pair_is_rejected(tmp_path: Path) -> None:
    """A state edge the firmware cannot wire must not be expressible.

    Getting this wrong does not crash — it produces a model that runs with a
    dangling state and is quietly, subtly wrong, which is the worst failure
    mode the zoo can have.
    """
    text = MINIMAL.replace('  feeds_from = "stateN"\n', "")
    with pytest.raises(recipe.RecipeError, match="no `feeds_from`"):
        recipe.load(_write(tmp_path, text))


def test_state_pointing_at_a_nonexistent_output_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace('feeds_from = "stateN"', 'feeds_from = "does_not_exist"')
    with pytest.raises(recipe.RecipeError, match="not one of this graph's outputs"):
        recipe.load(_write(tmp_path, text))


def test_constant_input_without_a_value_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace("  value = 16000\n", "")
    with pytest.raises(recipe.RecipeError, match="no `value`"):
        recipe.load(_write(tmp_path, text))


def test_disabled_graph_must_explain_itself(tmp_path: Path) -> None:
    """A graph dropped without a reason is indistinguishable from one nobody
    tried, and the failure atlas is the zoo's primary product."""
    # Keys must sit in the [[graph]] table itself, before any sub-table.
    anchor = 'file = "onnx/model.onnx"\n'
    text = MINIMAL.replace(anchor, anchor + "enabled = false\n")
    with pytest.raises(recipe.RecipeError, match="skip_reason"):
        recipe.load(_write(tmp_path, text))

    ok = MINIMAL.replace(
        anchor, anchor + 'enabled = false\nskip_reason = "dynamic KV cache"\n'
    )
    r = recipe.load(_write(tmp_path, ok))
    assert r.enabled_graphs == []
    assert r.graphs[0].skip_reason == "dynamic KV cache"


def test_unpinned_named_dim_blocks_compilation(tmp_path: Path) -> None:
    text = MINIMAL.replace("[graph.pin]\nbatch_size = 1\n", "")
    graph = recipe.load(_write(tmp_path, text)).graphs[0]
    assert graph.unpinned == ["batch_size"]
    assert not graph.is_compilable
    assert graph.fix_parametric_shapes() is None


def test_anonymous_axes_are_not_reported_as_pinnable(tmp_path: Path) -> None:
    """`--fix-parametric-shapes` keys on names, so an unnamed axis has no
    command-line remedy. Listing it under `unpinned` would suggest one."""
    text = MINIMAL.replace('shape = ["batch_size", 512]', 'shape = ["?", 512]').replace(
        "[graph.pin]\nbatch_size = 1\n", ""
    )
    graph = recipe.load(_write(tmp_path, text)).graphs[0]
    assert graph.unpinned == []
    assert graph.anonymous_axes == [("audio", 0)]
    assert not graph.is_compilable


def test_duplicate_graph_ids_are_rejected(tmp_path: Path) -> None:
    text = MINIMAL + """
[[graph]]
id = "main"
file = "onnx/other.onnx"
"""
    with pytest.raises(recipe.RecipeError, match="duplicate graph ids"):
        recipe.load(_write(tmp_path, text))


def test_multi_graph_model_funnels_each_half_independently(tmp_path: Path) -> None:
    """The encoder/decoder case that motivated graph-level granularity.

    "Whisper doesn't work" is not a useful result. "The encoder compiles at a
    frozen window; the decoder is control flow around a growing KV cache" is.
    """
    text = MINIMAL + """
[[graph]]
id = "decoder"
file = "onnx/decoder_model_merged.onnx"
enabled = false
skip_reason = "single top-level If wrapping a growing KV cache"
"""
    r = recipe.load(_write(tmp_path, text))
    assert [g.id for g in r.graphs] == ["main", "decoder"]
    assert [g.id for g in r.enabled_graphs] == ["main"]


def test_quantize_overrides_load_and_typos_are_refused(tmp_path: Path) -> None:
    r = recipe.load(_write(tmp_path, MINIMAL + "\n[quantize]\nactivation_symmetric = true\n"))
    assert r.quantize == {"activation_symmetric": True}
    assert recipe.load(_write(tmp_path, MINIMAL)).quantize == {}
    with pytest.raises(recipe.RecipeError, match="activation_symetric"):
        recipe.load(_write(tmp_path, MINIMAL + "\n[quantize]\nactivation_symetric = true\n"))
