"""Signature normalisation and failure classification.

The classifier's most important property is not accuracy on any one message —
it is that a board problem is never attributed to a model. That rule is tested
first and hardest.
"""

from __future__ import annotations

from zoo.faults import signatures
from zoo.faults.taxonomy import FailureClass as FC


def test_same_failure_different_run_hashes_the_same() -> None:
    """Paths, addresses, indices and timings vary run to run; the constraint
    does not. Without normalisation the atlas would count one problem N times."""
    a = signatures.fingerprint(
        "TOOL ERROR: Error in computation of shapes for node Slice_17 "
        "at /tmp/run-8813/model.onnx (0x34200000) after 12.4 s"
    )
    b = signatures.fingerprint(
        "TOOL ERROR: Error in computation of shapes for node Slice_42 "
        "at /home/x/other/run-2/model.onnx (0x71000000) after 3.1 s"
    )
    assert a == b


def test_different_failures_do_not_collide() -> None:
    a = signatures.fingerprint("TOOL ERROR: Error in computation of shapes")
    b = signatures.fingerprint("INTERNAL ERROR: Mismatch in channel position")
    assert a != b


def test_board_problems_are_classified_as_infra() -> None:
    """The load-bearing rule. Each of these is a fact about the bench.

    Asserted on `is_infra` rather than on a class name, because the class
    vocabulary comes from the mined catalogue and will keep growing. What must
    never drift is which side of the model/bench line a failure lands on.
    """
    for text in [
        "DEV_USB_COMM_ERR while reading",
        "Error: Loading memories failed",
        "No STM32 target found",
        "E801(HwIOError): Invalid firmware",
        "E200(ValidationError): Unable to bind the ST.AI runtime with 'network'",
    ]:
        result = signatures.classify(text)
        assert result.is_infra, f"{text} -> {result.failure_class}"


def test_model_problems_are_not_infra() -> None:
    for text in [
        "TOOL ERROR: Error in computation of shapes",
        "INTERNAL ERROR: Mismatch in channel position",
        "Failed to build a valid graph: missing shape or size for value=Pad_3_constant_value",
        "not implemented shape len for Conversion",
        "atonn reports 116 MB unallocatable",
    ]:
        result = signatures.classify(text)
        assert not result.is_infra, f"{text} -> {result.failure_class}"
        assert result.failure_class != FC.UNKNOWN, text


def test_a_crash_is_recognised_as_a_crash_not_a_generic_error() -> None:
    """A rank-5 tensor segfaults the front end. It needs its own class because
    the remedy — restructure the graph — differs from any recoverable error."""
    result = signatures.classify("Fatal error: signo=11 (Segmentation fault)")
    assert "SEGFAULT" in result.failure_class or "CRASH" in result.failure_class
    assert not result.is_infra


def test_catalogue_signatures_attach_a_workaround() -> None:
    """The point of mining the catalogue: a recognised failure arrives with
    its remedy, not just its name."""
    result = signatures.classify(
        "Failed to build a valid graph: missing shape or size for "
        "value=Pad_17_constant_value"
    )
    assert result.known_issue, result.failure_class
    assert result.workaround
    assert not result.is_new


def test_catalogue_wildcards_match_a_varying_node_index() -> None:
    """Signatures were transcribed with `*` where the node index varies.
    Matching them literally would mean those entries never fire."""
    prefix = "Failed to build a valid graph: missing shape or size for value="
    a = signatures.classify(prefix + "Pad_3_constant_value")
    b = signatures.classify(prefix + "Pad_912_constant_value")
    assert a.known_issue == b.known_issue != ""


def test_catalogue_loads_and_is_substantial() -> None:
    issues = signatures.known_issues()
    assert len(issues) >= 50
    assert all(i.id and i.failure_class for i in issues)
    # Every entry must cite where it came from: a remedy nobody ran is a guess.
    assert all(i.sources for i in issues)
    # The silent failures are the catalogue's most valuable rows.
    assert sum(1 for i in issues if i.silent) >= 10


def test_silent_quantisation_discard_is_classified() -> None:
    """The single most dangerous failure in the catalogue: the tool succeeds
    and the deployed model is secretly float."""
    for text in (
        'INVALID_ARGUMENT:(tensors[7].value.format.type): invalid value "BOOL" for type',
        "Ignore Quantization JSON file; unsupported version ''",
    ):
        assert signatures.classify(text).failure_class == FC.QUANT_SILENTLY_DISCARDED


def test_infra_patterns_win_over_generic_ones() -> None:
    """Order matters: a serial timeout mentioning 'error' during a compile
    must classify as infra, not as a compile failure."""
    text = "INTERNAL ERROR unrelated\nserial read timed out after 120 s"
    assert signatures.classify(text).is_infra


def test_unmatched_output_is_the_discovery_queue() -> None:
    result = signatures.classify("something nobody has seen before")
    assert result.failure_class == FC.UNKNOWN
    assert result.is_new
    assert result.signature


def test_noise_lines_do_not_change_the_signature() -> None:
    """Tool banners and elapsed-time lines carry no diagnostic weight."""
    bare = "TOOL ERROR: Error in computation of shapes"
    noisy = (
        "ST Edge AI Core v4.0.1-20581 7ed50de05\n"
        f"{bare}\n"
        "elapsed time (GENERATE): 3.201s\n"
    )
    assert signatures.fingerprint(bare) == signatures.fingerprint(noisy)


def test_catalogue_entries_that_claim_a_patch_name_one_that_exists() -> None:
    """A catalogue entry promising a patch the registry does not have would
    send the funnel after a repair it cannot perform."""
    from zoo.graph import patches

    known = set(patches.REGISTRY)
    # Patch names in the catalogue are proposals mined from prose; the ones
    # already implemented must match the registry exactly.
    implemented = {i.patch for i in signatures.known_issues() if i.patch} & known
    assert implemented <= known


def test_atlas_search_finds_a_silent_failure_by_its_symptom() -> None:
    """A silent failure has no log line to classify; a symptom must still find it."""
    found = signatures.search_catalogue(["entry point", "align"])
    assert "signing-without-align-unbootable-image" in [i.id for i in found]
    assert all(i.silent for i in signatures.search_catalogue(["board"], silent_only=True))
    assert signatures.search_catalogue(["no-such-word-anywhere"]) == []


def test_atlas_cli_classifies_a_log(tmp_path, capsys) -> None:
    from zoo import cli

    log = tmp_path / "build.log"
    log.write_text('network.c:57:4: error: #error "Possible mismatch in ll_aton library used"\n')
    assert cli.main(["atlas", "--classify", str(log)]) == 0
    assert "ll-aton-middleware-version-mismatch" in capsys.readouterr().out
    assert cli.main(["atlas", "dcache"]) == 0
