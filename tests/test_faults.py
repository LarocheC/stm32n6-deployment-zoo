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
    """The load-bearing rule. Each of these is a fact about the bench."""
    for text, expected in [
        ("DEV_USB_COMM_ERR while reading", FC.STLINK_WEDGED),
        ("Error: Loading memories failed", FC.STLINK_WEDGED),
        ("No STM32 target found", FC.BOARD_NOT_ATTACHED),
        ("E801(HwIOError): Invalid firmware", FC.LOAD_FAILED),
        ("E200(ValidationError): Unable to bind the ST.AI runtime with 'network'", FC.LOAD_FAILED),
    ]:
        result = signatures.classify(text)
        assert result.failure_class == expected, text
        assert result.is_infra, text


def test_model_problems_are_not_infra() -> None:
    for text, expected in [
        ("TOOL ERROR: Error in computation of shapes", FC.FRONTEND_SHAPE_ERROR),
        ("INTERNAL ERROR: Mismatch in channel position", FC.FRONTEND_CHANNEL_POSITION),
        ("Failed to build a valid graph: missing shape or size for value=Pad_3_constant_value",
         FC.FRONTEND_SHAPE_ERROR),
        ("not implemented shape len for Conversion", FC.FRONTEND_IMPORT_ERROR),
        ("atonn reports 116 MB unallocatable", FC.ALLOC_FAILED_ALL),
    ]:
        result = signatures.classify(text)
        assert result.failure_class == expected, text
        assert not result.is_infra, text


def test_a_crash_is_recognised_as_a_crash_not_a_generic_error() -> None:
    """A rank-5 tensor segfaults the front end. It needs its own class because
    the remedy — restructure the graph — differs from any recoverable error."""
    result = signatures.classify("Fatal error: signo=11 (Segmentation fault)")
    assert result.failure_class == FC.FRONTEND_CRASH
    assert not result.is_infra


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
