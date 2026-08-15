"""The failure-class enum. This vocabulary is the zoo's scientific product.

A class is worth adding when it would change what someone *does* next. Two
failures that call for the same remedy should share a class even if the tools
phrase them differently, and one that needs a different remedy deserves its own
even if it is rare.
"""

from __future__ import annotations


class FailureClass:
    # -- statically detectable: a lint pass predicts these with no ST tool ---
    OP_UNSUPPORTED = "OP_UNSUPPORTED"
    OP_CONTROL_FLOW = "OP_CONTROL_FLOW"
    OP_SW_POLICY_REJECT = "OP_SW_POLICY_REJECT"
    SHAPE_DYNAMIC_UNPINNABLE = "SHAPE_DYNAMIC_UNPINNABLE"
    SHAPE_DIM_TOO_LARGE = "SHAPE_DIM_TOO_LARGE"
    RANK_TOO_HIGH = "RANK_TOO_HIGH"
    OPSET_TOO_HIGH = "OPSET_TOO_HIGH"
    IR_TOO_HIGH = "IR_TOO_HIGH"
    RANK1_GRAPH_INPUT = "RANK1_GRAPH_INPUT"
    INITIALIZERS_AS_INPUTS = "INITIALIZERS_AS_INPUTS"

    # -- fetch --------------------------------------------------------------
    FETCH_ERROR = "FETCH_ERROR"
    NO_ONNX_PUBLISHED = "NO_ONNX_PUBLISHED"

    # -- ST front end -------------------------------------------------------
    FRONTEND_IMPORT_ERROR = "FRONTEND_IMPORT_ERROR"
    FRONTEND_SHAPE_ERROR = "FRONTEND_SHAPE_ERROR"
    FRONTEND_CHANNEL_POSITION = "FRONTEND_CHANNEL_POSITION"
    FRONTEND_CRASH = "FRONTEND_CRASH"

    # -- quantisation -------------------------------------------------------
    QUANT_FAILED = "QUANT_FAILED"
    QUANT_ACCURACY_LOSS = "QUANT_ACCURACY_LOSS"
    #: The worst kind: atonn discards the quantisation JSON and compiles the
    #: model as float, with no error. It runs; it is silently the wrong
    #: precision. Detected by a post-condition on the reported weight bytes.
    QUANT_SILENTLY_DISCARDED = "QUANT_SILENTLY_DISCARDED"

    # -- compilation --------------------------------------------------------
    ALLOC_FAILED_ONCHIP = "ALLOC_FAILED_ONCHIP"
    ALLOC_FAILED_ALL = "ALLOC_FAILED_ALL"
    CODEGEN_ERROR = "CODEGEN_ERROR"
    COMPILE_TIMEOUT = "COMPILE_TIMEOUT"
    SW_FALLBACK_PRESENT = "SW_FALLBACK_PRESENT"  # only under --disable-sw-fallback

    # -- host validation ----------------------------------------------------
    HOST_VALIDATE_MISMATCH = "HOST_VALIDATE_MISMATCH"
    IO_LAYOUT_MISMATCH = "IO_LAYOUT_MISMATCH"

    # -- on target ----------------------------------------------------------
    TARGET_HANG = "TARGET_HANG"
    TARGET_OUTPUT_CORRUPT = "TARGET_OUTPUT_CORRUPT"
    LATENCY_BUDGET_EXCEEDED = "LATENCY_BUDGET_EXCEEDED"
    ONTARGET_ACCURACY_LOSS = "ONTARGET_ACCURACY_LOSS"

    # -- infrastructure: never a model verdict ------------------------------
    LOAD_FAILED = "LOAD_FAILED"
    LOADER_NO_SUCCESS_MARKER = "LOADER_NO_SUCCESS_MARKER"
    NETWORK_IDENTITY_MISMATCH = "NETWORK_IDENTITY_MISMATCH"
    SERIAL_TIMEOUT = "SERIAL_TIMEOUT"
    STLINK_WEDGED = "STLINK_WEDGED"
    BOARD_NOT_ATTACHED = "BOARD_NOT_ATTACHED"
    CANARY_DRIFT = "CANARY_DRIFT"
    TOOLCHAIN_MISCONFIGURED = "TOOLCHAIN_MISCONFIGURED"

    # -- harness self-assessment --------------------------------------------
    #: A failure the lint pass should have predicted and did not. Recorded so
    #: the funnel can measure its own precision and recall rather than being
    #: trusted; the lint rules are known to under-approximate.
    LINT_GAP = "LINT_GAP"
    LINT_FALSE_POSITIVE = "LINT_FALSE_POSITIVE"

    UNKNOWN = "UNKNOWN"


#: Classes that describe the bench, not the model. Events carrying one of these
#: are logged for board-reliability statistics and excluded from every verdict.
INFRA_CLASSES = frozenset(
    {
        FailureClass.LOAD_FAILED,
        FailureClass.LOADER_NO_SUCCESS_MARKER,
        FailureClass.NETWORK_IDENTITY_MISMATCH,
        FailureClass.SERIAL_TIMEOUT,
        FailureClass.STLINK_WEDGED,
        FailureClass.BOARD_NOT_ATTACHED,
        FailureClass.CANARY_DRIFT,
        FailureClass.TOOLCHAIN_MISCONFIGURED,
    }
)

#: Classes a lint pass is expected to predict before any ST tool runs. Used to
#: score the funnel's own accuracy: a compile failure in one of these classes
#: that lint did not flag is a LINT_GAP.
STATICALLY_PREDICTABLE = frozenset(
    {
        FailureClass.OP_UNSUPPORTED,
        FailureClass.OP_CONTROL_FLOW,
        FailureClass.SHAPE_DYNAMIC_UNPINNABLE,
        FailureClass.SHAPE_DIM_TOO_LARGE,
        FailureClass.RANK_TOO_HIGH,
        FailureClass.OPSET_TOO_HIGH,
        FailureClass.IR_TOO_HIGH,
        FailureClass.RANK1_GRAPH_INPUT,
        FailureClass.INITIALIZERS_AS_INPUTS,
    }
)


def is_infra(failure_class: str) -> bool:
    return failure_class in INFRA_CLASSES
