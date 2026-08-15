"""Post-training static int8 quantisation, to ST's documented requirements.

The requirements are not folklore — they are stated in ST Edge AI Core's own
`quantization.html`, and several of them contradict what a default ONNX Runtime
invocation would do:

  static only        dynamic / weight-only quantisation is explicitly
                     unsupported, and a hybrid model is silently converted back
                     to float at import. It would deploy as fp32 while looking
                     quantised
  QuantFormat.QDQ    QOperator is "not tested" and only partially supported
  QuantType.QInt8    for BOTH activations and weights. QUInt8 is "not
                     recommended and not tested", and the NPU rejects unsigned
                     activations outright
  per-channel        symmetric per-channel weights, asymmetric per-tensor
                     activations — the "ss/sa" scheme ST names
  MinMax             on a representative dataset, never on noise for anything
                     whose accuracy will be quoted

The consequence worth repeating: **never deploy a vendor's pre-quantised
file.** `onnx-community/*_int8.onnx` are ORT dynamic or QOperator artifacts,
`qualcomm/*` `w8a8` assets are AIMET uint8-asymmetric, and `opencv/*_int8bq`
are block-quantised. All are useful accuracy references and all are the wrong
scheme for this backend.
"""

from zoo.quant.calib import CalibrationSpec, get_provider, providers  # noqa: F401
