# First on-board results

Five models compiled for the Neural-ART NPU, four measured on an
STM32N6570-DK. int8 static QDQ, synthetic calibration.

| model | ms | on-target cos | pool | epochs (HW/SW) | weights KB | act KB | predicted ms | measured ÷ predicted |
|---|---:|---:|---|---|---:|---:|---:|---:|
| mnist-12 | **0.11** | 1.0000 | on-chip | 10 (8/0) | 6 | 14 | 0.020 | ×5.2 |
| handpose @224 | **10.15** | 0.9398 | on-chip | 53 (50/0) | 1,175 | 1,372 | 7.45 | ×1.4 |
| pphumanseg @192 | **55.01** | 1.0000 | weights-in-flash | 96 (76/15) | 1,517 | 2,032 | 16.63 | ×3.3 |
| yunet @640 | **161.61** | 0.9990 | activations-in-PSRAM | 53 (49/0) | 1,681 | 4,481 | 28.11 | ×5.7 |
| mobilenet_v2 @224 | — | — | weights-in-flash | 56 (54/0) | 3,854 | 2,058 | — | loader failure (infra) |

## The compiler agrees with ST, exactly

ST publishes measured figures for MobileNetV2-1.0 @224 on this board:
**3,812 KB weights, 2,058 KB internal activations.** The zoo's compile produced
**3,854 KB and 2,058 KB**. The activation figure matches to the kilobyte; the
weight figure is 1% off, which is the quantiser's per-channel scale and
zero-point tensors.

That is the strongest evidence so far that the pipeline — fetch, pin, patch,
quantise, compile — is doing the same thing ST's own model zoo does.

## The prediction gap tracks memory placement, not model size

`network_c_info.json` carries per-node cycle estimates, so every compile yields
a predicted latency with no board involved. Comparing prediction to measurement
across the four:

| placement | measured ÷ predicted |
|---|---:|
| all on-chip (handpose) | ×1.4 |
| weights in flash (pphumanseg) | ×3.3 |
| activations in PSRAM (yunet) | ×5.7 |

The prediction models compute. It does not model waiting for memory. So the
ratio is not noise — it is a **memory-boundedness indicator**, available before
the board is touched, and it orders the three placements exactly as the
architecture says it should.

mnist-12's ×5.2 is the exception that fits: at 0.106 ms the fixed per-inference
overhead dominates, so the ratio says nothing about memory there.

## The activation cliff, measured

yunet at 640×640 spills to PSRAM and takes **161.6 ms**. ST measured the same
model at 320×320 at **6.74 ms**. Same architecture, 4× the pixels, **24× the
time** — considerably worse than the 4× a compute-bound model would show, and
consistent with ST's own DeepLabv3 (5.6×) and FastDepth (19.5×) measurements.

The zoo's screen predicted this: `face_detection_yunet` was flagged
`activations-in-psram` at the budget stage, before anything was compiled. The
recipe carries a note to re-export at 320. That is now a measurement rather than
a recommendation.

## Synthetic calibration is visibly not free

handpose came out at **cos 0.9398** where the others are 0.999+. Its recipe
calibrates on Gaussian noise, like all of these. For a landmark regressor whose
outputs are coordinates rather than logits, arbitrary activation ranges cost
real accuracy.

This is exactly why the ⚠ marker exists and why tier-2 promotion requires a real
provider. The number is not wrong — it is a correct measurement of a model
calibrated on noise, and it should not be read as anything else.

## Open

- **mobilenet_v2 fails to load** with no success marker, three attempts, after
  compiling cleanly. Classified infra, so it has not become a model verdict.
  Prime suspect is the RAM-resident image: the validation firmware links into
  AXISRAM1 at 1024K and mobilenet's weights are 3.8 MB in external flash, so the
  flash write may be the step that is failing. Worth checking against
  `stm32n6_reloc.mpool` / `--relocatable`.
- **15 software epochs in pphumanseg** — its `Resize` nodes, as lint predicted.
  Candidate for the `--expand-softmax`-style question: is there a rewrite that
  moves them to hardware?
- Every number here is one load and one validate. The policy calls for 3 loads
  and 10 invokes before a row is trustworthy; that gate is not implemented yet,
  so treat these as first light rather than as the leaderboard's final word.
