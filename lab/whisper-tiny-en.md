# whisper-tiny.en — the ASR spike

Source: `onnx-community/whisper-tiny.en`. Running as a parallel track alongside
the easy-model funnel, with partial credit defined up front: *encoder only,
frozen window, decoder on the host*.

## The decoder is out, and not for the reason usually given

The usual framing is "the KV cache grows, so shapes are not static". True, but
it understates it. `decoder_model_merged.onnx` is **one top-level `If` node** —
the whole network lives inside its two branches, which select the with-past and
without-past paths. Structurally this is the same problem as
`onnx-community/silero-vad`, and it is fatal before the cache is even reached:
a statically scheduled epoch graph cannot branch.

A head-scan of the first 8 MB of the 118 MB file is enough to see it:

```
$ zoo lint <decoder>
nodes=1  ops={'If': 1}
```

Unmerged `decoder_model.onnx` / `decoder_with_past_model.onnx` avoid the `If`
but not the growing cache. Not attempted; recorded as skipped with a reason.

## The encoder is more tractable than expected

495 nodes. Op census: **304 hardware, 164 plumbing (`Constant`), 27 `SW_INT`**.
No `Einsum`, no `Where`, no `Expand`, no `LayerNormalization` op — LayerNorm is
already decomposed into `ReduceMean/Sub/Pow/Sqrt/Div`, all of which map. Max
rank 3. No control flow.

The attention cost is smaller than the MatMul count suggests. Of 32 `MatMul`
nodes, **only 8 have two dynamic operands** — the Q·Kᵀ and scores·V pairs across
4 layers. The other 24 are Q/K/V/output projections against constant weights,
which reach hardware. So:

| | software instances |
|---|---|
| default profile | 27 SW_INT + 8 dynamic MatMul = **35** of 495 nodes |
| `transformer` profile (`--expand-softmax`) | 23 + 8 = **31** |

## Freezing the window is not enough — and the first answer was wrong

Pinning `encoder_sequence_length` from 3000 to 500 produces a graph that does
not load:

```
Node (/Add_2) Op (Add) [ShapeInferenceError] Incompatible dimensions
```

`embed_positions.weight` is a `(1500, 384)` learned absolute positional
embedding, added to what is now a `(1, 250, 384)` activation. It has to be
sliced to `(250, 384)`, which is exact — position *i* of a shorter window is
still position *i*.

This is worth dwelling on, because the *first* budget measurement said 5 s
needed 2.30 MB and therefore fitted the ~2.88 MB on-chip pool. That number came
from a graph too broken to shape-infer: 174 tensors had no resolvable shape and
were silently skipped. It looked like a green light. The only reason it did not
become a conclusion is that `budget.analyse` reports `unresolved_tensors` and
refuses to name a placement when it is non-zero.

With `pin_dims` + `slice_positional_embedding`, every shape resolves, the graph
runs in ONNX Runtime, and the real numbers are:

| window | frames | encoder out | peak activation (fp32) | placement |
|---|---|---|---|---|
| 30 s | 3000 | 1500 | **112.61 MB** | beyond PSRAM |
| 10 s | 1000 | 500 | 13.54 MB | PSRAM |
| 5 s | 500 | 250 | **4.99 MB** | PSRAM |
| 2 s | 200 | 100 | **2.00 MB** | on-chip |

So the fp32 on-chip window is about **2 seconds**, not 5.

## What to do next

int8 activations should be roughly 4× smaller, which would put the 5-second
window near 1.25 MB and back on-chip. That is an expectation, not a result —
measure it after quantisation, and record the measured number rather than the
projection.

Order of work:

1. Quantise the 5 s pinned+sliced encoder, static QDQ int8, real speech
   calibration (LibriSpeech dev-clean is enough; synthetic would make the
   fidelity number meaningless).
2. Re-measure peak activation on the quantised graph. If ≤ 2.88 MB, this is an
   on-chip 5-second Whisper encoder.
3. `stedgeai analyze` without `--st-neural-art` first — cheap front-end import
   gate, and it yields the M55 baseline for free.
4. Compile with `onchip` and with `transformer`, and compare epoch counts. The
   interesting number is not latency but how many of those 31–35 software
   epochs survive `--expand-softmax`.
5. Weights are 32.8 MB fp32 → about 8.2 MB int8, which sits in octoFlash
   without complaint. Weight locality is not the constraint here; activations
   are.

## Open questions

- Does `--expand-softmax` actually move all 4 `Softmax` nodes to hardware, or
  does the expansion cost more epochs than it saves? Only the compile report
  answers this.
- The 8 dynamic `MatMul`s are irreducible without changing the architecture.
  What do they cost as software epochs, in ms? `object_tracking_vittrack` is
  queued as the cleaner experiment for pricing one software epoch.
- Whisper's mel front end is not in this graph. On-device it would be CMSIS-DSP
  on the M55, as in ST's audio getting-started package. Not costed yet.
