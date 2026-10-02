# Defensible numbers

Follow-up to `first-board-results.md`, which ended with four things owed: an
evidence policy that was written down but not implemented, a fidelity number
produced from Gaussian noise, a model the screen said should be re-exported at
half the resolution, and a Whisper encoder that int8 was expected to bring
on-chip.

Three of the four are now answered, and one of the answers is the opposite of
what was expected.

> **Measured.** Every graph in the zoo now carries a board number taken under
> the full policy — 7 rows, all `trusted`, none stale. It took four physical
> replugs to get there: the probe wedged after 9, 8, ~10 and ~11 loads, and the
> attempts lost to it are recorded as infra failures and excluded from every
> verdict, which is the fold's one editorial rule doing its job.

## 1. The determinism gate

Policy called for 3 loads × 10 invokes plus a canary. That is now what
`zoo measure` does, and every row carries the verdict it earned.

**The unit of repetition is a full reload, not a repeated invoke.** This is the
whole design. A load that fails quietly leaves the previous firmware resident,
and the next `validate` times *that* — plausibly, and wrongly. Ten invokes
inside one load re-measure the same possibly-wrong firmware more precisely;
they say nothing about whether the number reproduces. So the gate reports the
coefficient of variation **across loads**, and the within-load spread is
recorded beside it rather than folded in.

The within-load spread turned out to be free. `validate` already prints it:

```
duration    : 0.106 ms by sample (0.103/0.115/0.005)
```

— mean, then min/max/std across the samples of that run. The old parser took
the first number and discarded the parenthetical. Both are now recorded, which
means the cheapest possible evidence — is this model's own latency noisy? —
costs one extra regex rather than one extra load.

Four verdicts, ordered so that the bench outranks the model:

| verdict | meaning |
|---|---|
| `quarantined` | the canary moved more than `canary_drift_frac`; the bench changed, so nothing measured beside it is evidence about a model |
| `insufficient` | fewer successful loads than `min_loads`; a single measurement cannot be stable, only unsupported |
| `unstable` | enough loads, but they disagreed by more than `unstable_cv` |
| `trusted` | met the bar |

A gate verdict never edits a number. An unstable row keeps its median and is
rendered as unstable; a quarantined row keeps its measurements too, because a
quarantined measurement is evidence about the bench and discarding it would
lose that. The leaderboard grew an `evidence` column — `3x10 ✓ 0.4%` and
`1x4 !` are visibly different claims.

**The canary is mnist-12**, read once before each row. It is the cheapest graph
in the zoo to load, and its calibration is now deliberately frozen on synthetic
data while every other vision recipe moved to real images: the canary's job is
to produce the same number every session, so that a change in it means the
bench moved rather than that a dataset was re-materialised.

Cost, for the five-model leaderboard: 5 canary loads + 15 model loads against
the 5 loads a single-shot pass would use. That is the price of the whole
exercise, and it buys the difference between a number and evidence.

### What it found

The three rows that got through agree with themselves to an almost absurd
degree:

| graph | median | across-load CV | within-load std | gate |
|---|---:|---:|---:|---|
| mnist-12 | 0.104 ms | 0.55% | 0.004 | trusted |
| yunet @320 | 6.342 ms | 0.01% | 0.004 | trusted |
| handpose @224 | 10.298 ms | 0.006% | 0.004 | trusted |
| mobilenet_v2 @224 | 22.107 ms | 0.002% | 0.005 | trusted |
| pphumanseg @192 | 56.220 ms | 0.05% | 0.102 | trusted |
| yunet @640 | 163.760 ms | 0.0004% | 0.004 | trusted |
| whisper encoder @5 s | 10,935.052 ms | 0.05% | 9.6 | trusted |

Three independent firmware reloads of a 163 ms network reproduce it to the
third decimal. That is worth knowing precisely because it was not knowable
before: a bench that is this deterministic *when it works* means the risk here
was never measurement noise, it was measuring the wrong thing — stale firmware,
a wedged probe, a superseded artifact. Those are exactly the failures a
repeated-invoke policy would not have caught and a repeated-*reload* policy
does.

**pphumanseg is the gate paying for itself.** On its first attempt load 1
returned 56.136 ms — close enough to the 55.01 ms this document's predecessor
published that it would have been waved through — and then loads 2 and 3 failed
the success marker three attempts each. The row sat as `insufficient` for a
session, keeping its number and claiming nothing, until a clean window produced
56.171 / 56.220 / 56.221. Under the old single-shot policy that first load
*was* the row; under this one it was a hypothesis that happened to be right,
and the difference was not knowable at the time.

**The free within-load spread turns out to detect software epochs.** Every
pure-hardware graph reports a within-load std of 0.004–0.005 ms regardless of
whether its latency is 0.1 ms or 163 ms. The two graphs with software epochs
break that flat line completely:

| graph | software epochs | within-load std |
|---|---:|---:|
| mnist-12, handpose, mobilenet, yunet ×2 | 0 | 0.004–0.005 ms |
| pphumanseg | 15 | 0.094 ms |
| whisper encoder | 184 | 9.6 ms |

Three orders of magnitude of latency, and the accelerator path holds its
dispersion constant at 4 microseconds; the Cortex-M55 path does not. That is a
software-epoch detector hiding inside a number `validate` was already printing
and the old parser was discarding.

It also means the honest tuning is to keep the gate and stop worrying about the
CV threshold: at 0.02 the `unstable` verdict has enormous headroom on this
bench, so anything that trips it is a real signal.

### The cost the bench imposes

The probe wedged again nine loads into the session — two gated rows and one
canary read. The catalogue's entry for `DEV_USB_COMM_ERR` predicts this
exactly: it is triggered by the loader's powerdown reset "and sometimes simply
by a completed validate run", and only a physical replug clears it.

That is the real price of the policy on this hardware. At four loads per row
(canary plus three reloads), a wedge every eight to ten loads is roughly two
rows per replug. The gate is not what breaks the bench — a single-shot pass
would hit the same wall four times more slowly — but it does mean a full
leaderboard refresh is now a multi-replug operation, and that is a fact about
the bench worth stating rather than a reason to weaken the evidence bar.

`zoo measure --loads N --invokes M --no-canary` exists for iteration, and the
row records what was actually used — a row measured with `--loads 1` is
permanently distinguishable from one that met the bar.

## 2. Real calibration, and what it actually changed

`tiny-imagenet` was not on this machine — `~/butterfly/data/tiny-imagenet-200/`
contains only the `val_format.py` helper — so it was fetched from
`zh-plus/tiny-imagenet` and materialised as 512 JPEGs.
`VoiceBank-DEMAND-16k` was present, but as a Hugging Face parquet snapshot
rather than as audio files; 256 clean clips were extracted to WAV. Both now sit
under `~/datasets/`, and the recipes point at them by path.

### The input convention is part of the model

Every one of these graphs starts at its first `Conv` — the normalisation is
external, so a corpus has to be put through the *model's own* convention or the
activation ranges are wrong from the first layer. These were read from source
rather than remembered:

| model | convention | from |
|---|---|---|
| handpose | `[0,1]` RGB | `opencv_zoo/mp_handpose.py`: `blob / 255.` |
| pphumanseg | `[-1,1]` RGB | `pphumanseg.py`: `/255`, `-= 0.5`, `/= 0.5` |
| mobilenet_v2 | `[-1,1]` RGB | its own `preprocessor_config.json` |
| yunet | **raw 0–255 BGR** | OpenCV `FaceDetectorYN::detect` calls `blobFromImage(pad_image)` with default arguments — no scale factor, no mean |

yunet is the one that would have been silently wrong: calibrating it on `[0,1]`
sets every activation scale 255× too small. The convention is now a
`[calibration.options]` block in each recipe, next to a comment naming the
source line it came from.

### The evaluation data was noise too

The larger finding is that the old fidelity numbers were not merely calibrated
on noise — they were *scored* on noise. `_fidelity` generated Gaussian inputs
regardless of what the provider was. So a model calibrated on noise and graded
on noise agreed with itself, and reported a healthy cosine for it.

The fidelity check now draws held-out samples from the same provider with a
different seed (held out because a quantiser graded on its own calibration set
flatters itself), and records `fidelity_real_inputs` alongside the score.

That makes the 2×2 measurable. For `mobilenet_v2`:

| calibrated on ↓ · scored on → | synthetic | real |
|---|---:|---:|
| **synthetic** | 0.9926 | 0.9394 |
| **real** | 0.9072 | **0.9702** |

and for `handpose`:

| calibrated on ↓ · scored on → | synthetic | real |
|---|---:|---:|
| **synthetic** | 0.9993 | 0.9983 |
| **real** | 0.9990 | **0.9996** |

The diagonal is the point. Each artifact scores best on the distribution its
scales were set from, which is exactly what a self-agreement measurement does.
Reading only the old top-left cell, mobilenet's quantisation looked like a
0.9926; on the images it will actually see, that same artifact is a 0.9394, and
calibrating on images halves the remaining error to 0.9702.

So real calibration did not make the numbers prettier. It made them honest, and
in mobilenet's case the honest number is *worse* than the one it replaces.

### The on-target cosine was never an accuracy either

This is the pass's most consequential finding, and the board handed it over
immediately. Re-measured with real calibration, yunet's on-target cosine
**fell** — 0.9990 to 0.8527 at 640, and 0.7222 at 320. A model calibrated on
real images scored *worse* on the board than the same model calibrated on
noise.

The reason is in `stedgeai validate --help`:

```
--range MIN MAX   range of values to generate the random input data (default: [0 1])
```

`validate` invents its own inputs — uniform noise in **[0, 1]** — unless given
data. yunet is fed raw **0–255** BGR pixels, so its activation scales are
calibrated for 255× more range than the data it was being graded on: the entire
validation input collapses into the bottom fraction of the first quantisation
step. The 0.8527 is that mismatch. It is not a deployment accuracy, and neither
was the 0.9990 it replaced.

Which turns the old numbers inside out. The previous on-target column looked
healthy — 0.9990, 1.0000, 0.9987 — because those models were calibrated on
Gaussian noise and then graded on uniform noise, and the two distributions are
close enough for the artifact to agree with itself. **The whole column was a
self-agreement measurement**, and its worst entry was the honest one: handpose's
0.9398 was a synthetically calibrated model failing to agree with itself even
under favourable conditions.

The fix is that `validate` accepts `-vi`, so the model can be graded on the
corpus it was calibrated for. `zoo measure` now writes the recipe's own
calibration data to `.npy` — held out by seed, as with the host fidelity check
— and passes it. Every board event records `ontarget_input_source`, because a
row measured on real pixels and a row measured on `[0, 1]` noise are not
comparable and should never share a column silently.

The format could not be verified without hardware (`--mode host` is refused for
Cortex-M55 targets), so the next session tests it on the canary first, where a
rejected argument costs ten seconds rather than a row.

### On handpose's 0.9398 — measured

`-vi` works (`.npy`, verified on the canary first), so handpose has now been
measured with real images at both ends: calibrated on them, and graded on
held-out ones. The result is **0.888**, against the old 0.9398.

The two numbers are not comparable and the new one is not a regression. 0.9398
was a noise-calibrated model graded on `[0, 1]` noise; 0.888 is a real model
graded on real pixels, and it is the first accuracy figure in this zoo that
means what the column header says.

What makes it interesting is the comparison alongside it. mobilenet_v2,
measured the same way in the same session, scores **0.995**. So handpose's
divergence is a property of that model rather than of the method: a landmark
regressor whose outputs are pixel coordinates diverges further between ST's
int8 realisation and ONNX Runtime's QDQ semantics than a classifier does. The
original ⚠ was pointing at something real; it just had the mechanism wrong.

Also worth recording: switching handpose to real images changed 3 of its 67
activation scales by more than 2×, one of them by 10.7×.

### Two bugs found on the way

- `calibration_samples` was always 0: the `Reader` that was counted was not the
  one `quantize_static` consumed. A zero sample count is now an audit failure
  in its own right — an empty calibration reader is not an error anywhere in
  ONNX Runtime, it simply leaves every activation range unmeasured.
- The compiler's `memory_footprint.weights` is **not** the model's weights.
  yunet has 186 parameter buffers at both resolutions, totalling 80 KB at 640
  and 77 KB at 320, while that field reads 1,681 KB at 640×640 and 78 KB at
  320×320. A twentyfold difference for the same 53,104 parameters: it includes
  something that scales with input resolution, which learned weights cannot.
  The leaderboard's weights column now uses the summed parameter buffers
  (`param_bytes`). Where the two agree — mobilenet, which is the row that
  matched ST's published figure — nothing changes.

## 3. yunet at 320: no re-export required

The recipe carried a note to "re-export at 320×320". It did not need one.

For a fully convolutional network the input dimension is a constant in the
file and every interior shape is derived from it. Setting the input and
re-running shape inference produces exactly the graph the re-export would have
— without the training environment, the weights repository, or the export
script, none of which the zoo has for a downloaded model. Two things can make
that unsafe, and both are visible in the graph, so both are checked before the
rewrite rather than discovered as a compiler backtrace:

- a `Reshape` whose target is a literal rather than containing `-1`;
- a `Resize` driven by `sizes` rather than `scales`.

yunet passes both: 12 Reshapes, all `[1, -1, k]`, and 2 scale-driven Resizes.
The 320 graph is a `resolution` block in the recipe, and it is a second row on
the leaderboard rather than a replacement — input size is a deployment
decision, and on this part it is usually the decisive one.

| | 640×640 | 320×320 |
|---|---:|---:|
| MACs | 344 M | 86 M |
| peak activation, int8 (analytic) | 8.25 MB | 2.11 MB |
| compiler activations | 4,481 KB | **1,139 KB** |
| parameters | 80 KB | 80 KB |
| profile that compiled | `extram` | **`onchip`** |
| placement | activations-in-PSRAM | **all on-chip** |
| epochs | 54 (50 HW / 0 SW) | 52 (48 HW / 0 SW) |
| predicted latency | 30.47 ms | **5.15 ms** |
| **measured latency** (3×10, trusted) | **163.76 ms** | **6.34 ms** |
| measured ÷ predicted | ×5.4 | **×1.2** |

**25.8× faster, from a rewrite that needed no export.** That is the whole claim
the funnel was making, measured: the screen flagged `activations-in-psram` at
the budget stage before anything was compiled, the recipe carried a note to
re-export at 320, and acting on that note — without leaving the zoo — moves the
model from 163.76 ms to 6.34 ms.

Three things corroborate it. The compiler puts 320's activations at
**1,139 KB** against **ST's published 1,130 KB**, a second independent
agreement with ST's own figures after mobilenet's matched to the kilobyte. The
measurement lands at **6.34 ms against ST's 6.74 ms** for the same model at the
same resolution. And the prediction gap collapses from **×5.4 to ×1.2** — the
memory-boundedness indicator from the previous document, behaving exactly as it
claimed to: once the activations are on-chip, the compiler's compute-only
estimate is nearly right, because there is no longer anything to wait for.

Re-measured with `-vi`, yunet @640's on-target cosine goes **0.8527 → 0.9689**,
which is the mismatch hypothesis confirmed on hardware: the model was never the
problem, the `[0, 1]` validation data was.

The 320 row, measured the same way, sits at **0.8589** — reproduced to six
digits across two sessions, so it is a real property and not noise. Its *host*
int8 cosine is 0.9979, essentially identical to 640's 0.9974, so the extra
divergence appears only on device and only at the lower resolution. That is an
open question rather than a finding; it is recorded here so that the next
person does not read the 320 row's cosine as a quantisation problem, which the
host numbers say it is not.

## 4. The Whisper encoder: int8 made it worse

The expectation on record was that int8 activations would be roughly 4×
smaller, putting the 5-second window near 1.25 MB and back on-chip. It was
labelled an expectation rather than a result, which is fortunate, because it is
wrong in the wrong direction:

| | fp32 | int8 |
|---|---:|---:|
| peak activation (analytic) | 4.99 MB | **9.47 MB** |
| compiler activations | — | 16,385 KB |
| placement | PSRAM | PSRAM |

The compiler's own allocation agrees with the direction, so this is not an
artifact of the zoo's accounting. The mechanism is in the epoch table: the
quantised encoder compiles to **391 epochs, 184 of them software**. Every
software epoch works in float, so each boundary between an accelerated region
and a software one needs a dequantised copy — 27 `QuantizeLinear` and 27
`DequantizeLinear` epochs mark exactly those boundaries. int8 did not shrink
the activations; it added a second representation of them.

What is in those software epochs, from the compiler's own table:

```
 72  Conv
 27  QuantizeLinear
 27  DequantizeLinear
 18  GlobalAveragePool(float)      LayerNorm's mean
  9  Sub(float)  9 Pow(float)  9 Sqrt(float)  9 Div(float)
  4  Softmax
```

### `--expand-softmax` works, and does not help enough

The lab notes asked whether the `transformer` profile's softmax expansion
actually moves the 4 `Softmax` nodes to hardware, or costs more epochs than it
saves. The compile report answers it:

| | default (`extram`) | `transformer` |
|---|---:|---:|
| epochs | 391 | 399 |
| software epochs | 184 | 180 |
| software `Softmax` | 4 | **0** |
| compiler activations | 16,385 KB | **8,764 KB** |
| predicted latency | 111.35 ms | 130.14 ms |

So: it does what it claims — all four `Softmax` epochs move off software — and
it nearly halves the activation footprint, at about 17% more predicted cycles.
It is still 8.76 MB against a 2.88 MB pool, so the 5-second window stays in
PSRAM either way.

### The 72 software convolutions are the real cost

The dominant term is not attention and not softmax. It is 72 software epochs
each containing a `Conv`, listed by the compiler as `Conv` rather than
`Conv(float)` — so integer software, not a float fallback. Their buffers are
named `Gemm_255_gemm_167_0_conv_688` and shaped `[1, 250, 1, 384]` at 8 bits
with ~37 MMAC each: these are the encoder's projection matmuls, lowered to 1×1
convolutions by the front end, against constant weights — the case ST's mapping
table says *should* reach hardware.

Why they do not is the open question this leaves. It is worth answering,
because it is the difference between an encoder that is memory-bound and one
that is 130 ms of Cortex-M55.

Also worth recording: the encoder's host int8 cosine is **0.8295** against fp32
on real speech, with real mel calibration. That is far below anything else in
the zoo and is consistent with transformer activations being hard to quantise
per-tensor. A 5-second Whisper encoder on this part is not blocked on memory
alone.

### Measured: 10.9 seconds for a 5-second window

The board settles it. Three reloads, gate trusted, across-load CV 0.05%:

| | |
|---|---:|
| median latency | **10,935.05 ms** |
| real-time factor (5 s window) | **2.19** |
| predicted | 111.35 ms |
| **measured ÷ predicted** | **×98** |
| on-target cosine, real speech | 1.0000 |

Ten point nine seconds to encode five seconds of audio — a real-time factor of
2.19 on a model whose whole point was to keep up with a microphone.

The ×98 deserves its own note, because it retires a reading from the previous
document. That document proposed the measured-over-predicted ratio as a
**memory-boundedness indicator**, on the evidence of ×1.4 on-chip, ×3.3 weights
in flash, ×5.7 activations in PSRAM. The Whisper encoder is also
activations-in-PSRAM and comes in at ×98, seventeen times worse than the worst
of those. So the ratio is not measuring memory placement. It is measuring **how
much of the graph the compiler's cycle model can see at all** — and the cycle
model sees hardware epochs. yunet at 320, which is pure hardware and on-chip,
lands at ×1.2; this encoder, 184 of whose 391 epochs are software, lands at
×98. Where the two coincided before, it was because software epochs and
external memory happened to travel together.

That makes the ratio *more* useful, not less, but it has to be read as a
compute-coverage indicator: a big number means the compiler is not modelling
most of what the model does.

And the cosine is 1.0000. The encoder is numerically perfect on target, on real
speech, and more than twice too slow to use — which is exactly the kind of row
the zoo exists to record rather than discover in a demo.

## Where this leaves the four asks

| ask | status |
|---|---|
| determinism gates | implemented and run to completion: **all 7 graphs trusted**, across-load CV ≤ 0.55%. Two rows spent a session as `insufficient` rather than being published on one load |
| real calibration | done for all five vision graphs and the Whisper encoder; both corpora materialised locally; fidelity measured on held-out real data at host *and* on target. handpose 0.888, mobilenet 0.995 — the first on-target numbers here that are accuracies rather than self-agreement |
| yunet @320 | **163.76 → 6.34 ms**, on-chip, 3×10 trusted; 1,139 KB against ST's 1,130 KB and 6.34 ms against ST's 6.74 ms |
| Whisper 5 s int8 | answered, negatively, twice over: int8 doubles the activation footprint rather than quartering it, and the measured encoder runs at **RTF 2.19** — 10.9 s for a 5 s window, ×98 the prediction |

## A bug the session found

`silero-vad` reached the **quantize** stage, having been rejected at lint for
being fifteen `If` nodes. `measure_graph` skipped the screen because
`prepared.onnx` was already on disk — but that file is written *before* lint
renders its verdict, so its existence proves the graph was repaired, not that
it was accepted. A cached artifact from a previously rejected screen walked
straight past the rejection. The stage now re-lints the prepared graph before
quantising, which costs nothing and is the only thing standing between a
rejected model and the rest of the pipeline.

## Next

1. Find out why 72 constant-weight 1×1 convolutions land in software epochs on
   the Whisper encoder. It is the difference between a memory-bound encoder and
   130 ms of Cortex-M55, and it is the largest single lever left in the zoo.
2. pphumanseg's 15 software epochs (`Resize`) now have two symptoms worth
   chasing: a ×3.3 prediction gap, and twenty times the within-load dispersion
   of any pure-hardware graph.
3. Why yunet @320 diverges more on device than @640 (0.8589 against 0.9689)
   while their host int8 cosines are indistinguishable.
4. Promotion. yunet @320 (6.34 ms, on-chip, 0.19 of a 30 fps frame budget) and
   handpose (10.30 ms, 0.31) are the candidates for a tier-3 firmware demo with
   live camera I/O. Nothing in the zoo is `DEPLOYED` yet; everything is
   `MEASURED`.

A note on the bench for whoever runs this next. Across four sessions the probe
wedged after 9, 8, ~10 and ~11 loads, always with the same `DEV_USB_COMM_ERR`,
always needing a physical replug; the fifth session ran 12 loads clean, so the
count is a tendency rather than a rule. Budget two rows per replug, and use
`zoo measure --graph <id>` to re-measure one graph of a multi-graph recipe —
without it, refreshing yunet @320 costs a full pass over @640 as well, which is
half a session. The bracket now
re-checks probe health after any failed load and abandons the row instead of
spending `loader_retries` attempts per remaining load on a probe that cannot
answer; that fired for the first time on yunet @320 and turned what would have
been six futile load attempts into one diagnosis, printed with the remedy.

One thing the sessions did establish that no single session could: yunet @640
measured **163.760 / 163.760 / 163.761 ms** in one session and **163.760 /
163.760 / 163.761 ms** in another, across a physical replug and a power cycle.
Cross-session reproducibility was the canary's founding assumption, and it now
has evidence behind it.
