# mobilenet_v2_1.0_224 — the toolchain baseline

Source: `onnx-community/mobilenet_v2_1.0_224`. Chosen not because it is
interesting but because it should be boring: a pure convolutional classifier
that ought to map entirely to hardware. If this does not work, nothing does.

## Screen

100 nodes, **all 100 hardware-mapped** — `Conv 52, Clip 35, Add 10,
GlobalAveragePool, Flatten, Gemm`. Zero software epochs, zero
frontend-only ops, no control flow, max rank 4, max dim 1280. One symbolic
dimension (`batch_size`), pinned to 1.

Every `Conv`/`Gemm` operand is directly constant, so nothing falls back to the
Cortex-M55. The simplifier finds nothing to fold — the export is already clean.

## The budget module agrees with ST

fp32 peak activation: **9.2 MB**, weights **13.3 MB**. Both come down roughly
4× under int8, giving about **2.3 MB** of activation.

ST publishes a measured figure for this exact model at this exact resolution on
this exact board: **2058 KB internal activation, 3812 KB weights**. So the
analytic fp32 peak divided by four lands within about 12% of ST's measured int8
number.

That is worth more than it sounds. The budget module is analytic — it computes
a liveness maximum over inferred shapes and cannot know what the real allocator
will do. Having it land close to an independently measured value on a model of
this size is evidence the liveness model is not wildly wrong. It is *not*
evidence it will hold for a graph with awkward reuse; the documented case where
an analytic 1.95 MB became "116 MB unallocatable" is the standing reminder.

## Why it says "activations-in-psram"

Because it is screening the **fp32** graph. 9.2 MB exceeds the ~2.88 MB on-chip
pool, so the honest answer before quantisation is PSRAM. Once the int8 stage
exists this should become `all-on-chip`, matching ST. Worth re-reading this note
after the first quantised screen — if it does not, the discrepancy is the
finding.

## Next

This is the model to **measure Rule B with**: compile it twice, once with the
`onchip` profile and once with `extflash`, and publish the ratio. ST's own
numbers put its weights at 3812 KB int8, which does not fit the 2816 KB on-chip
pool alongside activations, so it is a natural candidate for the
weights-in-flash comparison. Expect roughly 1.6× based on the ConvFSENet
measurement (4.40 ms on-chip vs 7.14 ms from flash), and 12–20 ms overall.
