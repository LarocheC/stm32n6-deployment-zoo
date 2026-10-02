# citrinet-256-gamma025: the ASR counter-example

Source: `OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx`, a pre-exported ONNX of
NVIDIA NeMo's `stt_en_citrinet_256_gamma_0_25` (CC-BY-4.0). The bring-up
happened in [stm32n6-stt](https://github.com/LarocheC/stm32n6-stt) (the local
atlas cites it as `stm32n6-stt:path`). It went all
the way to a push-to-talk captioner on the DK. This note records what the zoo
should take from it, and points at that repo for everything else.

## Why it is here

whisper-tiny-en's encoder is the zoo's only measured ASR row: 391 epochs, 184 of
them in software, activations in PSRAM, 10,935 ms (`RESULTS.md`). Citrinet is
the counter-example on the same board and the same compiler. It is a
convolutional CTC encoder with no software epochs, every activation on-chip, and
a measured 124.0 ms invoke for an 8 s window. "A conv encoder is what this part
is for" should be a leaderboard row and not only prose. The recipe is
`models/audio/citrinet-256-gamma025.toml`; it is disabled until its patches
exist (see below).

## What went into the zoo from it

- **Atlas.** Seven constraints and two corrections came from its Gate 3
  write-up: signing without `-align`, the 512 KiB app slot, the silent PSRAM
  spill, the ST Makefile traps, `ai_dpu`'s model check, and the HOTPLUG/UR
  access matrix. The ll_aton correction matters most: the version guard covers
  the dev number too.
- **Budget fix.** `peak_activation_fused` was double-counting hoisted
  weight-DequantizeLinear outputs, so it reported 11,250,052 B for a graph that
  the compiler places on-chip.
- **Calibration front end.** `log_mel_nemo` in `zoo/quant/calib.py` is
  bit-identical to stm32n6-stt's `model/fe.py`. Swapping the `2**-24` log floor
  for `1e-2` takes dev-clean WER from 5.83 % to 30.80 %.
- **Policy.** Activation symmetry is now a `[quantize]` key with a per-recipe
  override, because this model's M55 front end quantises with no offset term.

## What it found on silicon (stm32n6-stt README, "On silicon"; board/GATE4.md)

The Gate 2 compile (628 epochs, 0 software) passed every compile-stage check and **hung the NPU**.
Two Neural-ART defects, both in depthwise convolutions, had to be found on the
board and worked around in the graph:

1. A **stride-2 depthwise** convolution stalls the NPU forever, and the stall
   follows the operator across two compiler schedules. The fix
   (`model/fold_stride2.py`) folds the decimation into the following pointwise
   convolution. It is bit-exact and was applied at 3 sites.
2. An **activation accelerator driving a convolution accelerator's data port**
   (`ACTIV 1 → CONVACC 0/1/2 port 0`) stalls forever. atonn produces it when a
   `Reshape` separates a `Relu` from its producing convolution. Keeping the
   `Relu` on the 4-D tensor (`model/break_relu_chain.py`, 84 sites) avoids it,
   and the result is *faster*: 448 epochs against 618. A 9-node reproducer is in
   `board/REPRO-blocker2.md`.

Both compile cleanly and report 0 software epochs. A compile postcondition
cannot catch them; only running on the board does. The deployed build has 448
epochs, 0 software, 0 hybrid, 300 kB cpuRAM2 + 425 kB npuRAM6, no PSRAM, and
9.726 MB of weights at 0x70400000. It measured 124.035 ms, 2.1 % under the
compiler's estimate. Reading each input from memory-mapped flash first makes it
140.0 ms (board/GATE4.md Round 20).

## Before this recipe can run end to end

- None of the six graph rewrites is a zoo patch yet, and each owes the parity
  gate. stm32n6-stt's `model/` directory is the reference.
- Four of them run before quantisation and fit the recipe's `patches` list.
  `fold_stride2` and `break_relu_chain` run on the QDQ graph, after
  quantisation, and the funnel has no stage for that: recipe patches run at
  screen time, before `stage_quantize`. Without those two the graph compiles
  cleanly and then hangs the board. The two ways forward are a
  post-quantisation patch stage, or fp32 formulations of both rewrites. Nobody
  has tested whether ORT's QDQ pass would put the Q/DQ pairs back where the
  rewrites need them.
- `[postconditions]` is not read by the loader yet. It now holds the deployed
  build's numbers plus `activ_to_convacc_links = 0`, the check that would have
  caught the second stall at compile time. The exact byte counts are derived,
  because stm32n6-stt keeps its compiler reports out of git.

## Accuracy, for context (stm32n6-stt, not re-measured here)

The int8 cost at 8 s is 4.91 % → 5.41 % WER on the host (n=373, 95 % CI
[+0.07, +0.94]). Device and host agree: 5.81 % vs 5.92 % on 64 utterances,
paired p = 0.897. Live accented speech measures around 30 %. That is a model
problem, followed up in a separate retraining repo, not a port problem.
