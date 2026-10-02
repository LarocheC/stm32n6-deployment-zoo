# STM32N6 knowledge hub

Start here before deploying anything on the **STM32N6570-DK**, whether you are a
person or an agent. Several projects have put models on this board, and each one
lost days to a defect that another project had already found. This page collects
what they learned and says where each piece of evidence lives.

The rule for what belongs here: a fact about **this part, this toolchain, this
board or ST's packages** belongs in this repo, because the next model will hit
it again. A fact about one model stays in that model's repo, and this page links
to it.

## 1. Before you debug, search the atlas

`zoo/faults/known_issues.toml` holds the failure atlas, keyed on **symptoms**.
Every entry gives a cause, a fix and the sources that back it. The board and
toolchain are not needed to search it:

```bash
uv sync
uv run zoo atlas                          # counts per section
uv run zoo atlas stall depthwise          # every word must appear
uv run zoo atlas --silent board           # only failures where the tool reports success
uv run zoo atlas --classify build.log     # match a log against the verbatim signatures
uv run zoo atlas --id signing-without-align-unbootable-image
```

Read these first. They are the most expensive silent failures recorded so far:

| entry | what you see |
|---|---|
| `signing-without-align-unbootable-image` | a completely dead board with byte-perfect flash. `-align` is mandatory on CubeProgrammer 2.21+, and ST's own `bm.mk` omits it |
| `app-slot-512k-overflow-into-weights` | a clean build and a clean flash, then garbage outputs. The app slot between 0x70100000 and 0x70180000 is 512 KiB |
| `psram-spill-with-zero-sw-epochs` | 0 SW and 0 hybrid epochs and a *lower* cycle total, with activations in PSRAM anyway |
| `bool-quant-json-discards-all-quantization` | a model that compiles, runs, and is secretly float |
| `int8-input-independent-constant-output` | the same output for every input |
| `ll-aton-middleware-version-mismatch` | `#error "Possible mismatch in ll_aton library used"`. All four version components must match, dev number included |

## 2. Toolchain pins

These are the versions this bench is pinned to (`config/toolchain.example.toml`).
A result is only meaningful next to the build that produced it.

| tool | version | why it matters |
|---|---|---|
| ST Edge AI Core | 4.0.1-20581 | generated code requires ll_aton **1.1.3-275** exactly; ST's GettingStarted-Audio package ships 1.1.3-262 (atlas: `ll-aton-middleware-version-mismatch`) |
| Arm GNU toolchain | 13.3.rel1 | validated by ST for the M55, but it cannot build ST's audio example: `Projects/GS/Makefile` passes `-fcyclomatic-complexity`. Use GNU Tools for STM32 from CubeCLT (atlas: `st-makefile-passes-fcyclomatic-complexity`) |
| STM32CubeProgrammer | 2.21+ (2.22.0 on the stt bench) | signing needs `-align`; check that the word at +0x70 of the signed image equals `Reset_Handler` before flashing |
| STM32CubeCLT | 1.21.0 | provides `ST-LINK_gdbserver` for `n6_loader.py` |

Never override `OPT` on ST's make command line: doing so silently drops every
`OPT +=` in the Makefile (atlas: `make-command-line-opt-override-discards-appends`).

## 3. Memory: the numbers that decide placement

| fact | value | source |
|---|---|---|
| screening on-chip pool (cpuRAM2 + npuRAM3-6) | 2,883,576 B usable, 2,883,584 B declared; 8 B reserved per pool | `config/policy.toml` `[budget]` |
| ST audio app pool (cpuRAM2 + npuRAM6) | 1,507,328 B. ST's default, **not** a hardware ceiling: AXISRAM3/4/5 are powered and unclaimed | `config/policy.toml` |
| signed app slot | 0x70100000 to 0x70180000 = 524,288 B; move the weights to 0x70400000 for 3 MB | atlas `app-slot-512k-overflow-into-weights` |
| the cliff | weights streamed from octoFlash cost about 1.6x; activations in PSRAM cost 5-20x | README, `docs/first-board-results.md` |
| how to check for a spill | grep the per-pool placement line in `network_generate_report.txt`. Neither the epoch table nor the cycle total shows it | atlas `psram-spill-with-zero-sw-epochs` |

`zoo/graph/budget.py` predicts placement before compiling. It now counts hoisted
weight-`DequantizeLinear` outputs as weights; before that fix it reported
Citrinet-256 as `activations-in-psram` at 11.25 MB when the compiler placed it
on-chip.

## 4. Operators

`uv run zoo ops <Op>` crosses ST's accelerator table with the front-end
vocabulary (README, "Why an operator oracle comes first"). The facts that most
often change a design:

- `Softmax` runs in software unless the compile passes `--expand-softmax`, and no
  stock profile passes it.
- `MatMul`, `Gemm` and `Conv` reach hardware only with a constant operand, which
  is why conv encoders are fast here and self-attention is not.
- Grouped and depthwise convolutions reach hardware with compiler 4.0.1-20581,
  but ST's operator page never promises it ("group" appears zero times in r1.3).
  Pin it with a compile postcondition, as `models/audio/citrinet-256-gamma025.toml`
  does.

### Two NPU stalls a clean compile will not show

Both compile with 0 software and 0 hybrid epochs, and then hang the part forever
with no fault and no output. The compile report shows neither, so each needs its
own static check (below) or a run on the board. Both were found on Citrinet-256
in stm32n6-stt (`board/GATE4.md`) and fixed bit-exactly in the graph:

| entry | trigger | fix |
|---|---|---|
| `stride2-depthwise-conv-stalls-npu` | any depthwise conv with stride 2; lint for `group > 1` and `stride > 1` | move the stride to the following 1x1 conv (`model/fold_stride2.py`) |
| `activ-to-convacc-stream-link-stalls-npu` | a `Reshape` between a conv and its `Relu`, so atonn chains the Relu into the next depthwise conv's data port | keep the Relu on the 4-D tensor (`model/break_relu_chain.py`); check for zero `ACTIV -> CONVACC` links in network.c |

When the NPU hangs, use the atlas's notes on instrumenting the epoch callback.
Through the stai API the callback set with `LL_ATON_RT_SetNetworkCallback` never
fires, and `epoch_num` is the trace counter + 2
(`sw-dequantizelinear-epoch-hang`, `stai-layer-discards-epoch-callback`). Also
remember that a flash boot re-enumerates the USB serial port, so the first ~8 s
of output can be lost (`flash-boot-output-lost-in-usbipd-reattach-gap`).

## 5. Measured on this board

The zoo's own leaderboard is `RESULTS.md`. A row marked ✓ met the evidence bar
in `config/policy.toml`: at least three full reloads of ten invokes each, with
agreement across reloads (`docs/defensible-numbers.md`). Numbers measured in
other repos, with their sources:

| model | measured | on the DK | conditions | source |
|---|---|---:|---|---|
| Citrinet-256 encoder, 8 s window | per invoke | **124.035 ms** (140.0 ms when the input was just read from flash) | 448 epochs, all on-chip, weights in octoFlash | stm32n6-stt README "On silicon"; `board/GATE4.md` Round 20 |
| Citrinet log-mel front end (M55) | per 8 s window | 136.0 ms | float32 C, matches the host on all but 6 of 960,000 int8 values | stm32n6-stt README |
| LiSenNet nc24, streaming | per 16 ms frame | **2.79 ms** · 29.7 µJ | `n6-noextmem`, validate | eco8 `deploy/stm32n6/ONBOARD_MEASUREMENT.md`; N6Net `N6NET_POWER.md` |
| LiSenNet nc24, windowed T=64 | per frame | 1.15 ms (73.63 ms per window) | `n6-allmems-O3` | eco8 `ONBOARD_MEASUREMENT.md` |
| NSNet2 `monarch_full` (sparse) | per frame | 2.128 ms | `n6-noextmem`, validate | eco8 `NSNET2_DEPLOYMENT_NOTES.md` |
| NSNet2 dense | per frame | 22.94 ms, not real-time | weights in octoFlash, memory-bound | eco8 `NSNET2_DEPLOYMENT_NOTES.md` |
| ConvFSENet 192-384, streaming | per frame | 4.40 ms | `n6-noextmem`, profiler | eco8 `ONBOARD_MEASUREMENT.md` |
| N6Net v2 C128 (seeded weights) | per frame | 2.597 ms · 935.3 µJ | `n6-noextmem-ec` | N6Net `N6NET_POWER.md` |
| DNSMOS forward, 1 s crop | per window | 528.2 ms | `n6-allmems-O3`, 5 MB of activations in hyperRAM | dnsmos `examples/convfsenet_ondevice/ONBOARD_RESULTS.md` |
| PESQ predictor (182 k params) | per inference | 37.6 ms | `n6-noextmem` | dnsmos `ONBOARD_RESULTS.md` |
| mobilenet_v2 @224 | per inference | 22.11 ms | weights in flash, 3x10 ✓ | `RESULTS.md` |
| whisper-tiny encoder | per 5 s window | 10,935 ms | 184 SW epochs, activations in PSRAM | `RESULTS.md` |

Three cautions before quoting any of them:

- No repo has measured speech quality (PESQ or DNSMOS of enhanced audio) **on
  the device**. Every PESQ above is host-side. The board gives latency, energy
  and agreement with the host graph.
- The N6Net energy figures subtract a busy-loop "idle" (565 mW), not a WFI idle
  (atlas: `busy-loop-idle-baseline-inflates-energy`).
- `validate` and `npu_profiler` agree on the same graph. The "~1 ms offset" in
  older notes came from comparing two different graphs
  (`validate-vs-profiler-latency-offset`). Still label every number with its
  method.

The full tables, with sweeps, ensemble folds, the energy campaign and the
compiler estimates kept apart from the measurements, are in
[`docs/measurements-elsewhere.md`](docs/measurements-elsewhere.md).

## 6. Reusable pieces in other repos

Before writing glue for the board, check whether it already exists. Paths are
in each repo; the full list is at the end of `docs/measurements-elsewhere.md`.

- **Build, sign and flash without the IDE**: eco8-neaixt `deploy/stm32n6/Makefile`
  and `scripts/{generate,flash,doctor}.sh`, which sign with `-align`; stm32n6-stt
  `board/flash_and_verify.sh`, which refuses an image that would reach the weights
  and reads the flash back.
- **Multi-input / recurrent-state models in ST's app**: eco8-neaixt
  `deploy/stm32n6/app/ai_dpu_se_stream.{c,h}`, plus `host/gen_io_layout.py`,
  which generates the I/O layout header from the compiled network.
- **Audio front end on the M55**: stm32n6-stt `firmware/src/citrinet_fe.c` (NeMo
  log-mel, bit-exact against its Python oracle), and its microphone/AGC notes in
  `firmware/AUDIO-INPUT.md`.
- **Graph patches with parity gates for ST Edge AI 4.0.1**: dnsmos_exported
  `examples/convfsenet_ondevice/patch_for_stedgeai.py`; stm32n6-stt
  `model/fold_stride2.py` and `model/break_relu_chain.py` for the two stalls;
  eco8-neaixt N6Net `n6net/export_npu.py` (`quantize_prelu_slopes`,
  `tie_fifo_qparams`, `native_full_height`).
- **Compile checks**: stm32n6-stt `compile/score_build.py` (epochs, pools, and
  any stream-switch link silicon has not executed); eco8-neaixt N6Net
  `host/npu_cycle_report.py` (static cycles and energy from
  `network_c_info.json`).
- **Measurement harnesses**: eco8-neaixt N6Net `scripts/measure_n6net.sh` (load
  with retries and a hard gate, validate, profile) and `power/run_campaign.sh`
  (two-pass FNB58 energy); `power/probes/run_probes.sh` stops at the first hang,
  so a wedge costs one cell.
- **Reporting a defect to ST**: dnsmos_exported `package_st_report.sh` with
  `ST_BUG_REPORT.md`, and stm32n6-stt's 9-node `board/REPRO-blocker2.md`.

## 7. Where the rest of the N6 knowledge lives

| repo | what it holds about the N6 |
|---|---|
| [stm32n6-stt](https://github.com/LarocheC/stm32n6-stt) | Citrinet-256 ASR end to end: the two NPU stalls and their graph fixes (`model/`, `board/GATE4.md`, a 9-node reproducer in `board/REPRO-blocker2.md`), OTP (`board/OTP.md`), the build/sign/flash recipe (`board/BUILD.md`), the M55 log-mel front end and microphone notes (`firmware/FRONTEND.md`, `firmware/AUDIO-INPUT.md`) |
| [eco8-neaixt](https://github.com/LarocheC/eco8-neaixt) | speech enhancement on the N6: the LiSenNet deployment, the NPU-hardened variants and `deploy/stm32n6/`; newer N6 work on the `N6Net` branch |
| [dnsmos_exported](https://github.com/LarocheC/dnsmos_exported) | DNSMOS as an int8 metric and a trainable loss graph on the N6: on-device training, a Neural-ART defect report for ST with a reproducer (`examples/convfsenet_ondevice/`) |

Other N6 work (model retraining, papers, power measurement, a generative enhancer) lives in private
repos; their findings reach this hub as atlas entries and measured results.

The atlas cites files as `repo:path` (for example `stm32n6-stt:board/GATE4.md`), where `repo` is one
of the repositories above and a bare path is this repo.

## 8. Adding to the hub

- **A new failure.** Add an `[[issue]]` to `zoo/faults/known_issues.toml`, in its
  section. The sections are ordered by what a reader can do about an entry: lint
  rules, graph patches, compile postconditions, board invariants, infra retries,
  then document-only. Within a section, silent entries come first, then the rest
  alphabetically. Key the entry on the symptom a reader will see, not on the
  cause you eventually found. Cite `sources`, because a remedy nobody ran is a
  hypothesis. Give an `error_signature` only if you have the verbatim text, and
  check it with `zoo atlas --classify`.
- **A model.** Write a recipe in `models/` and a lab note in `lab/` (prose only).
  Anything in `lab/` that gets used twice moves into `zoo/` with a test.
- **From another repo.** stm32n6-stt's `zoo-contrib/` showed the pattern:
  prepare ready-to-apply files there, verify them against a scratch copy of
  this repo, and apply them here in one PR.
