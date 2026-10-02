# Measured elsewhere: STM32N6570-DK results in eco8-neaixt and dnsmos_exported

Collected 2026-10-02 for `KNOWLEDGE.md`. These numbers come from other repos,
not from the zoo's evidence gate (`docs/defensible-numbers.md`). Each row says
how it was measured and cites its file, so check that before comparing rows.
For Citrinet-256 (stm32n6-stt) see `lab/citrinet-256-gamma025.md`.

Branch labels: **main** =
eco8 `main` (2026-09-04), **N6Net** = eco8 `N6Net` (2026-09-17), **ens** = eco8
`ensemble-study` (2026-07-31; identical copies on `block-design` and
`sparse-masks-rowfusion`), **dnsmos** = dnsmos_exported
`claude/dnsmos-onnx-training-wsxieh` (to 2026-08-05). Line numbers refer to
that branch's copy of the file. On main, `RESULTS_LISENNET.md` is
`docs/models/lisennet.md`, `RESULTS_CONVFSENET.md` is `docs/models/convfsenet.md`
and `LISENNET_NPU_HANDOVER.md` is `docs/targets/stm32n6-lisennet-npu.md`.

## Common conditions

- Board: STM32N6570-DK (STM32N657), Cortex-M55 at 800 MHz, Neural-ART NPU at
  1 GHz (`deploy/stm32n6/ONBOARD_MEASUREMENT.md:94` main;
  `deploy/stm32n6/N6NET_POWER.md:3-4` N6Net). dnsmos adds NOC 400 MHz and NIC
  900 MHz (`examples/convfsenet_ondevice/ST_BUG_REPORT.md:49`).
- Compiler: ST Edge AI Core 4.0.1 everywhere; dnsmos pins v4.0.1-20581
  (`ST_BUG_REPORT.md:46`); the runtime reports LL_ATON atonn-v1.1.3-275 and the
  N6Net validation firmware says "Compiled with GCC 14.3.1"
  (`deploy/stm32n6/results/n6net_power/power/pass1/n6net_b3__n6-noextmem-ec/uart.txt`).
- Firmware: ST's `NPU_Validation` app, RAM-resident, loaded by `n6_loader.py`
  over ST-LINK/gdb. `n6-noextmem` = all weights and activations in on-chip
  npuRAM; `-ec` adds `--enable-epoch-controller`
  (`deploy/stm32n6/n6net_neuralart.json` N6Net); `n6-allmems-O3` = weights in
  octoFlash.
- Methods: **validate** = `stedgeai validate --mode target`, "duration ... by
  sample" (usually 10 or 20 samples); **profiler** = `npu_profiler.py`;
  **fw loop** = the power firmware's own DWT timing.
- **Nothing in these repos measures speech quality (PESQ/DNSMOS of enhanced
  audio) on the device.** Every PESQ is host-side (ORT, 824-utterance VBD
  test). What the board gives is latency, energy and output fidelity
  (cosine / score agreement with the host graph). N6NET_POWER.md:11-13 says so
  explicitly for N6Net.
- Weights: N6Net graphs are seeded-random (latency/power only); rows marked ‡
  in eco8 tables use re-initialised weights.

## A. Per-frame latency on silicon (eco8)

| model (int8) | measured | value | conditions | source |
|---|---|---:|---|---|
| ConvFSENet 192-384 streaming | latency/frame | 7.14 ms (RTF 0.45), NPU core 3.73 ms, 27% util | `n6-allmems-O3` (weights octoFlash) | main `deploy/stm32n6/ONBOARD_MEASUREMENT.md:90` |
| ConvFSENet 192-384 streaming | latency/frame | 7.21 ms; split NPU/SW/SW-ctrl 51.7/20.9/27.4% | weights xSPI flash, validate, 10 samples, "preliminary, one run" | N6Net `RESULTS_CONVFSENET.md:72-75` |
| ConvFSENet 192-384 streaming | latency/frame | **4.40 ms** (RTF 0.275), NPU core 1.26 ms, 81% util, mask cos 0.990 vs FP32 | `n6-noextmem`, profiler | main `ONBOARD_MEASUREMENT.md:91,105` |
| ConvFSENet native-dilation variant (Y1) | latency/frame | 5.42 ms, cos 0.9997 | `n6-noextmem`, validate; reverted | main `deploy/stm32n6/TODO.md:81` |
| LiSenNet nc24 windowed (emit_T=64) | per window / per emitted frame | 73.63 ms/window (min 73.04 / max 74.07 / std 0.32) = 1.15 ms/frame, RTF 0.072, cos 0.99829 | `n6-allmems-O3`, validate | N6Net `LISENNET_NPU_HANDOVER.md:18,140`; main `ONBOARD_MEASUREMENT.md:101` |
| LiSenNet nc24 windowed | where the time goes | NPU core 20.3 ms (27.7%), 6 SW epochs 38.1 ms (stride-3 convs 23.2 ms, Gathers 14.9 ms) | profiler | N6Net `RESULTS_LISENNET.md:438-441` |
| LiSenNet nc24 streaming (17-state FIFO) | latency/frame | **2.79 ms** (2.791; RTF 0.174), cos 0.9941 (random-state feeds) | `n6-noextmem`, validate | main `ONBOARD_MEASUREMENT.md:102`; N6Net `LISENNET_NPU_HANDOVER.md:24` |
| LiSenNet nc24 streaming | profiler total | 2.784 ms, NPU core 0.718 ms (26%) | `n6-noextmem`, profiler | main `ONBOARD_MEASUREMENT.md:118-119` |
| NSNet2 `monarch_full` (sparse) | latency/frame | **2.128 ms** (2.123/2.146), RTF 0.13, cos 0.99979 | `n6-noextmem`, validate | main `deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md:231-233` |
| NSNet2 `monarch_8` (sparse) | latency/frame | 2.891 ms (2.885/2.910), RTF 0.18, cos 0.99994 | `n6-noextmem`, validate | same, `:231-233` |
| NSNet2 dense baseline | latency/frame | 22.94 ms, RTF 1.43 (not real-time), memory-bound | `n6-allmems-O3`, profiler | main `NSNET2_DEPLOYMENT_NOTES.md:96`, `ONBOARD_MEASUREMENT.md:106` |

### A2. LiSenNet sweep, 2026-07-14 (ens only)

All `stedgeai generate` -> `n6_loader` -> validate, `--fix-parametric-shapes
"{'B':1}"`; streaming `n6-noextmem`, windowed `n6-allmems-O3` (ens
`deploy/stm32n6/ONBOARD_MEASUREMENT.md:134-138`). Raw numbers in ens
`paper/data/board_results.csv` (streaming + windowed emit_T=64, ms per
inference) and `paper/data/win1_results.csv` (stateless emit_T=1).

| variant | streaming ms/frame | windowed T=64 ms/frame (ms/window) | stateless T=1 ms/frame | source |
|---|---:|---:|---:|---|
| hardened nc20 ‡ | 2.59 (2.588) | 0.93 (59.791) | 29.93 | ens `ONBOARD_MEASUREMENT.md:106,142`; `board_results.csv:5-6`; `win1_results.csv:2` |
| hardened nc24 | 2.79 (2.789) | 1.15 (73.643) | 32.81 | `:107,143`; `board_results.csv:13-14`; `win1_results.csv:3` |
| hardened nc28 ‡ | 3.15 (3.147) | 1.45 (92.926) | 40.01 | `:108,144`; `board_results.csv:7-8` |
| + dilation 16 ‡ | 3.63 (3.633) | 1.82 (116.784) | 74.72 | `:109,145`; `board_results.csv:9-10` |
| + 3 blocks (deep) ‡ | 4.88 (4.877) | 5.72 (366.253, spills to hyperRAM) | 127.16 (allmems) | `:110,146`; `board_results.csv:11-12` |
| relu6-deep (deploy) | **4.83** (4.825) | 3.09 (197.994) | 119.86 (allmems) | `:111,147`; `board_results.csv:2-3` |
| relu6-deep, fp32 decoder ("hybrid") | - | 33.99 (2175.560), RTF 2.1 | - | `:148`; `board_results.csv:4` |
| GRU-over-time nc24 ‡ (t=1 cell as 1x1 convs) | **1.822** | - | - | ens `RESULTS_LISENNET.md:374`; `board_results.csv:15` (cos column "DUMMY") |
| GRU-over-time relu6-deep ‡ | **2.181** | - | - | ens `RESULTS_LISENNET.md:376`; `board_results.csv:16` |

Repeatability: nc24 re-measured 11 days later at 2.789 / 73.643 ms vs 2.791 /
73.633 ms, byte-identical compile (ens `ONBOARD_MEASUREMENT.md:175-177`).

### A3. Ensemble folds of the nc20-shaped streaming graph, 2026-07-31 (ens)

`n6-noextmem`, profiler (validate failed on the fold graphs).

| graph | ms/frame | HW / hybrid / SW ms | source |
|---|---:|---|---|
| K=1 baseline | 2.598 (b=16; validate 2.599) | 1.412 / 0.445 / 0.275 | ens `deploy/stm32n6/ensemble_sweep/README.md:86` |
| K=2 fold, grouped everything (v1) | 7.387 (b=4) | 1.980 / 0.712 / 4.057 | `:87` |
| K=3 fold v1 | 10.584 (b=1) | 2.592 / 0.958 / 6.025 | `:88` |
| K=2 block-diagonal-dense fold (v2) | **3.762** | 2.04 / 0.66 / 0.44 | `:126` |
| K=3 bd-fold (v2) | 5.312 | 2.69 / 0.90 / 0.85 | `:127` |

Raw profiler tables: ens `deploy/stm32n6/ensemble_sweep/prof_*.log`.

## B. Latency and board energy, N6Net campaign 2026-09-17 (N6Net)

FNB58 inline on CN8 (JP2 = 5V_USB_SNK), debug on CN6; board power at the 5 V
input; energy = (saturated - bracketed idle) / inference rate, mean of 2
counterbalanced passes; "mW rt" = one inference per 16 ms minus idle; board
"idle" 565 mW (`N6NET_POWER.md:21-25`). **Caveat**: that idle is a HAL_GetTick
busy loop (analyser caveat in every `analyze.txt`; matched-wait minus idle is
-9.2 to -11.6 mW in all 22 runs, `power/campaign_runs.csv`). Latency is
validate; all 22 runs passed a gate of fw-loop timing within 3.2% of validate
(`N6NET_POWER.md:42-43`). MACs are validate's `macc`.

| model | profile | MACs/frame | ms/frame | uJ/inference | nJ/MAC | mW sat | mW rt | source |
|---|---|--:|--:|--:|--:|--:|--:|---|
| n6net_b1 (seeded) | noextmem-ec | 21.7 M | 0.619 | 148.0 ± 0.4 | 6.8 | +239 | +23.4 | `deploy/stm32n6/N6NET_POWER.md:29` |
| n6net_b3 (seeded) | noextmem-ec | 65.0 M | 1.611 | 444.9 ± 0.4 | 6.8 | +276 | +56.7 | `:30` |
| n6net_b3 (seeded) | noextmem | 65.0 M | 2.447 | 470.7 ± 1.4 | 7.2 | +192 | +56.0 | `:31` |
| n6net_v2_c96 (seeded) | noextmem-ec | 94.7 M | 1.154 | 471.5 ± 0.1 | 5.0 | +409 | +59.9 | `:32` |
| n6net_v2 C128 (seeded) | noextmem-ec | 168.2 M | 2.597 | 935.3 ± 1.8 | 5.6 | +360 | +69.6 | `:33` |
| n6net_v2_fullband, rewritten pool (seeded) | noextmem-ec | 95.5 M | 1.314 | 504.7 ± 0.0 | 5.3 | +384 | +64.2 | `:34` |
| n6net_v2_fullband_native (seeded) | noextmem-ec | 95.4 M | 1.257 | 489.7 ± 0.6 | 5.1 | +390 | +60.9 | `:35` |
| n6net_v2_fullband before 5588e0f | noextmem-ec | 95.4 M | **hangs** | - | - | - | - | `:36` |
| anchor NSNet2 monarch_20 (trained graph from a private repo) | noextmem | 0.28 M | 0.789 | 9.8 ± 0.6 | 35 | +12.6 | +4.1 | `:37` |
| anchor NSNet2 blockdiag_full | noextmem | 0.73 M | 0.674 | 17.6 ± 0.2 | 24 | +26.5 | +5.5 | `:38` |
| anchor ConvFSENet (graph from a private repo) | noextmem | 0.72 M | 3.108 | 22.1 ± 1.0 | 31 | +7.2 | +4.7 | `:39` |
| anchor LiSenNet streaming (HF conv-hardened) | noextmem | 1.44 M | 2.788 | 29.7 ± 0.3 | 21 | +11.0 | +2.8 | `:40` |

Per-run detail (min/max/std, cycles, random-input cosine):
`deploy/stm32n6/results/n6net_power/n6_latency_matrix.csv:2-12`; both energy
bases and duty-cycled energy: `results/n6net_power/power/campaign_cells.csv`;
raw FNB58 traces, UART marks and analysis per run under
`results/n6net_power/power/pass{1,2}/`.

Board probes for the full-band hang (validate, n6-noextmem-ec, seeded): control
0.100 ms, gap_only 0.105, gap (broadcast Add) 0.109, pool_native 0.193 (all cos
>= 0.999999), fullh_only hangs (`deploy/stm32n6/N6NET_COMPILE.md:350-356`).

Note: these are the only N6 energy figures in either repo. main's
`docs/targets/stm32n6.md:75-79` (2026-09-04) still says "no power or energy
number for the STM32N6"; the measured power work in eco8's local tree
(POWER_MEASUREMENT_20260810.md, npu_structure_bench/) was never pushed.

## C. dnsmos_exported on silicon (2026-08-03/04)

| artifact (int8 unless noted) | measured | value | conditions | source |
|---|---|---:|---|---|
| ConvFSENet trunk, windowed emit_T=64 (trained) | per window | 31.75 ms/window = 0.50 ms/frame (RTF 0.031), X-cross cos 0.988, 126.3 MMAC/window | `n6-noextmem`, validate | `examples/convfsenet_ondevice/ONBOARD_RESULTS.md:18` |
| ConvFSENet trunk, per-frame streaming (9 states) | latency/frame | 4.236 ms (RTF 0.265); 68 epochs (29 HW / 27 hybrid / 12 SW), 1.63 MMAC/frame | `n6-noextmem`, validate | `ONBOARD_RESULTS.md:139-140`; `ST_BUG_REPORT.md:30` |
| DNSMOS forward, 1 s crop | per window | 528.2 ms (sigma 0.04 over 10; RTF 0.53); 35 epochs (14/2/19) | `n6-allmems-O3` (5.0 MB acts in hyperRAM) | `ONBOARD_RESULTS.md:19` |
| ConvFSENet head backward (T=63) | per update | 34.2 ms; 10 epochs (2 HW / 8 SW) | `n6-noextmem` | `ONBOARD_RESULTS.md:201-202` |
| DNSMOS loss graph, 0.25 s crop | per window | 4.80 s, all 205 epochs run (outputs wrong: ST defect) | `n6-allmems-O3`, 6.7 MB acts (3.97 MB hyperRAM) | `ONBOARD_RESULTS.md:215-216` |
| DNSMOS loss graph, 0.25 s, bare LL_ATON runner | per clip | 4254344844 cycles = 5317 ms at 800 MHz (same wrong output) | baked-input firmware | `ST_BUG_REPORT.md:133` |
| DNSMOS loss graph, 1 s crop | - | hangs in epoch 65 of 204 | `n6-allmems-O3` and `-O1` | `ONBOARD_RESULTS.md:20`; `ST_BUG_REPORT.md:186-197` |
| PESQ predictor (181,650 params), LearnableSigmoid head | per inference | 37.6 ms; 35 epochs (12 HW / 0 / 23 SW) | `n6-noextmem` | `ONBOARD_RESULTS.md:643, 617` |
| PESQ predictor, sigmoid head | per inference | 37.6 ms; 33 epochs (12 / 0 / 21) | `n6-noextmem` | `ONBOARD_RESULTS.md:817-818, 829` |

On-device output fidelity (not speech quality):
- PESQ predictor, device vs host int8: mean 0.0056 PESQ, max 0.028 over six
  candidates (`ONBOARD_RESULTS.md:638`); sigmoid head mean 0.0060, max 0.018
  (`:829`).
- DNSMOS forward: mos cos 0.996 but a systematic +0.23 mean bias (mae 0.245 on
  10 random samples; |dmos| up to 0.46 on a real clip) (`ONBOARD_RESULTS.md:24-27`).
- PESQ loss graph (fp32): score exact (3.292 / 1.104 / 3.647), 64.7 kB gradient
  corrupt by 1e9-1e11 (`ST_BUG_REPORT.md:94-98`).

## D. Compiler / analytic estimates - NOT measurements

- N6Net static NPU estimates from `network_c_info.json` power_estimates (est.
  NPU ms/frame): v1 b3 time_w 3.55, time_c 1.49, time_split 1.44, b1 0.55; v2
  C80 0.72, C96 0.81, C128 1.71, C192 on octoFlash 17.8; full-band variants
  0.84-1.54 (N6Net `N6NET_COMPILE.md:51-59, 111-119, 165-175`). Against the
  board: energy 3.3-5x low, time 13-66% low (`N6NET_POWER.md:81-85`).
- Ensemble fold estimates from an epoch-count regression: K=2 3.5-4.3 ms, K=3
  4.2-5.1 ms (ens `ensemble_sweep/README.md:37-43`); measured 7.387 / 10.584
  ms for v1 folds.
- dnsmos feasibility budget: analytic MAC-derived timing, explicitly "NONE of
  the timing was measured on this project's board" at the time
  (`FEASIBILITY.md`; summarised in the atlas entry
  mac-derived-latency-budget-unvalidated).

## E. Discrepancies to resolve before quoting

- ConvFSENet: 4.40 ms (profiler, eco8's graph, June) vs 3.108 ms (validate,
  the "ConvFSENet" anchor from a private repo, Sept). The anchor lists 0.72 M MACs vs
  eco8's 1.47 M MACC/frame, so it is probably a different graph (inference);
  its random-input cos 0.734 is flagged "worth a look" (`N6NET_POWER.md:44-47`).
- N6NET_POWER.md quotes compiler time estimates b3 1.39 / v2 1.56 ms, while
  N6NET_COMPILE.md's tables list 1.44 / 1.71 ms for what look like the same
  graphs.
- The "validate runs ~1 ms above the profiler" note in eco8 is not borne out
  where the same graph was measured both ways (LiSenNet 2.79 vs 2.784 ms;
  nc20 2.599 vs 2.598 ms). Label the method per row anyway.
- GRU rows in `paper/data/board_results.csv` carry cos "DUMMY": re-initialised
  weights, latency only.

## F. Reusable firmware and tooling (paths in each repo)

eco8-neaixt, main (also on N6Net):
- `deploy/stm32n6/Makefile`, `config.mk`, `scripts/{generate,flash,doctor}.sh` - non-interactive generate -> io-layout -> build -> sign (`-align`) -> flash (`make deploy`), `make doctor` config check.
- `deploy/stm32n6/app/ai_dpu_se_stream.{c,h}`, `app/model_io_layout.h` - multi-input/output recurrent-state glue replacing the stock one-in/one-out `ai_dpu.c`.
- `deploy/stm32n6/host/gen_io_layout.py` - generates `model_io_layout.h` from the compiled network's I/O.
- `deploy/stm32n6/host/export_blockdiag_npu.py` (`export_monarch_npu.py` on ens) - rank-2 Slice+MatMul+Concat re-export of block-diagonal/Monarch GRU models.
- `deploy/stm32n6/cloud/dev_cloud_bench.py` (+ `.env.example`) - ST Edge AI Developer Cloud board-farm benchmark client (`make bench-cloud`).
- `deploy/stm32n6/n6_loader.config.json` - n6_loader configuration for the DK.
- `deploy/stm32n6/scripts/build_gate0d_probe.py` - generate-only Gate-0 probe graph builder; `scripts/run_windowed_eval.sh` - windowed-model host eval.
- `deploy/stm32n6/ONBOARD_MEASUREMENT.md` - step-by-step load/validate/profile checklist (WSL + usbipd).
- `lisennet/export_onnx.py` (`--streaming` / `--windowed --emit_T`, empty-Pad fix) and `lisennet/quant_onnx.py` (signed per-channel QDQ, state-threaded calibration); `convfsenet/quant.py` (prologue exclusion, `skip_optimization`).

eco8-neaixt, N6Net:
- `deploy/stm32n6/scripts/compile_n6net.sh` - generate with a profile and print placement, epoch split and the cycle report.
- `deploy/stm32n6/host/npu_cycle_report.py` - static ops/cycles/U_MAC/energy report from `network_c_info.json`, artefact epochs excluded.
- `deploy/stm32n6/n6net_neuralart.json` - `n6-noextmem` and `n6-noextmem-ec` profiles.
- `deploy/stm32n6/scripts/measure_n6net.sh` - export -> compile -> load (3 retries, hard gate) -> validate -> npu_profiler sweep into `latency.tsv`.
- `deploy/stm32n6/power/run_campaign.sh` - validate + 2-pass FNB58 power campaign + summary (drives a power harness that is not in these repos).
- `deploy/stm32n6/power/probes/run_probes.sh` - validate cells one by one, stop at the first hang.
- `deploy/stm32n6/host/fullband_probes.py` - real-shape probe graphs to isolate a compiler rewrite.
- `n6net/export_npu.py` - streaming export (`time_split`), ORT parity, int8 post-passes `quantize_prelu_slopes` and `tie_fifo_qparams`; `n6net/model_v2.py::native_full_height`.
- `deploy/stm32n6/results/n6net_power/` - raw FNB58 traces, UART marks and per-run analysis, a template for committing evidence.

eco8-neaixt, ens (ensemble-study / block-design):
- `deploy/stm32n6/scripts/measure_streaming.sh` - load with retry, refuse to validate a failed load, validate one streaming graph.
- `deploy/stm32n6/ensemble_sweep/profile_one.sh` - generate + load + npu_profiler for one graph.
- `deploy/stm32n6/ensemble_sweep/analyze_prof.py` - npu_profiler log -> HW/hybrid/SW ms decomposition.
- `deploy/stm32n6/ensemble_sweep/parse_fit.py` - parse generate reports, fit latency vs epochs.
- `deploy/stm32n6/ensemble_sweep/sanitize_pads.py` - strip empty Pad `constant_value` inputs (atonn segfault fix) from an existing ONNX.
- `deploy/stm32n6/ensemble_sweep/build_group_probe.py` - grouped/depthwise conv mapping probe.
- `deploy/stm32n6/ensemble_sweep/fold_channels.py`, `fold_members.py` - fold K models into one graph (block-diagonal-dense rule), bit-exact.
- `paper/data/make_table.py` + `compile_facts.json` / `board_results.csv` - regenerate a results table from compile reports and board CSVs.

dnsmos_exported (`examples/convfsenet_ondevice/` unless noted):
- `patch_for_stedgeai.py` - consolidated ST Edge AI 4.0.1 graph patches with ORT parity gates.
- `atonn_shim.sh` - strips BOOL entries from `*_Q.json` between the front end and atonn.
- `firmware_loss/{orchestrator_loss.c,build_and_run.sh,gen_loss_input.py}` - bare LL_ATON runner with input baked into the image; build, flash, gdb-load and pyserial UART capture at 921600 (IEEE-754 bit printing, no float printf).
- `run_pesq_on_target.py` - feed candidates to an on-target model through ai_runner and compare against the host int8 graph.
- `package_st_report.sh` + `ST_BUG_REPORT.md` - template for packaging a vendor defect report with reproducers.
- `budget.py` - QDQ-aware MAC and activation-peak budget for a graph.
- `export_streaming_trunk.py` - per-frame FIFO export with threaded-state calibration.
- `src/dnsmos_trainable/verify.py` - export-time STM32N6 device lint (`check_device_constraints`).
