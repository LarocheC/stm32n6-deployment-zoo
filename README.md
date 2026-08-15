# STM32N6 Deployment Zoo

Take arbitrary models off Hugging Face, push them at an **STM32N6570-DK**
(Cortex-M55 @ 800 MHz + Neural-ART NPU @ 1 GHz), and record — reproducibly —
which ones survive, how fast they run, and exactly which toolchain limitation
killed the ones that didn't.

The failure atlas is the primary product. Latency rows are the secondary one.

## Quickstart

```bash
uv sync
cp config/toolchain.example.toml config/toolchain.toml   # edit paths for this machine
uv run zoo doctor            # every tool located and version-checked, no board needed
uv run zoo ops MatMul Softmax LayerNormalization
```

## Why an operator oracle comes first

On this part **latency is epoch-bound, not MAC-bound** below roughly a megabyte
of weights. Every op that misses the accelerator becomes a Cortex-M55 software
epoch, and each one costs an NPU pipeline teardown, a memory round-trip and
cache maintenance. So the cheapest useful question — answerable in under a
second, with no board and no compiler run — is *how many of this graph's ops
actually reach hardware?*

`zoo ops` answers it by crossing two sources, because either alone is wrong:

| Source | What it really is |
|---|---|
| `stneuralart_operator_support.html` | ST's **accelerator** mapping table: 101 ONNX ops tagged `HW` / `SW_INT` / `SW_FLOAT` |
| `stedgeai supported-ops` | The **front-end parser** vocabulary: 137 ops, including `Einsum`, `Where`, `TopK`, `LSTM`, `GRU` — things the importer accepts and the NPU cannot run |

Their difference is a real tier: 36 ops the parser accepts that ST's mapping
table never mentions. Treating those as supported is how you end up with a
model that imports cleanly and then runs entirely on the CPU.

Two caveats in that table change verdicts, so both are parsed rather than
eyeballed:

- **`Softmax` is gated, not free.** The table says `HW`; the comment says
  *"SW_INT otherwise"*, and no profile ST ships passes `--expand-softmax`. The
  honest default is a software epoch, and the flag that fixes it is named.
- **`MatMul` / `Gemm` / `Conv` reach hardware only when an operand is
  constant.** A convolution's weights qualify; an attention score matrix does
  not. That one line is why conv encoders are fast here and self-attention is
  not, and it cannot be answered from an op histogram — it needs the graph.

```
$ uv run zoo ops Softmax
  Softmax  →  SW_INT   (HW with --expand-softmax)
      The --expand-softmax NPU compiler is requested to enable softmax
      expansion into operations supported by the hardware. SW_INT otherwise.
```

## Layout

```
config/     toolchain paths (machine-local), policy thresholds, compilation profiles
zoo/        the reusable core — must never import from lab/, models/ or firmware/
models/     one recipe per model attempt (the zoo's contents)
results/    append-only events.jsonl + derived leaderboard
lab/        raw notebook, prose only — no code
firmware/   live-I/O demos for the handful of promoted models
```

Anything in `lab/` that gets used twice moves into `zoo/` with a test. That is
the rule that keeps the notebook from eating the core.

## Compilation profiles

`config/profiles/zoo_neuralart.json.in` is a template; `zoo profiles`
materialises it with **absolute** memory-pool paths. ST ships its profiles with
relative paths, which is why prior work on this machine had to prefix every
command with `cd ~/stedgeai/install/4.0/scripts/N6_scripts` — and consequently
could not version-control its profiles or keep the compiler's scratch
directories out of the vendor install.

The screening ladder is `onchip → extflash → extram → allmems`. **Which rung a
model lands on is itself a result**: weights streaming from octoFlash cost
about 1.6×, but activations spilling to external memory cost 5–20×.

Beyond the ladder there are gates (`pure-*`, using `--disable-sw-fallback` for
an unambiguous *"is this 100 % NPU?"*), experiments (`*-ec`, enabling the epoch
controller that ST documents as a recommended default and ships in no profile),
a `transformer` profile turning on the recognition passes, and three
diagnostics for bisecting wrong-output bugs.

## Drafting a recipe

Recipe authoring, not compilation, is the real bottleneck — hand-transcribing
per-input names, shapes, dtypes and roles for thirty heterogeneous models costs
more than every compile combined, and it is exactly the kind of copying that
produces silent errors. `zoo init` reads the graph and fills in everything
mechanical:

```
$ uv run zoo init onnx-community/mobilenet_v2_1.0_224
  model  (14.0 MB)
      ir=9 opset=11 nodes=100 max_dim=1280 rank=4
      op tiers: {'HW': 100}
      symbolic dims to pin: ['batch_size']
  wrote models/vision/mobilenet_v2_1.0_224.toml — 7 TODO(s) to resolve
```

Roles are inferred (`sr` as a foldable constant, `state`→`stateN` paired as a
feedback edge, the largest float input as the feature), and every guess it
could not make is a `TODO`. **The unit of compilation is a graph, not a model**
— Whisper is an encoder and a decoder, and "Whisper doesn't work" is not a
useful result where "the encoder compiles at a frozen window, the decoder is
control flow around a growing KV cache" is.

For files too large to pull just to draft against, a single HTTP Range request
recovers the complete operator census without downloading: `GraphProto.node` is
field 1 and `initializer` is field 5, so the nodes land before the weights. 4 MB
of a 33 MB Whisper encoder is enough for all 495 of its nodes.

## Status

Built:

- toolchain discovery, version pinning and `zoo doctor` (23 checks, board-free)
- profile materialisation with absolute mpool paths, and the mpool parser
- the operator oracle (`zoo ops`)
- the ONNX probe — local full parse and remote head scan, transitive constant
  propagation for the `MatMul`/`Gemm`/`Conv` operand question, subgraph op
  counting so a control-flow wrapper cannot masquerade as a one-node graph
- the recipe schema and the `zoo init` auto-drafter
- the fetch layer for plain Hugging Face blobs

Next: the results store (`results/events.jsonl` and the derived leaderboard),
then the static funnel — lint → shape → patch → budget. See `.claude/plans/`
for the full plan.
