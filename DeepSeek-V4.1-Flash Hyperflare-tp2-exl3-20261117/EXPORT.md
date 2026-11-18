# Exporting the package yourself

This guide builds the Hyperflare package of this folder from Diffbot's EXL3 pack: toolchain, fork, pack, export,
checks. Budget **about 1-1.5 hours of mostly unattended time** (41 min of export on our machine with a warm kernel
cache) and **~650 GiB of free disk**.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 2 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 each, peer-to-peer enabled | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (251 GiB usable; DDR5-6000, 8 channels) | serving commits ~140.5 GiB planned (151 GiB RSS measured) with the Engram tables in RAM ([README, Memory](README.md#memory)); the export needs ~80 GiB free on top of what else runs |
| Disk | ~650 GiB free | pack ~322 GiB, package ~314 GiB, kernel cache ~7 GiB. Put the package on a fast NVMe: every start reads it |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | the export compiles the `xmoe` and merge kernels for `sm_120a` with it; nvcc 12.x is too old |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | the fork's Python environment (Python 3.12) |
| Locked memory | `memlock unlimited` for your user | Hyperflare pins host memory, e.g. `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

## 2. Build the fork

The engine is the public fork [AzallazA/hyperflare-inference](https://github.com/AzallazA/hyperflare-inference) at tag
`hyperflare-tp2-exl3-20261117` (upstream NVIDIA Hyperflare `70069281` plus the fork's changes, including the EXL3 recipe and its `xmoe`
kernels).

```bash
git clone https://github.com/AzallazA/hyperflare-inference.git flare
cd flare
git checkout hyperflare-tp2-exl3-20261117                                     # 036cde24

uv sync --locked --extra prepare --extra test          # Python environment in .venv (builds Heliopause from this tree)
cargo build --locked --release --manifest-path flare-cli/Cargo.toml   # flare-cli/target/release/flare
```

Apply the fork's FlashInfer patches to the installed FlashInfer. They patch files inside `.venv`, so re-apply them
after every `uv sync`. (`tp3-patches/overlay/` is the TP3 build's b12x overlay; this build does not use it.)

```bash
SP=$(.venv/bin/python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parents[1])")
for p in tp3-patches/*.patch; do patch -p1 -N -d "$SP" < "$p"; done
```

The `xmoe` and merge kernels are not prebuilt: the first export compiles them into `.brrr-ext/` in the checkout (a
few minutes) and later exports reuse them.

Check the build:

```bash
flare-cli/target/release/flare --version
.venv/bin/ignition export --help | head -3
```

## 3. Get the pack

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --revision 6110453b901e4b51da66f539f3802b5c7f18e8e8 \
  --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw          # ~322 GiB
```

The pack is used as published: no checkpoint view, no conversion. Hyperflare reads its `config.json`, the EXL3
experts, DeepSeek's FP8 tensors and the int4 Engram tables directly. Its `recipe/` and `patches/` folders are not used.

**The directory name becomes the model name.** The package lands in
`~/.cache/flare/packages/models/<directory name>/gb202-tp2-<timestamp>`, `flare models list` shows it as
`<directory name>`, and `flare run <directory name>` serves it. `serve.sh` defaults to
`PACKAGE=DeepSeek-V4.1-Flash-1017-EXL3-3bpw`; with another directory name, serve with `PACKAGE=<your name>` (or the
package path).

## 4. Export

From the fork checkout, with both GPUs idle (stop any server first):

```bash
CKPT=~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw /path/to/this-folder/export.sh --plan-only   # CPU only: plan check
CKPT=~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw /path/to/this-folder/export.sh               # the release build
```

`export.sh` sets the release settings and runs the fork's `tp3-tools/ds41-exl3-export-tp2.sh`, whose header
documents every switch. `--dry-run` prints the environment and the exact `ignition export` command without running
anything. The settings:

| Setting (variable) | Value | Meaning |
| --- | --- | --- |
| `FLARE_DS41_HOST_EXPERT_LAYERS` (`PARK`) | `2-4,6-39` | layers that keep part of their routed experts in host RAM; layers 0, 1 and 5 keep all 384 in VRAM |
| `FLARE_DS41_EXL3_HOT_LAYERS` (`HOT_LOW`, `HOT_HIGH` in `export.sh`) | `344` on layers 2-4, 6-19; `316` on 20-39 | resident (hot) experts per layer in VRAM, out of 384; the rest are read from host RAM |
| `FLARE_DS41_EXL3_EXPORT_HOT` (`EXPORT_HOT`) | `308` | VRAM rows mapped per split layer during the export only (tuning headroom); the package uses the counts above |
| `FLARE_DS41_EXPERT_COUNTS_DIR` (`COUNTS`) | `expert-order/` | route counts that decide which experts are resident (blended 0.6 code / 0.4 prose, collected on this pack at TP2) |
| `FLARE_DS41_EXL3_PREFILL` (`MOE_PREFILL`) | `do` | prefill MoE on Diffbot's decode-once kernel |
| `FLARE_DS41_EXL3_PREFILL_GROUPS`, `_PREFILL_SCRATCH_ROWS`, `_PREFILL_COLD_SIDE` | `2`, `4096`, `1` | prefill MoE scratch in 2 groups and 4,096-route passes; host experts on a side stream beside the decode-once kernels (−294 MiB/GPU) |
| `FLARE_DS41_EXL3_WOA_SINGLE` | `1` | keep one `wo_a` layout, drop the unused tile-major copy (−688 MiB/GPU) |
| `FLARE_DS41_EXL3_MERGE_TOPK`, `_MERGE_EPILOGUE` | `1`, `1` | resident-expert lookup folded into the routing prep; routed + shared expert outputs added in the MoE epilogue (bit-exact) |
| `FLARE_DS41_EXL3_MERGE_ROPEQ`, `_BULK_MIN_ROWS` | `0`, `0` | measured, no gain: off |
| `FLARE_DS41_EXL3_COLD_WIDTH` | `16` | host-expert decode launches 16 wide |
| `FLARE_DS41_EXL3_COLD_SIDE` | `16` | host-expert launch on a side stream with 16 SMs |
| `FLARE_DS41_EXL3_SHARED_SIDE` | `1` | shared expert on a side stream beside the router |
| `FLARE_DS41_EXL3_FUSED_PREP`, `_FUSED_PREP_MAX_ROWS` | `1`, `24` | fused routing prep for graphs up to 24 rows |
| `SIDE_STREAMS` | `1` | indexer chain and DSpark context projection on side streams |
| `--max-concurrency` (`MAXC`) | `8` | request slots |
| `--prefill-chunk-size` (`PREFILL`) | `4096` | prefill chunk |
| `--mtp-draft-tokens` (`DRAFTS`) | `5` | DSpark drafts (NVFP4 draft head, FP8 Markov head); must match `serve.sh` |
| `--mtp-only` (`MTP_ONLY`) | on | no graphs for serving without speculation |
| `--image-concurrency` (`IMAGE_CONCURRENCY`) | `1` | one image feature slot (a second image request waits; text requests are still admitted) |
| `FLARE_TUNE_SCRATCH_TOLERANCE`, `_MIN_SAVING_MIB` | `0.05`, `16` | prefer the lower-scratch kernel within 5% of the fastest |

Also on from the fork script's defaults: host-row embedding gather, vision tower in host RAM with staging, VRAM
reclaim, PDL in the mHC norm kernels, vocabulary-parallel sampling. Every switch is recorded in the package's
`tp3-config.json`.

### What to expect, and how long

Stage times of the release export (kernel cache warm from earlier exports of this pack, 5 compiler workers):

| Stage (log line) | Time |
| --- | --- |
| Load weights | 254 s |
| Construct graph catalog | 64 s |
| Prepare and tune kernels | 117 s |
| Compile model forwards | 764 s |
| Load and warm compiled forwards | 287 s |
| Prepare serving graphs | 377 s |
| Export native core | 551 s |
| **Total** | **~41 min** |

A first export on a new machine also tunes kernels from scratch and builds the `xmoe` extension; expect 1-1.5 h.

When you serve it (step 5) with `serve.sh` defaults, the `Allocate cache` line should show about **2,140,800 KV
tokens** and `Prepare kernels` should log `CUDA stack limit 128 B per thread (... largest kernel stack 120 B)`.
Another machine, driver or clock setting can land a little higher or lower.

## Trading speed for KV

See [KV-TRADEOFF.md](KV-TRADEOFF.md): how the resident expert counts (`HOT_LOW` / `HOT_HIGH`) trade KV against decode
speed in both directions, with example settings. To choose the resident experts from your own traffic, see
[EXPERT-ORDER.md](EXPERT-ORDER.md).

## 5. Check and serve

```bash
flare-cli/target/release/flare models list            # DeepSeek-V4.1-Flash-1017-EXL3-3bpw, version gb202-tp2-<timestamp>
FLARE=flare-cli/target/release/flare /path/to/this-folder/serve.sh
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015` (about a minute with the Engram tables
loading into RAM). Then a quick benchmark:

```bash
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
  --concurrency 1,4,8 --contexts 32k --output result.json
```

**Less host RAM:** `ENGRAM=one ./serve.sh` puts one Engram table on disk (~41 GiB less RAM, no measurable speed cost);
`ENGRAM=disk ./serve.sh` puts both on disk (~87 GiB less RAM, decode -1 to -3%, prefill -10 to -20%). Rows are read
from the package through a row cache (`ENGRAM_CACHE` sets its size). Needs a fast NVMe; measured numbers in
[README, Memory](README.md#memory).

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tuning fails with `OutOfMemoryError ... failed to prepare with configuration` | a cold tuning cache can need more free VRAM than serving (seen on our TP3 build, not on this one). Export once with lower counts (e.g. `HOT_LOW=320 HOT_HIGH=300`) to fill the shared kernel cache, then export the release setting again; the second export reuses the cache |
| `refusing: ... is set` from the export script | a debug or BRRR variable is set in your shell; unset it (BRRR runs MXFP4 experts only) |
| `... is not an EXL3 pack` | `CKPT` does not point at the pack directory (the one with `config.json`) |
| `xmoe` / merge kernel build fails (`nvcc`, `ninja`) | `CUDA_HOME` must point at CUDA 13.1 (`nvcc` with `sm_120a`); `ninja` comes with `uv sync` |
| Export dies with `runtime patch context is missing or ambiguous` or a torch/Triton version error | re-run `uv sync --locked --extra prepare --extra test`; do not upgrade torch or Triton by hand |
| The server refuses to load with a stack-limit error | `STACK_LIMIT` is below the largest stack of a registered kernel (logged in `Prepare kernels`). Use `STACK_LIMIT=256`, or `STACK_LIMIT=` for the driver default |
| `serve.sh` cannot find the model | the pack directory had another name (step 3); `flare models list` shows it; serve with `PACKAGE=<name>` or the package path |
| First `flare run` of a new package asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal |
| The command does nothing and exits 0 | your shell's `env` is not `/usr/bin/env` (uv installs one at `~/.local/bin/env`); the scripts here call `/usr/bin/env` explicitly |
| The server fails to start or the system swaps | not enough host RAM for the Engram tables: `ENGRAM=one` or `ENGRAM=disk`, or a smaller `PREFIX_CACHE` |
| HTTP 400 "Thinking mode does not support this tool_choice" | `tool_choice` `"required"` or a named function with thinking on; send it with `chat_template_kwargs: {"enable_thinking": false}` |
| Disk fills up after several exports | `flare models remove <model>@<version>`, then `flare models gc` |
