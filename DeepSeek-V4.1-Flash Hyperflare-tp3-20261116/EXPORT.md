# Exporting the package yourself

This guide builds the Hyperflare package in this folder from scratch. It covers the toolchain, the patched fork, the
checkpoint, the export, and the checks to run before serving. Budget **about 2 hours of mostly unattended time**
and **~1.1 TB of free disk**.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (251 GiB usable; DDR5-6000, 8 channels) | **256 GB is the minimum for this build.** Serving with both FP8 Engram tables in RAM commits 217.3 GiB at `PREFIX_CACHE=12G` (dry run); at boot 21.0 GiB stayed available, and 10.7 GiB at the lowest point of a 16-stream KV stress. The export itself left 193 GiB available (it needs ~50 GiB). With less RAM see `ENGRAM_RAM=layer1` in step 5. The 20261111 MXFP4 build committed 145.7 GiB |
| Disk | ~1.1 TB free | DeepSeek checkpoint ~510 GB, package ~470 GiB, kernel tuning cache 60-80 GB. Tested on PCIe Gen5 NVMe (12 GB/s reads) |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | nvcc 12.x is too old for SM120. The export also compiles the BRRR layer kernel with it |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | the fork's Python environment (Python 3.12, managed by uv) |
| Locked memory | `memlock unlimited` for your user | Hyperflare pins host memory; for example `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

`check_env.sh` in this folder checks all of the above plus the fork state (step 2), free disk and idle GPUs. It is
read-only and prints one `PASS` / `WARN` / `FAIL` line per requirement; `WARN` lines are steps you have not done yet.
Run it once the clone from step 2 exists and again after each step:

```bash
/path/to/this-folder/check_env.sh <fork checkout>      # exit 1 if anything FAILed; MIN_FREE_GB=<n> if checkpoints live on another drive
```

The exact versions behind every number in this folder (driver, toolkit, torch, FlashInfer, benchmarks) are in
[versions.txt](versions.txt). torch's `cu132` wheel next to the 13.1 toolkit is expected, not a mismatch.

## 2. Build the fork

The fork is upstream NVIDIA Hyperflare (formerly Flare) at `70069281` plus `source.patch` from this folder.
(Releases up to 20261111 were based on `972e28e4`, the 2026-11-10 sync builds on `cbf21a12`; the 2026-11-14 port moved
the base to `70069281`. A checkout of an older base needs a fresh `git checkout`, not a patch on top.)

```bash
git clone https://github.com/NVIDIA/hyperflare-inference.git flare
cd flare
git checkout 70069281cb3ace4365b86f4bfbbcb409da89434b
git apply /path/to/this-folder/source.patch           # applies cleanly to 70069281

uv sync --locked --extra prepare --extra test         # Python environment in .venv (FlashInfer @ aa6c4221 = 0.7.3, CUTLASS DSL 4.8.1, torch 2.14.1+cu132, Triton 3.8.0)
cargo build --locked --release --manifest-path flare-cli/Cargo.toml  # the flare binary: flare-cli/target/release/flare
```

Hyperflare's torch/Triton fixes are not a separate step: `ignition.runtime_patches` applies them in memory during
`ignition export` and checks them against the pinned torch 2.14.1 / Triton 3.8.0.

**Apply the FlashInfer patches** to the installed FlashInfer. They patch files inside `.venv`, so re-apply them
after every `uv sync`:

```bash
SP=$(.venv/bin/python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parents[1])")
for p in /path/to/this-folder/tp3-patches/*.patch; do patch -p1 -d "$SP" < "$p"; done
```

| Patch | Why |
| --- | --- |
| `flashinfer-dcp-a2a-world3.patch`, `flashinfer-tuning-world3.patch` | PCIe all-to-all and its tuning tables for world size 3 |
| `flashinfer-engram-mxfp4.patch` | packed-FP4 path in the Engram lookup kernel, for NVIDIA's MXFP4 tables (not used by the FP8 view; kept so the MXFP4 view still exports) |
| `flashinfer-tuning-scratch-preference.patch` | `FLARE_TUNE_SCRATCH_TOLERANCE`: among race candidates within the tolerance of the fastest, pick the smaller workspace (`export.sh` sets 0.05) |
| `overlay/flashinfer-pcie-twoshot-pull-world3.patch` | **not applied to the venv.** Adds the two-shot pull all-reduce as a tuning candidate (`B12X_PCIE_TWOSHOT_PULL=1`). It goes into a separate copy of b12x (the overlay) that `export.sh` puts first on the import path |

Build the overlay once, from the fork checkout, after the venv patches:

```bash
VENV=$PWD/.venv B12X_OVERLAY_PATCHES=/path/to/this-folder/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch \
  tp3-tools/make-b12x-overlay.sh $PWD/.b12x-overlay-twoshot      # prints "overlay ready: ..."
```

`flashinfer-tiny-decode-swiglu.patch` is gone since the FlashInfer aa6c4221 pin: FlashInfer now applies `swiglu_limit`
inside the `tiny_decode` W4A8 MoE kernel (outputs corrupted mid-reply at TP3 without it on older pins).

Check the build:

```bash
flare-cli/target/release/flare --version
.venv/bin/ignition export --help | head -3
```

## 3. Build the checkpoint

The package is exported from a **checkpoint view**: symlinks to every file of DeepSeek's original snapshot. Shards
47-48 hold the two Engram tables in FP8 (189 GiB). No NVIDIA download is needed; the 20261111 build took MXFP4 versions
of those two shards (97 GiB) from `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170   # ~510 GB of tensors
/path/to/this-folder/make_checkpoint_view.sh \
  ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  ~/ds41-1017-engram-fp8
# expect: two "Engram table F8_E4M3 ... ok" lines and
#         "96085 tensors in 48 shards, missing 0; payload 510286023000 vs index total_size 510286023000"
```

DeepSeek's own `config.json` and `model.safetensors.index.json` are used unchanged (its `total_size` matches the FP8
payload).

**The directory name becomes the model name.** Hyperflare names the exported model after the checkpoint directory: the
package lands in `~/.cache/flare/packages/models/<directory name>/gb202-tp3-<timestamp>`, `flare models list` shows
it as `<directory name>`, and `flare run <directory name>` serves it. `serve.sh` defaults to
`PACKAGE=ds41-1017-engram-fp8`, the name used here; if you name the view differently, serve with `PACKAGE=<your name>`
(or the package path).

## 4. Export

From the fork checkout, with GPUs idle (stop any server first). **Export twice on a fresh machine:** with a cold
kernel tuning cache, tuning at `HOT=342` runs out of VRAM. A `HOT=300` export first fills the shared cache, and the
second export reuses it:

```bash
CKPT=~/ds41-1017-engram-fp8 B12X_OVERLAY=$PWD/.b12x-overlay-twoshot HOT=300 /path/to/this-folder/export.sh   # warms the tuning cache
CKPT=~/ds41-1017-engram-fp8 B12X_OVERLAY=$PWD/.b12x-overlay-twoshot /path/to/this-folder/export.sh           # the release build (HOT=342)
```

`DRAFTS` (default 5) must be the same at export and in `serve.sh`: the package is captured for one draft count. The
release export ran with `COMPILE_WORKERS=3` (a power-supply limit of the test system; the default sizes the compiler
pool automatically). The export does not change with it, only its duration.

`export.sh` sets these and runs `ignition export`. Every value below is the default; the variable in parentheses
changes it:

| Setting | Value | Meaning |
| --- | --- | --- |
| `FLARE_DS41_HOST_EXPERT_LAYERS` | `20-33` | routed-expert layers that keep part of their experts in host RAM |
| `FLARE_DS41_HOT_EXPERTS` (`HOT`) | `342` | experts per parked layer kept in VRAM (out of 384); the rest live in host RAM |
| `FLARE_DS41_MERGED_BANK` | `1` | one VRAM + host-RAM expert bank per parked layer (one MoE call) |
| `FLARE_DS41_EXPERT_COUNTS_DIR` | `expert-order/` | route counts that decide which experts are hot |
| `FLARE_DS41_BRRR_LAYERS` | `0-39` | layers that use the fused BRRR layer tail (compiled from `brrr/kernels/decoder.cu` at export) |
| `FLARE_DS41_BRRR_CKPT` | `$CKPT` | checkpoint the BRRR tail reads its weight layout from |
| `FLARE_DS41_BRRR_PARKED_MAX_ROWS` (`PARKED_ROWS`) | `0` | parked layers use the BRRR tail at every graph width (20261111: only up to 12 rows) |
| `FLARE_DS41_BRRR_WOA_UNEVEN` / `_WOA_SMALL` (`WOA_UNEVEN`, `WOA_SMALL`) | `1` / `0` | BRRR job layout: uneven wo_a split at ≤ 8 rows; own extension build |
| `FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST` | `1`, `1` | token embedding and vision tower in device-visible host RAM |
| `FLARE_DS41_EMBED_ROWS` (`EMBED_ROWS`) | `1` | vectorised host-row embedding gather instead of b12x's lookup in decode graphs |
| `FLARE_DS41_VISION_STAGE` | `1` | stage the vision tower's weights through two VRAM slots per block (cold image 0.11 s instead of 1.70 s) |
| `FLARE_DS41_AR_ROWS` | `1024` | size the prefill all-reduce runtime for 1024 rows |
| `FLARE_DS41_AR_PART_ROWS`, `FLARE_DS41_ENGRAM_PART_ROWS` (`AR_PART_ROWS`, `ENGRAM_PART_ROWS`) | `768`, `768` | prefill all-reduce parts of at most 768 rows for the decoder and the Engram projection (5 × 768 + 256 per 4096-row chunk instead of 4 × 1024) |
| `FLARE_DS41_ENGRAM_CHUNK_ROWS`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL` | `1024`, `1` | Engram projection in 1024-row chunks; `engram_residual` shares a buffer |
| `FLARE_DS41_INDEX_SIDE_STREAM` (`INDEX_SIDE`) | `1` | indexer query chain on a side stream |
| `FLARE_DS41_CONTEXT_SIDE_STREAM` (`CONTEXT_SIDE`) | `1` | DSpark context projection on a side stream beside verification |
| `FLARE_DS41_ENGRAM_SIDE_LAYERS` (`ENGRAM_SIDE`) | unset | Engram side stream; off (layer 14 measured level, not adopted) |
| `FLARE_DS41_VRAM_RECLAIM` (`RECLAIM`) | `1` | shared Engram token map, trimmed vocabulary gather, no draft-vocabulary channel, shared vocabulary columns |
| `B12X_PCIE_TWOSHOT_PULL` (`TWOSHOT`) | `1` | race the two-shot pull all-reduce beside the upstream candidates; needs `B12X_OVERLAY` |
| `B12X_MLA_DECODE_REMAINDER_STREAM` (`ATTN`) | `1` | each attention decode's 16-head grid on a side stream beside the 8-head remainder grid |
| `B12X_MHC_PDL` (`MHC_PDL`) | `1` | programmatic dependent launch in b12x's mHC norm kernels |
| `FLARE_TUNE_SCRATCH_TOLERANCE`, `_MIN_SAVING_MIB` | `0.05`, `16` | prefer the lower-scratch kernel within 5% of the fastest |
| `FLARE_EXPORT_LIBRARY_SCRATCH` (`LIBRARY_SCRATCH`) | code default (on) | cuBLAS's own 64 MiB pool is kept out of the export's reachability; `0` = off. The release package was exported from code before this change; on the same settings it gave the same KV |
| `FLARE_DS41_TP3_VOCAB_PARALLEL`, `FLARE_DS41_TP3_GRAPH_GIB` | `1`, `1` | vocabulary-parallel sampling at TP3 (fixes top_p at 1.0); the planner's graph-memory estimate |
| `--max-concurrency` (`MAXC`) | `16` | request slots the package captures graphs for (20261111: 8) |
| `--mtp-only` | on | omit the graphs for serving without speculation (the package needs `--mtp-draft-tokens` > 0 at serve time) |
| `--prefill-chunk-size` (`PREFILL`) | `4096` | prefill chunk |
| `--mtp-draft-tokens` (`DRAFTS`) | `5` | DSpark drafts, NVFP4 draft head, FP8 Markov head |

Every switch is recorded in the package's `tp3-config.json`.

### What to expect, and how long

Stage wall times of the release export (warm tuning cache, 3 compiler workers), from its log:

| Stage (log line) | Time |
| --- | --- |
| Load weights | 46.5 s |
| Construct graph catalog | 13.3 s |
| Prepare and tune kernels | 200.2 s (warm cache; the cold `HOT=300` warm-up tuned for 955 s) |
| Compile model forwards | 767.0 s |
| Load and warm compiled forwards | 205.8 s |
| Prepare serving graphs | 402.4 s |
| Export native core | 393.0 s |
| **Wall time to `Exported cache entry`** | **2,076 s (34 min)** |

When you then serve it (step 5) with `serve.sh` defaults, the server log should show about **5,403,008 KV tokens** in
the `Allocate cache` line. Without the CUDA stack limit (`STACK_LIMIT=`) it is 5,155,456. `flare run <package> --dry-run`
predicts 4,996,096: the dry run does not model the stack limit or the measured idle reserve exactly.

### Variants: more KV for less speed

The hot-expert count is the main lever. Each hot expert per parked layer moved to host RAM frees ~79 MiB per GPU,
about **92K KV tokens** (measured before the port). Decode gets slower because more experts stream over PCIe. Only
`HOT=342` was measured on this build; check the `Allocate cache` line and bench your own export before relying on
another value.

## 5. Check and serve

```bash
flare-cli/target/release/flare models list                    # model ds41-1017-engram-fp8 (= the view's directory name), version gb202-tp3-<timestamp>, newest = default
FLARE=flare-cli/target/release/flare /path/to/this-folder/serve.sh            # serves ds41-1017-engram-fp8; PACKAGE=<name>@<version> or a path for another one
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015`, about 55 s with the Engram tables
loading into RAM. Keep `DRAFTS` equal to the export's.

**Less host RAM:** `ENGRAM_RAM=layer1 ./serve.sh` keeps layer 1's table in RAM and reads layer 14's from disk through a
32 GiB row cache (`ENGRAM_CACHE`). Measured on the FP8 package before the round-2 switches (both at a 32G prefix
cache): the dry run commits 177.8 instead of 237.3 GiB of host RAM, the lowest available RAM was 63.0 GiB, decode at
c1-c16 was level with both tables in RAM, and an 8-stream agent replay gave 18% less output. `ENGRAM_RAM=0`
(both tables on disk) is not measured with the FP8 tables.

Then run the corruption check, which compares sampled decode tokens against a teacher-forced prefill and must
report `TRUE corruptions 0`, and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 10
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
  --concurrency 1,4,8,16 --contexts 32k --output result.json
python3 /path/to/this-folder/tools/bench_steps.py result.json   # steps/s and tokens/step per cell
```

Optional: `tools/llm-bench-flare-steps.patch` (`git apply` inside `llm-inference-bench`, tested on v0.9.1; the
benchmark is MIT-licensed, see its README) makes the benchmark's own summary show the "MTP-normalized decode steps/s"
table for Hyperflare, computed from its draft-round counters.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tuning fails with `OutOfMemoryError ... failed to prepare with configuration` | A **cold tuning cache** needs more free VRAM than serving. Export with `HOT=300` first, then the target: the tuning cache in `~/.cache/flare/kernels` is shared and the second export reuses it. A new draft count or a new `MAXC` needs its own warm-up export |
| `export.sh` stops with `TWOSHOT=1 needs B12X_OVERLAY` | build the overlay (step 2) and pass `B12X_OVERLAY=<dir>`, or export with `TWOSHOT=0` (upstream all-reduce candidates; not the measured build) |
| Export stops at `KeyError: 'FLARE_DS41_BRRR_CKPT'` | you ran `ignition export` directly; use `export.sh` or set `FLARE_DS41_BRRR_CKPT` to the checkpoint view |
| BRRR kernel compile fails (`nvcc`, `ninja`) | `CUDA_HOME` must point at CUDA 13.1 (`nvcc` with `sm_120a`); `ninja` comes with `uv sync`. Compiled kernels are cached in `.brrr-ext/` in the checkout |
| `index total_size does not match tensor payload bytes` | the view does not link DeepSeek's own index, or a shard is incomplete: rebuild the view with `make_checkpoint_view.sh` |
| Export dies with `runtime patch context is missing or ambiguous` or a torch/Triton version error | `ignition.runtime_patches` does not match the installed torch/Triton: re-run `uv sync --locked --extra prepare --extra test`; do not upgrade torch or Triton by hand |
| The server refuses to load with a stack-limit error | `--cuda-stack-limit` / `FLARE_CUDA_STACK_LIMIT` is below the largest stack of a kernel the package registers (DeepSeek release package: 176 B, logged in the `Prepare kernels` line), or a kernel's stack cannot be bounded. Use `STACK_LIMIT=256` (default) or `STACK_LIMIT=` (driver default 1,024 B, fewer KV tokens) |
| HTTP 400 "Thinking mode does not support this tool_choice" | a request with `tool_choice` `"required"` or a named function while thinking is on. DeepSeek's reference rejects it the same way (before: HTTP 500 when the model answered without a call). Send it with `chat_template_kwargs: {"enable_thinking": false}` to force the call |
| Chat requests return 400 for `thinking` / `clear_thinking` | Hyperflare's thinking toggle is `chat_template_kwargs: {"enable_thinking": true}` (or `false`); vLLM's `thinking` / `clear_thinking` fields are rejected |
| The command does nothing and exits 0 | your shell's `env` is not `/usr/bin/env` (uv installs one at `~/.local/bin/env`); the scripts here call `/usr/bin/env` explicitly |
| Corruption check reports true corruptions | FlashInfer older than aa6c4221 without the old `flashinfer-tiny-decode-swiglu.patch` (`check_env.sh` reports it) |
| `serve.sh` fails because `ds41-1017-engram-fp8` is not found | the checkpoint view had another directory name, and the model name follows it (step 3). `flare models list` shows the name; serve with `PACKAGE=<name>` or the package path |
| The server fails to start or the system swaps at boot | not enough host RAM for both FP8 tables (step 1): use `ENGRAM_RAM=layer1`, or a smaller `PREFIX_CACHE` |
| First `flare run` of a new package asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (deleting folders by hand frees nothing: files are shared) |

## History

**Update 2026-11-16 (this release).** New base `70069281` (the 2026-11-14 port onto upstream's compiler-planned memory
scheme), FP8 Engram view instead of MXFP4, 16 request slots, the round-2 export switches and the b12x two-shot overlay
(step 4 table), serve defaults budget 32, prefix cache 12G, TTFT guard order `shortest`, prefill merge 512 and CUDA
stack limit 256 (`serve.sh` header).

**Update 2026-11-08.** `source.patch` now carries the BRRR kernel hygiene pass (cold-first parked jobs, per-chunk attention
flags, a constants contract between kernel and config, a cross-rank agreement check), vision weight staging
(`FLARE_DS41_VISION_STAGE=1`, the new `export.sh` default: 0.11 s per image instead of 1.70 s), one resolution point for
Engram table storage in Helios, `flare run ... --dry-run` (KV capacity without a serve) and three GPU equivalence tests.
It was regenerated from the fork after its history was cleaned for publication (experiment scripts removed, tools
under `tp3-tools/`), and verified to reproduce the fork tree exactly on a clean `972e28e4` checkout. Measurements in
MEASUREMENTS.md ("2026-11-08").

**Update 2026-11-07 (review pass).** `source.patch` now also carries the fixes from a code review of this release. None of
them change the exported package or its numbers; they matter for serving and for other setups:

| Fix | Effect |
| --- | --- |
| Helios counted the TP3 merged expert banks (VRAM prefix + host tail) as all-VRAM in its GPU admission check and in `flare models describe` | the check could reject a package that fits by a few GiB; fixed in all six accounting sites |
| `FLARE_ENGRAM_RAM_TABLES` tables were not counted toward `--memory-limit` host accounting, and the disk row-cache budget was split across tables no longer on disk | correct host accounting; disk tables get the whole cache budget |
| `--history-drafts` inputs were advertised for every DeepSeek export but only consumed under vocabulary-parallel sampling | TP4 exports behave like upstream again (TP3 unchanged) |
| Guards: the BRRR tail checks its attention-output dtype and RoPE table; the parked-layer gate requires an actual host tail; the export fails loudly if b12x ever re-materialises host-parked experts (TP4-style tiling would) | no silent garbage or VRAM over-commit on future changes |
| Three opt-in experiments measured level and left switched off (`FLARE_DS41_BRRR_SPARSE`, `FLARE_DS41_FUSED_QKV_NORM`, `FLARE_DS41_QB_ATOMIC`). The FlashInfer patch they rely on, `flashinfer-dense-splitk-deferral.patch`, is fork-only and not shipped here (`flashinfer-tuning-route-reuse.patch` was dropped at the FlashInfer 85ea6e76 sync: upstream FlashInfer #78 makes whole-batch route reuse part of the default MoE tuning workload); `tp3-patches/` holds only the four the default build needs | default build untouched; see MEASUREMENTS.md |

A `flare` binary built from the updated patch serves the published package as before (checked: same KV count, corruption check clean).
