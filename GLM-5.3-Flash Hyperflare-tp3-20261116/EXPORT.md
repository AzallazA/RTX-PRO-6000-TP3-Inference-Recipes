# Exporting the packages yourself

This guide builds the Hyperflare (formerly Flare; the binary is still `flare`) packages in this folder from scratch:
the TP3 package (3 GPUs, up to 32 requests) and the TP2 package (2 GPUs, up to 8 requests), both with image input.
Both come from the same fork checkout, the same Python environment and the same checkpoint; only the export script
differs. Budget **up to an hour of mostly unattended time per export** on a machine that has exported GLM before
(longer on a fresh machine, while kernel tuning fills its cache) and **~550 GB of free disk** for the checkpoint plus
both packages.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 (TP2: GPUs 0 and 1 of the same machine) | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (DDR5-6000, 8 channels) | Serving: the host prefix cache (`PREFIX_CACHE=32G`; the server log reports 39.7 GiB at TP3 and 30.9 GiB at TP2), the parked vision tower (373 MiB per GPU) plus, at TP2, the token embedding |
| Disk | ~550 GB free | checkpoint ~178 GB (166 GiB), TP3 package 174 GiB, TP2 package 168 GiB, plus the kernel tuning cache (`~/.cache/flare/kernels`) |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | nvcc 12.x is too old for SM120 |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | Hyperflare's Python environment (Python 3.12, managed by uv) |
| Locked memory | `memlock unlimited` for your user | Hyperflare pins host memory; for example `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

The exact versions behind every number in this folder are in [versions.txt](versions.txt).

## 2. Build the Hyperflare fork

The fork is upstream NVIDIA Hyperflare `main` at `70069281` plus our TP3 work (the DeepSeek V4.1 TP3 port, the GLM
TP3/TP2 work and the GLM VRAM reclaim switches). It is published as the public fork
https://github.com/AzallazA/spark-local-inference at tag `hyperflare-glm-20261116`. The same tree builds the DeepSeek
V4.1 20261116 package; one `flare` binary serves both models.

Without the tag: clone upstream, `git checkout 70069281`, then `git apply /path/to/this-folder/source.patch`
(applies cleanly; the result is the tag's tree).

```bash
git clone https://github.com/AzallazA/spark-local-inference.git flare
cd flare
git checkout hyperflare-glm-20261116

uv sync --locked --extra prepare --extra test               # Python environment in .venv (pulls FlashInfer @ aa6c4221)
cargo build --locked --release --manifest-path flare-cli/Cargo.toml   # the flare binary: flare-cli/target/release/flare
```

**Apply the FlashInfer patches** to the installed FlashInfer. They patch files inside `.venv`, so re-apply them
after every `uv sync`. The fork's `tp3-patches/` and this folder's copy are the same files:

```bash
SP=$(.venv/bin/python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parents[1])")
for p in tp3-patches/*.patch; do patch -p1 -d "$SP" < "$p"; done
```

| Patch | Why |
| --- | --- |
| `flashinfer-dcp-a2a-world3.patch` | PCIe all-to-all and DCP exchange at world size 3 (the TP3 DCP head exchange and candidate gather) |
| `flashinfer-tuning-world3.patch` | tuning tables for world size 3 |
| `flashinfer-engram-mxfp4.patch`, `flashinfer-tuning-scratch-preference.patch` | DeepSeek V4.1 patches; applied in the measured environment (one venv serves both models). The scratch preference acts only when `FLARE_TUNE_SCRATCH_TOLERANCE` is set, which the GLM scripts do not do |

`flashinfer-tiny-decode-swiglu.patch` and `flashinfer-dense-splitk-deferral.patch` of the 20261112 folder are not
needed on this FlashInfer pin.

`overlay/flashinfer-dcp-gather-pull-world3.patch` is **not** applied to `.venv`. The export scripts build a separate
copy of b12x with it (`.b12x-overlay/` in the checkout, through the fork's `tp3-tools/make-b12x-overlay.sh`) and put
that copy first on `PYTHONPATH`. It makes the world-3 DCP head gather pull peers' rows instead of pushing them
(923 → 63 µs at 48 rows; +7-8% decode at c1/c4/c8 when measured on 2026-11-09). At two GPUs it compiles the same push
kernels as upstream. Build it after the venv patches (it copies the patched b12x). The DeepSeek release uses a
different overlay patch (two-shot pull); keep the two overlays in separate directories.

Check the build:

```bash
flare-cli/target/release/flare --version
.venv/bin/ignition export --help | head -3
```

## 3. Get the checkpoint

```bash
hf download nvidia/GLM-5.3-Flash-FP4-Local --revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
  --local-dir ~/glm53-fp4-local                             # ~178 GB, 40 shards
```

The checkpoint is used as published; no view or corrected index is needed.

**The directory name becomes the model name.** Hyperflare names the exported model after the checkpoint directory:
the packages land in `~/.cache/flare/packages/models/<directory name>/gb202-tp3-<timestamp>` and
`.../gb202-tp2-<timestamp>`, `flare models list` shows `<directory name>`, and `flare run <directory name>` serves
the default version. The serve scripts default to `PACKAGE=glm53-fp4-local`. **Both packages share that name**, so if
you export both, serve each by version (`PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp>`) or by path.

## 4. Export

From the fork checkout, with GPUs idle (stop any server first):

```bash
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp3.sh    # TP3, GPUs 0,1,2, CONC=32
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp2.sh    # TP2, GPUs 0,1, CONC=8
```

The scripts are copies of the tag's `tp3-tools/release/glm/export-glm-tp3.sh` and `export-glm-tp2.sh`. What they set:

| Setting | TP3 | TP2 | Meaning |
| --- | --- | --- | --- |
| `--gpus` | `0,1,2` | `0,1` | three GPUs are the only way to get TP3 (a 4-GPU host auto-selects TP4) |
| Decode context parallelism | DCP3 (`FLARE_GLM53_DCP3=1`, fork-only) | DCP2 (upstream) | the KV cache is split over the ranks: 768-token pages at TP3 (256 records per rank), 512 at TP2 |
| `FLARE_GLM53_EXPERT_COLUMNS` | `704` | — | routed experts 704 columns per rank instead of 768 (2048 padded to 2112, not 2304) |
| `FLARE_GLM53_EXPERT_ACTIVATION` | `a16` | `a16` | 16-bit activations into the NVFP4 experts at decode; the recipe default on SM120, pinned here |
| `FLARE_GLM53_KDA_SIDE_CHAIN` | `1` | `1` | the KDA gate/decay projections run beside the q/k/v GEMM |
| `FLARE_GLM53_EMBED_HOST` | — | `1` | the replicated BF16 token embedding in device-visible host memory; the VRAM goes to KV |
| `FLARE_GLM53_TRIAL_POOL` (`RECLAIM=1`) | `separate` | `separate` | **new:** the eager "Measure graph memory" trial graphs get a pool of their own, released before the serving capture |
| `FLARE_GLM53_SLAB_SEGMENT` (`RECLAIM=1`) | `1` | — | **new:** the preparation-only context-parallel prefill slab takes whole 2 MiB granules |
| `FLARE_GLM53_ONESHOT_FIT` (`RECLAIM=1`) | — | `1` | **new:** the one-shot all-reduce sized for the 512 rows that use it (refused at three ranks) |
| `FLARE_BF16_BMM_CUBLAS` | `0` | `0` | **new:** GLM's BF16 head BMMs race Triton tiles only (contains a cuBLAS over-read on SM120) |
| `B12X_OVERLAY` | `.b12x-overlay` | `.b12x-overlay` | world-3 pull gather (step 2) |
| `--mtp-only` | yes | yes | no graphs for plain (no-draft) decode; the package refuses `--mtp-draft-tokens 0` |
| `--max-concurrency` (`CONC`) | `32` | `8` | request slots the package captures graphs for |
| `--mtp-draft-tokens` / `--prefill-chunk-size` | `5` / `4096` | `5` / `4096` | the recipe defaults |
| Images (`VISION`, default 1) | `--image-concurrency 4` | `--image-concurrency 2` | `FLARE_GLM53_VISION_STAGE=1` parks the vision tower (blocks and tail) in host memory and stages it per image. `VISION=0` exports text-only (`--language-model-only`) |
| `COMPILE_WORKERS` | unset | unset | `--compile-workers`; our exports used 3 (set in the Hyperflare config) to stay inside the test system's power-supply limit |

`RECLAIM=0` exports the 20261112 memory layout (switches unset; the code paths are upstream's). The export KV
hygiene of the DeepSeek 20261116 release (cuBLAS's own 64 MiB pool kept out of the export's reachability) is on by
default in this code and was on for these exports.

Our TP2 export timed the rank-local MoE expert tuning races one GPU at a time (`B12X_SERIAL_RACES`, a local b12x patch
that is not part of this folder; a power-supply precaution on the test machine). It changes only how candidates are
timed, not what they compute; without it the races run on both GPUs at once, as in the TP3 export.

### What to expect, and how long

On the test machine (tuning cache warm from earlier GLM exports on the same FlashInfer pin):

| | TP3 package 202611170053 | TP2 package 202611170201 |
| --- | --- | --- |
| Wall time | 35 min | 17 min (resumed the tuning database of an earlier TP2 export) |
| Load weights | 53 s | — |
| Prepare and tune kernels | 118 s | 270 s |
| Compile model forwards (3 compiler workers) | 765 s | — |
| Graph catalog | 74 graphs (text-only: 68) | 56 graphs (text-only: 50) |
| Reclaim lines in the log | `trial graph pool released: 3574 of 3574 MiB returned before the serving capture`; `prefill slab 528.26 MiB in 530 MiB of its own segments` | `trial graph pool released: 2990 of 2990 MiB ...`; `one-shot all-reduce sized for 512 rows (4 MiB per call) instead of 4096` |
| Package on disk | 174 GiB | 168 GiB |

A fresh machine starts with an empty tuning cache and takes longer. If tuning runs out of VRAM on a cold
cache, export once with a lower `CONC` (for example `CONC=8`) to fill the shared tuning cache, then export again
with the release setting.

The export ends with:

```text
[    OK    ] Commit core.  <store>/models/glm53-fp4-local/gb202-tp3-<timestamp>  0.0s
Exported cache entry: <store>/models/glm53-fp4-local/gb202-tp3-<timestamp>
```

## 5. Check and serve

KV capacity without loading the model (needs the driver and visible GPUs, but no GPU memory):

```bash
flare-cli/target/release/flare run glm53-fp4-local@gb202-tp3-<timestamp> --gpus 0,1,2 --served-model-name GLM-5.3-Flash \
  --prefix-cache-memory 32G --mtp-draft-tokens 5 --dry-run
# release packages at 97,281 MiB free per GPU (--dry-run-free-memory): TP3 15,025,152 KV tokens; TP2 (--gpus 0,1) 3,177,472
```

Serve (from the fork checkout, so the default `FLARE` path resolves):

```bash
PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp> /path/to/this-folder/serve-glm-tp3.sh
PACKAGE=glm53-fp4-local@gb202-tp2-<timestamp> /path/to/this-folder/serve-glm-tp2.sh
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015`. The `Allocate cache` line reports
the KV count actually allocated; on the test machine (export chain binary, no stack-limit override):

```text
TP3: [    OK    ] Plan cache.  per GPU: 61.48–61.49 GiB loaded · 32.92 GiB cache budget  0.0s
     [    OK    ] Allocate cache.  98.7 GiB · 14834688 KV tokens · 507 shared snapshot slots · limited by memory  3.4s
TP2: [    OK    ] Plan cache.  per GPU: 84.14 GiB loaded · 10.26 GiB cache budget  0.0s
     [    OK    ] Allocate cache.  20.5 GiB · 3049984 KV tokens · 118 shared snapshot slots · limited by memory  0.4s
```

Then run the corruption check (decode tokens against a teacher-forced prefill; must report `TRUE corruptions 0`),
the tool-call test and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 20
python3 /path/to/this-folder/tools/tool_calls.py 60 32            # 60 prompts at concurrency 32 (8 for TP2); expect 60/60
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model GLM-5.3-Flash \
  --concurrency 1,8,32 --contexts 32k --duration 30 --skip-prefill --output result.json   # 1,4,8 for TP2
```

`verify_fix.py`, `tool_calls.py` and `glm_common.py` read `MODEL` (default `GLM-5.3-Flash`) and `FLARE_URL`
(default `http://127.0.0.1:15015`) from the environment.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| First `flare run` of a new package fails with `core signer is not a configured publisher`, or asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal (or serve with the same config the export used: `CONFIG=` at export means `--config` at serve) |
| `--mtp-draft-tokens 0` is refused | the packages are exported with `--mtp-only`; serve with 5 drafts |
| `serve-glm-tp*.sh` stops with an unknown argument `--scheduler-ttft-order` | the `flare` binary predates the fork tag: rebuild it (step 2), or serve with `TTFT_ORDER=` |
| The server refuses `FLARE_CUDA_STACK_LIMIT` | the limit is below the largest registered kernel stack of the package: serve with `STACK_LIMIT=` (driver default) |
| `serve-glm-tp*.sh` cannot find `glm53-fp4-local` | the checkpoint directory had another name, and the model name follows it (step 3). `flare models list` shows the name; serve with `PACKAGE=<name>@<version>` or the package path |
| The server starts the TP2 package on three GPUs or the TP3 package on two (or refuses) | both packages share the model name; give the version (`PACKAGE=<name>@gb202-tp3-...`) |
| Overlay build fails in `patch` | the venv patches (step 2) are not applied, or the venv FlashInfer is not the `aa6c4221` pin. Re-run `uv sync --locked ...`, apply `tp3-patches/*.patch`, delete `.b12x-overlay/` and export again |
| The export log lacks the `trial graph pool released` line | the export ran with `RECLAIM=0`, or without the fork's Python on `PYTHONPATH` (run the script from the fork checkout) |
| Requests with `enable_thinking: false` still reason | GLM-5.3-Flash always thinks (its chat template has no switch). Use `reasoning_effort` `low` or `high`; any other value renders `max` |
| Temperature or top_p below 0.95 has no effect | the GLM recipe allows 0.95-1.0 for both and clamps other values into that range |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (files are shared between versions) |
