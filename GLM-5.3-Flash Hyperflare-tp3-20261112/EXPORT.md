# Exporting the packages yourself

This guide builds the Hyperflare (formerly Flare; the binary is still `flare`) packages in this folder from scratch:
the TP3 package (3 GPUs, up to 32 requests) and the TP2 package (2 GPUs, up to 8 requests), both with image input. Both come from the same
fork checkout, the same Python environment and the same checkpoint; only the export script differs. Budget **about
1 hour of mostly unattended time per export** on a machine that has exported GLM before (longer on a fresh machine,
while kernel tuning fills its cache) and **~550 GB of free disk** for the checkpoint plus both packages.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 (TP2: GPUs 0 and 1 of the same machine) | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (DDR5-6000, 8 channels) | Serving: the host prefix cache (`PREFIX_CACHE=32G`; the server log reports 39.7 GiB at TP3 and 30.9 GiB at TP2), the parked vision tower (373 MiB per GPU) plus, at TP2, the 2.4 GiB token embedding |
| Disk | ~550 GB free | checkpoint ~178 GB (166 GiB), TP3 package ~177 GB, TP2 package ~168 GB, plus the kernel tuning cache (`~/.cache/flare/kernels`, shared by all models; ours is 143 GB after many DeepSeek and GLM exports) |
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

The fork is upstream NVIDIA Hyperflare `main` at `cbf21a12` plus our TP3 work (the DeepSeek V4.1 TP3 port, the GLM TP3
port, upstream PR #253 and `--mtp-only`, upstream's open PRs #122/#124/#125/#231, the TTFT guard, and GLM vision tower
staging with its lean working memory). It is published
as the public fork https://github.com/AzallazA/spark-local-inference at tag `hyperflare-glm-20261112` (one squashed commit,
`ac9136b0`). One release commit builds both packages: the TP2-only code (the host-memory
embedding, the world-2 branch of the pull gather) is opt-in or compiled only at two ranks, so the TP3 path is the one
the TP3 package was built from.

Without the tag: clone upstream, `git checkout cbf21a12d118217b8ad4e6bfa565562a90addeec`, then
`git apply /path/to/this-folder/source.patch` (316 files, 138 new; applies cleanly). The patch matches the release
commit except that local paths in the fork's notes, scripts and test defaults are replaced by `/path/to/...`
placeholders.

```bash
git clone https://github.com/AzallazA/spark-local-inference.git flare
cd flare
git checkout hyperflare-glm-20261112

uv sync --locked --extra prepare --extra test               # Python environment in .venv (pulls FlashInfer @ 85ea6e76)
cargo build --locked --release --manifest-path flare-cli/Cargo.toml   # the flare binary: flare-cli/target/release/flare
```

There is no `python -m ignition.patches` step on this base: upstream now applies its torch/Triton fixes in memory.
The binary moved from `flare/target/release/flare` (DeepSeek folders up to 20261111) to
`flare-cli/target/release/flare`.

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
| `flashinfer-tiny-decode-swiglu.patch`, `flashinfer-engram-mxfp4.patch`, `flashinfer-dense-splitk-deferral.patch` | DeepSeek V4.1 fixes and experiments; applied in the measured environment (one venv serves both models), inert for GLM as far as we know |

`overlay/flashinfer-dcp-gather-pull-world3.patch` is **not** applied to `.venv`. The export scripts build a separate
copy of b12x with it (`.b12x-overlay/` in the checkout, through the fork's `tp3-tools/make-b12x-overlay.sh`) and put
that copy first on `PYTHONPATH`. It makes the world-3 DCP head gather pull peers' rows instead of pushing them
(923 → 63 µs at 48 rows; +7-8% decode at c1/c4/c8 when measured on 2026-11-09). At two GPUs it compiles the same push
kernels as upstream. Build it after the venv patches (it copies the patched b12x).

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

The fork's `tp3-tools/glm53-export-tp3.sh` and `tp3-tools/glm53-export-tp2.sh` run the same command when given the
release switches (TP3: `MTP_ONLY=1 DCP=1 FLARE_GLM53_EXPERT_ACTIVATION=a16 FLARE_GLM53_KDA_SIDE_CHAIN=1
FLARE_GLM53_EXPERT_COLUMNS=704 CONC=32`; TP2: `GPUS=0,1 MTP_ONLY=1 FLARE_GLM53_EXPERT_ACTIVATION=a16
FLARE_GLM53_KDA_SIDE_CHAIN=1 FLARE_GLM53_EMBED_HOST=1`; both plus `VISION=1 FLARE_GLM53_VISION_STAGE=1` and
`IMAGE_CONCURRENCY=4` at TP3 or `2` at TP2), but they assume `CUDA_HOME=/usr/local/cuda-13.1`.

What the scripts set:

| Setting | TP3 | TP2 | Meaning |
| --- | --- | --- | --- |
| `--gpus` | `0,1,2` | `0,1` | three GPUs are the only way to get TP3 (a 4-GPU host auto-selects TP4) |
| Decode context parallelism | DCP3 (`FLARE_GLM53_DCP3=1`, fork-only) | DCP2 (upstream) | the KV cache is split over the ranks: 768-token pages at TP3 (256 records per rank), 512 at TP2 |
| `FLARE_GLM53_EXPERT_COLUMNS` | `704` | — | routed experts 704 columns per rank instead of 768 (2048 padded to 2112, not 2304); shared expert stays 768 |
| `FLARE_GLM53_EXPERT_ACTIVATION` | `a16` | `a16` | 16-bit activations into the NVFP4 experts at decode; the recipe default on SM120, pinned here |
| `FLARE_GLM53_KDA_SIDE_CHAIN` | `1` | `1` | the KDA gate/decay projections run beside the q/k/v GEMM, without two staging copies per layer (same weights) |
| `FLARE_GLM53_EMBED_HOST` | — | `1` | the replicated BF16 token embedding (1,210 MiB per GPU) in device-visible host memory; the VRAM goes to KV |
| `B12X_OVERLAY` | `.b12x-overlay` | `.b12x-overlay` | world-3 pull gather (step 2) |
| `--mtp-only` | yes | yes | no graphs for plain (no-draft) decode; the package refuses `--mtp-draft-tokens 0` |
| `--max-concurrency` (`CONC`) | `32` | `8` | request slots the package captures graphs for |
| `--mtp-draft-tokens` | `5` | `5` | the recipe default (keep 5 when serving) |
| `--prefill-chunk-size` | `4096` | `4096` | the recipe default |
| Images (`VISION`, default 1) | `--image-concurrency 4` | `--image-concurrency 2` | vision tower kept; `FLARE_GLM53_VISION_STAGE=1` parks its 373 MiB per GPU in host memory and stages it through 22 MiB of VRAM per image. `VISION=0` exports text-only (`--language-model-only`): +2.6% KV at TP3, +7.2% at TP2 |
| Collective tuning | static | static | the TP3 runtime collective ladder stops at 24 MiB, a 4096-row GLM prefill chunk reduces 32 MiB per call |

### What to expect, and how long

On the test machine (tuning cache warm from earlier GLM exports on the same FlashInfer pin):

| | TP3 package 202611130000 | TP2 package 202611130307 |
| --- | --- | --- |
| Wall time | 25 min | 22 min |
| Load weights | 58 s | 55 s |
| Prepare and tune kernels | 128 s | 664 s |
| Graph catalog | 74 graphs (text-only: 68) | 56 graphs (text-only: 50) |
| Operator scratch | 578 MiB | 367 MiB |
| Package on disk | ~177 GB | ~168 GB |

The text-only packages of the same base (less warm cache, new FlashInfer pin) took 41 min (TP3) and 28 min (TP2); the
first GLM exports on this machine (cold GLM shapes, 8 slots, 2026-11-08) took 39 min and 33 min.
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
# release packages at 97,281 MiB free per GPU (--dry-run-free-memory): TP3 13,393,152 KV tokens
# (text-only 13,737,984); TP2 (--gpus 0,1) 2,414,592 (text-only 2,589,184)
```

Serve (from the fork checkout, so the default `FLARE` path resolves):

```bash
PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp> /path/to/this-folder/serve-glm-tp3.sh
PACKAGE=glm53-fp4-local@gb202-tp2-<timestamp> /path/to/this-folder/serve-glm-tp2.sh
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015`. The `Allocate cache` line reports
the KV count actually allocated; on the test machine it was 2-8% below the dry run because less memory was free at
start than the dry run assumed:

```text
TP3: [    OK    ] Plan cache.  per GPU: 64.88–64.92 GiB loaded · 29.35–29.39 GiB cache budget  0.0s
     [    OK    ] Allocate cache.  88.0 GiB · 13130496 KV tokens · 449 shared snapshot slots · limited by memory  4.2s
TP2: [    OK    ] Plan cache.  per GPU: 86.56 GiB loaded · 7.71 GiB cache budget  0.0s
     [    OK    ] Allocate cache.  15.4 GiB · 2231808 KV tokens · 86 shared snapshot slots · limited by memory  0.7s
```

Then run the corruption check (decode tokens against a teacher-forced prefill; must report `TRUE corruptions 0`),
the tool-call test and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 20
python3 /path/to/this-folder/tools/tool_calls.py 60 32            # 60 prompts at concurrency 32 (8 for TP2); expect 60/60
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model GLM-5.3-Flash \
  --concurrency 1,4,8,16,32 --contexts 32k --duration 60 --skip-prefill --output result.json   # 1,4,8 for TP2
```

`verify_fix.py`, `tool_calls.py` and `glm_common.py` read `MODEL` (default `GLM-5.3-Flash`) and `FLARE_URL`
(default `http://127.0.0.1:15015`) from the environment.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| First `flare run` of a new package fails with `core signer is not a configured publisher`, or asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal (or serve with the same config the export used: `CONFIG=` at export means `--config` at serve) |
| `--mtp-draft-tokens 0` is refused | the packages are exported with `--mtp-only` (no plain decode graphs); serve with 5 drafts |
| `serve-glm-tp*.sh` stops with an invalid-value error for `--scheduler-policy` | the `flare` binary predates the fork tag: rebuild it (step 2) |
| `serve-glm-tp*.sh` cannot find `glm53-fp4-local` | the checkpoint directory had another name, and the model name follows it (step 3). `flare models list` shows the name; serve with `PACKAGE=<name>@<version>` or the package path |
| The server starts the TP2 package on three GPUs or the TP3 package on two (or refuses) | both packages share the model name; give the version (`PACKAGE=<name>@gb202-tp3-...`) |
| Overlay build fails in `patch` | the venv patches (step 2) are not applied, or the venv FlashInfer is not the `85ea6e76` pin. Re-run `uv sync --locked ...`, apply `tp3-patches/*.patch`, delete `.b12x-overlay/` and export again |
| Requests with `enable_thinking: false` still reason | GLM-5.3-Flash always thinks (its chat template has no switch). Use `reasoning_effort` `low` or `high`; any other value renders `max` |
| Temperature or top_p below 0.95 has no effect | the GLM recipe allows 0.95-1.0 for both and clamps other values into that range (defaults: temperature 1.0, top-p 0.95) |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (deleting folders by hand frees nothing: files are shared) |
