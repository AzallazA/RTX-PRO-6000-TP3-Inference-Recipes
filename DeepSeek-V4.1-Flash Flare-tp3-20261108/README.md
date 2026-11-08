# DeepSeek-V4.1-Flash-1017 on NVIDIA Flare at TP3 — 20261108

DeepSeek-V4.1-Flash-1017 served by **NVIDIA Flare at tensor parallel 3 with the full 1M-token context** on three
96 GB RTX PRO 6000 Blackwell cards. This build supersedes [Flare-tp3-20261107](../DeepSeek-V4.1-Flash%20Flare-tp3-20261107/README.md)
(same settings, same checkpoint view, same KV: a drop-in replacement). Changes since then:

- the BRRR fused layer tail got a hygiene pass: host (cold) expert jobs first in parked layers, per-chunk attention
  flags, a kernel/config constants contract checked at export, a cross-rank agreement check (a mismatch would deadlock
  the in-kernel all-reduce);
- vision weight staging (`FLARE_DS41_VISION_STAGE=1`, the new export default): the host-parked vision tower's weights
  are copied block by block through two VRAM slots on a side stream, so a cold image takes 0.11 s instead of 1.70 s,
  with no KV cost and the same 60-image accuracy;
- Helios resolves Engram table storage (RAM or disk, per table) in one place; `flare run ... --dry-run` prints the
  memory plan and KV capacity of a package without touching the GPUs;
- three GPU equivalence tests (fused layer tail vs Flare's stock path with FP64 per-operator bounds, expert parking
  layouts, staged vision) and the fork handbook `docs/TP3-GUIDELINES.md`;
- (tag `flare-tp3-20261108.1`) one typed `TP3Config` object replaces the recipe's `FLARE_DS41_*` environment reads
  (same variable names, validated once, recorded as `tp3-config.json` beside the package's `model.safetensors`), and
  the dead experiment paths (BRRR prologue, QB_ATOMIC probe) are removed. Package 081024 from this tag passed the full
  c8 qualification below and is level with the 20261108 package over 8 alternating rounds.

Text speed is unchanged: 32K c1/c4/c8 ≈ 306/719/973 tok/s over four alternating rounds against the 20261107 package
(304/720/978), 0 true decode corruptions, 2,461,184 KV tokens. The fork's experiment with wide-row restreaming in the
fused kernel (stage 2) measured -5% at c4 and is not included. Full numbers in [MEASUREMENTS.md](MEASUREMENTS.md)
("2026-11-08").

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT) for all dense and expert weights; the two Engram tables in MXFP4
from [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (MIT).
Runtime: [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) `main` @ `972e28e4` plus this fork
(`source.patch`, branch `ds41-tp3-v3`; the same code is the public fork https://github.com/AzallazA/spark-local-inference at tag `flare-tp3-20261108.1`), FlashInfer @ `1f8ed8a3` (upstream's pin) plus four patches in `tp3-patches/`.
The fused layer tail is adapted from [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0).

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download the two public checkpoints (~412 GB), build the fork, export twice (~30 min each; the first run also
tunes kernels, 20-30 min more). `./check_env.sh <fork checkout>` is a read-only preflight for all of it. Then:

```bash
FLARE=<fork>/flare/target/release/flare ./serve.sh   # port 15015, model DeepSeek-V4.1-Flash, ~55 s; PACKAGE=<name or path> if the view was not named ds41-1017-engram-mxfp4
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**
unless marked 350 W. Every measurement, including the settings sweeps, is in [MEASUREMENTS.md](MEASUREMENTS.md).

## Headline numbers

| | 20261107 and this build (level), 450 W | same, 350 W | 20261105, 350 W | vLLM Kraken TP3 `.11` |
| --- | --- | --- | --- | --- |
| KV cache | **2,373,760 tokens** at 16G prefix cache, 2,461,184 at the default 32G (2.4 × 1M) | same | 2,991,488 | 2,185,581 |
| Decode 32K, c1 / c4 / c8 (tok/s) | **298-305 / 716-738 / 957-975** | 275 / 644 / 850 | ~254 / ~595 / ~805 | 231-260 / 470-490 / ~790 |
| Decode 32K, c1 engine steps/s | ~114-122 | ~110 | ~94 | ~97 |
| Decode 128K, c1 / c4 / c8 | 296 / 708 / 956 | 253 / 627 / 817 | — | — |
| Decode 512K, c1 / c4 | 275 / 657 | 262 / 540 | — | — |
| Prefill 32K / 128K (tok/s) | **14,507-14,805 / 14,619-14,645** | 12,931 / 12,844 | 12,548 / 12,160 | 14,195 / 12,716 |
| Coding peak (`--coding-peak`, mean) | 456-482 tok/s | 432 | 379-398 (median) | — |
| Startup | ~55 s | | ~60 s | ~3.5 min |
| eval-43 / tool calls | **43/43** / **60/60** | | 43/43 / 60/60 | 42/43 / — |
| lavd / hotel-lights (30 each) | **30/30** / **30/30** at c8 | | 30/30 / 30/30 at c4 | 30/30 / 30/30 |
| LiveBench coding | **82.2** | | 82.9 | 81.9 |
| c8 KV stress | **PASS** | | PASS | PASS |

Decode uses `llm_decode_bench.py --concurrency 1,4,8 --contexts 32k --coding-peak` (`llm-inference-bench`,
ignore-EOS decode cells). Ranges are the means of separate sessions; single runs vary by ±2-3%. Steps/s is engine
speed with draft acceptance divided out; tok/s also depends on how predictable the text is.

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
2. [More KV: fewer hot experts](#more-kv-fewer-hot-experts) · [Which settings for me](#which-settings-for-me) · [Engram on disk (less RAM)](#engram-on-disk-less-ram) · [Memory](#memory)
3. [Quality and long context](#quality-and-long-context) · [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Coming from 20261107 or 20261105](#coming-from-20261107-or-20261105) · [Building the package](#building-the-package) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

## Test system

| Part | Spec |
| --- | --- |
| GPUs | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), each on PCIe Gen5 x16 |
| GPU settings | **450 W** power limit, **+75 MHz core / +6000 MHz memory offset** (memory clock 16,365 MHz); ECC off; persistence mode on |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, 1 socket, 1 NUMA node |
| Motherboard | ASUS Pro WS WRX90E-SAGE SE |
| RAM | 256 GB DDR5: 8 × 32 GB Kingston Fury (KF560R32-32), 8 channels at **6000 MT/s** |
| Package drive | Crucial T710 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Checkpoint drive | Samsung 9100 PRO 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Software | Ubuntu, NVIDIA driver 615.71.09, CUDA 13.1 |

What matters most for speed, in our measurements:

- **Power.** On Flare, 450 W over 350 W is **+10-14% decode, +12-14% prefill, +12% coding** on this build. On the vLLM
  build only prefill gained. **600 W is not worth it:** one run at 600 W measured 297 / 757 / 1,002 tok/s at c1 / c4 / c8
  and coding 460 (450 W: 305 / 716 / 975, coding 465), within a few percent and about 240 W more at c8. Export at the
  power and clocks you will serve with: kernel tuning measures the cards as they run.
- **GPU memory clock.** Decode reads weights every step; the +6000 offset is part of every number here.
- **PCIe generation and topology.** The three GPUs exchange data twice per layer over PCIe (no NVLink), and cold experts
  stream from host RAM over PCIe. Gen5 x16 per GPU on one CPU root complex is the tested setup.
- **Host RAM.** With Engram in RAM the server uses ~148 GiB. With less RAM, see [Engram on disk](#engram-on-disk-less-ram).

## What this build contains

Cumulative description of the fork's features (the step from 20261105 to 20261107 is explained in the
[20261107 folder](../DeepSeek-V4.1-Flash%20Flare-tp3-20261107/README.md#what-changed-since-20261105); the changes since 20261107 are listed at the top).

| Change | Effect |
| --- | --- |
| Rebased onto upstream Flare `972e28e4` (FlashInfer `1f8ed8a3`: attention masking fix, copy ordering, draft prefix checkpoints, top-p) | same speed as the old base (A/B within noise), +175K KV tokens on the same settings |
| **BRRR fused layer tail** on all 40 layers (`ignition/recipes/deepseek41/brrr.py`, `brrr/kernels/decoder.cu`): one kernel for each layer's post-attention work. It reads Flare's own weights: the CSF expert scales are decoded in-kernel, and the dense FP8 weights match the checkpoint byte for byte, so no VRAM is spent on repacked copies. Parked layers (20-33) use it only in graphs of ≤12 rows; wider steps are relatively costlier there | c1 +6%, c4 +3%, c8 level (natural-EOS A/B vs the 20261105 layout at the same hot-expert count) |
| Token embedding (1.23 GiB/GPU) and vision tower (`FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST`) in device-visible host RAM | VRAM back to KV and hot experts; text speed unchanged |
| Vision tower weights staged through two VRAM slots per block (`FLARE_DS41_VISION_STAGE`, 2026-11-08) | each block's weights are copied into VRAM on a side stream while the previous block computes: cold image latency 0.11 s instead of 1.70 s (single image, 1 token), 190 MiB of activation-pool slots, no KV change, same 60-image accuracy |
| Prefill all-reduce runtime sized for 1024 rows instead of the 4096-row chunk (`FLARE_DS41_AR_ROWS`) | ~600 MiB/GPU back, prefill unchanged |
| 342 hot experts per parked layer instead of 300 | +7% / +4% / +5% at c1 / c4 / c8 (natural EOS, 350 W, same session; MEASUREMENTS.md), at 2.37M KV instead of 6.25M |
| Adaptive-verification alpha 0.8 instead of 0.6 | within noise of 0.6 (sweep in MEASUREMENTS.md) |
| Verification row cap 24 (`--mtp-max-verification-token-budget 24`, was 128 = no cap) | 4-run A/B: c1 +1.7%, c4 +0.4%, c8 +2.4%, coding +1%; output unchanged (only how many drafts are checked per step) |

## How it fits in 3 × 96 GB

**Merged hot/cold expert bank.** Routed-expert layers 20-33 keep their **342 most-routed experts in VRAM** and the
other 42 in pinned host memory. Each layer's experts sit in **one contiguous virtual range**: a VRAM prefix and a
host-NUMA tail, mapped with CUDA VMM. One MoE kernel call per layer reads both, so PCIe reads of cold experts overlap
the hot experts' compute. Expert IDs are remapped hot-first using route counts recorded from real traffic (a blend of
0.6 code and 0.4 prose). Each `expert-order/*.json` carries its provenance in a `source` field:
`blend 0.6 code (544K tok live traffic) + 0.4 prose (109K tok llm-inference-bench), 2026-10-30; layers 20-33 un-permuted`.
The counts were recorded with the vLLM build's route monitor; the fork ships no route-dump tool, so they cannot be
regenerated on Flare. The blend is general-purpose (code plus prose) and every number here was measured with it.

**MXFP4 Engram tables in RAM.** The two Engram n-gram tables take 189 GiB in FP8. NVIDIA's MXFP4 versions are
97 GiB, and both are held fully in host RAM. The b12x Engram lookup gets a packed-FP4 path that decodes bit-exact
against a reference.

**Vocabulary-parallel sampling at TP3.** Each GPU samples over its third of the vocabulary, which frees ~1 GiB per GPU.
Side effect: top_p is fixed at 1.0, the model's default.

**Prefill in 4096-token chunks** (+15-16% prefill over 2048). **Speculative decoding:** DSpark (the checkpoint's 3
`mtp.*` layers) with 5 drafts, an NVFP4 draft head, an FP8 Markov head and adaptive verification (alpha 0.8, at most 24 verification rows per step).

**Fixes required for TP3 at all** (unchanged from 20261105): FlashInfer `tiny_decode` honors `swiglu_limit`
(`tp3-patches/flashinfer-tiny-decode-swiglu.patch`, without it replies corrupt mid-way), world-size-3 PCIe
all-to-all and tuning tables, the planner context cap, one shared PCIe all-reduce runtime, vocabulary-parallel
kernels at TP3.

## More KV: fewer hot experts

Each hot expert per parked layer moved to host RAM frees **~79 MiB per GPU, about 92K KV tokens**. Decode gets
slower because more experts stream over PCIe. Export with `HOT=<n> ./export.sh`; everything else stays the same.

| `HOT` | KV tokens | Measured | Use it for |
| --- | --- | --- | --- |
| **342** | **2.37M** | **this release** (numbers above) | two 1M-token sessions |
| 300 | 6.25M | 350 W, natural EOS, 32K: 269 / 610 / 800 tok/s against 288 / 634 / 838 for 342 in the same session: 4-7% slower | six 1M-token sessions |

> **Approximate.** KV count ≈ 2.37M + (342 − HOT) × 92K tokens (measured: 92.2K); only 342 and 300 were measured on this fork.
> The speed cost depends on how often your traffic routes to the experts that became cold. Measure your own export
> (`Allocate cache` in the server log prints the KV count; [EXPORT.md](EXPORT.md) has the bench command).

## Which settings for me

| Your situation | Export (`export.sh`) | Serve (`serve.sh`) | Expect |
| --- | --- | --- | --- |
| 256 GB RAM, up to two 1M-token sessions (the tested setup) | defaults (`HOT=342`) | defaults | the numbers above: 2.37M KV, ~305 / 738 / 975 tok/s at 32K c1 / c4 / c8, 450 W |
| ≤ ~128 GB RAM | defaults | `ENGRAM_RAM=0` | ~59 GiB host RAM instead of ~148; decode ~4% slower, prefill the same ([Engram on disk](#engram-on-disk-less-ram)) |
| Six 1M-token sessions at once | `HOT=300` | defaults | 6.25M KV; decode 4-7% slower (350 W: 269 / 610 / 800 against 288 / 634 / 838 tok/s) |
| Many long sessions you return to | defaults | `PREFIX_CACHE=32G` | ~78 warm 127K-token prompts instead of ~37; a host-cache hit answers in 0.14 s instead of ~9 s of re-prefill; 16 GiB more RAM (MEASUREMENTS.md "Reuse test") |
| Heavy image use | defaults (`FLARE_DS41_VISION_STAGE=1`) | `FLARE_DS41_VISION_HOST=0` plus a lower `HOT` | staging already gives in-VRAM image latency (0.11 s per image) at no KV cost; the VRAM-resident tower is no longer needed |
| 350 W cards | defaults, exported at your power limit (tuning measures the cards as they run) | defaults | 10-14% slower: 275 / 644 / 850 tok/s at 32K, prefill 12,931 |

`DRAFTS` must be the same at export and serve (default 5 in both scripts): the package is captured for one draft count.

## Engram on disk (less RAM)

`ENGRAM_RAM=0 ./serve.sh` reads both Engram tables from the NVMe drive with direct I/O instead of holding them in RAM.
The tables never enter the page cache: it stayed at 13 GiB through the whole test.

| 450 W | Engram in RAM | Engram on disk |
| --- | --- | --- |
| Server host RAM | ~148 GiB | **~59 GiB** |
| Prefill 32K / 128K | 14,507 / 14,645 | 14,540-14,591 / 14,667-14,722 (same) |
| Decode 32K c1 / c4 / c8 | 303 / 719 / 957 | 292 / 692 / 926 (−3-4%) |
| Coding mean | 482 | 463 (−4%) |

The Engram row cache (`--embedding-table-cache-size`, default 4 GiB) answered 64% of lookups. 16 GiB gave the same
speed, because the misses are rows seen for the first time. This is measured on Gen5 NVMe (12 GB/s); slower drives
will cost more, mostly in prefill.

## Memory

| Per GPU | |
| --- | --- |
| Loaded (weights, graphs, scratch) | ~92.3 GiB |
| KV cache | 2.1 GiB (6.3 GiB over the 3 GPUs) → 2,373,760 tokens |

Host RAM: Engram tables ~97 GiB, host model memory (cold experts, embedding, vision) 16 GiB, prefix cache 16 GiB
(`PREFIX_CACHE=32G` or `48G` costs only RAM; no speed change measured).

## Quality and long context

The same protocol as the vLLM Kraken TP3 releases ([qualification/](qualification/RESULTS.md)), with thinking on,
`reasoning_effort` max, temperature 1 and top-p 1.0. It runs on the live package without restarts, at concurrency 8.

| Check | Result |
| --- | --- |
| eval-43 (thinking off) | **43/43** |
| Tool calls: 60 tool-requiring prompts at c8 | **60/60** |
| lavd, 30 runs at c8 | **30/30** exact |
| hotel-lights, 30 runs at c8 | **30/30** exact |
| Long retrieval: three markers at 128K and 900K tokens | **pass / pass** |
| Long generation after a 581,537-token conversation, 4 seeds | **4/4**, natural stop, 16.7K-28.1K tokens, 8-gram repetition ≤ 0.0011 |
| Decode bench with natural EOS | **0 loops**; 293.8 / 707.7 / 964.3 tok/s (c1/c4/c8) |
| LiveBench coding (128 questions) | **82.2** (LCB generation 74.4, coding completion 90.0) |
| KV stress: 2 waves × 8 × ~530K unique prompts + pinger | **PASS**: 16/16, pinger 136/136, 9.5M prompt tokens, min free VRAM 82 MiB |
| Corruption check (teacher-forced comparison, 10 runs) | 0 true corruptions |
| Vision (60-image set and hard set, both temperatures) | 60/60 and 58-59/60, same as vLLM `.11` (measured on the pre-rebase fork with the same host-RAM vision tower; ~3 s slower per image) |

### Qualification of this folder's packages (2026-11-08, c8, 450 W, k5, alpha 0.8, balanced, cap 24, 32G prefix cache)

Same suite as the 20261107 folder (`qualification/qual_all.sh`), run against the served package; raw results in
`qualification/results/run-20261108-c8/` (package 080909, tag `flare-tp3-20261108`) and
`qualification/results/run-20261108.1-c8/` (package 081024, tag `flare-tp3-20261108.1`).

| Test | 080909 | 081024 |
| --- | --- | --- |
| 43-question eval, thinking off | 43/43 | 43/43 |
| Tool calls, 60 prompts at c8 | 60/60, 0 wrong tool | 60/60, 0 wrong tool |
| lavd, 30 runs at c8 | 30 exact | 30 exact |
| hotel-lights, 30 runs at c8 | 30 exact | 30 exact |
| Long retrieval, 128K / 900K | pass / pass | pass / pass |
| Long generation after a 581K-token conversation, 4 seeds | 4/4, worst 8-gram repeat 0.0006 | 4/4, worst 0.0008 |
| Decode bench with natural EOS, 32K c1/c4/c8 (tok/s) | 293 / 717 / 964, 0 loops | 290 / 717 / 963, 0 loops |
| LiveBench coding (2024-11-25, 32K budget, temperature 1) | 83.9 | 81.6 |
| KV stress at c8 (2 waves x 8 x ~530K unique prompts + pinger, KV oversubscribed) | PASS, 16/16, pinger 120/120, 9.5M prompt tokens | PASS, 16/16, pinger 117/117, 9.5M prompt tokens |

LiveBench moves by a question or two between runs at temperature 1 (81.5-84.2 across our MXFP4 runs); the two
scores above bracket the 20261107 package's 82.2.

## Measured and not adopted

| Idea | Result |
| --- | --- |
| 7 DSpark drafts (best k7: alpha 1.0, throughput) | c4 −6%, c8 −3%, coding +2%, c1 level; 153K fewer KV tokens |
| Scheduler `throughput` instead of `balanced`, alpha 0.6 / 1.0 | all within ~2% at k5 |
| BRRR fused prologue (attention input side) | no gain: the work it fuses already overlaps on side streams |
| BRRR in parked layers at every width | c8 −2.4%; gated to ≤12 rows instead |
| Larger Engram row cache in disk mode (16 GiB) | no change; misses are first-touch rows |
| Earlier ideas (separate hot/cold calls, global placement, FP8 Engram in RAM, history drafting, joint tuning) | see the [20261105 README](../DeepSeek-V4.1-Flash%20Flare-tp3-20261105/README.md#measured-and-not-adopted) |

## Known limitations

- **Hardware-specific package:** 3 × 96 GB SM120 (RTX PRO 6000) only, max concurrency 8. Other setups need a fresh export.
- **top_p is fixed at 1.0** (vocabulary-parallel sampling); temperature is fixed at 1.0 by Flare's DeepSeek recipe.
- **Images**: the vision tower lives in host RAM and its weights are staged through VRAM per block
  (`FLARE_DS41_VISION_STAGE=1`, default since 2026-11-08): 0.11 s cold latency per image instead of 1.70 s without staging.
  `FLARE_DS41_VISION_STAGE=0` restores the direct host-RAM reads.
- **Bench loops:** `llm_decode_bench` decode cells use `ignore_eos`, which forces generation past the answer; a rare
  repetition loop there is a bench artifact (the model emits its end token, then invents tool output). With
  `--respect-eos` the same cells show 0 loops.
- Not upstreamable as written: Flare's contribution rules exclude model-specific offload policy and AI-authored
  GPU-execution changes. This is a personal fork.

## Coming from 20261107 or 20261105

**From 20261107:** drop-in. Same base commit `972e28e4`, same checkpoint view, same serve flags; apply this folder's
`source.patch` on a clean checkout (or fetch fork tag `flare-tp3-20261108`), export once, serve. Image requests get the
staged vision tower automatically; the `tp3-tools/` and `docs/tp3/` additions need nothing from you.

**From 20261105:**

- **New base commit.** The fork now sits on upstream `972e28e4` (20261105: `78b18b1c`) and the old `source.patch`
  conflicts with it. Make a fresh clone, or in the existing checkout drop the old patch (`git checkout -- . && git clean -fd`)
  and `git checkout 972e28e44781d8d0c84d70ae434db92c36076a43` before applying this folder's `source.patch`.
- Re-run `uv sync --locked --extra prepare --extra test`, then `uv run --locked --extra prepare python -m ignition.patches`
  (a step the 20261105 guide did not have) and the four FlashInfer patches in `tp3-patches/` (unchanged since 20261105,
  but `uv sync` reinstalls FlashInfer). `./check_env.sh <fork>` confirms the state.
- The kernel tuning cache in `~/.cache/flare/kernels` carries over and shortens the warm-up export; keep it.
- Old packages coexist with the new one (same model name `ds41-1017-engram-mxfp4`, new version). Remove one with
  `flare models remove ds41-1017-engram-mxfp4@gb202-tp3-<timestamp>`, then `flare models gc` to free the space.
- Serve flags changed: verification row cap 24 instead of 128 (`BUDGET`) and alpha 0.8 instead of 0.6 (`ALPHA`). The
  new `serve.sh` has them; the old command still runs the new package but gives up the 1-2% measured for the cap.

## Building the package

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
# from this folder
./check_env.sh <fork>                                                                 # read-only preflight: toolchain, fork state, disk, idle GPUs
./make_checkpoint_view.sh <deepseek snapshot> <nvidia NVFP4 snapshot> ~/ds41-1017-engram-mxfp4   # 307 GB + 105 GB of downloads
# from the fork checkout (export.sh uses its .venv)
cd <fork>
CKPT=~/ds41-1017-engram-mxfp4 HOT=300 /path/to/this-folder/export.sh   # warms the kernel tuning cache (a cold cache runs out of memory at HOT=342)
CKPT=~/ds41-1017-engram-mxfp4 /path/to/this-folder/export.sh           # the release build, HOT=342
```

The model name is the checkpoint directory's name: this view exports as `ds41-1017-engram-mxfp4`, which is what
`serve.sh` runs by default. Another directory name needs `PACKAGE=<that name>` when serving.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `check_env.sh` | read-only preflight: toolchain, memlock, fork commit and patches, disk, idle GPUs (PASS / WARN / FAIL) |
| `versions.txt` | exact versions behind the numbers: driver, CUDA, Python, torch, FlashInfer, Rust, uv, Flare and benchmark commits, GPU settings |
| `MEASUREMENTS.md` | every measurement of this build: power, settings sweeps, k5 vs k7, prefix cache, Engram on disk |
| `ENGRAM-PRECISION.md` | MXFP4 vs FP8 Engram tables: every quality score we have for each, from the vLLM and Flare releases |
| `serve.sh` / `export.sh` | serve the package / export it from a checkpoint view |
| `make_checkpoint_view.sh`, `checkpoint/` | build the checkpoint from the two public releases (config + corrected index) |
| `source.patch` | the fork's changes against Flare `972e28e4`, generated from fork tag `flare-tp3-20261108.1` (149 files, 103 new: the adapted `brrr/` kernels, vision staging, the Engram and KV-planning changes in Helios, GPU equivalence tests, the fork handbook `docs/TP3-GUIDELINES.md`; see the update notes in EXPORT.md) |
| `expert-order/` | per-layer route counts that set the hot-expert placement |
| `tp3-patches/` | the four FlashInfer patches |
| `tools/` | `bench_steps.py` (steps/s from a bench JSON), `verify_fix.py` (corruption check), `llm-bench-flare-steps.patch` (optional, for `llm-inference-bench` v0.9.1); the tool-call test is `qualification/tool_calls.py` |
| `qualification/` | the qualification suite (adapted for Flare, concurrency 8) and its results |

## Licenses

DeepSeek weights and NVIDIA's NVFP4 release: MIT (texts in `licenses/`). Flare: Apache-2.0 (`licenses/LICENSE`,
`licenses/NOTICE` and `licenses/THIRD_PARTY_NOTICES.md` kept; modified files marked). FlashInfer, which
`tp3-patches/` modifies: Apache-2.0 (`licenses/FlashInfer-LICENSE`, `licenses/FlashInfer-NOTICE`). BRRR: Apache-2.0
(`licenses/BRRR-LICENSE`, `licenses/BRRR-NOTICE`). `llm-inference-bench`, which `tools/llm-bench-flare-steps.patch`
modifies, is MIT (its README). The fused decoder kernel in `brrr/kernels/decoder.cu` is modified for TP3 and Flare: CSF
scale decode, Flare's dense weight layout, RoPE from Flare's frequencies, hot-expert slot mapping and Flare
peer-storage buffers. The `flare` binary links only libc/libstdc++; NVIDIA's CUDA Python wheels are used only for
exporting and are not redistributed.

## Credits

NVIDIA Flare and FlashInfer b12x; DeepSeek for the model; NVIDIA for the MXFP4 Engram tables; Fraser Price for BRRR;
Local Inference Lab for `llm-inference-bench`.
