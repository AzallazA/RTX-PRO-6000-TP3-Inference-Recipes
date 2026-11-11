# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261111

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`, so
> every command in this folder is unchanged. Earlier folders (`Flare-tp3-2026110x`) keep the old name.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB
RTX PRO 6000 Blackwell cards. This build supersedes [Flare-tp3-20261108](../DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md):
same checkpoint view and base commit, a new export and a rebuilt `flare` binary (see [Coming from 20261108](#coming-from-20261108)).
Changes since then:

- **+22% KV cache from graph scratch (export default):** the Engram projection, reduce and mix run in 1024-row chunks
  (−300 MiB per GPU) and `engram_residual` reuses the `ffn_hc.residual` buffer (−165 MiB per GPU). At the same
  settings (342 hot experts, verification row cap 24, 32G prefix cache) the KV cache grows from **2,467,200 to 3,014,656
  tokens** (2.9 × 1M). Decode is level: c1 / c4 / c8 32K 302 / 735 / 987 against 305 / 731 / 991 tok/s for the
  20261108.1 package over four alternating rounds (0.99 / 1.01 / 1.00×); prefill 32K / 128K 14,547 / 14,448 against
  14,763 / 14,541 tok/s (−1.5% / −0.6%); a 6,000-token prompt gives the same text and logprobs within 0.00006.
- **First-token latency under load (TTFT guard, `serve.sh` default `SCHEDULER=interactive`):** Hyperflare's scheduler
  used to let a new prompt wait behind decoding streams for seconds. The new `--scheduler-policy interactive`
  (balanced plus a 50 ms first-token target) forces the waiting prompt's prefill into the next rounds, interleaved with
  decode. Measured on this build, median of 3 alternating rounds: a 1.9K-token prompt arriving while 3 / 7 streams
  decode gets its first token after **0.36 / 0.38 s instead of 2.0 / 4.0 s**; a 4-stream agent replay (growing
  history, ~1.9K-token tool results) finishes **519 turns instead of 448** in 300 s with p90 TTFT 0.93 s instead of
  1.97 s; plain decode c1 / c4 / c8 300 / 708 / 965 against 302 / 713 / 942 tok/s (level).
- `export.sh` gains `MAXC` (request slots the package captures graphs for; 8 is the qualified setting) and
  `PARKED_ROWS` (the BRRR parked-layer row limit, default 12; 24 was measured for this release and not adopted, see
  [Measured and not adopted](#measured-and-not-adopted)). The fork's BRRR kernel keeps its CSF
  descriptors in registers, builds with `-fno-gnu-unique`, and carries three measured-level experiments behind
  default-off switches (split tail, fast route, counter parity).

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT) for all dense and expert weights; the two Engram tables in MXFP4
from [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (MIT).
Runtime: [NVIDIA Hyperflare](https://github.com/NVIDIA/hyperflare-inference) (formerly Flare) `main` @ `972e28e4` plus
this fork (`source.patch`; the same code is the public fork https://github.com/AzallazA/spark-local-inference at tag
`hyperflare-tp3-20261111`), FlashInfer @ `1f8ed8a3` (upstream's pin) plus four patches in `tp3-patches/`.
The fused layer tail is adapted from [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0).

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download the two public checkpoints (~412 GB), build the fork, export twice on a fresh machine (~30-40 min each; the
first run also tunes kernels, 20-30 min more). `./check_env.sh <fork checkout>` is a read-only preflight for all of it. Then:

```bash
FLARE=<fork>/flare/target/release/flare ./serve.sh   # port 15015, model DeepSeek-V4.1-Flash, ~55 s; PACKAGE=<name or path> if the view was not named ds41-1017-engram-mxfp4
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.
Every measurement is in [MEASUREMENTS.md](MEASUREMENTS.md); raw results under `qualification/results/`.

## Headline numbers

| | this build, 450 W | 20261108, 450 W | vLLM Kraken TP3 `.11` |
| --- | --- | --- | --- |
| KV cache | **3,014,656** (verification row cap 24, 32G prefix cache; 2.9 × 1M) | 2,467,200 | 2,185,581 |
| Decode 32K, c1 / c4 / c8 (tok/s) | **301 / 734 / 988** (mean of 5 alternating rounds, interactive scheduler) | 298-305 / 716-738 / 957-986 | 231-260 / 470-490 / ~790 |
| Decode 128K / 512K, c1 | 289 / 271 (1 run) | 296 / 275 | — |
| Prefill 32K / 128K (tok/s) | 14,547 / 14,448 (graph scratch A/B) | 14,507-14,805 / 14,619-14,645 | 14,195 / 12,716 |
| First token, 1.9K-token prompt while 3 / 7 streams decode (p50) | **0.36 / 0.38 s** | 2.0 / 4.0 s | — |
| Startup | ~55 s | ~55 s | ~3.5 min |
| eval-43 / tool calls | **42/43** / **60/60** | 43/43 / 60/60 | 42/43 / — |
| lavd / hotel-lights (30 each, c8) | **30/30** / **30/30** | 30/30 / 30/30 | 30/30 / 30/30 |
| LiveBench coding | **83.5** | 81.6-83.9 | 81.9 |
| c8 KV stress | **PASS** | PASS | PASS |

Decode uses `llm_decode_bench.py --concurrency 1,4,8 --contexts 32k` (`llm-inference-bench`, ignore-EOS decode cells,
60 s each). Single runs vary by ±2-3%; this build's numbers are the means of alternating rounds in one session (the
release A/B, `qualification/results/release-ab/`; two of the five c4 cells were stopped by the bench's loop guard, an
`ignore_eos` artifact described under [Known limitations](#known-limitations), so c4 is the mean of three). The
first-token row uses `tools/ttft_under_load.py probe --streams 3|7`.

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
2. [More KV: fewer hot experts](#more-kv-fewer-hot-experts) · [Which settings for me](#which-settings-for-me) · [Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams) · [Engram on disk (less RAM)](#engram-on-disk-less-ram) · [Memory](#memory)
3. [Quality and long context](#quality-and-long-context) · [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Coming from 20261108](#coming-from-20261108) · [Building the package](#building-the-package) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

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

- **Power.** On Hyperflare (20261107-20261108 builds), 450 W over 350 W is **+10-14% decode, +12-14% prefill, +12% coding** on the 20261107 build. On the vLLM
  build only prefill gained. **600 W is not worth it:** one run at 600 W measured 297 / 757 / 1,002 tok/s at c1 / c4 / c8
  and coding 460 (450 W: 305 / 716 / 975, coding 465), within a few percent and about 240 W more at c8. Export at the
  power and clocks you will serve with: kernel tuning measures the cards as they run.
- **GPU memory clock.** Decode reads weights every step; the +6000 offset is part of every number here.
- **PCIe generation and topology.** The three GPUs exchange data twice per layer over PCIe (no NVLink), and cold experts
  stream from host RAM over PCIe. Gen5 x16 per GPU on one CPU root complex is the tested setup.
- **Host RAM.** With Engram in RAM the server uses ~148 GiB. With less RAM, see [Engram on disk](#engram-on-disk-less-ram).

## What this build contains

Cumulative description of the fork's features (the step from 20261105 to 20261107 is explained in the
[20261107 folder](../DeepSeek-V4.1-Flash%20Flare-tp3-20261107/README.md#what-changed-since-20261105), 20261108 in its
[folder](../DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md); the changes since 20261108 are listed at the top).

| Change | Effect |
| --- | --- |
| Rebased onto upstream Hyperflare `972e28e4` (FlashInfer `1f8ed8a3`: attention masking fix, copy ordering, draft prefix checkpoints, top-p) | same speed as the old base (A/B within noise), +175K KV tokens on the same settings |
| **BRRR fused layer tail** on all 40 layers (`ignition/recipes/deepseek41/brrr.py`, `brrr/kernels/decoder.cu`): one kernel for each layer's post-attention work. It reads Hyperflare's own weights: the CSF expert scales are decoded in-kernel, and the dense FP8 weights match the checkpoint byte for byte, so no VRAM is spent on repacked copies. Parked layers (20-33) use it only in graphs of ≤12 rows (`PARKED_ROWS`) | c1 +6%, c4 +3%, c8 level (natural-EOS A/B vs the 20261105 layout at the same hot-expert count) |
| Token embedding (1.23 GiB/GPU) and vision tower (`FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST`) in device-visible host RAM | VRAM back to KV and hot experts; text speed unchanged |
| Vision tower weights staged through two VRAM slots per block (`FLARE_DS41_VISION_STAGE`, 2026-11-08) | each block's weights are copied into VRAM on a side stream while the previous block computes: cold image latency 0.11 s instead of 1.70 s (single image, 1 token), 190 MiB of activation-pool slots, no KV change, same 60-image accuracy |
| Prefill all-reduce runtime sized for 1024 rows instead of the 4096-row chunk (`FLARE_DS41_AR_ROWS`) | ~600 MiB/GPU back, prefill unchanged |
| 342 hot experts per parked layer instead of 300 | +7% / +4% / +5% at c1 / c4 / c8 (natural EOS, 350 W, same session; MEASUREMENTS.md), at 2.37M KV instead of 6.25M (both at the then verification cap 128) |
| Adaptive-verification alpha 0.8 instead of 0.6 | within noise of 0.6 (sweep in MEASUREMENTS.md) |
| Verification row cap 24 (`--mtp-max-verification-token-budget 24`, was 128 = no cap) | 4-run A/B: c1 +1.7%, c4 +0.4%, c8 +2.4%, coding +1%; output unchanged (only how many drafts are checked per step) |
| Graph scratch (`FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1`, export default since this build) | Engram projection/reduce/mix in 1024-row chunks and `engram_residual` on `ffn_hc.residual`: −465 MiB per GPU of graph scratch → KV 2,467,200 → 3,014,656 tokens; decode level, prefill −0.6 to −1.5% |
| TTFT guard (`--scheduler-policy interactive` = balanced + `--scheduler-ttft-target-ms 50`; `serve.sh` default since this build) | a prompt still without output 50 ms after arrival gets the next rounds (prefill chunks interleaved with decode, one other round between two forced chunks): first token 0.36-0.38 s instead of 2-4 s at c4/c8 load, decode level |

## How it fits in 3 × 96 GB

**Merged hot/cold expert bank.** Routed-expert layers 20-33 keep their **342 most-routed experts in VRAM** and the
other 42 in pinned host memory. Each layer's experts sit in **one contiguous virtual range**: a VRAM prefix and a
host-NUMA tail, mapped with CUDA VMM. One MoE kernel call per layer reads both, so PCIe reads of cold experts overlap
the hot experts' compute. Expert IDs are remapped hot-first using route counts recorded from real traffic (a blend of
0.6 code and 0.4 prose). Each `expert-order/*.json` carries its provenance in a `source` field:
`blend 0.6 code (544K tok live traffic) + 0.4 prose (109K tok llm-inference-bench), 2026-10-30; layers 20-33 un-permuted`.
The counts were recorded with the vLLM build's route monitor; the fork ships no route-dump tool, so they cannot be
regenerated on Hyperflare. The blend is general-purpose (code plus prose) and every number here was measured with it.

**MXFP4 Engram tables in RAM.** The two Engram n-gram tables take 189 GiB in FP8. NVIDIA's MXFP4 versions are
97 GiB, and both are held fully in host RAM. The b12x Engram lookup gets a packed-FP4 path that decodes bit-exact
against a reference.

**Vocabulary-parallel sampling at TP3.** Each GPU samples over its third of the vocabulary, which frees ~1 GiB per GPU.
Side effect: top_p is fixed at 1.0, the model's default.

**Prefill in 4096-token chunks** (+15-16% prefill over 2048). **Speculative decoding:** DSpark (the checkpoint's 3
`mtp.*` layers) with 5 drafts, an NVFP4 draft head, an FP8 Markov head and adaptive verification (alpha 0.8, at most 24 verification rows per step).

**Graph scratch.** The Engram n-gram projection, its all-reduce and the mix run in 1024-row chunks, and the
`engram_residual` activation shares the `ffn_hc.residual` buffer, which frees ~465 MiB per GPU of captured-graph
scratch for KV (3,014,656 tokens instead of 2,467,200).

**Fixes required for TP3 at all** (unchanged from 20261105): FlashInfer `tiny_decode` honors `swiglu_limit`
(`tp3-patches/flashinfer-tiny-decode-swiglu.patch`, without it replies corrupt mid-way), world-size-3 PCIe
all-to-all and tuning tables, the planner context cap, one shared PCIe all-reduce runtime, vocabulary-parallel
kernels at TP3.

## More KV: fewer hot experts

Each hot expert per parked layer moved to host RAM frees **~79 MiB per GPU, about 92K KV tokens**. Decode gets
slower because more experts stream over PCIe. Export with `HOT=<n> ./export.sh`; everything else stays the same.

| `HOT` | KV tokens | Measured | Use it for |
| --- | --- | --- | --- |
| **342** | **3.01M** | **this release** (numbers above) | two 1M-token sessions plus ~0.9M |
| 300 | ~6.9M estimated with graph scratch (6.25M measured at cap 128 without it) | 350 W, natural EOS, 32K: 269 / 610 / 800 tok/s against 288 / 634 / 838 for 342 in the same session: 4-7% slower | six 1M-token sessions |

> **Approximate.** KV count ≈ 3,014,656 + (342 − HOT) × 92K tokens (measured: 92.2K before graph scratch) at the
> default verification row cap 24 (cap 128 gives ~87K fewer); only 342 was measured with graph scratch.
> The speed cost depends on how often your traffic routes to the experts that became cold. Measure your own export
> (`Allocate cache` in the server log prints the KV count; [EXPORT.md](EXPORT.md) has the bench command).

## Which settings for me

| Your situation | Export (`export.sh`) | Serve (`serve.sh`) | Expect |
| --- | --- | --- | --- |
| 256 GB RAM, up to two 1M-token sessions (the tested setup) | defaults (`HOT=342`) | defaults | the numbers above: 3.01M KV, 301 / 734 / 988 tok/s at 32K, 450 W |
| ≤ ~128 GB RAM | defaults | `ENGRAM_RAM=0` | ~59 GiB host RAM instead of ~148; decode ~4% slower, prefill the same ([Engram on disk](#engram-on-disk-less-ram)) |
| Six 1M-token sessions at once | `HOT=300` | defaults | ~6.9M KV estimated (graph scratch not measured at 300); decode 4-7% slower (350 W, before graph scratch: 269 / 610 / 800 against 288 / 634 / 838 tok/s) |
| Many long sessions you return to | defaults | `PREFIX_CACHE=32G` | ~78 warm 127K-token prompts instead of ~37; a host-cache hit answers in 0.14 s instead of ~9 s of re-prefill; 16 GiB more RAM (MEASUREMENTS.md "Reuse test") |
| Heavy image use | defaults (`FLARE_DS41_VISION_STAGE=1`) | `FLARE_DS41_VISION_HOST=0` plus a lower `HOT` | staging already gives in-VRAM image latency (0.11 s per image) at no KV cost; the VRAM-resident tower is no longer needed |
| 350 W cards | defaults, exported at your power limit (tuning measures the cards as they run) | defaults | 10-14% slower: 275 / 644 / 850 tok/s at 32K, prefill 12,931 |

`DRAFTS` must be the same at export and serve (default 5 in both scripts): the package is captured for one draft count.
| Single user, throughput only (no first-token concern) | defaults | `SCHEDULER=balanced` | running streams never pause for a new prompt; new prompts wait behind decode under load ([Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams)) |

## Scheduler trade-off: first token vs running streams

The GPUs run one step at a time. When a new prompt arrives while other requests are decoding, its prefill either runs
now (the running streams pause for those steps) or waits until the scheduler gets to it (the new prompt waits). Every
policy picks a point between the two; none removes it. `serve.sh` takes `SCHEDULER=`:

| `SCHEDULER` | New prompt, first token (c4 / c8 load, p50) | Running streams while that prompt is prefilled (c4 / c8, tok/s) | Good for |
|---|---|---|---|
| `interactive` (default) | **0.36 / 0.37 s** | 190 / 308 (largest pause 0.30-0.31 s) | chat, agents, tools: anything waiting on first tokens |
| `throughput` | 1.50-1.56 / 2.72-2.90 s | 537-593 / 1035-1062 | in between |
| `balanced` (Hyperflare's own default) | 1.97-2.06 / 3.95-4.07 s | 597-626 / 1093-1108 (largest pause 0.25 s) | batch jobs where nobody waits on a first token |

A 1.9K-token prompt arriving while 3 / 7 streams decode (`tools/ttft_under_load.py probe`, this package, 11-10; 3 rounds per policy,
interactive round 1). The streams' slowdown lasts only while the new prompt is prefilled (well under a second for
a short prompt, longer for a long one); averaged over a run, decode is level (300 / 708 / 965 against 302 / 713 / 942
tok/s at c1 / c4 / c8), and an agent replay finishes 519 turns instead of 448 in 300 s with `interactive`.

To turn the guard off: `SCHEDULER=balanced ./serve.sh` (no re-export; the package is the same). Mixed prefill+decode
steps, as vLLM runs them, would soften the pause slightly; we estimated about +0.5-3% for DS41, not built.

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

Disk mode reads the tables from the checkpoint files the package was exported from, so the checkpoint must be at the
same absolute path at serve time as at export. For the view from EXPORT.md that includes the Hugging Face cache files
its symlinks point to: moving or pruning that cache breaks disk serving.

## Memory

| Per GPU | |
| --- | --- |
| Loaded (weights, graphs, scratch) | ~92.3 GiB |
| KV cache | 2.6 GiB (7.9 GiB over the 3 GPUs, the served `Allocate cache` line) → 3,014,656 tokens (graph scratch package, cap 24) |

Host RAM: Engram tables ~97 GiB, host model memory (cold experts, embedding, vision) 16 GiB, prefix cache 16 GiB
(`PREFIX_CACHE=32G` or `48G` costs only RAM; no speed change measured).

## Quality and long context

The same protocol as the vLLM Kraken TP3 releases ([qualification/](qualification/RESULTS.md)), with thinking on,
`reasoning_effort` max, temperature 1 and top-p 1.0. It runs on the live package without restarts, at concurrency 8,
served with `serve.sh` defaults (interactive scheduler). Raw results: `qualification/results/run-hyperflare-c8/`.

| Check | This build (package 202611111227) | 20261108.1 (package 081024) |
| --- | --- | --- |
| Corruption check (teacher-forced comparison, 10 runs) | 0 true corruptions | 0 |
| eval-43 (thinking off) | **42**/43 | 43/43 |
| Tool calls: 60 tool-requiring prompts at c8 | **60**/60 | 60/60 |
| lavd, 30 runs at c8 | **30/30** exact | 30/30 |
| hotel-lights, 30 runs at c8 | **30/30** exact | 30/30 |
| Long retrieval: three markers at 128K and 900K tokens | **pass / pass** | pass / pass |
| Long generation after a 581,537-token conversation, 4 seeds | **4/4**, worst 8-gram repeat 0.0013, natural stop | 4/4, worst 8-gram repeat 0.0008 |
| Decode bench with natural EOS, 32K c1/c4/c8 | **300 / 713 / 950** tok/s, 0 loops | 290 / 717 / 963, 0 loops |
| LiveBench coding (2024-11-25, 128 questions, 32K budget, temperature 1) | **83.5** (LCB_generation 73.1, coding_completion 94.0) | 81.6 (080909: 83.9) |
| KV stress: 2 waves × 8 × ~530K unique prompts + pinger, KV oversubscribed | **PASS**, 16/16, pinger 68/68 | PASS, 16/16, pinger 117/117 |
| Vision (60-image set and hard set) | not re-run: vision path unchanged since 20261108 (staging, 0.11 s per image) | 60/60 and 58-59/60 |

LiveBench moves by a question or two between runs at temperature 1 (81.5-84.2 across our MXFP4 runs).

Quick checks run on the way (same package family, 450 W): graph scratch package with the balanced scheduler: corruption
check 0, eval 42/43, tools 60/60, lavd 30/30, hotel-lights 29/30, c8 stress PASS
(`qualification/results/graph-scratch-ab-20261109/`); the same package with the interactive scheduler: corruption check
0, eval 42/43, tools 60/60, c8 stress PASS (`qualification/results/ttft-quality-20261110/`).

## Measured and not adopted

| Idea | Result |
| --- | --- |
| BRRR in parked layers up to 24 rows (`PARKED_ROWS=24`) at c8 | release A/B against 12, two exports from the same code, 5 alternating rounds: c8 +2.7 / +1.6 / −0.8 / +3.1% in the four rounds where both completed the c8 cell (the gate asked for ≥ +1% in every round), c1 1.02×, c4 1.00× (0.98× in round 1), 128K / 512K c1 0.98× / 0.97× (1 run). Within export-to-export variance; 12 stays the default (`qualification/results/release-ab/`) |
| BRRR "item 6" (split tail, fast route, counter parity; default-off switches in the fork) | split tail and counter parity slower in the layer A/B; fast route saves 0.5-2 µs per layer but is level end to end (c1 1.004×, c4 0.989×, c8 0.998×) |
| BRRR wide-row restreaming (stage 2) | c4 −5%, c1 −3% |
| Scheduler `throughput` against first-token waits | partial: c4 / c8 probe p50 1.55 / 2.83 s (balanced 2.04 / 4.0 s, interactive 0.36 / 0.38 s) and fewer agent turns per tok/s |
| 7 DSpark drafts | c4 −6%, c8 −3%, coding +2%, c1 level; fewer KV tokens |
| BRRR fused prologue, BRRR in parked layers at every width, larger Engram row cache on disk | see the [20261108 README](../DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md#measured-and-not-adopted) |

## Known limitations

- **Hardware-specific package:** 3 × 96 GB SM120 (RTX PRO 6000) only. Other setups need a fresh export.
- **Interactive scheduler trade-off:** forced prefill rounds interleave with decode, so decoding streams see slightly
  slower while a new prompt is prefilled (c4: 190 instead of ~600 tok/s for those steps; largest inter-token gap
  0.30-0.31 s instead of 0.25 s); averaged over a run their throughput is level. `SCHEDULER=balanced` restores the old
  order ([Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams)).
- **Max concurrency 8** is the qualified package. `MAXC=16` exports a 16-slot package (more graphs, fewer KV tokens);
  it is not part of this release's qualification.
- **top_p is fixed at 1.0** (vocabulary-parallel sampling); temperature is fixed at 1.0 by Hyperflare's DeepSeek recipe.
- **`reasoning_effort`** in chat requests accepts `low`, `high`, `xhigh`, `max` or an integer from 1 through 100;
  other values, including vLLM's `"medium"`, are rejected with HTTP 400.
- **Images**: the vision tower lives in host RAM and its weights are staged through VRAM per block
  (`FLARE_DS41_VISION_STAGE=1`, default since 2026-11-08): 0.11 s cold latency per image instead of 1.70 s without staging.
  `FLARE_DS41_VISION_STAGE=0` restores the direct host-RAM reads.
- **Bench loops:** `llm_decode_bench` decode cells use `ignore_eos`, which forces generation past the answer; a rare
  repetition loop there is a bench artifact (the model emits its end token, then invents tool output). With
  `--respect-eos` the same cells show 0 loops.
- Not upstreamable as written: Hyperflare's contribution rules exclude model-specific offload policy and AI-authored
  GPU-execution changes. This is a personal fork of Hyperflare.

## Coming from 20261108

- **Same base commit** `972e28e4`, same checkpoint view, same FlashInfer patches. Apply this folder's `source.patch` on
  a clean checkout (`git checkout -- . && git clean -fd`, then `git apply`), or fetch fork tag `hyperflare-tp3-20261111`.
- **Rebuild the binary** (`cargo build --release --manifest-path flare/Cargo.toml`): the TTFT guard is in Helios. An
  older binary rejects `--scheduler-policy interactive`; the new binary also serves older packages.
- **Export again** (`export.sh`, defaults): graph scratch changes the package. A tuning cache warmed by the 20261108
  export is enough; the graph-scratch export ran in 36 min with no `HOT=300` warm-up. A fresh machine still needs the
  warm-up export first (EXPORT.md).
- **Serve flags:** `serve.sh` now passes `--scheduler-policy interactive`. `SCHEDULER=balanced ./serve.sh` gives the old
  behaviour (same decode speed, first tokens wait behind decode under load).
- Old packages coexist with the new one (same model name `ds41-1017-engram-mxfp4`, new version). Remove one with
  `flare models remove ds41-1017-engram-mxfp4@gb202-tp3-<timestamp>`, then `flare models gc`.
- Coming from 20261107 or 20261105: read the [20261108 notes](../DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md#coming-from-20261107-or-20261105) first (20261105 used another base commit).

## Building the package

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
# from this folder
./check_env.sh <fork>                                                                 # read-only preflight: toolchain, fork state, disk, idle GPUs
./make_checkpoint_view.sh <deepseek snapshot> <nvidia NVFP4 snapshot> ~/ds41-1017-engram-mxfp4   # 307 GB + 105 GB of downloads
# from the fork checkout (export.sh uses its .venv)
cd <fork>
CKPT=~/ds41-1017-engram-mxfp4 HOT=300 /path/to/this-folder/export.sh   # warms the kernel tuning cache (needed before any HOT above ~300; also with a 20261105 cache)
CKPT=~/ds41-1017-engram-mxfp4 /path/to/this-folder/export.sh           # the release build, HOT=342
```

The model name is the checkpoint directory's name: this view exports as `ds41-1017-engram-mxfp4`, which is what
`serve.sh` runs by default. Another directory name needs `PACKAGE=<that name>` when serving.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `check_env.sh` | read-only preflight: toolchain, memlock, fork commit and patches, disk, idle GPUs (PASS / WARN / FAIL) |
| `versions.txt` | exact versions behind the numbers: driver, CUDA, Python, torch, FlashInfer, Rust, uv, Hyperflare and benchmark commits, GPU settings |
| `MEASUREMENTS.md` | every measurement of this build: power, settings sweeps, k5 vs k7, prefix cache, Engram on disk |
| `ENGRAM-PRECISION.md` | MXFP4 vs FP8 Engram tables: every quality score we have for each, from the vLLM and Hyperflare releases |
| `serve.sh` / `export.sh` | serve the package / export it from a checkpoint view |
| `make_checkpoint_view.sh`, `checkpoint/` | build the checkpoint from the two public releases (config + corrected index) |
| `source.patch` | the fork's changes against Hyperflare `972e28e4`, generated from fork tag `hyperflare-tp3-20261111` (164 files, 110 new: the adapted `brrr/` kernels, vision staging, the Engram and KV-planning changes in Helios, GPU equivalence tests, the fork handbook `docs/TP3-GUIDELINES.md`; see the update notes in EXPORT.md) |
| `expert-order/` | per-layer route counts that set the hot-expert placement |
| `tp3-patches/` | the four FlashInfer patches |
| `tools/` | `ttft_under_load.py` (below), `bench_steps.py` (steps/s from a bench JSON), `verify_fix.py` (corruption check), `llm-bench-flare-steps.patch` (optional, for `llm-inference-bench` v0.9.1); the tool-call test is `qualification/tool_calls.py` |
| `qualification/` | the qualification suite (adapted for Hyperflare, concurrency 8) and its results |
| `tools/ttft_under_load.py` | first-token latency under decode load (`probe`) and an agent-style replay (`agent`); stdlib only |
| `qualification/results/` | this release's full suite (`run-hyperflare-c8/`), the stage A rounds (`release-ab/`), the graph-scratch A/B (`graph-scratch-ab-20261109/`), the TTFT guard A/B (`ttft-ab-20261110/`) and its quick checks (`ttft-quality-20261110/`) |

## Licenses

DeepSeek weights and NVIDIA's NVFP4 release: MIT (texts in `licenses/`). Hyperflare (formerly Flare): Apache-2.0 (`licenses/LICENSE`,
`licenses/NOTICE` and `licenses/THIRD_PARTY_NOTICES.md` kept; modified files marked). FlashInfer, which
`tp3-patches/` modifies: Apache-2.0 (`licenses/FlashInfer-LICENSE`, `licenses/FlashInfer-NOTICE`). BRRR: Apache-2.0
(`licenses/BRRR-LICENSE`, `licenses/BRRR-NOTICE`). `llm-inference-bench`, which `tools/llm-bench-flare-steps.patch`
modifies, is MIT (its README). The fused decoder kernel in `brrr/kernels/decoder.cu` is modified for TP3 and Hyperflare: CSF
scale decode, Hyperflare's dense weight layout, RoPE from Hyperflare's frequencies, hot-expert slot mapping and Hyperflare
peer-storage buffers. The `flare` binary links only libc/libstdc++; NVIDIA's CUDA Python wheels are used only for
exporting and are not redistributed.

## Credits

NVIDIA Hyperflare (formerly Flare) and FlashInfer b12x; DeepSeek for the model; NVIDIA for the MXFP4 Engram tables; Fraser Price for BRRR;
Local Inference Lab for `llm-inference-bench`.
