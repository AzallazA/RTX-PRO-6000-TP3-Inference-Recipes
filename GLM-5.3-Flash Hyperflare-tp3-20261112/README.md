# GLM-5.3-Flash on NVIDIA Hyperflare at TP3 and TP2 — 20261112

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`.

GLM-5.3-Flash (NVIDIA's FP4-Local checkpoint) served by **Hyperflare with the full 1M-token context and image
input**, in two shapes:

- **TP3 / DCP3** on three 96 GB RTX PRO 6000 Blackwell cards: up to **32 concurrent requests**, **13.13M KV
  tokens** (12.5 × 1M), decode **279 / 630 / 866 / 1,290 / 1,790 tok/s** at c1 / c4 / c8 / c16 / c32 (32K context),
  prefill ~11,750 tok/s, 4 images encoded at once. Upstream Hyperflare supports TP1, TP2 and TP4 for GLM; TP3 and its
  decode context parallelism over three GPUs are this fork's port.
- **TP2 / DCP2** on two of the same cards (upstream's own rank count), leaving the third GPU free: up to **8
  concurrent requests**, **2.23M KV tokens** (2.1 × 1M), decode **248 / 512 / 727 tok/s** at c1 / c4 / c8, prefill
  ~11,970 tok/s, 2 images at once.

Images cost little: the vision tower's weights live in host memory and are staged through 22 MiB of VRAM per image
(the DeepSeek V4.1 build's method), and its working memory is laid out to fit inside the prefill's. Exported
text-only (`VISION=0`), the same build holds 2.6% (TP3) or 7.2% (TP2) more KV and decodes at the same speed
([Images](#images)).

This is the first Hyperflare GLM folder; the earlier GLM folders here are vLLM builds
([Kraken-tp3-CSF](../GLM-5.3-Flash%20Kraken-tp3-CSF/README.md) and older), which serve a different checkpoint (see
[Compared with the vLLM Kraken GLM build](#compared-with-the-vllm-kraken-glm-build)).

Weights: [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) @
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` (NVIDIA Open Model Agreement; base model
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT), used as published.
Runtime: [NVIDIA Hyperflare](https://github.com/NVIDIA/hyperflare-inference) `main` @ `cbf21a12` plus this fork (the
public fork https://github.com/AzallazA/spark-local-inference at tag `hyperflare-glm-20261112`; the
same changes as `source.patch` in this folder),
FlashInfer (b12x) @ `85ea6e76` plus the patches in `tp3-patches/`.

**How to get it:** build the packages yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download the checkpoint (~178 GB), build the fork, export (~25 min per package on our machine with a warm tuning
cache; a first export on a new machine takes longer while kernel tuning fills its cache). Then, from the fork checkout:

```bash
PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp> /path/to/this-folder/serve-glm-tp3.sh   # 3 GPUs, port 15015, model GLM-5.3-Flash
PACKAGE=glm53-fp4-local@gb202-tp2-<timestamp> /path/to/this-folder/serve-glm-tp2.sh   # 2 GPUs (0,1)
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.

## Headline numbers

| | **TP3 / DCP3** (3 GPUs, 32 slots) | **TP2 / DCP2** (2 GPUs, 8 slots) |
| --- | --- | --- |
| KV cache served (dry run) | **13,130,496** (13,393,152) | **2,231,808** (2,414,592) |
| Decode 32K, c1 / c4 / c8 (tok/s) | **279 / 630 / 866** | **248 / 512 / 727** |
| Decode 32K, c16 / c32 (tok/s) | **1,290 / 1,790** | — (8 slots) |
| Decode 128K, c1 / c4 / c8 (tok/s) | **276 / 615 / 851** | **246 / 499 / 714** |
| Decode 128K, c16 / c32 (tok/s) | **1,256 / 1,745** | — |
| Prefill 32K / 128K (tok/s) | **11,763 / 11,721** | **11,980 / 11,956** |
| Image encode: small image / 8,000-token image / 4 small images at once | 0.13 s / 1.51 s / 0.54 s | 0.14 s / 1.46 s / 0.60 s |
| Image slots (images encoded at once) | 4 | 2 |
| Startup (launch to ready) / host RAM of the server | **35 s** / 88 GiB | ~30 s (weights load 21 s) / 75 GiB (text-only package) |
| GPQA Diamond (198 questions, effort max, 32K) / tool-eval-bench | **89.4%** / **92**/100 | **90.9%** / **92**/100 |
| eval-43 / tool calls (60 prompts) / corruption check | 42/43 / 60/60 at c32 / 0 | 43/43 / 60/60 at c8 / 0 |
| Image questions (60 chart/diagram + 60 harder value reads) | 60/60, 51/60 | 59/60, 49/60 |
| KV stress at full concurrency | pass (64/64 at c32) | pass (16/16 at c8) |

Decode and prefill: averages of repeated `llm_decode_bench.py` runs (`llm-inference-bench` v0.9.1, ignore-EOS decode
cells, 60 s each) on the packages above, served with the scripts' defaults of the time (prefix cache 32G, 5 drafts, `balanced`
scheduler, which decodes the same as `interactive`; TP3 alpha 0.8, TP2 alpha 1.0). Decode loses only 1-3% from 32K to 128K context at either size.

The dry-run KV count is the planner's figure at 97,281 MiB free per GPU; the served count on the test machine was
2.0% (TP3) and 7.6% (TP2) lower because less memory was free at start. At TP2 each GiB per GPU is worth ~319K KV
tokens, so TP2 is more sensitive to whatever else holds GPU memory.

The text checks (eval-43, tool calls, corruption check, long retrieval and long generation; GPQA, tool-eval-bench and
hotel-lights at TP3) ran on packages of the same commit exported text-only (TP2 GPQA, tool-eval-bench and hotel-lights
ran on the TP2 image package); the image commits only touch the vision tower
(its weight staging and working memory), so the text path, kernels and serve flags are the same. The image packages
were checked for what the image commits change: memory (KV, full-concurrency stress), speed, and images
([Quality](#quality)).

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [How it fits: TP3](#how-it-fits-tp3) · [How it fits: TP2](#how-it-fits-tp2)
2. [Images](#images) · [TP3 or TP2](#tp3-or-tp2) · [Which settings for me](#which-settings-for-me) · [Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams) · [Memory](#memory)
3. [Quality](#quality) · [Measured and not adopted](#measured-and-not-adopted) · [Compared with the vLLM Kraken GLM build](#compared-with-the-vllm-kraken-glm-build) · [Known limitations](#known-limitations)
4. [Building the packages](#building-the-packages) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

## Test system

| Part | Spec |
| --- | --- |
| GPUs | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), each on PCIe Gen5 x16 (TP2 used GPUs 0 and 1) |
| GPU settings | **450 W** power limit, **+75 MHz core / +6000 MHz memory offset** (memory clock 16,365 MHz); ECC off; persistence mode on |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, 1 socket, 1 NUMA node |
| Motherboard | ASUS Pro WS WRX90E-SAGE SE |
| RAM | 256 GB DDR5: 8 × 32 GB Kingston Fury (KF560R32-32), 8 channels at **6000 MT/s** |
| Package drive | PCIe Gen5 x4 NVMe |
| Software | Ubuntu, NVIDIA driver 615.71.09, CUDA 13.1 |

The cards exchange data over PCIe only (no NVLink). Unlike the DeepSeek V4.1 TP3 build, GLM at TP3 and TP2 keeps
**every language-model weight in VRAM** (TP2 only parks the token embedding in host memory, and both park the 373 MiB
vision tower), so host RAM and drive speed matter little once the server is up. Export at the power and clocks you
will serve with: kernel tuning measures the cards as they run.

## What this build contains

| Change | Effect |
| --- | --- |
| **TP3 port of upstream's `glm53` recipe** (fork-only): per-rank shard geometry for three ranks, zero-padded loaders for the parts that do not divide by 3 (checkpoint bytes untouched), rank count 3 declared, world-3 collectives | GLM runs on 3 GPUs; TP1/TP2/TP4 stay byte-identical to upstream (tests) |
| **DCP3** (fork-only, `FLARE_GLM53_DCP3=1` at export): decode context parallelism over the three ranks, 768-token pages, a 3-way candidate merge and a 72-head query/LSE exchange | KV **12.39M instead of 4.21M** (2.94×, 8 slots, 11-09 builds); decode c1 −1.5%, c4 −3.4%, c8 −1.2% |
| **World-3 pull gather** (b12x overlay, `tp3-patches/overlay/`): the DCP head gather pulls peers' rows instead of pushing | head gather 923 → 63 µs at 48 rows; **+7.0 / +8.2 / +8.4%** at c1 / c4 / c8 (11-09) |
| **KDA side chain** (`FLARE_GLM53_KDA_SIDE_CHAIN=1`): the KDA gate/decay projections run beside the q/k/v GEMM, two staging copies per layer dropped | +2.2 / +1.3 / +1.1% at c1 / c4 / c8, outputs bitwise identical in the kernel tests (TP3, 11-10) |
| **704 expert columns per rank** at TP3 (`FLARE_GLM53_EXPERT_COLUMNS=704`): the 2048-wide routed experts padded to 2112 instead of 2304 | **KV +19%** (14.70M vs 12.39M at 8 slots), c1 +4%, c4 +5.6%, c8 level (11-10) |
| **32 request slots** at TP3 | c16 / c32 served with captured graphs; costs ~1.55M KV against an 8-slot export (13.15M vs 14.70M, 11-10/11-11 builds) |
| Upstream PR #253 (dead prefill storage reused for KV, now upstream) and upstream `--mtp-only` (no plain-decode graphs) | KV +2.0% (13.15M → 13.42M at TP3), decode level |
| Adaptive-verification alpha **0.8** at TP3, Hyperflare's default **1.0** at TP2 (serve flag) | TP3: c32 sweep winner (0.6 gave c32 +4.3% but c16 0.99×). TP2: 1.0 against 0.8, 2 alternating rounds at 32K: 253 / 504 / 710 against 247 / 496 / 702 tok/s at c1 / c4 / c8 (1.0 ahead by 1-2% in every cell) |
| **Token embedding in host memory** at TP2 (`FLARE_GLM53_EMBED_HOST=1`, fork-only): the replicated 1,210 MiB BF16 embedding per GPU in device-visible host RAM | with #253: KV 2.38M instead of 1.90M at TP2; decode c8 −1.75%, prefill ~−1% (11-11) |
| **Upstream base**: Hyperflare `main` @ `cbf21a12`, FlashInfer `85ea6e76`, plus upstream's open PRs #122 (compile headroom setting), #124/#125 (faster core cache payloads), #231 (Triton tile vs cuBLAS race for GLM's BF16 `kv_b_proj`) | in a same-session alternating A/B against the build before it: decode 0.99-1.01× (c8 +5%) at TP3, +3-7% at TP2, KV +0.8-1.5% |
| **Vision tower staging** (`FLARE_GLM53_VISION_STAGE=1` with `VISION=1`, fork-only): the replicated vision weights (373 MiB per GPU) in device-visible host memory, staged through 22 MiB of VRAM slots per image | see [Images](#images) |
| **Lean vision working memory** (fork-only): the vision blocks run in place in one buffer laid out by lifetime, temporaries in 1/8 row bands | image packages' extra memory per GPU 1.48 → 0.66 GiB at TP3: KV cost −5.3% → **−2.5%**; image features **bit-identical** to the previous dataflow on GPU |
| **TTFT guard** in Helios (`--scheduler-policy interactive`, same code as the DeepSeek release), **on by default** | see [Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams) |
| GLM tool-call parsing fixes: upstream #179 (keep arguments that do not match their declared type), #59's diagnostics, a `</arg_value>` inside a value is value text | tool calls 60/60 |

Each "Effect" was measured during development on the build of that day (date in brackets), each against the
build before it; the current numbers are in [Headline numbers](#headline-numbers).

## How it fits: TP3

GLM-5.3-Flash has 64 attention heads (KDA linear attention and sparse MLA), 2048-wide routed experts, a 12288-wide
dense MLP and a 154,880-token vocabulary. Three ranks get:

| Dimension | Split over 3 ranks |
| --- | --- |
| KDA and MLA heads (64) | **24 / 24 / 16** from the checkpoint; rank 2 carries 8 zero heads so every rank runs the same 24-head shapes (compiled forwards and scratch are shared by all ranks). 22 heads per rank (64 padded to 66) fails in the b12x sparse-MLA kernel, which is why the split is uneven |
| Indexer heads (32) | replicated |
| Routed experts (2048) | padded to 2112, **704 columns per rank** |
| Shared expert (2048) | padded to 2304, 768 per rank |
| Dense MLP (12288) | 4096 per rank, exact |
| Vocabulary (154,880) | 3 × 51,648 (64-aligned); the 64 extra logits are sliced off |
| Vision tower | replicated on every rank, parked in host memory and staged per image ([Images](#images)) |

Padding is zero weights added at load; the extra heads, columns and rows are inert by the kernels' own arithmetic
(checked by CPU and GPU tests on every rank). Per GPU this leaves ~65 GiB loaded and **~29 GiB for KV**, which DCP3
spreads over the three ranks: each token's KV record lives on one rank, so one 768-token page costs 256 records per
rank. Prefill runs in 4096-token chunks with one shared PCIe all-reduce runtime (static collective tuning).
Speculative decoding: the checkpoint's MTP layer with 5 drafts and adaptive verification.

## How it fits: TP2

TP2 is upstream's own geometry (32 heads, 1,024 expert columns and 77,440 vocabulary rows per rank, no padding) with
upstream's DCP2 (512-token pages). The weights take ~83 GiB per card, which leaves only ~7.7 GiB per GPU for KV
(TP3: ~29 GiB). Parking the token embedding in host memory moves 1.2 GiB per GPU to KV (~+0.38M tokens), at about
−1.75% decode at c8. Everything else is upstream's TP2 path plus the shared improvements (a16, KDA side chain,
`--mtp-only`, #253, the upstream sync, vision staging).

## Images

GLM-5.3-Flash reads images through a 24-block vision tower (up to 8,000 image tokens per image; larger images are
scaled down by the frontend). Upstream keeps that tower resident on every rank. This build:

- **Parks the vision weights in host memory** (373 MiB per GPU, device-visible) and copies each block into a 22 MiB
  VRAM staging area just before it runs, the copy overlapped with compute (`FLARE_GLM53_VISION_STAGE=1`, the same
  method as the DeepSeek V4.1 build). The staged run is bit-identical to the resident tower (CUDA test, 16 cases).
- **Fits the tower's working memory inside the prefill's.** The upstream dataflow keeps full-size FP32 copies
  (normalization, rotary, the clamped SwiGLU) alive at once: capturing the largest image graph grew the shared graph
  pool by ~2 GiB and left 632 MiB more in every package. This build runs the blocks in place in one buffer laid out by
  lifetime (832 MiB at the largest image, below the 1,283 MiB prefill scratch it shares) and computes FP32 temporaries
  in 1/8 row bands. The resulting image features are **bit-identical** to the upstream dataflow on GPU (real weights,
  8 chart images and an 8,000-token composite, graph replay), and repeat bit-identically run to run.
- **Encodes several images at once:** 4 feature slots at TP3 (`--image-concurrency 4`), 2 at TP2, 64 MiB per slot
  per GPU. More images in one request, or more requests with images, queue for a free slot.

| | TP3 (4 slots) | TP2 (2 slots) |
| --- | --- | --- |
| KV vs text-only (dry run) | 13,393,152 vs 13,737,984: **−2.5%** (with the upstream dataflow: −5.3%) | 2,414,592 vs 2,589,184: **−6.7%** |
| Extra fixed memory per GPU | 0.66 GiB | 0.48 GiB |
| First image after start / next small image | 0.15 s / 0.13 s | 0.15 s / 0.14 s |
| 4,096-token image / 8,000-token image | 0.70 s / 1.51 s | 0.71 s / 1.46-1.53 s |
| 4 small images in parallel (wall) | 0.54 s | 0.60 s |
| Smoke: small, near-max, over-max (downscaled), 2 and 4 images in one request | 5/5 correct | 5/5 correct |
| DS41 vision sets (60 chart/diagram questions + 60 harder value reads), T 1.0, effort low | **60/60, 51/60** (upstream-dataflow package, same day: 60/60, 49/60) | **59/60, 49/60** |
| Text decode vs text-only | level (0.99-1.00× at c1 / c8, one same-session round) | level (0.98-1.01×) |

Image latency is measured with one-token answers (`glm_images.py latency`); the question sets are the DeepSeek V4.1
release's vision qualification sets, answered with GLM's thinking at effort `low`. `VISION=0` at export gives the
text-only packages (more KV, no images).

## TP3 or TP2

| | TP3 | TP2 | TP3 / TP2 |
| --- | --- | --- | --- |
| GPUs | 3 | 2 (third GPU free) | |
| KV tokens (served) | 13,130,496 | 2,231,808 | **5.9×** |
| Slots / image slots | 32 / 4 | 8 / 2 | |
| Decode c1 / c4 / c8 (32K) | 279 / 630 / 866 | 248 / 512 / 727 | **1.13 / 1.23 / 1.19×** |
| Weights per GPU | ~65 GiB | ~87 GiB (incl. scratch) | |

TP3 is faster at every concurrency and holds about six times the KV: twelve 1M-token sessions instead of two.
TP2 is for a machine with two cards, or one that needs the third card for something else.

## Which settings for me

| Your situation | Export | Serve | Expect |
| --- | --- | --- | --- |
| 3 cards, many users or agents (the tested TP3 setup) | `export-glm-tp3.sh` (defaults: `CONC=32`, images, 4 image slots) | `serve-glm-tp3.sh` | the TP3 numbers above |
| 3 cards, text only, maximum KV | `VISION=0 export-glm-tp3.sh` | `serve-glm-tp3.sh` | +2.6% KV (13.74M dry run), no images |
| 3 cards, at most 8 requests at once, maximum KV | `CONC=8 export-glm-tp3.sh` | `serve-glm-tp3.sh` | **not measured on this build.** On the 11-10/11-11 builds an 8-slot text-only package held 14.70M KV against 13.15M for 32 slots, at c1 / c4 / c8 271 / 614 / 835 against 272 / 608 / 794 tok/s |
| 2 cards (the tested TP2 setup) | `export-glm-tp2.sh` (images, 2 image slots) | `serve-glm-tp2.sh` | the TP2 numbers above |
| 2 cards, text only | `VISION=0 export-glm-tp2.sh` | `serve-glm-tp2.sh` | +7.2% KV (2.59M dry run), no images |
| Batch jobs, nobody waits on a first token | as above | `SCHEDULER=balanced` | same decode; new prompts wait 3-22 s under load ([Scheduler trade-off](#scheduler-trade-off-first-token-vs-running-streams)) |
| Many long sessions you return to | as above | `PREFIX_CACHE=64G` (or more) | more warm prompts in host RAM; costs only RAM (not measured for GLM) |
| 350 W cards | export at your power limit | defaults | not measured for GLM; the DeepSeek TP3 build lost 10-14% at 350 W |

Keep 5 drafts (the export's). 3 and 5 drafts measured level at TP2 and TP3 (adaptive verification offers ~1.8
drafts per round either way). Capping the verification rows does not help at c32 either: `BUDGET=32` forces
one-row rounds at 32 requests but each still costs ~22 ms (1,365-1,407 tok/s), `BUDGET=48` gave 1,239, the default
1,689-1,718 (image package, 32K, 60 s).

## Scheduler trade-off: first token vs running streams

The GPUs run one step at a time. When a new prompt arrives while other requests are decoding, its prefill either runs
now (the running streams pause for those steps) or waits until the scheduler gets to it (the new prompt waits). Every
policy picks a point between the two; none removes it. The serve scripts take `SCHEDULER=`; **the default is
`interactive`** (Hyperflare's `balanced` plus a TTFT guard, as in the DeepSeek release); `SCHEDULER=balanced` gives
Hyperflare's own default:

| TP3 image package, same session, 2 alternating rounds each | `balanced` | **`interactive` (default)** |
|---|---|---|
| First token p90, 1.9K-token prompt while 3 / 7 / 15 / 31 streams decode | 2.92 / 6.13 / 11.59 / 22.30 s | **0.47 / 0.48 / 0.51 / 0.53 s** |
| Agent replay, 8 streams, 300 s: turns / output tok/s / TTFT p50 | 327 / 253 / 4.18 s | **424 / 303 / 0.61 s** |
| Agent replay, 16 streams, 300 s: turns / output tok/s / TTFT p50 | 325 / 302 / 8.52 s | **478 / 347 / 0.95 s** |
| Decode tok/s c1 / c4 / c8 / c16 / c32 (32K) | 266 / 588 / 820 / 1,211 / 1,694 | 267 / 587 / 837 / 1,213 / 1,703 (1.00-1.02×) |

`tools/ttft_under_load.py probe` (first token) and `agent` (replay of coding-agent turns: growing history, ~1.9K-token
tool results, staggered arrivals); decode with `llm_decode_bench`, 32K, 60 s; arms alternated bal / int / bal / int
on fresh servers. With the guard, agents finish 30-47% more turns and 15-20% more output in the same time, and plain
decode is unchanged. The cost is local: while a new prompt prefills, the running streams slow down for those steps
(the "during" decoder rate in `probe` falls to roughly a quarter for ~0.1-0.4 s). Use `SCHEDULER=balanced` for pure
batch traffic where no one waits on a first token. Not measured at TP2 (same guard code, same default).

**What the guard does (Helios scheduler, `--scheduler-policy interactive` = balanced + `--scheduler-ttft-target-ms 50`).**
Each round the scheduler checks the oldest request still without its first token. Once it has waited 50 ms, that
round may only run work that moves it forward (its prompt prefill, prefix-cache restore or image encoding). After each
forced prefill chunk at least one normal round lets the other streams decode, so a long prompt does not freeze them
for its whole prefill. With the target at 0 (`balanced`, `throughput`) the code path is inert and scheduling matches
upstream exactly (a trace test checks this, including a GLM TP3 simulation). The full description is in the
[DeepSeek release](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#scheduler-trade-off-first-token-vs-running-streams).
No re-export is needed to switch; the package is the same.

## Memory

| Per GPU | TP3 (image package) | TP2 (image package) |
| --- | --- | --- |
| Loaded (weights, graphs, scratch; `Plan cache` line) | 64.88-64.92 GiB | 86.56 GiB |
| KV cache budget | 29.35-29.39 GiB (88.0 GiB over 3 GPUs → 13,130,496 tokens served) | 7.71 GiB (15.4 GiB over 2 GPUs → 2,231,808 tokens served) |
| Weights loaded at start | 176.2 GiB over 3 GPUs + vision tower in host memory | 167.9 GiB over 2 GPUs + embedding and vision tower in host memory |
| Operator scratch (shared, export log) | 578 MiB | 367 MiB |

Host RAM: the prefix cache (`PREFIX_CACHE=32G`; the server log reports `memory 39.7 GiB (8031 slots, split by rank)`
at TP3 and `30.9 GiB (9375 slots)` at TP2) plus the server process: **88 GiB** resident for the TP3 image package (prefix cache included), 75 GiB measured for
the text-only TP2 package.

## Quality

Checks on the live packages, served with the `balanced` scheduler (scheduling only; outputs use the same kernels under `interactive`), thinking on (GLM always
thinks), temperature 1.0, top-p 0.95. Effort: eval-43 and image questions `low`, tool calls `high`, GPQA `max`
(32K tokens). The text checks ran on the text-only packages of the same commit ([Headline numbers](#headline-numbers),
"What carried means"); the image packages added the image, memory and speed checks.

| Check | TP3 | TP2 |
| --- | --- | --- |
| Corruption check (teacher-forced comparison, 20 runs, 6,000 tokens) | **0** true corruptions | **0** |
| eval-43 | **42**/43 (c32) | **43**/43 (c8) |
| Tool calls: 60 tool-requiring prompts | **60**/60 at c32 | **60**/60 at c8 |
| GPQA Diamond (198 questions, 32K tokens) | **89.4%** (177/198, Wilson 95% 84.3-93.0), 1 hit max tokens | **90.9%** (180/198, Wilson 95% 86.1-94.2), 1 hit max tokens; on the image package |
| tool-eval-bench | **92**/100 | **92**/100 (image package) |
| hotel-lights, 30 runs | **30/30** (a first run: 26/30, 2 wrong, 2 truncated) | 26/30 on the image package (2 unparseable answers, 1 truncated, 1 wrong: 49 for 48); 26/30 text-only (2 unparseable, 2 wrong) |
| Long retrieval: three markers at 128K and 900K tokens | pass / pass | pass / pass |
| Long generation after a ~530K-token conversation (2 seeds) | pass (10,086 and 20,363 tokens, worst 8-gram repeat 0.0005) | pass (11,164 and 17,340 tokens, 0.0004) |
| KV stress at full concurrency, KV oversubscribed: image package | **pass**: 64/64 long requests at c32, pinger 246/246, 0 bad log lines, min free VRAM 44 MiB | **pass**: 16/16 at c8, pinger 139/139, min free 96 MiB |
| Image questions, smoke, latency | [Images](#images) | [Images](#images) |
| Determinism (same request, fresh prefill, across a restart) | first token 24/24 identical; seeded multi-token answers 2/9 identical | same pattern |

hotel-lights is judged by the details on GLM: its baseline is 26-30/30 on every GLM build we have run (vLLM
included), and its misses are mostly truncated answers (thinking at max effort) or an unparseable answer format, not
wrong answers. The KV stress pushes 2 waves × 32 (TP3) or × 8 (TP2) requests of ~530K unique prompt tokens; "min free"
below 480 MiB is flagged as a warning by the kit, nothing failed.

## Measured and not adopted

| Idea | Result |
| --- | --- |
| a4 expert activations (4-bit activations into the NVFP4 experts at decode) | +7% c1, +3-5% c4/c8, but prose perplexity +1.9%, code +0.2-0.34%, KL 4.8-6.4× the noise floor (302K teacher-forced positions); task tests equal. Measurable loss: a16 kept |
| Local Inference Lab's CSF-QAD checkpoint (the one the vLLM Kraken build serves) instead of FP4-Local | faster (a4: c1 +1-2%, c4 +3-6%, c8 +0.5-1%) but perplexity +0.36% (a16 vs a16) and ~1M less KV. FP4-Local kept |
| No DCP at TP3 (DCP1) | KV 4.21M instead of 12.39M for ~1.5-3% more decode speed; DCP3 kept |
| FP32 per-rank partials in the DCP3 merge | would remove the extra BF16 rounding ([Known limitations](#known-limitations)); estimated from measured merge timings at −1% c1, −3% c4, −3.5% c8 (not built on GPU); not pursued |
| NVFP4 draft head + MXFP8 draft `eh_proj` ("draft diet") | c8 −0.8%, hotel-lights 27/30 with 3 wrong answers; rejected |
| Resident expanded expert scales | +5.3% c1, +2% c4/c8, −0.48M KV (a16, 11-09); not adopted |
| 768 expert columns per rank (the first TP3 layout) | 16% less KV than 704 (12.39M vs 14.70M) and 4-5% slower at c1/c4, c8 level (11-10) |
| 3 drafts instead of 5 | level within ±2% at every cell, TP2 and TP3 |
| Alpha 0.6 | c32 +4.3% but c16 0.99×; 0.8 kept |
| Verification row budget 32 / 48 at TP3 | c32 1,365-1,407 / 1,239 tok/s against 1,689-1,718 by default; c1-c16 level or lower |
| Token embedding in host memory at TP3 | kept to TP2, where each GiB of VRAM is worth far more KV in relative terms (+17% per GiB at TP2, +4% at TP3) |
| Images with the upstream vision dataflow (staging only) | KV −5.3% at TP3 instead of −2.5%; same images, same speed |

## Compared with the vLLM Kraken GLM build

[GLM-5.3-Flash Kraken-tp3-CSF](../GLM-5.3-Flash%20Kraken-tp3-CSF/README.md) (vLLM, 20261015, 32 slots) at **350 W**
with the CSF-QAD checkpoint, MTP3, DCP3: KV 10,485,760, decode 266-280 / 768-795 / 1,588-1,639 tok/s at c1 / c8 / c32.
This build at 450 W with FP4-Local, MTP 5, DCP3: KV 13,130,496 served with images, 279 / 866 / 1,790 tok/s. Power,
checkpoint and draft width all differ, so this is not an A/B. The two checkpoints differ in quality too (see the
CSF-QAD row in [Measured and not adopted](#measured-and-not-adopted)).

## Known limitations

- **Hardware-specific packages:** 3 × or 2 × 96 GB SM120 (RTX PRO 6000) only. Other setups need a fresh export.
- **Thinking is always on.** GLM-5.3-Flash's chat template has no switch: `enable_thinking: false` is ignored without
  an error. `reasoning_effort` `low` or `high` selects that effort; any other value (including `none` and `medium`)
  gives `max`. `chat_template_kwargs.clear_thinking=true` empties reasoning in assistant turns before the last user
  message.
- **Sampling:** temperature and top_p are clamped to 0.95-1.0 (defaults 1.0 and 0.95), the recipe's ranges.
- **Not bit-identical run to run.** The first token of a fresh prompt repeats exactly across restarts; multi-token
  answers mostly do not (2 of 9 identical), because the scheduler picks draft widths from live timings. Two exports of
  the same commit can also differ in the last bits (kernel tuning chooses per export).
- **DCP3 rounding:** with DCP3 each rank's partial attention is rounded to BF16 once more before the merge. In short
  contexts decode agrees slightly less with a teacher-forced prefill than without DCP (top-1 0.955 vs 0.974 on ~1,000
  greedy positions); at a 1,500-token context the two agree equally (0.927 vs 0.929). Quality checks above are on DCP3.
- **No plain decode graphs** (`--mtp-only`): the packages refuse `--mtp-draft-tokens 0`.
- **Images queue for a slot:** 4 images encode at once at TP3, 2 at TP2; more wait. Each image is at most 8,000 tokens
  (larger ones are scaled down by the frontend).
- **One model name for both packages:** TP2 and TP3 exports of the same checkpoint directory share the model name;
  serve by version.
- **Bench spread:** single 60 s cells vary by a few percent with draft acceptance; one TP3 round gave c4 713 and c32
  1,935 tok/s against ~610 and ~1,730 in the other two.
- **`SCHEDULER=balanced` under load:** a new prompt waits behind the decoding streams (p90 2.9 s at 3 streams,
  22 s at 31). The default `interactive` avoids this.
- Not upstreamable as written: Hyperflare's contribution rules exclude model-specific changes of this kind and
  AI-authored GPU-execution changes. This is a personal fork of Hyperflare.

## Building the packages

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
hf download nvidia/GLM-5.3-Flash-FP4-Local --revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 --local-dir ~/glm53-fp4-local
git clone https://github.com/AzallazA/spark-local-inference.git flare && cd flare && git checkout hyperflare-glm-20261112
uv sync --locked --extra prepare --extra test && cargo build --locked --release --manifest-path flare-cli/Cargo.toml
# apply tp3-patches/*.patch to the venv's FlashInfer (EXPORT.md step 2), then:
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp3.sh     # TP3 with images, ~25-40 min
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp2.sh     # TP2 with images, ~25-30 min
```

The model name is the checkpoint directory's name: `glm53-fp4-local`, which is what the serve scripts use by default.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the packages yourself |
| `export-glm-tp3.sh` / `export-glm-tp2.sh` | export the TP3 / TP2 package from the checkpoint (run from the fork checkout); images on by default, `VISION=0` for text-only |
| `serve-glm-tp3.sh` / `serve-glm-tp2.sh` | serve them (port 15015, model `GLM-5.3-Flash`; `SCHEDULER`, `ALPHA`, `BUDGET`, `PREFIX_CACHE`, `PACKAGE`, `GPUS` overrides) |
| `versions.txt` | exact versions behind the numbers: driver, CUDA, Python, torch, FlashInfer, Rust, uv, Hyperflare commit, packages, GPU settings |
| `tp3-patches/` | the FlashInfer patches for the venv, and `overlay/` (world-3 pull gather, applied to a separate b12x copy by the export scripts) |
| `tools/verify_fix.py`, `tools/glm_common.py` | corruption check (decode vs teacher-forced prefill), GLM settings shared by the tools |
| `tools/tool_calls.py` | tool-call test: N tool-requiring prompts at concurrency C, GLM markup leaks fail |
| `tools/ttft_under_load.py` | first-token latency under decode load (`probe`) and an agent-style replay (`agent`); stdlib only |
| `tools/bench_steps.py` | steps/s and tokens/step per cell from an `llm_decode_bench` JSON |
| `tools/glm_images.py` | image checks against a live server: `latency`, `smoke` (near-max, over-max, 2 and 4 images), `eval` (the DeepSeek V4.1 vision sets; `VISION_SETS=` points at them) |
| `source.patch` | the fork's changes against Hyperflare `cbf21a12`: `git diff cbf21a12 <release commit ac9136b0>` (316 files, 138 new; local paths in the fork's notes, scripts and test defaults replaced by `/path/to/...` placeholders and `$HOME`). Applies cleanly to a `cbf21a12` checkout |
| `qualification/` | the GLM qualification kit (`qual_all.sh`, `vision/` tower checks) and its sanitized results per package; `RESULTS.md` maps every check to its file |

## Licenses

Weights: NVIDIA Open Model Agreement for `nvidia/GLM-5.3-Flash-FP4-Local`; the base model GLM-5.3-Flash is MIT
(see the checkpoint's model card). Hyperflare (formerly Flare): Apache-2.0 (`licenses/LICENSE`, `licenses/NOTICE` and
`licenses/THIRD_PARTY_NOTICES.md` kept; modified files marked). FlashInfer, which `tp3-patches/` modifies:
Apache-2.0 (`licenses/FlashInfer-LICENSE`, `licenses/FlashInfer-NOTICE`). `llm-inference-bench` is MIT (its README).
NVIDIA's CUDA Python wheels are used only for exporting and are not redistributed.

## Credits

NVIDIA Hyperflare (formerly Flare) and FlashInfer b12x, including the upstream GLM-5.3-Flash recipe and the upstream
PRs this build carries; Z.ai for GLM-5.3-Flash; NVIDIA for the FP4-Local checkpoint; Local Inference Lab for
`llm-inference-bench` and the FlashInfer b12x fork.
