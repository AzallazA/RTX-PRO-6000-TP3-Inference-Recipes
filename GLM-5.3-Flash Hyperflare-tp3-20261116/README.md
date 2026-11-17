# GLM-5.3-Flash on NVIDIA Hyperflare at TP3 and TP2 — 20261116

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`.

GLM-5.3-Flash (NVIDIA's FP4-Local checkpoint) served by **Hyperflare with the full 1M-token context and image
input**, at TP3 / DCP3 on three 96 GB RTX PRO 6000 Blackwell cards (32 request slots) and at TP2 / DCP2 on two of them
(8 slots). This build supersedes [Hyperflare-tp3-20261112](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md). It
is a new upstream base, a new `flare` binary and new exports (see [Coming from 20261112](#coming-from-20261112)).
Changes since then:

- **More KV cache.** With the serve scripts on the release binary, TP3 serves **14,929,920 KV tokens** (14.2 × 1M;
  20261112: 13,130,496, **+13.7%**) and TP2 **3,113,472** (3.0 × 1M; 20261112: 2,231,808, **+39.5%**).
- **New upstream base.** The fork now sits on Hyperflare `70069281` (was `cbf21a12`), the base of the DeepSeek V4.1
  release [Hyperflare-tp3-20261116](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md); one fork tag and one
  `flare` binary serve both models. With the 20261112 settings the port alone served 13,833,984 KV tokens at TP3
  (+5.4%) and 2,518,016 at TP2 (+12.8%).
- **Export-time VRAM reclaim switches** (fork-only, on in the export scripts). They change which allocations an export
  keeps, not what a graph computes ([What this build contains](#what-this-build-contains)). On top of the port they
  add 1,000,704 KV tokens at TP3 (+7.2%) and 531,968 at TP2 (+21.1%).
- **Serve settings of the DeepSeek release:** the first-token guard serves the overdue request with the fewest prompt
  chunks left first (`TTFT_ORDER=shortest`), and the CUDA stack limit is set to 256 bytes (`STACK_LIMIT=256`; served KV
  +95,232 at TP3 and +63,488 at TP2 against the export chains' boots without it, on another build).
- **Decode and prefill level with 20261112** (single passes; [Headline numbers](#headline-numbers)).

**Released after the exports and their quick checks passed; the full qualification of these packages has not run
yet.** [Quality](#quality) lists what ran on these packages, on which binary, and what did not run.

Weights: [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) @
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` (NVIDIA Open Model Agreement; base model
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT), used as published.
Runtime: [NVIDIA Hyperflare](https://github.com/NVIDIA/hyperflare-inference) `main` @ `70069281` plus this fork
(`source.patch`; the same code is the public fork https://github.com/AzallazA/spark-local-inference at tag
`hyperflare-glm-20261116`), FlashInfer (b12x) @ `aa6c4221` (upstream's pin, 0.7.3) plus four patches in `tp3-patches/`
and one b12x overlay patch in `tp3-patches/overlay/`.

**How to get it:** build the packages yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download the checkpoint (~178 GB), build the fork, export (TP3 35 min, TP2 17 min on our machine with a warm
tuning cache; a first export on a new machine takes longer). Then, from the fork checkout:

```bash
PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp> /path/to/this-folder/serve-glm-tp3.sh   # 3 GPUs, port 15015, model GLM-5.3-Flash
PACKAGE=glm53-fp4-local@gb202-tp2-<timestamp> /path/to/this-folder/serve-glm-tp2.sh   # 2 GPUs (0,1)
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.
Every measurement is in [MEASUREMENTS.md](MEASUREMENTS.md); check results in [qualification/RESULTS.md](qualification/RESULTS.md).

## Headline numbers

| | **TP3 / DCP3** this build | TP3 20261112 (published) | **TP2 / DCP2** this build | TP2 20261112 (published) |
| --- | --- | --- | --- | --- |
| Package | `gb202-tp3-202611170053` | `gb202-tp3-202611130000` | `gb202-tp2-202611170201` | `gb202-tp2-202611130307` |
| Request slots / image slots | 32 / 4 | 32 / 4 | 8 / 2 | 8 / 2 |
| KV cache served | **14,929,920** | 13,130,496 | **3,113,472** | 2,231,808 |
| KV served without the stack limit / dry run | 14,834,688 / 15,025,152 | — / 13,393,152 | 3,049,984 / 3,177,472 | — / 2,414,592 |
| Decode 32K (tok/s), release binary | c1 / c8 / c32 **290.3 / 878.8 / 1,827.6** | c1 / c8 / c32 279 / 866 / 1,790 (c4 630, c16 1,290) | c1 / c8 **245.4 / 750.2** | c1 / c4 / c8 248 / 512 / 727 |
| Decode 32K (tok/s), export chain binary | c1 / c8 / c32 279.1 / 886.6 / 1,812.5 | | c1 / c4 / c8 246.5 / 519.2 / 762.8 | |
| Prefill 8K / 32K (tok/s) | 10,180 / **11,701** | — / 11,763 | 11,583 / **12,275** | — / 11,980 |
| Image: first / next small / 8,000-token / 4 small at once | 0.14 / 0.12 / 1.46 / 0.45 s | 0.15 / 0.13 / 1.51 / 0.54 s | not timed (smoke 5/5) | 0.15 / 0.14 / 1.46 / 0.60 s |
| Startup (launch to ready) | 30 s | 35 s | 29 s | ~30 s |

"Release binary": one boot of each package on the tested build of this tag's code (`27864a1d79e4`) with the serve
scripts' settings (interactive scheduler, TTFT order shortest, stack limit 256, prefix cache 32G, 5 drafts, TP3 alpha
0.8). "Export chain binary": the exports' own checks on an earlier build of the port (`67c691d29041`: no TTFT order
or stack-limit option, default first-token order). KV served depends on the free memory at start; the dry run is the
planner's figure at 97,281 MiB free per GPU.

**The columns use different methods.** This build's decode cells are single 30 s passes (`llm_decode_bench.py`,
`llm-inference-bench` v0.9.1, ignore-EOS, 32K context); the 20261112 columns are averages of repeated 60 s runs.
Single passes vary by a few percent with draft acceptance (TP3 c1 gave 279.1 and 290.3 on the two binaries), so read
decode as level with 20261112, not as a speed-up. The fair speed comparison is the export chain's pass against the
same settings without the reclaim switches on the same binary (local packages, not published): TP3 c1 / c8 / c32 279
/ 887 / 1,812 against 274 / 873 / 1,806, TP2 c1 / c4 / c8 247 / 519 / 763 against 245 / 517 / 748 tok/s. The switches
change memory only.

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [How it fits](#how-it-fits) · [Memory](#memory)
2. [Which settings for me](#which-settings-for-me) · [Scheduler](#scheduler)
3. [Quality](#quality) · [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Coming from 20261112](#coming-from-20261112) · [Going back to 20261112](#going-back-to-20261112) · [Building the packages](#building-the-packages) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

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

The cards exchange data over PCIe only (no NVLink). GLM keeps every language-model weight in VRAM (TP2 parks the token
embedding in host memory; both park the 373 MiB vision tower), so host RAM and drive speed matter little once the
server is up. Export at the power and clocks you will serve with: kernel tuning measures the cards as they run.

## What this build contains

Since 20261112:

| Change | Effect |
| --- | --- |
| **Upstream base `70069281`** (port of the fork's work, 2026-11-14): upstream's compiler-owned storage replaces the fork's graph scratch; FlashInfer `aa6c4221` (0.7.3) | with the 20261112 settings: KV 13,833,984 at TP3 (+5.4%) and 2,518,016 at TP2 (+12.8%); decode level, prefill +0.5-1% (TP3) |
| **Vision tail staged per image** (`FLARE_GLM53_VISION_STAGE=1` now stages the tower's blocks and its tail; the tail is copied per image, 108 MiB) | 22 MiB of VRAM staging slots per GPU, as before; image latency and smoke as before |
| The fork's lean vision working memory was **not carried** onto the new base: the base's compiler plans that memory | the largest image graph pool (976 MiB) still fits under the largest prefill pool (1,008 MiB at TP3, 1,072 MiB at TP2) |
| **Port fixes on GLM:** context-parallel candidate gathers write plain BF16 row slices (torch 2.14.1's Inductor miscompiled the FP32 views, which gave wrong answers above 1,024 prompt tokens in the first port export); GLM's BF16 head BMMs race Triton tiles only at export (`FLARE_BF16_BMM_CUBLAS=0`, contains a cuBLAS over-read on SM120); decode glue cleanup | a prefill probe (needles and counts from 128 to 33K tokens, images) is now an export gate |
| **`FLARE_GLM53_TRIAL_POOL=separate`** (TP3 and TP2, export, fork-only): the eager "Measure graph memory" trial graphs get their own pool, released before the serving capture | TP3: 3,574 MiB returned, kept graph pool 1,110 MiB per GPU (was 3,216); TP2: 2,990 MiB returned, 1,110 MiB kept (was 2,442) |
| **`FLARE_GLM53_SLAB_SEGMENT=1`** (TP3, export): the preparation-only context-parallel prefill slab (528 MiB) takes whole 2 MiB granules | no later tensor shares the slab's last segment and keeps the 530 MiB mapping in the package |
| **`FLARE_GLM53_ONESHOT_FIT=1`** (TP2, export): the one-shot all-reduce is sized for the 512 rows that use it instead of 4,096 (prefill of 1,024+ rows uses the copy-engine ring) | TP communication buffers 138 MiB per GPU (was 502) |
| Together (reclaim switches) | TP3 +1,000,704 served KV (+7.2%), TP2 +531,968 (+21.1%), against exports of the same settings without them on the same binary; decode and prefill level |
| **TTFT guard order `shortest`** (serve flag `--scheduler-ttft-order`, from the DeepSeek release) | the guard serves the overdue request with the fewest prompt chunks left first, so a short request is not held behind a long prompt's prefill |
| **CUDA stack limit 256 bytes** (`FLARE_CUDA_STACK_LIMIT`, from the DeepSeek release) | less per-thread stack reserved by the driver (largest GLM kernel stack: 32 B at TP3, 48 B at TP2): served KV +95,232 at TP3, +63,488 at TP2 (release binary with the limit against the export chains' boots without it) |
| **Release scripts in the fork** (`tp3-tools/release/glm/`) | the scripts in this folder are copies of the tag's |

Unchanged from [20261112](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#what-this-build-contains): the TP3 port
of upstream's `glm53` recipe (24 / 24 / 16 heads, padded loaders), DCP3 with 768-token pages, the world-3 pull gather
(b12x overlay), the KDA side chain, 704 expert columns per rank at TP3, 32 slots at TP3, `--mtp-only`, upstream PR
#253, adaptive-verification alpha 0.8 at TP3 and 1.0 at TP2, the token embedding in host memory at TP2, vision tower
staging, the TTFT guard (`interactive`, on by default) and the GLM tool-call parsing fixes. That folder explains each
with its measurement.

## How it fits

The shard geometry is unchanged from 20261112 ([TP3](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#how-it-fits-tp3),
[TP2](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#how-it-fits-tp2)): at TP3 every rank runs 24 KDA/MLA heads
(rank 2 has 8 zero heads), 704 routed-expert columns, a third of the dense MLP and of the vocabulary; DCP3 keeps each
token's KV record on one rank. TP2 is upstream's own geometry with DCP2. What changed is how much of each GPU the
package keeps outside the KV cache:

| Per GPU | TP3 this build | TP3 20261112 | TP2 this build | TP2 20261112 |
| --- | --- | --- | --- | --- |
| Loaded (weights, graphs, scratch; `Plan cache` line) | **61.49 GiB** | 64.88-64.92 GiB | **84.15 GiB** | 86.56 GiB |
| KV cache budget | **33.12-33.13 GiB** (99.3 GiB over 3 GPUs) | 29.35-29.39 GiB (88.0 GiB) | **10.47 GiB** (20.9 GiB over 2 GPUs) | 7.71 GiB (15.4 GiB) |

(Release binary with the stack limit; without it 32.92 GiB at TP3 and 10.26 GiB at TP2.)

At TP2 each GiB of VRAM per GPU is worth ~0.3M KV tokens against a 3.05M pool, which is why the reclaim adds more in
relative terms there.

## Memory

| | TP3 | TP2 |
| --- | --- | --- |
| Package on disk | 174 GiB | 168 GiB |
| Weights in host memory | vision tower (373 MiB per GPU, device-visible) | vision tower + token embedding (1.55 GiB per GPU mapped host memory in total) |
| Host prefix cache (`PREFIX_CACHE=32G`, server log) | 39.7 GiB (8,031 slots) | 30.9 GiB (9,375 slots) |

The dry-run KV count is the planner's figure at 97,281 MiB free per GPU; the served counts on the test machine (release
binary, stack limit 256) were 0.6% (TP3) and 2.0% (TP2) lower because less memory was free at start.

## Which settings for me

| Your situation | Export | Serve | Expect |
| --- | --- | --- | --- |
| 3 cards, many users or agents (the tested TP3 setup) | `export-glm-tp3.sh` (defaults: `CONC=32`, images, 4 image slots, `RECLAIM=1`) | `serve-glm-tp3.sh` | the TP3 numbers above |
| 2 cards (the tested TP2 setup) | `export-glm-tp2.sh` (8 slots, images, 2 image slots, `RECLAIM=1`) | `serve-glm-tp2.sh` | the TP2 numbers above |
| Text only, more KV | `VISION=0 export-glm-tp*.sh` | as above | not measured on this build; on 20261112 +2.6% KV at TP3, +7.2% at TP2 |
| The 20261112 memory layout | `RECLAIM=0 export-glm-tp*.sh` | as above | ~7% (TP3) / ~21% (TP2) less KV, same speed |
| Batch jobs, nobody waits on a first token | as above | `SCHEDULER=balanced` | same decode; new prompts wait behind decoding streams ([Scheduler](#scheduler)) |
| The 20261112 guard order | as above | `TTFT_ORDER=oldest` | the guard serves the oldest waiting request first |
| Many long sessions you return to | as above | `PREFIX_CACHE=64G` (or more) | more warm prompts in host RAM; costs only RAM (not measured for GLM) |

Keep 5 drafts (the export's); the packages refuse `--mtp-draft-tokens 0` (`--mtp-only`). The other 20261112 notes
(3 vs 5 drafts level, verification budgets 32 / 48 slower at c32) were measured on the earlier base and not repeated.

## Scheduler

The serve scripts default to `SCHEDULER=interactive` (Hyperflare's `balanced` plus a 50 ms first-token target), as in
20261112; its measurements (first-token p90 ~0.5 s instead of 3-22 s behind 3-31 decoding streams, 30-47% more agent
turns, decode level) are in the [20261112 README](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#scheduler-trade-off-first-token-vs-running-streams).
New: `TTFT_ORDER=shortest` (default). With many long prompts loading, the guard used to serve the oldest waiting
request, which is usually a long prompt; short requests then waited (~35 s per ping in a TP3 KV stress on the port
build, against ~12 s with `balanced`). With `shortest`, a test with 28 long prompts loading answered 21 short pings
against 5 with `oldest` (2026-11-15, port build). Agent traffic was not A/B-tested on GLM with this order; the
DeepSeek release measured it ([its README](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#scheduler)).
The prefill merge and tail merge of the DeepSeek release are not on for GLM.

## Quality

The packages were checked by the export chains, then booted once on the release binary. **The tiered qualification
(tier "export": same weights, new export) has not run on these packages yet.**

**Export chain checks** (binary `67c691d29041`; `serve-glm` defaults of the time: `interactive`, default first-token
order, no stack override, prefix cache 32G, 5 drafts, TP3 alpha 0.8, TP2 package alpha):

| Check | TP3 `gb202-tp3-202611170053` | TP2 `gb202-tp2-202611170201` |
| --- | --- | --- |
| KV served / dry run | 14,834,688 / 15,025,152 | 3,049,984 / 3,177,472 |
| First-token probe (24 prompts with logprobs, same-server repeat) | 24/24, repeat 12/12 identical; equal to the 20261112 package's stored probe 12/12 | 24/24, repeat 12/12; equal to the 20261112 package's stored probe 12/12 |
| Image smoke (small, near-max, over-max, 2 and 4 images) | 5/5 correct | 5/5 correct |
| Image latency | first 0.14 s, 8,000-token 1.46 s | not run |
| Prefill probe (needles and counts 128 to 33K tokens, 12 image cases) | PASS; one marginal case: 33K needle 30/32 found against 32/32 for the package without reclaim (2 omissions, p = 0.25, not significant); all other needles and images exact | PASS (needles 248/248, images 36/36) |
| Corruption check (decode vs teacher-forced prefill, 20 runs, 6,000 tokens) | not run in the chain | **0** true corruptions |
| Tool calls, 60 tool-requiring prompts | not run in the chain | **60/60** at concurrency 8 |
| Decode / prefill pass | see [Headline numbers](#headline-numbers) | see [Headline numbers](#headline-numbers) |
| Highest GPU temperature (export / speed pass) | 51 / 78 C | 52 / 71 C |

**Boot check on the release binary** (tested build `27864a1d79e4`; this tag's Rust code is that of the DeepSeek
20261116 release; `serve-glm-tp*.sh` settings: TTFT order shortest, stack limit 256), one boot per package:

| Check | TP3 | TP2 |
| --- | --- | --- |
| Boot, served KV, stack line | PASS: up in 30 s, 14,929,920 KV, stack limit 256 B (largest kernel stack 32 B) | PASS: up in 29 s, 3,113,472 KV, 256 B (largest 48 B) |
| Text smoke (17 × 23) | 391 | 391 |
| Image smoke | 5/5 correct | 5/5 correct |
| First-token probe 24 + same-server repeat 12 | 24/24, 12/12 identical | 24/24, 12/12 identical |
| Decode 32K, one 30 s pass | c1 / c8 / c32 290.3 / 878.8 / 1,827.6 tok/s (steps/s 114.2 / 459.8 / 759.8) | c1 / c8 245.4 / 750.2 (steps/s 104.8 / 312.3) |
| Prefill 8K / 32K | 10,180 / 11,701 tok/s | 11,583 / 12,275 tok/s |
| Server log and kernel log scan, KV unchanged at the end | PASS | PASS |
| Highest GPU temperature | 76 C | 71 C |

**Not run yet on these packages:** request checks (13 requests), tool calls at TP3, corruption check at TP3, image
latency at TP2, hotel-lights (TP3 at 32 streams, TP2 at 8), KV stress at full concurrency (TP3 32, TP2 8),
decode-vs-prefill consistency, eval-43, tool-eval-bench, GPQA Diamond, 128K decode, long retrieval 128K / 900K, long
generation, the image question sets, a same-session speed A/B. The 20261112 folder holds those results for the
20261112 packages ([its Quality section](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#quality)); there TP2
hotel-lights was 26/30 (2 unparseable answers, 1 truncated, 1 wrong) and the KV stress passed at TP3 (64/64) and TP2
(16/16).

## Measured and not adopted

| Idea | Result |
| --- | --- |
| `FLARE_GLM53_AR_ROWS=1024` at TP3 (prefill all-reduce in 1,024-row parts, smaller one-shot channel) | about +191K KV expected from the export memory map; not exported, a prefill A/B is owed; in the code, off by default |
| Token embedding in host memory at TP3 | +572K KV, but the embedding is read every decode step; kept to TP2 |
| Prefill merge of short message pieces (DeepSeek 20261116) on GLM | not measured beyond one pass; off |

The 20261112 list (a4 activations, CSF-QAD checkpoint, DCP1, FP32 DCP partials, draft diet, resident scales, 768
columns, 3 drafts, alpha 0.6, verification budgets, upstream vision dataflow) still applies:
[20261112](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#measured-and-not-adopted).

## Known limitations

- **Qualification not complete:** see [Quality](#quality).
- **Most checks ran on an earlier binary** (`67c691d29041`) than the release binary; the release binary was checked
  with one boot per package ([Quality](#quality)).
- **Hardware-specific packages:** 3 × or 2 × 96 GB SM120 (RTX PRO 6000) only. Other setups need a fresh export.
- **Thinking is always on.** GLM-5.3-Flash's chat template has no switch: `enable_thinking: false` is ignored without
  an error. `reasoning_effort` `low` or `high` selects that effort; any other value gives `max`.
- **Sampling:** temperature and top_p are clamped to 0.95-1.0 (defaults 1.0 and 0.95), the recipe's ranges.
- **Request handling differs from the DeepSeek build** (unchanged in this release): strict tool schemas and invalid
  JSON in history tool arguments get HTTP 400; a `required` or named `tool_choice` is checked after generation (an HTTP
  500 is possible); developer messages merge into the leading system message, so there is no prefix-cache reuse across
  them.
- **Not bit-identical run to run**, and two exports of the same commit can differ in the last bits (kernel tuning
  chooses per export). Our TP2 export timed the MoE expert tuning races one GPU at a time (a local b12x patch, a
  power-supply precaution on the test machine, not shipped); the TP3 export did not.
- **DCP3 rounding**, **no plain decode graphs**, **images queue for a slot**, **one model name for both packages**:
  as in [20261112](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md#known-limitations).
- **Rare `CUDA address collision` at start:** a restart helps.
- Not upstreamable as written: Hyperflare's contribution rules exclude model-specific changes of this kind and
  AI-authored GPU-execution changes. This is a personal fork of Hyperflare.

## Coming from 20261112

- **Rebuild everything:** new fork tag `hyperflare-glm-20261116` on base `70069281` (it does not descend from
  `hyperflare-glm-20261112`), new venv (FlashInfer `aa6c4221`; `flashinfer-tiny-decode-swiglu.patch` and
  `flashinfer-dense-splitk-deferral.patch` are gone, `flashinfer-tuning-scratch-preference.patch` is new), new overlay,
  new `flare` binary, new exports. Serving the 20261112 packages on the new binary was not tested.
- The new serve scripts pass `--scheduler-ttft-order` and set `FLARE_CUDA_STACK_LIMIT`; an older `flare` binary
  rejects them.
- The same `flare` binary serves the DeepSeek V4.1 20261116 package.

## Going back to 20261112

Use the [20261112 folder](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md) with a checkout of tag
`hyperflare-glm-20261112` (its own venv, binary and packages). Mixing packages and binaries of the two releases was
not tested.

## Building the packages

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
hf download nvidia/GLM-5.3-Flash-FP4-Local --revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 --local-dir ~/glm53-fp4-local
git clone https://github.com/AzallazA/spark-local-inference.git flare && cd flare && git checkout hyperflare-glm-20261116
uv sync --locked --extra prepare --extra test && cargo build --locked --release --manifest-path flare-cli/Cargo.toml
# apply tp3-patches/*.patch to the venv's FlashInfer (EXPORT.md step 2), then:
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp3.sh     # TP3 with images
CKPT=~/glm53-fp4-local /path/to/this-folder/export-glm-tp2.sh     # TP2 with images
```

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the packages yourself |
| `MEASUREMENTS.md` | every number in this README with its run |
| `export-glm-tp3.sh` / `export-glm-tp2.sh` | export the TP3 / TP2 package (run from the fork checkout); `VISION=0` text-only, `RECLAIM=0` 20261112 layout. Copies of the tag's `tp3-tools/release/glm/` |
| `serve-glm-tp3.sh` / `serve-glm-tp2.sh` | serve them (port 15015, model `GLM-5.3-Flash`; `SCHEDULER`, `TTFT_ORDER`, `STACK_LIMIT`, `ALPHA`, `BUDGET`, `PREFIX_CACHE`, `PACKAGE`, `GPUS` overrides). Copies of the tag's |
| `versions.txt` | exact versions behind the numbers |
| `tp3-patches/` | the FlashInfer patches for the venv, and `overlay/` (world-3 pull gather, applied to a separate b12x copy by the export scripts) |
| `tools/` | corruption check, tool-call test, first-token and agent replay probe, steps/s helper, image checks (unchanged from 20261112) |
| `source.patch` | the fork's changes against Hyperflare `70069281`: `git diff --binary 70069281 hyperflare-glm-20261116` (393 files, 184 new). Applies cleanly to a `70069281` checkout |
| `qualification/` | the GLM qualification kit of 20261112 (unchanged) and `RESULTS.md` with this release's check results in `results/` |
| `licenses/` | Hyperflare and FlashInfer license and notice files |

## Licenses

Weights: NVIDIA Open Model Agreement for `nvidia/GLM-5.3-Flash-FP4-Local`; the base model GLM-5.3-Flash is MIT
(see the checkpoint's model card). Hyperflare (formerly Flare): Apache-2.0 (`licenses/LICENSE`, `licenses/NOTICE` and
`licenses/THIRD_PARTY_NOTICES.md` kept; modified files marked). FlashInfer, which `tp3-patches/` modifies:
Apache-2.0 (`licenses/FlashInfer-LICENSE`, `licenses/FlashInfer-NOTICE`). `llm-inference-bench` is MIT (its README).
NVIDIA's CUDA Python wheels are used only for exporting and are not redistributed.

## Credits

NVIDIA Hyperflare (formerly Flare) and FlashInfer b12x, including the upstream GLM-5.3-Flash recipe; Z.ai for
GLM-5.3-Flash; NVIDIA for the FP4-Local checkpoint; Local Inference Lab for `llm-inference-bench` and the FlashInfer
b12x fork.
