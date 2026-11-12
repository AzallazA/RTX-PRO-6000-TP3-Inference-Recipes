# DeepSeek-V4.1-Flash-1017 Kraken TP3 — 20261019.12

DeepSeek-V4.1-Flash-1017 served at **tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000
Blackwell cards, from DeepSeek's original weights (FP8 dense layers, MXFP4 routed experts; nothing requantized) or,
recommended, the same weights with NVIDIA's 4-bit Engram tables so both tables fit in host RAM.

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT); Engram tables from
[nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (see
[README-engram-nvfp4.md](README-engram-nvfp4.md)). Runtime: Local Inference Lab's vLLM "karmic-kraken" image
(`karmic-kraken-beta-20261016.2`, vLLM `6d66eed35`) with FlashInfer's b12x kernels (Local Inference Lab fork @ `ea74ea61`)
and the TP3 changes in `src/` and the `patch-*/` folders.

Image: **`azallaza/ds41-kraken-tp3:20261019.12`** (`sha256:8d19fb462d4957ca78a2d29ea6dadd1c8218dd98296f2008fa196460e6f3ff47`); what changed is in
[New in .12](#new-in-12). Every tag since `20261019` shares all but a few layers with the previous one, so pulling over an
earlier tag downloads little. Older tags stay valid: `.11` (`sha256:c0d61091…`: b12x `5b065abe`, BF16 draft-logit cache, not
run-to-run deterministic; see [Coming from .11](#coming-from-11) to roll back), `.10` (BRRR's sorted indexer top-k on: 6-15% slower decode beyond 128K), `.9` (b12x attention, no BRRR sparse decode), `.8` (fused decoder with its own dense-weight copies), `.7` (no fused decoder layer), `.6` (no vocab-split sampling), `.5` (no BRRR kernels), `.4` (no cap-only adaptive verification), `.3` (stock Engram
tables only). The development history with per-step measurements is kept in [README-history.md](README-history.md).

```bash
docker pull azallaza/ds41-kraken-tp3:20261019.12
# one-time: hard-link NVIDIA's Engram shards next to the stock weights (both repos in the same HF cache)
python3 patch-nvfp4/make_engram_nvidia_variant.py --cache "$HF_CACHE_DIR"
# the tested setup: the engram-nvidia variant with both 4-bit Engram tables in host RAM
HF_CACHE_DIR=/path/to/huggingface MODEL=local/DeepSeek-V4.1-Flash-1017-engram-nvidia ENGRAM_RAM_LAYERS=1,14 \
  docker compose up -d      # port 15015, model name DeepSeek-V4.1-Flash; the first .12 start compiles its b12x kernels
docker logs -f ds41-kraken-tp3 2>&1 | grep -m1 'expert promotion armed and warmed'   # then send traffic
```

Host settings: 450 W/GPU for the `.12` measurements (the earlier-tag tables below are 350 W; 450 W raises prefill ~10%,
decode about level), +6000 memory offset, 256 GB host RAM, the checkpoint on a local NVMe drive.

## Contents

0. [New in .12](#new-in-12) · [Coming from .11](#coming-from-11)
1. [What it is](#what-it-is) · [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
2. [What is in this image](#what-is-in-this-image)
3. [Measurements](#measurements-on-this-image) · [Quality and stress](#quality-and-stress)
4. [Presets and switches](#presets-and-switches) · [Alternative weights](#alternative-weights)
5. [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations) · [Not tested](#not-tested)
6. [Files](#files) · [Variables](#variables) · [compose.yaml](#composeyaml) · [Credits](#credits)

## New in .12

`.12` = `.11` plus the changes below, each measured as it landed (A/B against the image before it) and then qualified
together against `.11` in one session ([Measurements](#12-against-11-both-with-engram-in-ram), [Quality](#qualification-of-12)).
Build steps: `patch-12/image/Dockerfile.12`, `Dockerfile.12-fp32`, `Dockerfile.12-det`.

| Change | Effect | Switch |
| --- | --- | --- |
| FlashInfer b12x `5b065abe` → `ea74ea61` (Local Inference Lab fork), vLLM port to its blockscaled-linear API, DS V4.1 attention with b12x's fused MXFP4 indexer (`patch-12/b12x-bump/`) | vs the same image without it: c1 +1.2%, c4 +7.6%, c8 +4.9%, prefill +2.5%, 512K c1 +4%; eval 43/43 | own cache volume (`runtime-cache-12`); `BRRR_RADIX_TOPK=0` |
| tiny_decode SwiGLU guard (re-applied on the new b12x) | 1-4 token MoE decode calls never use b12x's `tiny_decode` kernel, which has no SwiGLU clamp (DS V4.1 declares limit 10) | — |
| Vision staging (`patch-12/vision-staging/`) | the host-resident vision/aligner weights are staged through borrowed VRAM per image: first image 2.10 → 0.34 s, 6 in parallel 6.95 → 0.78 s; no VRAM held, text decode level | `DS41_VISION_STAGE=0` |
| Adaptive-verification skip (`patch-12/av-skip/`) | no per-step confidence copy + TP broadcast while the verify rows fit under `DS41_VERIFY_ROW_CAP`: c1 +0.6% steps/s, outputs identical | `DS41_AV_SKIP_UNDER_CAP=0` |
| Tool-call fixes: vLLM #62199, #62517, #62729, #64971 ported to the V4.1 frontend (`patch-12/tool-calls/`) | the `<｜DSML｜ tool_calls>` opener no longer leaks into content/reasoning; a literal image token in text returns HTTP 400; `add_generation_prompt` / `continue_final_message` honoured; `/v1/messages` sends one tool message per `tool_result` (results stay aligned after parallel calls); tools 60/60 | — |
| KV-aware prefill admission (`patch-12/kv-admission/`) | `max-parallel-prefills: auto` could deadlock when 8 unique ~600K-token prompts all started prefilling and filled the KV pool; admission now leaves room for what the prefills in flight still need, and a progress guard stops the pool filling with partial prefills: stress 16/16, 184/184 pings, 0 preemptions; agent traffic level | `DS41_KV_ADMISSION=0` (image ENV) |
| Lazy Engram host tables (`patch-12/lazy-engram/`) | the RAM tables are created after expert promotion and filled on the first real batch (6.0-6.7 s per rank, one rank at a time): boot low point 18.7 → 97.9 GB available; decode level | `ENGRAM_RAM_LAYERS` |
| BRRR draft-logit cache in FP32 (`patch-12/draft-fp32/`; recipes issue #2, found by [@epheien](https://github.com/epheien)) | the draft kernel sampled from FP32 logits but cached them BF16-rounded, so verification rebuilt a slightly different proposal distribution. Reporter's GPU probe (65,536 requests, standard rejection): token 1 target 0.100, observed 0.0855 with the BF16 cache, 0.0999 with FP32. Decode level | — |
| Deterministic serving (`patch-12/determinism/`) | identical requests in identical batches now give identical output: first-token logprob spread 0.000 and 8/8 identical 128-token decodes (before: top-1 logprob spread 0.1-1.5 run to run, the top token occasionally flipped); c1 -1.5%, c4 level, c8 -2%; 240 MiB/GPU less promotion | `DS41_DETERMINISTIC=0` |
| `oom_score_adj: -600` in `compose.yaml`, `B12X_COMPILE_WORKERS=2` | the host OOM killer picks other jobs before a vLLM worker; the cold first start no longer OOM-kills a worker while 24 compilers run next to ~150 GB of pinned tables | — |

**Determinism, in detail.** Two sources made repeated identical requests diverge: b12x's MoE combined each token's
top-k expert outputs with BF16 atomics (prefill and the draft model), and BRRR's fused decoder summed partial results
with FP32 atomics, so the summation order changed from run to run. `.12` makes the target model's MoE use b12x's
deterministic top-k output (one fixed-order FP32 reduction) and has the fused decoder accumulate into paired slots in a
fixed order for launches up to `DS41_BRRR_DET_ROWS=64` rows. It is deterministic for a given batch, **not
batch-invariant**: the output can still depend on what else is in the step (launches above 64 rows, i.e. mixed steps with
a long prefill, keep the old order). Re-checked on the published image (`sha256:8d19fb46…`): first-token spread 0.000000 and identical top-5 on 3 prompts × 8 repeats; 8/8 identical 128-token decodes on two prompts (max logprob difference 0.000000).

**SQL benchmark (recipes issue #1).** A user ([@epheien](https://github.com/epheien)) reported a lower pass rate on the
browser SQL benchmark at TP3 (`.11`, ~19/25) than on upstream's TP4 build (~23.5/25). Isolation on our side ruled out the
sampler, rejection mode, speculative decoding, Engram precision and prefix caching (all within run-to-run noise). The
reporter then found the two setups had loaded different checkpoints: with the **initial** DeepSeek-V4.1-Flash release,
TP3 on our image scores in the TP4 range (119/125 over 5 rounds), and TP4 with **1017** drops to 98/125. The gap is the
1017 checkpoint on this benchmark (a later checking query replaces the graded answer, or the tool loop hits its limit),
not the TP3 port. Our cross-check on `.12` (deterministic, same settings, 5 rounds each): the **initial** checkpoint scores
**23.4/25**, the 1017 checkpoint 18.2-19.0/25.

## Coming from .11

- Pull `.12` and recreate. Its b12x kernels compile into a new volume (`runtime-cache-12`) on the first start, which
  takes longer; later starts are warm. `B12X_COMPILE_WORKERS=2` (in `compose.yaml`) keeps that compile from running the
  host out of memory.
- **Block verification is now actually on.** Both images (`.11` and `.12`) have `rejection_sample_method: standard` built
  in; every measurement and qualification in this folder, for `.11` and `.12` alike, ran with a profile that switches it
  to `block`. The published `.11` compose did not mount that profile, so `.11` installs served **standard** rejection
  although this README described block. `compose.yaml` now mounts `ds41-flash.profile.yaml` (block) and
  `image-contract.json` (the image's contract updated for that file), so you get the tested setup. To keep standard
  rejection, remove those two lines (on the SQL benchmark the two were within noise: 19.4 vs 19.0/25).
- `ENGRAM_RAM_LAYERS` now selects `.12`'s lazily created tables (`1` = stock weights, the compose default; `1,14` =
  both 4-bit tables, the tested setup; empty = both on disk). `.11`'s at-boot tables moved to `ENGRAM_EAGER_RAM_LAYERS`:
  leave it empty on `.12` (setting it brings back the ~19 GB boot low point), set it when rolling back to `.11`.
- `BRRR_RADIX_TOPK` defaults to `0` (b12x's fused indexer top-k on the new b12x); `.11` used `blocks`.
- Output is run-to-run deterministic by default; `DS41_DETERMINISTIC=0` gives back ~1.5-2% decode.
- Roll back with `DS41_IMAGE=azallaza/ds41-kraken-tp3:20261019.11 DS41_CACHE_VOLUME=runtime-cache BRRR_RADIX_TOPK=blocks
  ENGRAM_EAGER_RAM_LAYERS=1,14` (`1` with the stock weights). Without the last one `.11` reads both Engram tables from disk
  (about 7.5% slower c1 decode, measured).

## What it is

| Layer | Source |
| --- | --- |
| Base image | `ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1…` (karmic-kraken-beta-20261016.2, vLLM `6d66eed35`, CUDA 13.4, PyTorch 2.14) |
| b12x kernels | FlashInfer b12x, Local Inference Lab fork @ `ea74ea61` (`.11`: `main` @ `5b065abe`): compact-scale MoE decode/prefill, inline CSF scale decoding, fused MXFP4 indexer, deterministic top-k output. Python only, compiled at startup |
| TP3 changes | 16 vLLM files in `src/` (`source.patch` against the base image); 8 more for the 4-bit Engram / NVFP4 support (`patch-nvfp4/`), 1 for cap-only adaptive verification (`patch-rowcap/`), 3 + the `brrr` package for the BRRR kernels (`patch-brrr/`), 2 + the updated `brrr` package for the fused decoder layer (`patch-fused/`), and the `.12` changes in `patch-12/` |
| Weights | the original `deepseek-ai/DeepSeek-V4.1-Flash-1017`; by default the `local/…-engram-nvidia` variant = the same shards with NVIDIA's two 4-bit Engram shards hard-linked in |

The base image already pads DS4.1's 64 attention heads to 72 (8 → 9 output groups) at TP3. What does not fit is
memory: the original checkpoint needs ~95 GiB per GPU before KV, caches and activations.

## How it fits in 3 × 96 GB

**Expert layers parked in host RAM, hottest experts promoted back to GPU.** Routed-expert layers 20-33 (of 40) start
in host-backed CUDA virtual memory (HOST_NUMA pages behind a device address range), so startup fits. Once the server is
healthy, `ds41-serve` arms promotion and sends a warm-up request; on that pass each parked layer remaps 16-expert chunks
of its weights to GPU memory **at the same virtual address** (captured CUDA graphs stay valid) until 1.5 GiB per GPU
remains free. With the default KV pool that is 19 of 24 chunks of the gate/up weights and 20 of 24 of the down weights.

**Hot-first expert order from real traffic.** At load, each parked layer's experts are relabeled most-routed-first
(router rows and expert weights together, so outputs are unchanged). The order (`expert-order-blend/`) was counted on
~650K routed tokens, 0.6 coding-agent traffic + 0.4 prose, so the promoted prefix holds the experts both workloads use:
~98.5% of routes hit a promoted expert and the parked layers run their MoE kernel at resident speed (~100 µs vs ~95).
A code-only order was fast on code and collapsed on prose; the sample must cover every workload served
([README-tuning.md](README-tuning.md) has the procedure to count your own).

**Compressed expert scales.** All routed-expert scales are compressed losslessly at load with FlashInfer's per-row
CSF encoder and decoded inline by the MoE kernels, which is what leaves room for promotion.

**Engram.** Two n-gram tables (layers 1 and 14). With the stock weights they are 94 GiB each: layer 1 pinned in host
RAM, layer 14 read from NVMe behind a 2 GiB-per-GPU host row cache. With NVIDIA's MXFP4 tables (16.2 GiB per rank
each) both sit in host RAM (`ENGRAM_RAM_LAYERS=1,14`): +6% decode and no disk reads while serving. The lookup kernel
decodes the packed 4-bit rows directly (`patch-nvfp4/`). Since `.12` the RAM tables are created after expert promotion and
filled on the first real batch, so a boot no longer dips to ~18 GB of available host memory.

**Decode.** DSpark speculative decoding (the checkpoint's 3 `mtp.*` layers), 5 drafts, block rejection sampling (the lil
profile `compose.yaml` mounts), draft logits cached in FP32 for verification, run-to-run deterministic reductions.
Verification is **cap-only adaptive**: every draft is verified while the decoding requests' target rows fit under
`DS41_VERIFY_ROW_CAP=24` (up to 4 requests), and above that drafts are trimmed best-confidence first across requests,
which is what keeps c8 fast without costing a lone user anything. Draft sampling and verification are **split by
vocabulary shard** across the three GPUs, so only a few words per step cross PCIe instead of the logits.

**Fused decoder layer.** For decode batches (up to 128 rows) every layer runs as one persistent kernel from BRRR:
attention output projection, both TP all-reduces over PCIe, the hyper-connection mixes, router, shared expert and the
routed experts, with the next layer's attention prologue fused in for batches of up to 8 rows. The TP3 port reads the
expert scales straight from FlashInfer's compressed (CSF) payload inside the kernel, and its dense weights (attention
output, router, shared expert) straight from vLLM's own tensors, so the fused layers keep CSF and the parking layout
above and cost no extra VRAM or launches.

**Sparse decode attention.** Decode attention (the 192-key window plus 512 indexer-selected keys on 38 of 40 layers)
runs BRRR's kernel, ported to 24 heads per GPU: two 16-head tensor-core tiles, the keys split over up to 12 blocks per
request row (64 keys each, which fits sm_120's 99 KB shared memory), and the per-block softmax results merged by
whichever block finishes last, so no thread-block cluster is needed. Matches b12x's attention to bf16 rounding
(cosine 0.999996 on live inputs). `DS41_SPARSE_DECODE=0` returns to b12x.

**Collectives.** b12x PCIe one-shot all-reduce for the small decode all-reduces; BRRR's phased PCIe all-reduce /
all-gather (rebuilt for 3 ranks) for the 16 KB..1 MB messages that used to fall back to NCCL ring.

## What is in this image

Everything below is built in; `compose.yaml` only mounts the launcher settings and the expert order. Effects are
`llm-inference-bench` 32K c1 unless noted, measured step by step as each change landed (details per step in
[README-history.md](README-history.md)).

| Change | Effect |
| --- | --- |
| TP3 port (padded heads/vocab/sinks, in-place VMM expert promotion, CSF scales) | the model runs at TP3 with 1M context; 226 tok/s short-prompt decode on `20261019` |
| `max-parallel-prefills: auto` | a short request behind a 420K-token prefill: TTFT 34 s → 0.6 s |
| NVFP4 drafter head, top-p 1.0 server default | model-card sampling; drafter head VRAM |
| Parked layers 20-25 → 20-33 with a 1.5 GiB reserve | +6% decode (more experts promoted per layer), KV-stress clean |
| Expert order counted on real traffic (blend) | +10% decode: 72 → 79.5 steps/s |
| NVIDIA 4-bit Engram tables, both in RAM | +6% decode: 79.5 → 84 steps/s; host RAM 150 → ~97 GB free |
| `max-num-seqs` 4 → 8 (graphs 48) | c8 611 tok/s aggregate; c8 KV stress PASS |
| Cap-only adaptive verification (`DS41_VERIFY_ROW_CAP=24`) | c6 512 → 635, c8 611 → 752 tok/s aggregate; c1/c4 unchanged |
| BRRR skinny FP8 GEMV + radix indexer top-k | 83 → 85.3 steps/s, coding mean 336 → 354 tok/s |
| BRRR phased PCIe all-reduce (3 ranks) | 85.3 → 86.5 steps/s, coding mean 365 tok/s |
| BRRR vocab-split draft sampling + verification (TP3 padding-aware) | 87.5 steps/s, 238 tok/s, c4 475 / c8 776 aggregate |
| BRRR fused decoder layer on all 40 layers, in-kernel CSF scale reader (`.8`) | 87.5 → 95 steps/s; coding mean 381-388 → 393-404; c8 776 → 779-788 |
| BRRR's sorted top-512 radix select for the indexer turned off (`.11`, `BRRR_RADIX_TOPK=blocks`) | c1 at 128K 90.9 → 95.6-96.8 steps/s, at 512K 77.4 → 89.6-90.6 (+6% / +15%); 32K unchanged |
| BRRR sparse decode attention at 24 heads (`.10`) | c1 95.7-97.1 → 98.0-100.5 steps/s (+3-4% on every preset); c8 level |
| Fused kernel reads vLLM's dense weights instead of repacked copies (`.9`) | ~1.6 GiB/GPU back to promotion (2x: 18+18 → 19+20 chunks); c8 779-788 → 801-818, c8 +3-10% on every preset |
| BRRR renderer fix | tool-result-only messages no longer start a new turn (reasoning kept inside agent tool loops); parallel tool results ordered by call |
| NVFP4 expert support (`nvidia/…-NVFP4` as a drop-in) | works, 73.8 steps/s on this stack (ceiling: scale layout + W4A16); kept as an option |
| `.12`: b12x `ea74ea61`, vision staging, AV skip, tool-call fixes, KV-aware prefill admission, lazy Engram, FP32 draft cache, deterministic serving | see [New in .12](#new-in-12): b12x c4 +7.6% / c8 +4.9%, determinism -1.5-2%, images ~6x faster, boot host-RAM low point 19 → 98 GB |

## Measurements on this image

### .12 speed

Average of several runs, `llm-inference-bench`, 450 W/GPU, 2x preset, engram-nvidia weights with both Engram tables in
host RAM (`ENGRAM_RAM_LAYERS=1,14`), deterministic serving on (the default):

| | `.12` |
| --- | ---: |
| Decode 32K, c1 / c4 / c8 (tok/s) | **290 / 604 / 988** |
| Decode 128K, c1 | **279** |
| Decode 512K, c1 | **271** |
| Prefill 32K / 128K (tok/s) | **16,593 / 15,224** |
| First image / 6 images in parallel | 0.34 s / 0.78 s (`.11`: 2.10 s / 6.95 s) |

### .12 against .11, both with Engram in RAM

Same cards, settings and weights, 450 W, 2026-11-11; `.11` with `ENGRAM_RAM_LAYERS=1,14` (its at-boot tables, as it was
published), 2 runs; `.12` from its qualification run, 2 rounds. Decode 32K, 60 s per cell; engine steps/s in brackets.
(The qualification's own `.11` rounds read the Engram tables from disk and are not used here.)

| | `.11` | `.12` | change (steps/s) |
| --- | ---: | ---: | --- |
| decode 32K c1, tok/s (steps/s) | 277 (106.0) | **291 (108.1)** | +2.0% |
| decode 32K c4 aggregate | 580 (210.1) | **583 (216.7)** | +3.1% |
| decode 32K c8 aggregate | 898 (388.0) | **979 (417.6)** | +7.6% |
| coding peak c1, mean / max tok/s | 446.5 / 480 (2 runs) | 451 / 476 (1 run) | about level |
| decode with natural stops (respect EOS), c1 / c4 / c8 | — | 282 / 555 / 930, 0 loops | |

Like-for-like steps (each against the image just before it, same session, same Engram placement):
`.11` → `.12` without the b12x update (vision staging, AV skip, tool fixes, KV admission, lazy Engram): decode c1/c4/c8
290 / 579 / 951 tok/s, level with `.11`; + b12x `ea74ea61`: c1 +1.2%, c4 +7.6%, c8 +4.9% (3 rounds), prefill +2.5%,
512K c1 +4%; + FP32 draft cache: level; + deterministic serving: c1 -1.5%, c4 level, c8 -2%. Images and boot host RAM
are the large changes; decode is +2% / +3% / +8% steps/s at c1 / c4 / c8 against `.11` with its tables in RAM.
KV is the same 2,185,581 tokens on both (fixed by the preset); `.12` keeps 6-7 more expert chunks per GPU in VRAM.

### Earlier tags

All at 32K context unless noted, image `.11` (2x) and `.10` (4x/6x/8x; same code at 32K), `local/…-engram-nvidia` weights, **350 W/GPU, +6000 memory offset**, with
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench):

```bash
python3 llm_decode_bench.py \
    --port 15015 \
    --model DeepSeek-V4.1-Flash \
    --concurrency 1,4,8 \
    --contexts 32k \
    --duration 30 \
    --coding-peak
```

steps/s is MTP-normalized engine steps; tok/s moves with the draft acceptance of each sampled run (temperature 1), steps/s
is the stable number. Each KV preset is one `docker compose` overlay (below); the KV comes out of expert promotion, so
more KV = fewer promoted experts. `2x` is two runs (ranges), `4x`/`6x`/`8x` one run each.

| preset (`./kv.sh`, `kv-<preset>.yaml`) = x 1M-token sessions | KV / GPU | capacity (tokens) | promoted w13 / w2 | c1 steps/s | c1 tok/s | c4 aggregate tok/s | c8 steps/s | c8 aggregate tok/s | coding mean / max tok/s | prefill 32K tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **2x (default)** | 2.25 GiB | 2,185,581 | 19/24 + 20/24 | **97.1-100.5** | **229-265** | 467-494 | 360.2-370.2 | **808-820** | 414-425 / 437-461 | 14,455-14,986 |
| 4x | 4.75 GiB | 4,614,187 | 17/24 + 18/24 | 93.8 | 254 | 460 | 344.2 | 751 | 365 / 380 | 14,437 |
| 6x | 6.5 GiB | 6,314,130 | 16/24 + 16/24 | 88.7 | 261 | 414 | 317.0 | 708 | 346 / 366 | 14,084 |
| 8x | 8.75 GiB | 8,499,917 | 14/24 + 15/24 | 82.5 | 232 | 378 | 289.9 | 632 | 312 / 339 | 13,631 |

Long context, 2x, `.11` (aggregate tok/s; c1 also steps/s). These rows exposed the regression `.11` fixes: with BRRR's
sorted indexer top-k on (`.6`-`.10`), c1 lost 6% at 128K and 15% at 512K.

| context | c1 tok/s (steps/s) | c4 | c8 |
| --- | ---: | ---: | ---: |
| 32K | 252-265 (97.1-99.7) | 467-492 | 812-814 |
| 128K | 273 (95.6-96.8) | 468 | 780 |
| 256K | 245 (92.5) | 435 | 749 |
| 512K | 227-244 (89.6-90.6) | 410-414 | — |

Against earlier tags on the same presets (350 W):

| preset | c1 steps/s `.7` / `.8` / `.9` / `.10` | c8 tok/s `.7` / `.8` / `.9` / `.10` | coding mean `.7` / `.8` / `.9` / `.10` |
| --- | --- | --- | --- |
| 2x | 87.4-87.9 / 94.8-95.0 / 95.7-97.1 / 98.0-100.5 | 771-782 / 779-788 / 801-818 / 808-820 | 381-388 / 393-396 / 398-406 / 414-423 |
| 4x | 84.4-85.8 / 90.8 / 91.3 / 93.8 | 713-742 / 713 / 771 / 751 | 339-342 / 353 / 366 / 365 |
| 6x | 80.4-81.2 / 85.3 / 86.2-86.8 / 88.7 | 693-704 / 645 / 690-705 / 708 | 316-328 / 321 / 337-339 / 346 |
| 8x | 74.8-75.0 / 79.2 / 79.1 / 82.5 | 595-598 / 587 / 622 / 632 | 280 / 278 / 310 / 312 |

(4x c8 on `.10`: 751 tok/s at the same 344 steps/s as `.9`; the tok/s gap is that run's draft acceptance.)

`.8` kept its own repacked copy of every layer's dense weights (~42 MiB/layer, ~1.6 GiB/GPU), which cost expert promotion
and showed at high concurrency (6x c8 645); `.9` reads vLLM's tensors directly (bit-identical output, ~5% slower per
fused layer in isolation, more than repaid by the promoted experts). `DS41_FUSED_DENSE_NATIVE=0` restores the `.8`
copies; `DS41_FUSED_LAYERS=0` the `.7` path; `DS41_SPARSE_DECODE=0` the `.9` attention.

Coding peak at higher concurrency: `--coding-peak` runs sequential cc1 only. The same prompt through
`--completion-stats --prompt "Write a Python script that implements the Sieve of Eratosthenes." --profile-concurrency 8
--profile-runs 40 --max-tokens 2000` gave 112.8 tok/s per request, ~800 tok/s aggregate on `.8` 2x (at cc1 the same
command reads 394.9 vs coding peak's mean 404.1).

Reading the presets: `6x` still promotes ~256 of 384 experts per parked layer, the set that carries most routes;
`8x` cuts into it (~224), so its cost per added 1M tokens is the steepest. Use `8x` when eight concurrent 1M-token
sessions are really needed; spare KV beyond the running sessions only feeds prefix caching.

## Quality and stress

### Qualification of .12

Run 2026-11-11 on the published image (`sha256:8d19fb46…`), 450 W, with the settings this folder's `compose.yaml` ships
(block rejection, 2x preset) and the engram-nvidia weights with both tables in RAM, at 8 concurrent where a test is concurrent. `.11` columns are the
same session unless marked (that session's `.11` read its Engram tables from disk, which affects speed only).

| Test | `.12` | `.11` |
| --- | --- | --- |
| Corruption check (`verify_fix`, 20 runs) | **0** true corruptions | — |
| 43 verifiable-answer questions, thinking off, temperature 0 | **42/43** (refusals 3/37) | 42/43 (2/37) |
| Tool calls, 60 at 4 concurrent | **60/60** | 60/60 |
| Tool-call port checks: streaming with thinking, non-streaming, parallel results on `/v1/messages`, `continue_final_message` | all pass; literal image token → HTTP 400 | — |
| LiveBench coding (release 2024-11-25, 32K budget) | **81.9** | `.10`: 81.5 / 81.5 / 82.5 |
| GPQA Diamond (198 questions, temperature 0, 32K token budget, 8 concurrent; `llm-inference-bench --test-profile gpqa-diamond`) | **89.4%** (177/198; Wilson 95% 84.3-93.0%), 8 hit the token budget before answering | — |
| tool-eval-bench (community leaderboard settings: hard mode, seed 42, temperature 0; `llm-inference-bench --tool-eval`) | **90/100**, 79/92 scenarios pass | — |
| hotel-lights, 30 runs at 8 concurrent | **30 exact** / 0 wrong / 0 truncated | 30/30 (c4) |
| lavd, 30 runs at 8 concurrent | **30 exact** / 0 wrong / 0 truncated | 30/30 |
| Retrieval of three markers at 128K / 900K; needle at 995K | pass / pass; pass | pass / pass (`.3`) |
| Long generation, 4 runs (16.5K-21K-token completions) | 4/4 clean, worst 8-gram repeat 0.0013 | 4/4, 0.0005 (`.3`) |
| Vision basic (60 images), right answer / strict format, T0 + T1 | **60 + 60** / 48 + 50 | 60 + 60 / 48 + 48 (release run) |
| Vision hard (60 images), right answer / strict format, T0 + T1 | **57 + 58** / 36 + 45 | 58 + 59 / 36 + 50 (release run) |
| Image stress: 4 × 10 photos of 6000×4000 | 4/4, min free VRAM 550 MiB, no OOM | 4/4 |
| Images during two ~470K-token prefills | 17/17, median 0.90 s; both long requests complete | — |
| c8 KV stress (below) | 16/16 long, 166/166 pings, 0 bad log lines, 0 restarts; **min free VRAM 193 MiB** (see below) | PASS (2026-11-03) |
| Run-to-run determinism (3 prompts × 8 first tokens, 2 × 8 decodes of 128 tokens) | **bit-identical**: first-token spread 0.000000, top-5 identical 8/8 (3 prompts); 1 distinct completion out of 8 on both decodes | spread 0.1-1.5, top token flips |

Vision: the prompts ask for a bare answer ("Reply with only the code."); the strict score counts an answer that comes
with an explanation ("Therefore, the answer is **cyan**.") as a miss. The right-answer column scores the answer itself.
Run to run the model switches between the two styles, which moves the strict score by several points at temperature 1.

### Earlier tags

Correctness set run on each shipped change (same protocol as the GLM Kraken TP3 releases: thinking on,
`reasoning_effort` max, temperature 1 unless noted):

| Test | Result on `.11` | Earlier tags |
| --- | --- | --- |
| 43 verifiable-answer questions, thinking off, temperature 0 (`compare_models.py`) | **42/43** (refusals 2/37) | 43/43 (`.10`), 42/43 (`.9`, bat-and-ball), 43/43 (`.8`), 41/43 (`.7`), 42/43 (`.4`-`.6`), 41-42 stock |
| LiveBench coding (release 2024-11-25, 32K budget) | not re-run (`.11` only changes the indexer top-k back to b12x's, the `.5` path); `.10`: 81.5 / 81.5 / 82.5, `.9` control 82.5 | 83.2 (`.8`, `.7`), 84.2 (`.6`), 81.9 (`.4`/`.5`), 81.3 stock Engram |
| hotel-lights, 30 runs at 4 concurrent, seed 103 | **30 exact** / 0 wrong / 0 truncated (mean 20,216 tokens) | 30/30 on every tag (`.7` mean 21,125) |
| lavd, 30 runs at 4 concurrent | — | 30 exact (`.3`) |
| Retrieval of three markers at 128K / 900K | — | pass / pass (`.3`) |
| Long generation after a 581K-token conversation, 4 seeds | — | 4/4 clean, worst 8-gram repeat 0.0005 (`.3`) |
| Vision basic / hard (60 + 60 images) | — | 60/60, 117/120 (`20261019`) |
| Coding task (`qualification/coding-task/`) | — | 18/18 tests passed (`.3`) |

Stress: 8 concurrent *unique* ~530K-token prompts in waves until >= 1.15x the KV pool has passed through (forces
eviction), with a short-prompt pinger throughout; min free VRAM sampled every 2 s; container log scanned for OOM /
mapping-failure / preemption / cuBLAS lines.

| Stress | Result |
| --- | --- |
| `.12`, 2x pool, c8 | **PASS on requests**: 16/16 long, 166/166 pings, 0 bad log lines, 0 restarts (17.2 min); min free VRAM **193 MiB** |
| `.11`, 2x pool, c8 | **PASS**: 16/16 long, 130/130 pings, 9.56M prompt tokens, min free VRAM 524 MiB, 0 bad log lines (26.5 min) |
| `.10` (sparse attention), 2x pool, c8 | PASS: 16/16 long, 137/137 pings, 9.56M prompt tokens, min free VRAM 490 MiB, 0 bad log lines (28 min) |
| `.9` (fused, vLLM dense weights), 2x pool, c8 | PASS: 16/16 long, 135/135 pings, 9.56M prompt tokens, min free VRAM 521 MiB, 0 bad log lines (27 min) |
| `.8` (fused decoder), 2x pool, c8 | PASS: 16/16 long, 231/231 pings, 9.56M prompt tokens, min free VRAM 511 MiB, 0 bad log lines (36 min) |
| `.7` (vocab-split), 2x pool, c8 | PASS: 16/16 long, 262/262 pings, 9.56M prompt tokens, min free VRAM 494 MiB, 0 bad log lines (39 min) |
| `.6` (BRRR kernels), 2x pool, c8 | PASS: 16/16 long, 136/136 pings, 9.56M prompt tokens, min free 543 MiB |
| `.5` (cap-only adaptive), 2x pool, c8 | PASS: 16/16, 127/127, 9.56M tokens, min free 593 MiB |
| `.4`, 2x / 4x / 6x / 8x pools, c4 | all PASS (min free 517-537 MiB; the 1.5 GiB reserve, not the KV size, governs stability) |

`.12` runs with less spare VRAM at the peak of this test (193 MiB against 490-593 MiB on earlier tags): it keeps 6-7
more expert chunks per GPU in VRAM, and the b12x update and deterministic reduction add small workspaces. Every request
and ping completed. If you hit an out-of-memory error under heavier load, raise the promotion reserve
(`RESERVE_BYTES=1879048192`, 1.75 GiB, instead of 1.5 GiB): a few experts stay in host memory and ~250 MiB more stays
free per GPU.
| 3-rank phased all-reduce extension (`patch-brrr/test_phased.py`) | 132 collectives bit/tolerance-equal to torch references |
| Sparse attention vs b12x on live inputs (`DS41_SPARSE_CHECK=1`, 3,135 calls, 1-48 rows, all three layer types) | cosine 0.999996, max relative difference 4-8e-3 (bf16 rounding) |
| Fused layer replay, vLLM dense weights vs `.8` copies (`patch-fused/replay_native.py`) | bit-identical outputs; memcheck 0 errors |
| Fused layer replay (`patch-fused/replay_csf.py`, real dumped inputs) | in-kernel CSF vs decoded scales equal within float-atomic noise (rel 7e-7); compute-sanitizer memcheck 0 errors |

LiveBench at temperature 1 moves by a question or two per run: across six runs on four tags the score ranged 81.5-83.2
(every miss a complete wrong answer, none truncated; most misses are the same hard questions on every tag).

Results and logs: `qualification/`.

## Presets and switches

**KV presets** (`./kv.sh 2x|4x|6x|8x`, or `docker compose -f compose.yaml -f kv-<preset>.yaml up -d --force-recreate`):
names are 1M-token sessions. `max-num-seqs` is 8, so `6x`/`8x` mean genuinely concurrent 1M-token sessions; spare KV
beyond the running sessions feeds prefix caching (a re-sent long conversation skips its prefill). Measured above.

**7-draft coding preset** (`docker compose -f compose.yaml -f k7.yaml up -d --force-recreate`): 7 DSpark drafts
with the row cap at 32. Higher coding peak (351 / 400 tok/s mean / max on `.5`) but slower on prose (c1 192 vs ~220,
c4 369 vs 460): every step does 8 rows of MoE routing instead of 6. For single-user coding sessions only.

**Adaptive verification**: `DS41_VERIFY_ROW_CAP=24` (default), `32` (5 requests untouched), `0` = upstream's
cost-model adaptive (trims a lone user to ~2 drafts: c1 -11%). `adaptive-verification: false` in `ds41-launch.yaml` =
every draft verified at any concurrency (c8 611 instead of 752).

**Fused decoder**: `DS41_FUSED_LAYERS=40` (default) fuses every layer; `N` fuses layers 0..N-1; `0` = the `.7` path
(no fused engine is built). `DS41_FUSED_DENSE_NATIVE=1` (default) reads vLLM's dense weights; `0` = `.8`'s copies.

**Indexer top-k**: `BRRR_RADIX_TOPK=0` (default since `.12`) = b12x's top-k (the new b12x fuses it into its MXFP4
indexer); `blocks` (`.11` default) keeps only BRRR's 2,048-block radix pick, `1` = `.10` (sorted top-512 radix on: slower
beyond 128K), `sorted` = that path alone.

**More than 8 sequences** (opt-in, short contexts only): `compose.seqs12.yaml` / `compose.seqs16.yaml` raise
`max-num-seqs` (CUDA graphs 72 / 96 rows) and turn on the load-scaled row cap (24 rows up to 8 decoding requests,
`DS41_VERIFY_ROW_CAP_HIGH=32` above). 32K aggregate: c12 ~973, c16 ~1,097 tok/s (8 sequences top out at ~810);
c1-c8 unchanged; no promotion cost. See Known limitations before using it with long prompts.

**`.12` switches**: `DS41_DETERMINISTIC=0` (previous arithmetic, ~1.5-2% faster, not reproducible run to run);
`DS41_BRRR_DET_ROWS` (largest fused-decoder launch summed in fixed order, 64); `DS41_VISION_STAGE=0` (vision weights read
from host memory per image, the `.11` path); `DS41_AV_SKIP_UNDER_CAP=0`; `DS41_KV_ADMISSION=0` (the `.11` scheduler).
The last three are image ENV: set them in an `environment:` block. Block rejection is the mounted lil profile: remove
the `ds41-flash.profile.yaml` and `image-contract.json` lines from `compose.yaml` for standard rejection.

**Sparse attention**: `DS41_SPARSE_DECODE=1` (default) = BRRR's decode attention; `0` = b12x's (= `.9`).
`DS41_SPARSE_CHECK=1` runs both on eager calls and logs the difference (debugging only).

**BRRR pieces**: `VLLM_DSPARK_SHARD_MARKOV_HEAD=0 VLLM_DSPARK_REPLICATE_MARKOV_W1=0` turns the vocab-split sampler off
(= `.6`); `BRRR_TP3_LOG=1` logs once per worker which kernels engaged.

**Expert order**: `B12X_EXPERT_ORDER_DIR=/opt/ds41/expert-order` = the image's baked order (an 18K-token sample,
~15% cold routes); `B12X_EXPERT_ROUTES_ALL=1` dumps route counts to build your own ([README-tuning.md](README-tuning.md)).

## Alternative weights

Any checkpoint that keeps the original FP8 layout (same tensor names, dtypes, shapes and index) is a drop-in via
`MODEL`. Verified on this recipe: the stock weights (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, set `ENGRAM_RAM_LAYERS=1`),
`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated` and `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` (both
third-party, not audited by us; lavd/hotel-lights 30/30, retrieval pass, 41-43/43 on the question set),
`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` (NVFP4 experts: 73.8 steps/s, eval 43/43), and the recommended
`local/DeepSeek-V4.1-Flash-1017-engram-nvidia` (stock weights + NVIDIA's Engram shards, made by
`patch-nvfp4/make_engram_nvidia_variant.py`). Details, download commands and the A/B tables:
[README-history.md](README-history.md) and [README-engram-nvfp4.md](README-engram-nvfp4.md).

## Measured and not adopted

| Change | Result |
| --- | --- |
| LMCache host-RAM prefix tier (`cache-mode: lmcache`, 48 or 16 GiB) | host thrash and OOM-killed workers: the engine-driven DS4.1 mode takes ~80 GB of host memory regardless of tier size, which the Engram tables already occupy |
| Fused attention prologue for batches above 8 rows (`DS41_FUSED_PROLOGUE_MAXM` 24 / 48) | neutral: c4 190-199 and c8 351-368 steps/s either way; b12x's separate GEMMs already match it there |
| Higher fixed row caps at 8 sequences (32 / 40 / 48) | c6 690 → 615 / 485 / 477, c8 808 → 787 / 680 / 616: 24 stays |
| 7 drafts on `.10` | c1 level, c4 420-425 vs 481-494, c8 746-756 vs 808-820: stays an opt-in coding preset |
| Decode context parallel (DCP3) | blocked in vLLM for V4.1; measured cost of its per-layer query gather + output merge over PCIe: 89 µs/layer at c1 with NCCL (+35% per step), best case ~8-10% at c1 and 20%+ at c8, no better than the 6x preset |
| Upstream adaptive verification (cost model, any cost scale) | c8 +19-21% but c1 -11..-28%: a 6-row verify step really costs ~25% more than a 3-row one here, so a lone user gets trimmed to 2 drafts |
| 7 drafts as the default | prose c1 192 vs 216-228, c4 369 vs 460 (coding peak +4-7%): kept as the `k7.yaml` preset |
| 6-aligned CUDA-graph capture sizes | neutral (padded rows are free) |
| EP all-reduce through the b12x PCIe path | neutral for decode (not on the steady decode path) |
| Host-CPU expert compute (KTransformers/Fiddler style) | ~10x DDR5-vs-GDDR7 bandwidth gap: one CPU expert costs a whole GPU MoE layer |
| Prefill chunks of 8192 / reserve 768 MiB / parking 20-29 / 4-6-7 drafts / indexer chunk 256 / MoE-only tuning | see history: OOM, no gain, or slower |

## Known limitations

- Promotion happens on the warm-up request after `/health`; requests before the `expert promotion armed and warmed`
  log line run with the parked layers in host memory (slower decode).
- Up to 8 concurrent sequences by default (`max-num-seqs: 8`, CUDA graphs up to 48 decode rows). The 12/16-sequence
  overlays are for short contexts: 16 unique ~265K-token prompts against the 2x pool livelocked the scheduler (12
  partly prefilled requests held 98.8% of the KV, none could finish). 8 sequences x 530K passes the same test.
- Time to first token is ~0.25 s even for short prompts (per-request host overhead of the eager prefill pass).
- Host RAM: ~150 GB in use while serving. With `.12`'s lazy tables a boot keeps ~98 GB available (`.11`: ~18 GB); the
  tables are filled on the first request (~6-7 s per GPU). Remove any leftover `/dev/shm/lmcache_*` pools before starting.
- Determinism is per batch, not batch-invariant: the same request can still produce different output when the rest of
  the batch differs (steps above 64 rows, e.g. a long prefill sharing the step, keep the old summation order).
- The first `.12` start compiles the new b12x kernels (longer first start; keep the host otherwise idle).
- Less VRAM headroom at peak than earlier tags: 193 MiB free in the c8 stress test (all requests completed). If you
  see out-of-memory errors, set `RESERVE_BYTES=1879048192` (1.75 GiB promotion reserve).

## Not tested

Benchmark accuracy scores beyond LiveBench coding, video inputs, other GPU power limits, more than 8 concurrent
requests (the configured limit), the stock (FP8) Engram tables on the BRRR tags (unchanged code paths). On `.12`
specifically: the 4x/6x/8x KV presets and the 12/16-sequence overlays (not re-measured), the stock weights with the lazy
layer-1 table (`ENGRAM_RAM_LAYERS=1`), the NVFP4 experts, standard rejection, and 350 W.

## Files

- `Dockerfile`: base image + FlashInfer `main` @ `5b065abe` + `src/`; `ds41-serve` (entrypoint); `ds41-launch.yaml`
  (lil launcher settings, mounted by `compose.yaml`: 8 slots, 4096-token chunks, graphs 48, 5 drafts, cap-only
  adaptive verification); `source.patch` (the 16 TP3 vLLM files); `build.log`, `image-inspect.json`.
- `patch-nvfp4/` (`.4`: 4-bit Engram lookup + NVFP4 experts, `make_engram_nvidia_variant.py`, converters, tests;
  write-up `README-engram-nvfp4.md`), `patch-rowcap/` (`.5`: cap-only adaptive verification),
  `patch-brrr/` (`.6`/`.7`: the BRRR port — `brrr/` package incl. kernels, `brrr.pth` hook, `vllm-patched/`,
  `stage1.patch`, `test_phased.py`, `image/Dockerfile`, `LICENSE-brrr`, `NOTICE-brrr`), `patch-fused/` (`.8`: the
  updated `brrr/` package with the TP3 fused decoder `kernels/decoder.cu` + `decoder.py`, `vllm-patched/` `nvidia/model.py`
  and `b12x.py`, `replay.py`/`replay_csf.py`/`replay_native.py` offline replay harnesses, `image/Dockerfile` (`.8`),
  `image/Dockerfile.9`, `image/Dockerfile.10`, `image/Dockerfile.11`; `vllm-patched/v1/worker/gpu/spec_decode/adaptive_verification.py` (`.11`, load-scaled cap); `vllm-patched/models/deepseek_v4_1/attention.py` (`.10`, sparse-check hook), license files).
- `patch-12/` (`.12`): `image/Dockerfile.12` (b12x update, vision staging, AV skip, tool calls, KV admission, lazy Engram),
  `image/Dockerfile.12-fp32`, `image/Dockerfile.12-det`; the files and patches each step copies (`b12x-bump/`,
  `vision-staging/`, `av-skip/`, `tool-calls/` with `PORT-NOTES.txt` and tests, `kv-admission/`, `lazy-engram/`,
  `draft-fp32/` with the reporter's `test_vp_cache_precision.py`, `determinism/`).
- `ds41-flash.profile.yaml` + `image-contract.json`: the lil profile with block rejection and the matching contract
  (mounted by `compose.yaml`).
- `compose.yaml`, `.env.example`, `kv.sh` + `kv-2x/4x/6x/8x.yaml`, `k7.yaml` + `ds41-launch-k7.yaml`,
  `compose.seqs12.yaml` / `compose.seqs16.yaml` + `ds41-launch-seqs12/16.yaml` (opt-in, short contexts),
  `expert-order-blend/` (the shipped order), `expert-order/` (the image's baked order), `compose.nvfp4.yaml` (dev overlay).
- `qualification/`: `run_qualification.py` and the GLM-derived checks, `vision/`, `coding-task/`, `kv-presets-stress/`
  (harness + per-preset results), `engram-nvidia/` (hotel-lights + eval on `.4`), `brrr-tp3/` (stress, eval, LiveBench
  answers, hotel-lights on `.6`/`.7`; `fused/` for `.8`: eval, LiveBench, hotel-lights, stress, speed and preset logs; `fused9/` for `.9`: eval, stress, preset logs; `sparse10/` for `.10`: eval, LiveBench x3 + `.9` control, hotel-lights, stress, presets, live check vs b12x; `topk11/` for `.11`: indexer top-k A/B at 32K-512K, row-cap and 12/16-sequence runs, c16 stress (livelock), eval, hotel-lights, c8 stress), `RESULTS.md`.
- `bench/`, `bench-20261019.3/`: the early prefill/needle/decode measurements and their scripts.
- `README-history.md`: the previous README with every update section; `README-tuning.md`: expert-order procedure.

## Variables

| Variable | Default | Values |
| --- | --- | --- |
| `HF_CACHE_DIR` | required | Hugging Face cache containing the weights |
| `MODEL` | `local/DeepSeek-V4.1-Flash-1017-engram-nvidia` | weights repo to serve (see [Alternative weights](#alternative-weights)) |
| `ENGRAM_RAM_LAYERS` | `1` | Engram layers held in pinned host RAM, created lazily after promotion (`.12`); `1,14` with the 4-bit tables (the tested setup), `1` with the stock 94 GiB tables; `1,14` with the stock tables needs ~189 GiB pinned host RAM (hosts with ~384 GB+, not tested); empty (`ENGRAM_RAM_LAYERS=`) = both tables read from disk |
| `KV_CACHE_MEMORY_BYTES` | `2415919104` (2x) | KV bytes per GPU; presets 2x/4x/6x/8x via `./kv.sh` or `kv-<preset>.yaml` |
| `DS41_VERIFY_ROW_CAP` | `24` | cap-only adaptive verification row ceiling; `0` = upstream cost-model adaptive |
| `VLLM_DSPARK_SHARD_MARKOV_HEAD` / `VLLM_DSPARK_REPLICATE_MARKOV_W1` | `1` / `1` | TP-sharded DSpark markov head for the vocab-split sampler; `0`/`0` = stock head, split sampler off (= `.6`) |
| `DS41_FUSED_LAYERS` | `40` | layers run by the fused decoder kernel; `0` = `.7` path |
| `BRRR_RADIX_TOPK` | `0` | indexer top-k: `0` (b12x, `.12`) / `blocks` (`.11`) / `1` (`.10`) / `sorted` |
| `DS41_DETERMINISTIC` / `DS41_BRRR_DET_ROWS` | `1` / `64` | run-to-run deterministic reductions; `0` = previous arithmetic |
| `B12X_COMPILE_WORKERS` | `2` | b12x kernel compilers per GPU worker (cold first start only; 8 OOM-killed a worker) |
| `DS41_CACHE_VOLUME` | `runtime-cache-12` | compile-cache volume; `runtime-cache` with the `.11` image |
| `DS41_VERIFY_ROW_CAP_HIGH` / `DS41_VERIFY_ROW_CAP_REQS` | `0` / `8` | row cap used above `REQS` decoding requests (`0` = off; the seqs12/16 overlays set 32) |
| `DS41_SPARSE_DECODE` | `1` | BRRR sparse decode attention; `0` = b12x attention (`.9`) |
| `DS41_FUSED_DENSE_NATIVE` | `1` | fused kernel reads vLLM's dense weights; `0` = per-layer copies (`.8`) |
| `BRRR_TP3_LOG` | `0` | `1` = log once per worker which BRRR kernels engaged |
| `B12X_EXPERT_ORDER_DIR` | `/opt/ds41/expert-order-blend` | expert promotion order; `/opt/ds41/expert-order` = baked order |
| `B12X_EXPERT_ROUTES_ALL` | `0` | `1` = dump route counts every 30 s to build your own order (~0.5-1% while on) |
| `DS41_OFFLOAD_FROM` / `DS41_OFFLOAD_TO` | `20` / `33` | parked expert layers; do not pass ~34 |
| `RESERVE_BYTES` | `1610612736` | GPU memory left free after promotion (1.5 GiB, stress-clean) |
| `ENGRAM_EAGER_RAM_LAYERS` | empty | `.11` rollback only: Engram tables created at boot in host RAM (`1,14` 4-bit, `1` stock); keep empty on `.12` |
| `ENGRAM_ROW_CACHE_ROWS` | `8388608` | host row cache for disk Engram tables; `0` = off |
| `DRAFT_NVFP4_HEAD` | `1` | DSpark drafter vocabulary head in NVFP4; `0` = BF16 |
| `DS41_LAUNCH_SETTINGS` | `/etc/ds41/launch.yaml` | lil launcher settings file (`k7.yaml` points it at `ds41-launch-k7.yaml`) |
| `DS41_IMAGE` | `azallaza/ds41-kraken-tp3:20261019.12` | image tag (`.11` … `.3` remain valid) |
| `HOST_PORT`, `GPU0`/`GPU1`/`GPU2` | `15015`, `0`/`1`/`2` | API port, GPU indices |

## compose.yaml

```yaml
# DeepSeek-V4.1-Flash-1017 TP3 release, image azallaza/ds41-kraken-tp3:20261019.12
# (karmic-kraken-beta-20261016.2 + FlashInfer b12x ea74ea61 + the TP3 port in src/, patch-nvfp4/, patch-brrr/,
#  patch-fused/ and patch-12/; .12 adds vision staging, tool-call fixes, KV-aware prefill admission, lazy Engram
#  host tables, the FP32 draft-logit cache and deterministic serving).
# Defaults: original deepseek-ai weights (MXFP4 experts, lossless), DSpark 5 drafts with
# cap-only adaptive verification (all 5 drafts verified up to 4 concurrent decodes), 8 concurrent sequences, 1M context, 4096-token prefill chunks, KV 2,185,581 tokens
# (2.08x 1M; see .env.example for the 4.4x / 6x presets), expert layers 20-33 parked in
# host RAM with their most-routed experts promoted to GPU after startup using the
# expert-order-blend/ order built from real traffic (2026-10-30), Engram layer 1 in host
# RAM and layer 14 read from the checkpoint (with the 4-bit Engram variants set
# ENGRAM_RAM_LAYERS=1,14 to hold both in RAM: faster, ~150 GB host). Needs 3x 96 GB GPUs
# and ~256 GB host RAM. Qualified at 450 W/GPU (350 W works; power mostly affects prefill).
services:
  ds41-kraken-tp3:
    image: "${DS41_IMAGE:-azallaza/ds41-kraken-tp3:20261019.12}"
    container_name: ds41-kraken-tp3
    # unless-stopped: auto-start on host reboot / daemon restart, but respect a manual `stop`.
    restart: "unless-stopped"
    stop_grace_period: 90s
    # Boot safety: the kernel OOM killer picks other host jobs (compilers, test runs) before a vLLM worker (~100 GB
    # RSS). No effect on serving.
    oom_score_adj: -600
    ipc: host
    # io_uring (disk Engram reads) needs seccomp unconfined; pinned host memory needs memlock.
    security_opt: [seccomp=unconfined]
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU0:-0}", "${GPU1:-1}", "${GPU2:-2}"]
              capabilities: [gpu]
    ports:
      - "${HOST_PORT:-15015}:15015"
    volumes:
      # Populated Hugging Face cache with deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10.
      # Engram layer 14 is read from these files during serving: keep them on an NVMe drive.
      - ${HF_CACHE_DIR:?Set HF_CACHE_DIR to your populated Hugging Face cache}:/root/.cache/huggingface:ro
      # b12x compile/tuning caches and the promotion arm file. .12 uses its own volume: its b12x (ea74ea61) kernels
      # differ from .11's, so the first start compiles them cold (longer first start; later starts are warm).
      # Rollback to .11: DS41_IMAGE=azallaza/ds41-kraken-tp3:20261019.11 DS41_CACHE_VOLUME=runtime-cache ENGRAM_EAGER_RAM_LAYERS=1,14.
      - ${DS41_CACHE_VOLUME:-runtime-cache-12}:/cache
      # Launcher settings from this release (max-num-seqs 8, graphs 48, 4096-token chunks, dspark k5) over the
      # copy baked into the image (which still says max-num-seqs 4).
      - ./ds41-launch.yaml:/etc/ds41/launch.yaml:ro
      # Expert promotion order built from real traffic (see B12X_EXPERT_ORDER_DIR below).
      - ./expert-order-blend:/opt/ds41/expert-order-blend:ro
      # DSpark block rejection sampling: the lil profile with rejection_sample_method: block (the image's copy says
      # standard) and the image contract with that file's hash (otherwise identical to the image's). This is the
      # configuration every release measurement and qualification used. Remove both lines for standard rejection.
      - ./ds41-flash.profile.yaml:/opt/lil/runtime/profiles/ds41-flash.yaml:ro
      - ./image-contract.json:/opt/lil/runtime/image-contract.json:ro
    environment:
      # Weights repo (from HF_CACHE_DIR). Any checkpoint in the original FP8 layout is a drop-in;
      # e.g. MODEL=mrfakename/DeepSeek-V4.1-Flash-1017-abliterated. See README "Alternative weights".
      MODEL: "${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
      # Fixed KV pool per GPU. 2.25 GiB = 2,185,581 tokens. More KV comes out of expert
      # promotion (fewer hot experts): 4.75 GiB = 4.61M tokens (~-5% decode), 6.5 GiB =
      # 6.31M tokens (~-10%), 8.75 GiB = 8.50M tokens (~-14%). Presets + measured numbers in .env.example (`./kv.sh`).
      KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-2415919104}"
      # Expert layers (ordinals FROM..TO) kept in host RAM at startup; after the server is
      # healthy, their most-routed experts move to GPU memory until RESERVE bytes remain free.
      # 20-33 (14 layers) measured best on llm-bench: 72.3 decode steps @32K vs 68.0 at
      # the old 20-25 (+6.3%). Widening spreads the fixed ~6 GiB of parking thinner per
      # layer (cold-fetch rate ~6.7%->0.5%); the win needs the 1.5 GiB reserve below (at
      # 1 GiB, 20-33 throws OOM-mapping retries under concurrent long-context). Do NOT go
      # past ~34 (20-34/36 OOM at load: cold chunks for too many layers exhaust host RAM),
      # and keep the draft layers 37-39 resident. cf=0.2 to wrap more layers under-promotes
      # and loses. See release README-tuning.md.
      DS41_OFFLOAD_FROM_ORDINAL: "${DS41_OFFLOAD_FROM:-20}"
      DS41_OFFLOAD_TO_ORDINAL: "${DS41_OFFLOAD_TO:-33}"
      # GPU memory left free after promotion: prefill scratch for 4096-token chunks AND
      # the serving cushion. 1.5 GiB pairs with 20-33 (stress-clean, min free ~480 MiB
      # under 4x600K concurrent); 1 GiB paired with the old 20-25. 768 MiB OOMs a 31K prompt.
      B12X_DEFERRED_BANK_RESERVE_BYTES: "${RESERVE_BYTES:-1610612736}"
      # Promotion order: per-layer expert route counts (expert-routes-rank0-layer<L>.json,
      # "counts" = 384 ints); the loader relabels each parked layer so its most-routed experts
      # fill the promoted prefix. expert-order-blend/ = 0.6 real coding traffic (544K routed
      # tokens) + 0.4 prose (109K), measured +10% decode steps vs the image's baked order
      # (which came from an 18K-token sample and left ~15% of routes cold). Point at
      # /opt/ds41/expert-order to use the baked order. README: "Update 2026-10-30".
      B12X_EXPERT_ORDER_DIR: "${B12X_EXPERT_ORDER_DIR:-/opt/ds41/expert-order-blend}"
      # 1 = count expert routes on every layer and dump /cache/all-expert-routes-rank<r>-layer<L>.json
      # every 30 s (cumulative) to build your own order. ~0.5-1% decode while on. See README.
      B12X_EXPERT_ROUTES_ALL: "${B12X_EXPERT_ROUTES_ALL:-0}"
      # Cap-only adaptive verification (image .5, ds41-launch.yaml adaptive-verification: true): every DSpark
      # draft is verified while the decoding requests' total target rows (1 + drafts each) fit under this cap;
      # above it drafts are trimmed, best-confidence first across requests. 24 = 4 requests untouched, c6 +24%,
      # c8 +23% aggregate, c1/c4 unchanged (2026-10-31). 0 = upstream's cost-model adaptive (c1 -11%);
      # to verify every draft at any concurrency set adaptive-verification: false in ds41-launch.yaml.
      DS41_VERIFY_ROW_CAP: "${DS41_VERIFY_ROW_CAP:-24}"
      DS41_VERIFY_ROW_CAP_HIGH: "${DS41_VERIFY_ROW_CAP_HIGH:-0}"
      DS41_VERIFY_ROW_CAP_REQS: "${DS41_VERIFY_ROW_CAP_REQS:-8}"
      # 1 = log once per worker which BRRR TP3 kernels engaged (skinny GEMM, radix top-k, phased all-reduce). Image .6.
      BRRR_TP3_LOG: "${BRRR_TP3_LOG:-0}"
      DS41_FUSED_LAYERS: "${DS41_FUSED_LAYERS:-40}"
      DS41_SPARSE_DECODE: "${DS41_SPARSE_DECODE:-1}"
      # Indexer top-k. .12 (b12x ea74ea61, fused MXFP4 indexer) uses b12x's top-k: 0. .11 used blocks.
      BRRR_RADIX_TOPK: "${BRRR_RADIX_TOPK:-0}"
      # Run-to-run deterministic serving (.12): the target model's MoE reduces in a fixed order (b12x deterministic
      # top-k output) and BRRR's fused decoder accumulates in paired slots for launches up to DS41_BRRR_DET_ROWS rows.
      # Identical requests in identical batches give identical output; c1 -1.5%, c8 -2% decode. 0 = previous arithmetic.
      DS41_DETERMINISTIC: "${DS41_DETERMINISTIC:-1}"
      DS41_BRRR_DET_ROWS: "${DS41_BRRR_DET_ROWS:-64}"
      # b12x kernel-compile processes per GPU worker. 8 (b12x default) x 3 ranks OOM-killed a worker during the cold
      # first start next to ~150 GB of pinned host tables; 2 compiles longer and fits. Only matters while compiling.
      B12X_COMPILE_WORKERS: "${B12X_COMPILE_WORKERS:-2}"
      # Image .7 bakes VLLM_DSPARK_SHARD_MARKOV_HEAD=1 / VLLM_DSPARK_REPLICATE_MARKOV_W1=1 (TP-sharded DSpark markov head,
      # needed by the vocab-split sampler). Set both to 0 here to get the stock replicated head; the split sampler then
      # disables itself (= image .6 behaviour, ~3-5% slower at c1).
      VLLM_DSPARK_SHARD_MARKOV_HEAD: "${VLLM_DSPARK_SHARD_MARKOV_HEAD:-1}"
      VLLM_DSPARK_REPLICATE_MARKOV_W1: "${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
      # Engram layers whose tables live in pinned host RAM. .12 creates them lazily: after expert promotion, filled
      # rank by rank on the first real batch (~6-8 s per rank), so the boot low point stays ~98 GB available instead
      # of ~19 GB. 1 = stock weights (94 GiB table, 31.5 GiB per GPU worker; layer 14 read from disk);
      # 1,14 = both 4-bit tables (the engram-nvidia / NVFP4 variants, 16.2 GiB per worker each: the tested setup).
      # 1,14 with the stock FP8 weights = both 94 GiB tables (~189 GiB pinned host RAM): only on hosts with ~384 GB+
      # RAM, not tested. Empty (ENGRAM_RAM_LAYERS=) = both tables read from disk (least RAM, slower decode).
      DS41_ENGRAM_LAZY_RAM_LAYERS: "${ENGRAM_RAM_LAYERS-1}"
      # At-boot RAM tables (.11 style): keep empty on .12 (lazy tables above); for a .11 rollback set
      # ENGRAM_EAGER_RAM_LAYERS=1,14 (4-bit tables) or 1 (stock), or .11 reads both tables from disk (~7.5% slower c1).
      DS41_ENGRAM_RAM_LAYERS: "${ENGRAM_EAGER_RAM_LAYERS-}"
      DS41_ENGRAM_LAZY_PARALLEL: "0"
      # Host RAM row cache for the disk Engram table, in rows (256 B each): 8388608 = 2 GiB
      # per GPU worker. 0 = read every row from the checkpoint.
      DS41_ENGRAM_ROW_CACHE_ROWS: "${ENGRAM_ROW_CACHE_ROWS:-8388608}"
      # DSpark drafter vocabulary head in NVFP4 (drafts only; every token is still verified by
      # the full model, so outputs are unchanged): +3-4% decode. 0 = BF16 head.
      VLLM_DS41_DRAFT_NVFP4_HEAD: "${DRAFT_NVFP4_HEAD:-1}"
      # Mount a different lil launcher settings file here to change context, drafts or chunking.
      DS41_LAUNCH_SETTINGS: "${DS41_LAUNCH_SETTINGS:-/etc/ds41/launch.yaml}"
      # Persist vLLM/Inductor/Triton compile caches to the runtime-cache volume too (default
      # is the container's ephemeral layer, recompiled on every `recreate`). The heavy b12x +
      # FlashInfer JIT caches already live under /cache; this covers the rest.
      VLLM_CACHE_ROOT: "${VLLM_CACHE_ROOT:-/cache/vllm}"
      TORCHINDUCTOR_CACHE_DIR: "${TORCHINDUCTOR_CACHE_DIR:-/cache/inductor}"
      TRITON_CACHE_DIR: "${TRITON_CACHE_DIR:-/cache/triton}"
      HF_HUB_OFFLINE: "1"
volumes:
  runtime-cache-12:
  runtime-cache:
```

## Credits

- **Model**: [DeepSeek](https://huggingface.co/deepseek-ai) — [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **Runtime, kernels and base image**: [Local Inference Lab](https://github.com/local-inference-lab) — the
  `karmic-kraken` vLLM fork and the [`ghcr.io/local-inference-lab/vllm`](https://github.com/local-inference-lab/blackwell-llm-docker)
  CUDA 13.4 runtime image this image is built from, the FlashInfer `b12x` attention/MoE/linear backends and DSpark
  speculative decoding, and [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) (the
  benchmark, hotel-lights and lavd profiles used throughout).
- **Decode kernels in `patch-brrr/` and `patch-fused/`**: [Fraser Price](https://huggingface.co/fraserprice) —
  [DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0; the
  skinny FP8 GEMV, radix top-k, phased PCIe all-reduce, vocab-split sampling, the fused decoder layer, sparse decode attention and the renderer
  tool-loop fix, written for TP4 and ported here to TP3 (`patch-brrr/`, `patch-fused/`); `LICENSE-brrr` and `NOTICE-brrr` are kept with the copies).
- **4-bit Engram tables and the NVFP4 checkpoint**: [NVIDIA](https://huggingface.co/nvidia) —
  [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (ModelOpt).
- **Adaptive verification** (the manager the cap-only patch builds on) and everything else under the hood:
  the [vLLM](https://github.com/vllm-project/vllm) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) projects.
- **Issue reports**: [@epheien](https://github.com/epheien) found the BF16-rounded draft-logit cache (recipes issue #2,
  with a CPU test and a GPU reproducer; the test ships in `patch-12/draft-fp32/`) and traced the SQL-benchmark gap of
  issue #1 to the checkpoint version.
- **Tool-call fixes in `.12`**: the authors of vLLM #62199, #62517, #62729 and #64971.
- **Long-context regression report**: a user testing on 3× RTX PRO 6000 Max-Q with real coding-agent traffic found
  that BRRR's sorted indexer top-k cost 13-15% beyond 256K, which led to the `.11` fix.
- **Quality checks**: [LiveBench](https://github.com/LiveBench/LiveBench) (coding, release 2024-11-25).
- Ideas reviewed while building this (not shipped in the image): [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)
  (sm_120 SASS kernels, FP4 KV) and NVIDIA's [Flare](https://github.com/NVIDIA/flare-local-inference) (MTP verification budgets).
