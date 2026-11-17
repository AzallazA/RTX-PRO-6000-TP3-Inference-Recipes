# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261116

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB
RTX PRO 6000 Blackwell cards. This build supersedes [Hyperflare-tp3-20261111](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md).
It is a new base commit, a new checkpoint view, a new export and a new `flare` binary (see [Coming from 20261111](#coming-from-20261111)).
Changes since then:

- **New upstream base.** The fork now sits on Hyperflare `70069281` (was `972e28e4`). Upstream's compiler-planned
  memory (one planned pool per compiled forward, one shared capture pool) replaced the fork's graph scratch. On the
  same settings that alone took the served KV cache from 3.19M to 5.02M tokens (2026-11-14, 8 slots, cap 24).
- **16 request slots** (was 8). Four export-time VRAM reclaim switches (`FLARE_DS41_VRAM_RECLAIM=1`) pay for most of the extra graphs.
- **KV cache 5,403,008 tokens** (5.2 × 1M) at 16 slots, verification cap 32 and a 12G prefix cache (20261111:
  3,014,656 at 8 slots, cap 24, 32G).
- **DeepSeek's own FP8 Engram tables, both in host RAM** (were NVIDIA's MXFP4 tables). No NVIDIA download is needed
  any more, but the tables take 189 GiB instead of 97 GiB. **A 256 GB machine is now the minimum** ([Memory](#memory)).
- **Prefill and decode work in the export.** Highlights: two-shot pull all-reduce in 768-row parts for prefill; the
  indexer chain and the draft context projection on side streams; PDL in the mHC norm kernels; the BRRR fused layer
  tail in every parked-layer graph; a host-row embedding gather. Plus decode glue cleanup, which removed several hundred
  kernel launches per step. Prefill 32K: **15,766 tok/s** (20261111: 14,547).
- **Scheduler and runtime.** Three serve-time changes: tail merge of the last short prompt piece, a merge of short
  message pieces at a request's own checkpoint (one prefill launch per agent turn instead of ~3), and a TTFT-guard
  order that serves the request with the fewest prompt chunks left first. Also new: a CUDA stack-limit option
  (+247,552 KV tokens) and a faster start and stop.
- **Request handling follows DeepSeek's reference.** A required or named `tool_choice` in thinking mode now returns
  HTTP 400 (was 500). `reasoning_effort` accepts `none`, `minimal` and `medium`. Malformed tool-call arguments are
  passed through. History tool arguments that are not JSON objects are rendered.

**Qualification is the owner's minimum tier for a weights change**, not the full suite of earlier releases. See
[Quality](#quality-and-long-context) for what ran and what did not. During qualification, a hardware fault on one GPU
(Xid 79, the card dropped off the PCIe bus) interrupted the first 16-stream KV stress attempt after 42 seconds; the
stress was rerun on the same build and settings, and its result follows: **PASS** (32/32 long requests, 212/212 pings,
lowest free VRAM 64 MiB; post-run checks and log scan clean).

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT), all 48 shards, including DeepSeek's FP8 Engram tables.
Runtime: [NVIDIA Hyperflare](https://github.com/NVIDIA/hyperflare-inference) `main` @ `70069281` plus this fork
(`source.patch`; the same code is the public fork https://github.com/AzallazA/spark-local-inference at tag
`hyperflare-tp3-20261116`), FlashInfer @ `aa6c4221` (upstream's pin, 0.7.3) plus four patches in `tp3-patches/` and one b12x
overlay patch in `tp3-patches/overlay/`.
The fused layer tail is adapted from [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0).

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download DeepSeek's checkpoint, build the fork and the b12x overlay, then export (34 min with a warm tuning
cache; a fresh machine first runs a `HOT=300` warm-up export). `./check_env.sh <fork checkout>` is a read-only
preflight. Then:

```bash
FLARE=<fork>/flare-cli/target/release/flare ./serve.sh   # port 15015, model DeepSeek-V4.1-Flash, ~55 s; PACKAGE=<name or path> if the view was not named ds41-1017-engram-fp8
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.
Every measurement is in [MEASUREMENTS.md](MEASUREMENTS.md); qualification results in [qualification/RESULTS.md](qualification/RESULTS.md).

## Headline numbers

| | this build, 450 W | 20261111 (published), 450 W |
| --- | --- | --- |
| Request slots / verification cap / prefix cache | 16 / 32 / 12G | 8 / 24 / 32G |
| KV cache | **5,403,008** (5.2 × 1M) | 3,014,656 |
| Decode 32K c1 / c8 / c16 (tok/s) | **319.5 / 978.8 / 1,331.5** (one 30 s pass each); same binary and package earlier the same day: c1 293.7-317.2, c8 989.2-1,024.4 (4 passes) | c1 / c4 / c8 301 / 734 / 988 (mean of 5 alternating rounds, 60 s cells); no c16 (8 slots) |
| Decode 32K c4 (tok/s) | 781.2 (one 30 s pass, same package, binary without the prefill merge) | 734 |
| Decode 128K / 512K, c1 | 320.8 / 292.7 (one run each) | 289 / 271 (one run each) |
| Prefill 32K (tok/s) | **15,766** (4 samples) | 14,547 |
| 8-stream agent replay, 300 s | 649 turns, 464.6 output tok/s, first token p50 / p90 0.42 / 0.72 s | not measured in this form (4-stream replay: 519 turns) |
| Host RAM committed by the server | **~217 GiB** (FP8 Engram tables in RAM, 12G prefix cache) | ~148 GiB (MXFP4 tables, 16G) |
| Startup | ~55 s | ~55 s |
| eval-43 / tool calls (60) | **42/43** / **60/60** | 42/43 / 60/60 |
| tool-eval-bench | **90**/100 (77 of 91 graded scenarios) | 91 (same harness, run after the release) |
| GPQA Diamond (one sample of 198) | **91.4%** (181/198; 4 answers truncated, WARN above 3) | 90.4% (run after the release) |
| hotel-lights, 30 runs | **30/30** exact, 0 truncated (16 streams) | 30/30 (8 streams) |
| KV stress (16 streams) | **PASS**: 32/32 long requests, 212/212 pings, lowest free VRAM 64 MiB (rerun; the first attempt was interrupted by a hardware fault) | PASS (8 streams) |
| Long retrieval 128K / 900K | **pass / pass** | pass / pass |
| LiveBench coding | not run | 83.5 |

Decode uses `llm_decode_bench.py --contexts 32k` (`llm-inference-bench`, ignore-EOS decode cells). **The two columns
use different methods.** The 20261111 column holds means of alternating rounds with 60 s cells. This build's cells are
single 30 s passes from the qualification run. Single passes of this build spread by up to ±4% at c1 and ±2% at c8
within one session (the same-day range in the table). Read c1 and c8 as level with 20261111, not as a speed-up. c16
is new: the package now captures graphs for 16 requests.

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [Memory](#memory)
2. [Which settings for me](#which-settings-for-me) · [Scheduler](#scheduler)
3. [Quality and long context](#quality-and-long-context) · [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Coming from 20261111](#coming-from-20261111) · [Going back to 20261111](#going-back-to-20261111) · [Building the package](#building-the-package) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

## Test system

| Part | Spec |
| --- | --- |
| GPUs | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), each on PCIe Gen5 x16 |
| GPU settings | **450 W** power limit, **+75 MHz core / +6000 MHz memory offset** (memory clock 16,365 MHz); ECC off; persistence mode on |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, 1 socket, 1 NUMA node |
| Motherboard | ASUS Pro WS WRX90E-SAGE SE |
| RAM | 256 GB DDR5 (251 GiB visible to Linux): 8 × 32 GB Kingston Fury (KF560R32-32), 8 channels at **6000 MT/s** |
| Package drive | Crucial T710 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Checkpoint drive | Samsung 9100 PRO 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Software | Ubuntu, NVIDIA driver 615.71.09, CUDA 13.1 |

What matters most for speed is unchanged from 20261111 (power, memory clock, PCIe topology; see
[its notes](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#test-system)). New in this build: **host RAM
is a hard requirement.** With both FP8 Engram tables in RAM the server commits ~217 GiB. At boot ~21 GiB of the
251 GiB stay available, and the lowest we saw under load was 10.7 GiB (16-stream KV stress of the same package, on
the binary before the prefill merge; 12.4 GiB in the same stress on this release's binary).

## What this build contains

Cumulative description of the fork's features. The 20261111 [contents table](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#what-this-build-contains)
still applies (BRRR fused layer tail, merged hot/cold expert bank with 342 hot experts in layers 20-33, embedding and
vision tower in host RAM with vision staging, vocabulary-parallel sampling, 4096-token prefill chunks, DSpark 5 drafts
with adaptive verification, TTFT guard). The rows below are new since then, in the order they landed. Each was
measured against the build before it, on the packages named in [MEASUREMENTS.md](MEASUREMENTS.md).

| Change | Effect when it landed |
| --- | --- |
| Port onto upstream `70069281`: compiler-planned memory replaces the graph scratch pool; FlashInfer `aa6c4221` (0.7.3); `--mtp-only` packages (no MTP-disabled graphs); export tuning prefers lower-scratch kernels within 5% of the fastest (`FLARE_TUNE_SCRATCH_TOLERANCE=0.05`) | served KV 3,189,376 → 5,024,384 (8 slots, cap 24); c4 / c8 / prefill / agent level, c1 301 vs 318 tok/s at the time |
| Prefill tail merge (`--scheduler-prefill-tail-merge true`; thinking requests without tools keep their last-message checkpoint) | prefill 8K +19%, 32K +7%; agent first-token p90 0.99 → 0.91 s at 8 streams |
| Fast restart: Engram tables in private 4 KiB-page memory, loader threads, fast exit (`FLARE_ENGRAM_PRIVATE_TABLES=1`, `FLARE_FAST_EXIT=1`) | start 62-64 → 47-48 s, stop 16-17 → 8 s, decode level |
| Decode glue cleanup (fully written buffers left uninitialised, operands as own tensors) | several hundred fewer kernel launches per decode graph; c1 steps/s 1.035× |
| Two-shot pull all-reduce for prefill (b12x overlay patch, `B12X_PCIE_TWOSHOT_PULL=1`), BRRR in every parked-layer graph (`PARKED_ROWS=0`), attention decode's 16-head grid on a side stream beside the 8-head remainder (`B12X_MLA_DECODE_REMAINDER_STREAM=1`), host-row embedding gather (`FLARE_DS41_EMBED_ROWS=1`) | c4 +4.6-4.7%, c8 +3.4-4.6%, prefill 32K +5%, c1 level |
| VRAM reclaim (`FLARE_DS41_VRAM_RECLAIM=1`: shared Engram token map, trimmed vocabulary gather, no draft-vocabulary channel, shared vocabulary columns) and **16 request slots** (`MAXC=16`) | served KV +2.7% while paying for the 16-slot graphs; c1-c8 level; c12 / c16 1,204 / 1,211 tok/s at cap 24 |
| Verification cap 32 (`BUDGET=32`, was 24) | c16 1,375 vs 1,226 tok/s, c12 1,234 vs 1,188; c1-c8 level; 36K fewer KV tokens |
| **FP8 Engram tables** (DeepSeek's own, 189 GiB) both in host RAM, prefix cache 12G (32G no longer fits) | c1 / c4 / c8 / c16 0.99 / 1.02 / 1.03 / 0.99× vs MXFP4; served KV −1.8% |
| Round-2 export switches: indexer chain on a side stream (`FLARE_DS41_INDEX_SIDE_STREAM=1`), DSpark context projection beside verification (`FLARE_DS41_CONTEXT_SIDE_STREAM=1`), prefill all-reduce parts of 768 rows for the decoder and the Engram projection (`FLARE_DS41_AR_PART_ROWS=768`, `FLARE_DS41_ENGRAM_PART_ROWS=768`), PDL in the mHC norm kernels (`B12X_MHC_PDL=1`), BRRR `wo_a` split for ≤ 8 rows (`FLARE_DS41_BRRR_WOA_UNEVEN=1`) | c1 1.076×, c4 1.011×, c8 0.992× (2 passes each); prefill 8K 1.046×, 32K 1.042×; dry-run KV +0.6% |
| CUDA stack limit 256 B per thread (`FLARE_CUDA_STACK_LIMIT=256`; the largest registered kernel needs 176 B; the load refuses a limit below it) | served KV 5,155,456 → **5,403,008** (+247,552) |
| TTFT guard order `shortest` (`--scheduler-ttft-order shortest`): of the requests past the 50 ms target, the one with the fewest prompt chunks left goes first; the oldest still gets at least every second guarded round | a short prompt no longer waits behind every earlier long prompt (scheduler simulation: mean first token 2.31 → 2.14 s, p90 3.88 → 4.12 s, 8 agents with 0.8K-14K-token turns) |
| Prefill merge of short message pieces (`FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512`): a chunk that starts at the request's own checkpoint runs past message boundaries closer than 512 tokens | prefill launches per agent turn 3.15 → 1.00; first token p50 0.79× / 0.75× and turns 1.09× / 1.10× at 8 / 16 streams; decode level (see [Scheduler](#scheduler) for the noise caveat) |
| Export KV hygiene: cuBLAS's own 64 MiB pool is kept out of the export's reachability (`FLARE_EXPORT_LIBRARY_SCRATCH`, default on in the code) | no KV change on this package's settings; the release package was exported before it (see [Measured and not adopted](#measured-and-not-adopted)) |

## Memory

**GPU (per GPU, release package, served):** 89.87 GiB loaded (weights, graphs, scratch), 4.76 GiB KV cache budget;
14.3 GiB of KV over the 3 GPUs = 5,403,008 tokens. The 20261111 build loaded ~92.3 GiB and had 2.6 GiB of KV per GPU.

**Host RAM** (system has 251 GiB visible):

| Serve mode | Committed host RAM (dry run) | Lowest MemAvailable measured | Speed vs both in RAM |
| --- | --- | --- | --- |
| **Both FP8 tables in RAM, prefix cache 12G (default)** | **217.3 GiB** (tables 188.8 GiB, host model memory 16.0 GiB, prefix cache 10.8 GiB) | 21.0 GiB at boot; 12.6 GiB in the 8-stream agent replay; 12.4 GiB in the 16-stream KV stress (10.7 GiB on the binary before the prefill merge) | — |
| Both FP8 tables in RAM, prefix cache 32G | 237.3 GiB | not started: predicted 0.4 GiB available | — |
| `ENGRAM_RAM=layer1`: layer 1's table in RAM, layer 14's from disk with a 32 GiB row cache, prefix cache 32G | 177.8 GiB | 63.0 GiB | decode c1-c16 level; 8-stream agent replay output 0.82× |
| `ENGRAM_RAM=0`: both tables from disk | not measured with FP8 tables | — | — |
| 20261111 (MXFP4 tables in RAM, 16G) | ~148 GiB (as published) | — | — |

The `layer1` row was measured on the first FP8 package (the code before the round-2 switches). Treat it as a guide.
Less than 256 GB of RAM: use `ENGRAM_RAM=layer1`, or stay on the [20261111 build](#going-back-to-20261111) with
the MXFP4 tables (97 GiB).

The FP8 export itself left 193 GiB of host RAM available at its lowest (243 GiB available before it), so exporting
needs about 50 GiB.

## Which settings for me

| Your situation | Export (`export.sh`) | Serve (`serve.sh`) | Expect |
| --- | --- | --- | --- |
| 256 GB RAM, up to 16 concurrent requests (the tested setup) | defaults | defaults | the numbers above: 5.40M KV, 16 slots |
| Less than 256 GB RAM | defaults | `ENGRAM_RAM=layer1` | ~60 GiB less committed host RAM; decode level, agent-style traffic ~18% less output (one 300 s replay) |
| Many long sessions you return to | defaults | `ENGRAM_RAM=layer1 PREFIX_CACHE=32G` | more warm prompts in the host prefix cache; same cost as the row above |
| More KV, less speed | `HOT=<n>` below 342 | defaults | each hot expert fewer frees ~92K KV tokens (measured on 20261111's base; not re-measured on this one) |
| Single user, throughput only (no first-token concern) | defaults | `SCHEDULER=balanced` | running streams never pause for a new prompt; new prompts wait behind decode under load |
| The 20261111 serving behaviour | defaults | `TTFT_ORDER=oldest PREFILL_MERGE= STACK_LIMIT=` | oldest-first guard, no message merge, recorded stack limit (247,552 fewer KV tokens) |
| Max-Q cards (~95.0 GiB) or 350 W cards | see [20261111](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#which-settings-for-me) | | not tested on this build |

`DRAFTS` must be the same at export and serve (default 5 in both scripts): the package is captured for one draft count.
`MAXC` (default 16) sets how many requests the package has graphs for; 8 was not measured on this build.

## Scheduler

`serve.sh` defaults: `--scheduler-policy interactive` (the TTFT guard from 20261111), `--scheduler-prefill-tail-merge true`,
`--scheduler-ttft-order shortest` and `FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512`.

**Prefill merge.** Agent turns resume at the previous turn's terminal checkpoint, so each new turn starts with the
answer's end token (1-2 tokens) up to the next message boundary. Every message boundary used to end a prefill chunk,
so that piece ran as its own prefill launch (~27 ms fixed cost each at agent context lengths), and the TTFT guard's
interleave then split the next message. With the merge, a chunk that starts at the request's own checkpoint runs
past message boundaries closer than 512 tokens. Those boundaries get no checkpoint. A later turn still resumes at the
prompt end, the terminal checkpoint, or the last message. Thinking requests without tools keep their last message.
The prefill-merge measurement (same binary, flag off / on / on / off, 300 s replays):

| Workload | Turns | Output tok/s | First token p50 / p90 | Prefill launches per turn |
| --- | --- | --- | --- | --- |
| 8 streams, agent-style | 1.086× | 1.014× | 0.79× / 0.82× | 3.15 → 1.00 |
| 8 streams, mixed (agent + chat) | 1.083× | 1.068× | 0.80× / 0.93× | 3.03 → 1.35 |
| 16 streams, agent-style | 1.097× | 1.244× | 0.75× / 0.71× | 3.23 → 1.01 |
| 16 streams, mixed | 1.124× | 1.079× | 0.83× / 0.93× | 3.07 → 1.36 |
| Decode 32K c1 / c8 | 1.023× / 1.011× (level) | | | |

**Read the 8-stream rows with care.** Two runs of the same flag differed by up to 11.8% in turns at 8 streams, so the
8-stream gain is inside the noise of this measurement. The pre-set adoption rule therefore said "hold". The owner
adopted it on the first-token times and the 16-stream result. `PREFILL_MERGE= ./serve.sh` turns it off.

**TTFT order.** With `oldest` (20261111) the guard always protects the oldest prompt still without output. While long
prompts load, that is a long prompt, so a short request waits for every earlier long prompt's prefill. `shortest`
picks the overdue request with the fewest prompt chunks left. The oldest one still gets every second guarded round,
so no request starves. Measured only in the scheduler simulation (see the table above) and as part of the integration
check of this build. `TTFT_ORDER=oldest ./serve.sh` restores the old order.

The [20261111 notes on the guard itself](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#scheduler-trade-off-first-token-vs-running-streams)
still apply (the first token arrives sooner while decoding streams pause briefly).

## Quality and long context

**This build was qualified with the owner's minimum tier for a weights change** (TIER=weights, plus long retrieval and
the 512K cell). Earlier releases ran the full suite. All results come from one run on the running server without
restarts, except the 16-stream KV stress and the checks after it, which come from the rerun on the same binary after the
hardware fault (new server start, same settings); package `gb202-tp3-202611160815`, served exactly as `serve.sh` serves
it, at 450 W. Details and gates are in
[qualification/RESULTS.md](qualification/RESULTS.md).

| Check | This build | 20261111 (package 202611111227) |
| --- | --- | --- |
| GPU kernel checks (`gpu_check`, 13 cases) | 13/13 | — |
| Smoke (thinking on / off) and tool-call smoke | 2/2, 3/3 | — |
| Images (5 cases incl. over-size and four images) | 5/5 served, 5/5 correct | not re-run |
| Corruption check (teacher-forced comparison, 10 runs) | **0** true corruptions | 0 |
| Decode vs prefill consistency (top-1 agreement over 284 positions) | 0.951 (gate ≥ 0.95), median Δlogprob 0.0081 | — |
| eval-43 (thinking off) | **42**/43 | 42/43 |
| Tool calls: 60 tool-requiring prompts at 8 streams | **60**/60 | 60/60 |
| tool-eval-bench (92 scenarios, 91 graded) | **90**/100 (77/91 pass) | 91 (78/91) |
| GPQA Diamond (one sample of 198, 8 streams, 32K tokens) | **91.4%** (181/198, 95% interval 86.7-94.6); 4 truncated without an answer (WARN: gate ≤ 3), 7 hit the token limit; paired with 20261111's run: 5 gained, 3 lost (p 0.73) | 90.4% |
| hotel-lights, 30 runs | **30/30** exact, 0 truncated (16 streams) | 30/30 (8 streams) |
| Long retrieval: three markers at 128K and 900K tokens | **pass / pass** | pass / pass |
| KV stress: 2 waves × 16 × ~530K unique prompts + pinger, KV oversubscribed | **PASS**: 32/32 long requests, 212/212 pings, 0 bad log lines, lowest free VRAM 64 MiB (above the 16 MiB gate, below the kit's 480 MiB margin), 25.6 min; post-run smoke 2/2, served KV unchanged, 0 true corruptions; log scan clean (rerun after a hardware fault interrupted the first attempt) | PASS (2 waves × 8) |
| lavd (30 runs), long generation after a 581K-token conversation, natural-EOS decode bench, LiveBench coding | **not run** | 30/30, 4/4, 300 / 713 / 950 tok/s with 0 loops, 83.5 |

**Not run, and why.** These stages did not run: LiveBench coding, long generation, lavd, the natural-EOS decode bench,
the 8-stream KV stress (the 16-stream stress replaces it), the 128K c8 decode cell and the vision sets. The owner's
tiered policy (2026-11-16) drops stages that have never failed on this model in our records, to release faster: lavd
always 30/30, long generation always 4/4, EOS 0 loops on natural EOS, LiveBench 81.3-84.2. The weights tier was chosen
because the FP8 Engram tables are a quantisation change. It adds GPQA Diamond to the decode-path and export checks.
Nothing here says those stages would pass; they were not measured on this build.

## Measured and not adopted

| Idea | Result |
| --- | --- |
| Engram side stream (layer 14's lookup, projection and all-reduce during layer 13; `ENGRAM_SIDE=14`) together with export KV hygiene | package 161317 against the release package: c1 / c4 / c8 0.97 / 0.99 / 1.02×, prefill 8K / 32K 1.00×, served KV identical, boot 76 s vs 51 s. Not adopted; the side stream stays off |
| Export KV hygiene in the release export | no KV change on these settings (dry-run 4,996,096 both); the check still reports one unclaimed 44 MiB span per GPU, under investigation. The code default is on; `LIBRARY_SCRATCH=0` reproduces the release export's code path |
| Verification cap 48 | c16 1,300 vs 1,375 tok/s at 32; c1-c8 level |
| Prefix cache 32G with both FP8 tables in RAM | does not fit in 256 GB (predicted 0.4 GiB available) |
| BRRR `SPIN_NS`, `TWO_SHOT_MIN`, `KS_WOB`, `PER_SM` build-time switches | kept as default-off switches; not part of the release export |
| Earlier ideas (7 drafts, BRRR stage 2, item 6, …) | see the [20261111 list](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md#measured-and-not-adopted) |

## Known limitations

- **256 GB of host RAM** with both FP8 tables in RAM; `ENGRAM_RAM=layer1` or the 20261111 build for less.
- **Minimum qualification** (see [Quality](#quality-and-long-context)): no LiveBench, long generation, lavd, EOS bench
  or 8-stream stress on this build.
- **Hardware-specific package:** 3 × 96 GB SM120 (RTX PRO 6000) only. Other setups need a fresh export.
- **The KV stress runs with little VRAM headroom:** the lowest free VRAM in the 16-stream stress of this release was
  64 MiB (80 MiB on the binary before the prefill merge; no request failed; earlier packages showed 80 MiB in their
  stress runs). The KV
  cache takes whatever VRAM the graphs leave, so this is the normal state of a full cache, not a leak.
- **Thinking mode and forced tool calls:** `tool_choice: "required"` or a named function with thinking on returns
  HTTP 400 ("Thinking mode does not support this tool_choice"), as DeepSeek's reference does. Turn thinking off
  (`chat_template_kwargs: {"enable_thinking": false}`) to force a call.
- **top_p is fixed at 1.0** (vocabulary-parallel sampling); temperature is fixed at 1.0 by Hyperflare's DeepSeek recipe.
- **The `--mtp-only` package requires speculation** at serve time (`DRAFTS` > 0, the default 5).
- **Bench loops:** `llm_decode_bench` decode cells use `ignore_eos`; a rare repetition loop there is a bench artifact.
- Not upstreamable as written: Hyperflare's contribution rules exclude model-specific offload policy and AI-authored
  GPU-execution changes. This is a personal fork of Hyperflare.

## Coming from 20261111

- **New base commit** `70069281` (was `972e28e4`). Start from a fresh clone at `70069281` and apply this folder's
  `source.patch`, or fetch fork tag `hyperflare-tp3-20261116`. The old checkout does not take the new patch.
- **New Python environment:** FlashInfer `aa6c4221` (0.7.3) and CUTLASS DSL 4.8.1 from upstream's `uv.lock`.
  `uv sync --locked --extra prepare --extra test`, then this folder's four FlashInfer patches. The tiny-decode
  swiglu patch is gone: FlashInfer now applies the limit itself.
- **Build the b12x overlay** for the two-shot pull all-reduce (EXPORT.md step 2); `export.sh` refuses to run without
  it unless `TWOSHOT=0`.
- **Rebuild the binary:** `cargo build --locked --release --manifest-path flare-cli/Cargo.toml`. The binary moved to
  `flare-cli/target/release/flare`.
- **New checkpoint view** `ds41-1017-engram-fp8` (`make_checkpoint_view.sh` here, DeepSeek's snapshot only). The
  MXFP4 view of 20261111 still exports, but this release was measured with FP8 tables only.
- **Export again** (`export.sh` defaults). The tuning cache from 20261111 does not carry over (new FlashInfer pin): a
  fresh `HOT=300` warm-up export comes first.
- **Serve flags:** `serve.sh` now passes `--prefix-cache-memory 12G`, `--mtp-max-verification-token-budget 32`,
  `--scheduler-prefill-tail-merge true` and `--scheduler-ttft-order shortest`. It also sets
  `FLARE_CUDA_STACK_LIMIT=256`, `FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512`, `FLARE_ENGRAM_PRIVATE_TABLES=1` and
  `FLARE_FAST_EXIT=1`.
- Old and new packages have different model names (`ds41-1017-engram-mxfp4` and `ds41-1017-engram-fp8`). Remove a
  version with `flare models remove <model>@gb202-tp3-<timestamp>`, then `flare models gc`.

## Going back to 20261111

Loading the 20261111 package with this build's binary was not tested (a 2026-11-12 package from the same era did
load on the ported binary). The supported way back is that folder's own build: check out `972e28e4`, apply the
[20261111 folder](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md)'s `source.patch`, and follow its
EXPORT.md. A 20261111 package you kept is served by that build's binary with its own `serve.sh`. On this
build, the closest you get to the 20261111 serving behaviour is `TTFT_ORDER=oldest PREFILL_MERGE= STACK_LIMIT= BUDGET=24`.

## Building the package

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
# from this folder
./check_env.sh <fork>                                                              # read-only preflight
./make_checkpoint_view.sh <deepseek snapshot> ~/ds41-1017-engram-fp8               # symlinks only; DeepSeek's snapshot, all 48 shards
# from the fork checkout (export.sh uses its .venv)
cd <fork>
VENV=$PWD/.venv B12X_OVERLAY_PATCHES=$PWD/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch \
  tp3-tools/make-b12x-overlay.sh $PWD/.b12x-overlay-twoshot                       # b12x copy with the two-shot pull patch
CKPT=~/ds41-1017-engram-fp8 B12X_OVERLAY=$PWD/.b12x-overlay-twoshot HOT=300 /path/to/this-folder/export.sh   # warms the kernel tuning cache
CKPT=~/ds41-1017-engram-fp8 B12X_OVERLAY=$PWD/.b12x-overlay-twoshot /path/to/this-folder/export.sh           # the release build
```

The model name is the checkpoint directory's name: this view exports as `ds41-1017-engram-fp8`, which is what
`serve.sh` runs by default.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `check_env.sh` | read-only preflight: toolchain, memlock, fork commit and patches, disk, idle GPUs (PASS / WARN / FAIL) |
| `versions.txt` | exact versions behind the numbers |
| `MEASUREMENTS.md` | every measurement of this build, with the package and settings of each |
| `ENGRAM-PRECISION.md` | MXFP4 vs FP8 Engram tables: every quality score we have for each |
| `serve.sh` / `export.sh` | serve the package / export it from a checkpoint view |
| `make_checkpoint_view.sh` | build the FP8 checkpoint view from DeepSeek's snapshot |
| `source.patch` | the fork's changes against Hyperflare `70069281`, generated from fork tag `hyperflare-tp3-20261116` (387 files, 178 new) |
| `expert-order/` | per-layer route counts that set the hot-expert placement (unchanged) |
| `tp3-patches/` | the four FlashInfer patches, and `overlay/` with the b12x two-shot pull patch |
| `tools/` | `ttft_under_load.py` (first-token latency and agent replay), `bench_steps.py`, `verify_fix.py` (corruption check), `consistency.py` (decode vs prefill), `llm-bench-flare-steps.patch` |
| `qualification/` | the qualification kit this build was checked with, its results and `RESULTS.md` |

## Licenses

DeepSeek weights: MIT (`licenses/DeepSeek-V4.1-Flash-1017-LICENSE`). NVIDIA's NVFP4 release (MIT,
`licenses/NVIDIA-DeepSeek-V4.1-Flash-1017-NVFP4-LICENSE`) is no longer used by the default view; the license stays for
the MXFP4 view of earlier folders and `flashinfer-engram-mxfp4.patch`. Hyperflare: Apache-2.0 (`licenses/LICENSE`,
`licenses/NOTICE` and `licenses/THIRD_PARTY_NOTICES.md` kept; modified files marked). FlashInfer, which `tp3-patches/`
modifies: Apache-2.0 (`licenses/FlashInfer-LICENSE`, `licenses/FlashInfer-NOTICE`). BRRR: Apache-2.0
(`licenses/BRRR-LICENSE`, `licenses/BRRR-NOTICE`). `llm-inference-bench`, which `tools/llm-bench-flare-steps.patch`
modifies, is MIT (its README). The `flare` binary links only libc/libstdc++; NVIDIA's CUDA Python wheels are used only
for exporting and are not redistributed.

## Credits

NVIDIA Hyperflare and FlashInfer b12x; DeepSeek for the model and the FP8 Engram tables; NVIDIA for the MXFP4 Engram
tables of earlier builds; Fraser Price for BRRR; Local Inference Lab for `llm-inference-bench` and `tool-eval-bench`.
