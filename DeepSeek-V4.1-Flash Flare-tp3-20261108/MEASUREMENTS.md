# Measurements — Flare TP3 20261107

Every measurement of this release's build. The fork is rebased onto upstream Flare `972e28e4` (FlashInfer `1f8ed8a3`);
"v2" below is the pre-rebase fork, the base of the 20261105 release.

| | |
| --- | --- |
| Package | `ds41-1017-engram-mxfp4@gb202-tp3-202611070243` |
| Build | 342 hot experts per parked layer (20-33), merged VRAM/host expert bank, BRRR fused layer tail on all 40 layers (parked layers only in graphs of 12 rows or fewer), token embedding and vision tower in host RAM, all-reduce sized for 1024 rows, MXFP4 Engram, prefill chunk 4096, 8 slots, 5 DSpark drafts |
| KV cache | 2,373,760 tokens (two 1M-token sessions) |
| Serve flags | `--mtp-draft-tokens 5 --mtp-adaptive-verification --mtp-max-verification-token-budget 24 --mtp-adaptive-verification-alpha 0.8 --scheduler-policy balanced --prefix-cache-memory 16G`, `FLARE_ENGRAM_RAM_TABLES=0` (both Engram tables in RAM) |
| Hardware | 3 × RTX PRO 6000 Blackwell Workstation, 256 GB DDR5-6000, PCIe Gen5 NVMe |

Unless noted, the benchmark is `llm_decode_bench.py --concurrency 1,4,8 --contexts 32k --coding-peak` (ignore-EOS),
with decode in tok/s and "coding" the coding-peak mean tok/s. Tables marked "user" were run by hand; the rest come from an
automated sweep with one minute between runs.

## Power and clocks (user)

Each column is the mean of the user's runs: 3 at 350 W, 2 at 450 W. Memory offset +6000 MHz in both; alpha 0.6 (the setting before the sweep).

| | 350 W, +0 core | 450 W, +75 core | change |
| --- | --- | --- | --- |
| 32K c1 / c2 / c4 / c8 | 275 / 443 / 644 / 850 | 303 / 507 / 719 / 957 | +10% / +14% / +12% / +13% |
| 32K c1 / c2 / c4 / c8 steps/s | 110.5 / 175.4 / 255.5 / 333.2 | 113.7 / 185.8 / 285.7 / 396.7 | |
| 128K c1 / c2 / c4 / c8 | 253 / 438 / 627 / 817 | 283 / 513 / 682 / 903 | +12% / +17% / +9% / +11% |
| 512K c1 / c2 / c4 | 262 / 385 / 540 | not run | |
| Prefill 32K / 128K | 12,931 / 12,844 | 14,507 / 14,645 | +12% / +14% |
| Coding mean | 432 | 482 | +12% |

On vLLM, the extra power only helped prefill (~+10%); decode was flat. On Flare it helps decode too.

**600 W (user, 1 run, release defaults, +75 / +6000):**

| 32K | 450 W (2-run mean) | 600 W | change |
| --- | --- | --- | --- |
| c1 / c4 / c8 tok/s | 305 / 716 / 975 | 296.9 / 757.1 / 1,002.2 | −3% / +6% / +3% |
| c1 / c4 / c8 steps/s (per stream) | 118.8 / 70.9 / 50.7 | 115.1 / 72.3 / 56.1 | |
| Coding mean | 465 | 460 | same |
| GPU power at c8, all three cards | ~1,350 W | ~1,594 W | +240 W |

Within a few percent of 450 W (c1 and coding flat; c4/c8 inside the 2-3% run-to-run spread or just above), for about
240 W more. 450 W stays the recommended setting. Everything else on this page was measured at **450 W, +75 core,
+6000 memory**.

## Release defaults (user, 450 W / +75 / +6000, k5, alpha 0.8, balanced)

Two runs of `llm_decode_bench` at 32K plus single runs at 128K and 512K, on `serve.sh` as shipped.

| | 32K run 1 | 32K run 2 | 128K | 512K |
| --- | --- | --- | --- | --- |
| c1 / c2 / c4 / c8 tok/s | 313.7 / 496.4 / 727.4 / 971.0 | 296.4 / 479.2 / 704.7 / 979.0 | 295.5 / — / 708.1 / 956.0 | 275.2 / — / 657.3 / — |
| c1 / c2 / c4 / c8 steps/s (aggregate) | 116.0 / 195.1 / 284.0 / 409.3 | 121.5 / 200.3 / 283.0 / 402.4 | 112.8 / — / 298.1 / 396.4 | 110.9 / — / 271.8 / — |
| Prefill 32K / 128K | 15,081 / 14,670 | 14,529 / 14,567 | | |
| Coding mean / max | 467 / 477 | 462 / 501 | | |

32K mean: 305 / 488 / 716 / 975 tok/s, prefill 14,805 / 14,619. Steps/s here is summed over the streams (c4 284 = ~71 per stream).

## Speculation settings sweep

### k5, adaptive-verification alpha × scheduler (1 run each)

| alpha / scheduler | c1 | c4 | c8 | coding |
| --- | --- | --- | --- | --- |
| 0.6 balanced | 306 | 715 | 948 | 462 |
| 0.6 throughput | 293 | 706 | 923 | 466 |
| 0.8 balanced | 301 | 714 | 955 | 480 |
| 0.8 throughput | 301 | 714 | 934 | 465 |
| 1.0 balanced | 293 | 713 | 952 | 460 |
| 1.0 throughput | 292 | 727 | 960 | 467 |

All six fall within ~2%. Repeat runs of the two best balanced settings:

| | runs | c1 | c4 | c8 | coding |
| --- | --- | --- | --- | --- | --- |
| 0.8 balanced | 4 | 296 | 711 | 955 | 472 |
| 0.6 balanced | 3 valid | 295 | 707 | 949 | 474 |

They tie within noise; **0.8 balanced** is the default. One extra 0.6 balanced run lost its c8 cell to the bench's
loop guard. That stream had already ended its reply (it emitted `<｜begin▁of▁sentence｜>`); ignore-EOS forced it on,
and it repeated invented tool output. Natural-EOS benches showed 0 loops.

### k7 (7 drafts, package `...070707`, 2,220,416 KV tokens), 1 run each

| alpha / scheduler | c1 | c4 | c8 | coding |
| --- | --- | --- | --- | --- |
| 0.6 balanced | 280 | 705 | 912 | 440 |
| 0.6 throughput | 291 | 668 | 892 | 478 |
| 0.8 balanced | 288 | 699 | 930 | 466 |
| 0.8 throughput | 295 | 684 | 924 | 466 |
| 1.0 balanced | 291 | 683 | 848 | 492 |
| 1.0 throughput | 296 | 699 | 936 | 471 |

### Final: best k5 vs best k7 (3 runs each)

| | c1 | c4 | c8 | coding | KV tokens |
| --- | --- | --- | --- | --- | --- |
| **k5, alpha 0.8, balanced** | 298 | 738 | 959 | 456 | 2,373,760 |
| k7, alpha 1.0, throughput | 299 | 698 | 930 | 467 | 2,220,416 |

k5 wins at c4 (+6%) and c8 (+3%) and keeps 153K more KV tokens. k7 is ~2% ahead on coding and ties at c1.

### Verification row cap (`--mtp-max-verification-token-budget`, 450 W, k5, alpha 0.8, balanced)

A full c8 step at k5 verifies 48 rows; the cap limits how many rows the adaptive verifier may use. Runs from a sweep
plus an alternating 128 / 24 / 128 / 24 confirmation:

| Cap | Runs | c1 | c4 | c8 | Coding |
| --- | --- | --- | --- | --- | --- |
| 128 (no cap; earlier default) | 4 | 301 | 731 | 963 | 468 |
| 48 | 2 | 298 | 721 | 968 | 464 |
| 32 | 1-2 | 298 | 726 | 954 | 463 |
| **24 (default)** | 4 (c8: 3) | **306** | **734** | **986** | **473** |

Each gap is inside the run-to-run spread, but all four favor 24 (+1.7% / +0.4% / +2.4% / +1%), so it became the
default. Higher core offsets at cap 24 (+100: 312 / 731 / 978, coding 471; +150: 308 / 722 / 972, coding 469) gave
nothing over +75.

### Prefix-cache memory (k5, alpha 0.8, balanced; 1 run each, 16G is the 4-run mean)

| `--prefix-cache-memory` | c1 | c4 | c8 | coding | host RAM free |
| --- | --- | --- | --- | --- | --- |
| 16G | 296 | 711 | 955 | 472 | ~97 GiB |
| 32G | 296 | 712 | 951 | 468 | 84 GiB |
| 48G | 290 | 723 | 955 | 464 | 70 GiB |

A larger prefix cache doesn't change decode speed; it costs only host RAM.

**Reuse test.** 80 distinct ~127K-token prompts (~10M tokens, far more than the 2.37M-token GPU pool), then a sample
re-sent oldest-first, time to first token:

| Re-sent prompt | 16G host cache | 32G host cache |
| --- | --- | --- |
| 0-4, 10 (oldest) | ~9.1 s (cold) | ~9.1 s (cold) |
| 20, 30, 40, 50 | ~9.0 s (cold) | **0.14 s** |
| 60, 70 (still in the GPU pool) | 0.14 s | 0.14 s |
| first send (cold prefill) | ~9.0 s | ~9.1 s |

A 127K-token prompt takes ~390 MB of host cache (~3.1 KB/token): 16G keeps ~37 such prompts warm (~4.5M tokens),
32G ~78 (~9.5M), on top of the ~18 in the GPU pool. A host hit is ~65x faster than re-prefilling. Use
`PREFIX_CACHE=32G` if you return to many long sessions; it leaves ~84 GiB of RAM free instead of ~97.

## Engram tables on disk (user)

Both tables were read from disk: `--embedding-table-storage disk` without `FLARE_ENGRAM_RAM_TABLES`. The
page cache was dropped before boot. Flare reads the tables directly from NVMe and the page cache stayed at 13 GiB,
so these are true disk numbers. A machine with much less RAM should see the same speed.

| | Engram in RAM (450 W, user) | Engram on disk, 4 GiB row cache | Engram on disk, 16 GiB row cache |
| --- | --- | --- | --- |
| Prefill 32K / 128K | 14,507 / 14,645 | 14,540-14,591 / 14,667-14,722 | not recorded |
| 32K c1 / c4 / c8 | 303 / 719 / 957 | 292 / 692 / 926 (−4% / −4% / −3%) | 287 / 687 / 926 (1 run) |
| 32K c1 / c4 / c8 steps/s | 113.7 / 285.7 / 396.7 | 111.8 / 288.8 / 368.4 | 111.4 / 282.7 / 371.1 |
| Coding mean | 482 | 463 (−4%) | 464 |
| Server host RAM (RSS) | ~148 GiB | **59 GiB** | 71.5 GiB |

Engram counters for the two 4 GiB runs: 64% row-cache hit rate, 19 GB read from NVMe, ~29 s total disk wait.
With 16 GiB the hit rate was 60%: the cache held 2.7M rows out of a 113M-row capacity, so it never filled and
every miss was a row seen for the first time. The bench sends new prompts every time, so a bigger row cache can't help
here; it can only help workloads that revisit the same text. Disk mode is a good low-RAM option: decode about 4% slower, prefill unchanged, about 90 GiB less host RAM.

## Hot experts and KV (350 W, natural EOS, 1 run each, same session)

| `HOT` | Package | KV tokens | 32K c1 / c4 / c8 | c1 / c4 / c8 steps/s | Corruption check |
| --- | --- | --- | --- | --- | --- |
| 342 | `...070243` | 2,373,760 | 288.0 / 633.9 / 837.6 | 108.4 / 65.8 / 44.3 | 0 |
| 300 | `...070212` | 6,247,680 | 269.3 / 609.6 / 799.5 | 103.4 / 64.2 / 50.7 | 0 |

Each hot expert fewer adds (6,247,680 − 2,373,760) / 42 = **92.2K KV tokens**.

## v3 vs v2 (same 2x build features, user command, 3 runs each)

| | c1 | c4 | c8 | coding | KV tokens |
| --- | --- | --- | --- | --- | --- |
| v3 (`...070243`) | 274 | 628 | 824 | 437 | 2,373,760 |
| v2 (`...062024`) | 284 | 630 | 828 | 431 | 2,199,040 |

Speed is the same within noise. v3 has 175K more KV tokens. This A/B ran at 350 W.

## Qualification (k5, alpha 0.8, balanced, 450 W)

The run used 8 concurrent requests wherever the protocol allows (the H300 release used 4). It took about 1 hour.

| Test | Result | H300 release |
| --- | --- | --- |
| eval-43 (thinking off) | **43 / 43** | 43 / 43 |
| Tool calls | **60 / 60**, c8 | 60 / 60 |
| lavd, 30 runs, c8 | **30 / 30** exact | 30 / 30 |
| hotel-lights, 30 runs, c8 | **30 / 30** exact | 30 / 30 |
| Long retrieval 128K / 900K | **pass / pass** | pass / pass |
| Long generation after a 581K conversation | **4 / 4**, natural stop, repeat_8gram ≤ 0.0011 | 4 / 4 |
| Natural-EOS bench 32K c1 / c4 / c8 | 294 / 708 / 964, **0 loops** | 250 / 588 / 794, 0 loops |
| LiveBench coding (2024-11-25) | **82.2** (generation 74.4, completion 90.0) | 82.9 |
| c8 KV stress (2 waves × 8 × 530K + pinger) | **PASS**: 16/16, pinger 136/136, 9.5M prompt tokens, min free VRAM 82 MiB | PASS |
| Corruption check (`verify_fix.py 10`) | 0 true corruptions (k5 and k7) | 0 |

Raw qualification outputs are in `qualification/results/`.

## 2026-11-08: kernel hygiene, stage-2 wide rows, vision staging (450 W, k5, alpha 0.8, balanced, cap 24, 32G prefix cache)

Three alternating rounds per build, `llm_decode_bench` 32K, 60 s per cell, tok/s. Raw results: `qualification/results/2026-11-08-ab/`.

| Build | c1 | c4 | c8 | verify_fix |
|---|---|---|---|---|
| default `...070243` | 301, 305, 304 | 725, 723, 729 | 919*, 981, 972 | 0 true corruptions |
| BRRR kernel hygiene (cold-first parked jobs, per-chunk attention flags, constants contract, cross-rank check) | 303, 337, 294 | 732, 717, 728 | 980, 982, 977 | 0 |
| the same plus an extra PDL trigger at the end of the layer kernel (earlier run, 2 rounds) | 304, 299 | 711, 716 | 974, 975 | 0 |
| stage 2: wide-row restreaming (OTOK/SPAN 48), parked gate 12 | 286, 299, 298 | 673, 704, 695 | 938, 938, 953 | 0 |
| stage 2, parked layers ungated | 291, 285, 297 | 706, 696, 691 | 942, 954 | 0 |

\* one bench cell with a loop artifact, excluded from the mean. Verdicts: the hygiene changes are level and are in `source.patch`;
the extra PDL trigger cost ~2% at c4/c8 and was removed; stage 2 is a regression (-5% at c4) and is not shipped.

Vision staging (`FLARE_DS41_VISION_STAGE=1`, now the export default): single image, one output token, fresh server, six images:
cold 0.11 s per image (0.11-0.14) vs 1.70 s (1.68-1.72) without staging. 60-image set at both temperatures: 49/60 with and without
staging (the 11 misses are the "grid" items answered verbosely, a parser artifact of this runner); text `verify_fix` 0 true
corruptions; KV 2,461,184 both. Under the 4-concurrent qualification runner the median request time fell from 7.9 s to 1.6 s.

KV dry run: `flare run <package> <serve flags> --dry-run` (new) reproduces the recorded "Allocate cache" lines of five packages
exactly. It also shows that the 2,373,760 → 2,461,184 KV change between the 16G and 32G configurations was the verification
row cap (24 drops 13 speculative graphs, ~76 MiB per rank), not the prefix cache (which costs ~10 MiB of device headroom).

### Qualification, 2026-11-08 (c8, both packages)

See the README's "Qualification of this folder's packages" table; raw results under `qualification/results/run-20261108-c8/`
and `run-20261108.1-c8/`. Export-to-export variance measured the same day (identical code re-exported: c1/c4/c8 298/731/988
vs 302/726/983 over 4 rounds) and session drift (~2% at c8) are recorded in the fork handbook, section D.1.
