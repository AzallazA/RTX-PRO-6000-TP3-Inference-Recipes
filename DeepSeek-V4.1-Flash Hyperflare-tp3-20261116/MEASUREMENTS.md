# Measurements: Hyperflare TP3 20261116

All runs: 3 × RTX PRO 6000 Blackwell Workstation, **450 W, +75 MHz core, +6000 MHz memory**, k5 drafts, adaptive
verification alpha 0.8, interactive scheduler, tail merge on. Decode cells are `llm_decode_bench.py` (`llm-inference-bench`,
ignore-EOS, 32K context unless stated). "Pass" = one cell run; the cell length is given per section. Agent replays are
`tools/ttft_under_load.py agent --streams N --warmup 30 --duration 300`. Packages are named by their export timestamp
(`gb202-tp3-<timestamp>`). The release package is **`ds41-1017-engram-fp8@gb202-tp3-202611160815`**.

Measurements of earlier builds (power, clocks, k5 vs k7, prefix cache sizes on the old base, Engram on disk with MXFP4
tables, hot-expert counts) are in the [20261111 MEASUREMENTS.md](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/MEASUREMENTS.md).
They were taken on the old base commit and are not repeated here.

## 1. Release run (2026-11-16 15:02, qualification run)

Package 160815, release binary (with the prefill merge), `serve.sh` defaults: 16 slots, cap 32, prefix cache 12G,
stack limit 256, TTFT order shortest, prefill merge 512. Served KV **5,403,008**. One pass per cell.

| Cell | Result |
| --- | --- |
| Decode 32K c1 / c8 / c16, 30 s | **319.5 / 978.8 / 1,331.5** tok/s; steps/s 122.8 / 350.7 / 301.0; acceptance 2.60 / 2.35 / 2.21; 0 errors |
| Decode 128K c1, 30 s | **320.8** tok/s, steps/s 119.2, acceptance 2.69 |
| Decode 512K c1, 60 s | **292.7** tok/s, steps/s 112.8, acceptance 2.59 |
| Prefill 32K (`--prefill-only`, 4 samples) | **15,766** tok/s |
| Agent replay, 8 streams, 300 s | **649 turns**, 464.6 output tok/s, first token p50 / p90 / max 0.42 / 0.72 / 1.23 s, 0 errors |

The qualification compared the decode cells with the round-2 export run (section 4, older binary, KV 5,155,456,
2 passes each): c1 0.955×, c8 0.937× (flagged WARN, outside ±4% / ±2.5%), c16 0.985×. The same binary and package
gave c1 293.7-317.2 and c8 989.2-1,024.4 tok/s over four 30 s passes earlier the same day (section 6). We report
the measured values and make no speed claim at c1 or c8 against the previous local builds.

During qualification, a hardware fault on one GPU (Xid 79, the card dropped off the PCIe bus) interrupted the first
16-stream KV stress attempt after 42 seconds; the stress was rerun on the same build and settings, and its result follows.
The 16-stream KV stress on this release's
binary and package (2026-11-16 18:50): PASS, 32/32 long requests, 212/212 pings, 0 bad log lines, lowest free VRAM
64 MiB, lowest host MemAvailable 12.4 GiB, 25.6 min; see [qualification/RESULTS.md](qualification/RESULTS.md). Section
3 has the stress of the same package on the binary before the prefill merge.

## 2. Against the published 20261111 numbers

| | 20261116 | 20261111 (published) | Method difference |
| --- | --- | --- | --- |
| KV tokens | 5,403,008 | 3,014,656 | 16 vs 8 slots, cap 32 vs 24, prefix cache 12G vs 32G |
| Decode 32K c1 / c4 / c8 | 319.5 / 781.2 / 978.8 | 301 / 734 / 988 | here: one 30 s pass each (c4 from section 3); 20261111: mean of 5 alternating rounds, 60 s cells |
| Decode 32K c16 | 1,331.5 | — | 20261111 had 8 slots |
| Decode 128K / 512K c1 | 320.8 / 292.7 | 289 / 271 | one run each on both |
| Prefill 32K | 15,766 | 14,547 | here: 4 samples in one run; 20261111: graph-scratch A/B |
| Agent replay | 8 streams: 649 turns, 464.6 output tok/s | 4 streams: 519 turns | different stream count; not comparable |

## 3. Integration check of the serving binary (2026-11-16 06:14)

Binary = fork commit `174f524d` (in the history of tag `hyperflare-tp3-20261116`; frontend reference fixes,
required-thinking 400, CUDA stack-limit option, TTFT guard order), package 160815, cap 32, prefix cache 12G, `FLARE_CUDA_STACK_LIMIT=256`, TTFT order shortest; no
prefill merge. One 30 s pass per cell.

| Item | Result |
| --- | --- |
| Served KV | **5,403,008** vs 5,155,456 on the previous binary without the stack limit (+247,552, +4.8%) |
| Boot log | the 15 `[ OK ]` lines equal the previous binary's except the stack, cap and KV numbers |
| Decode 32K c1 / c4 / c8 / c16 | 322.7 / 781.2 / 1,078.8 / 1,391.6 tok/s |
| KV stress, 2 waves × 16 × ~530K unique prompts + pinger | 32/32 long requests, 207/207 pings, 0 bad log lines, lowest free VRAM 80 MiB, 25.8 min |
| hotel-lights, 30 runs at 16 streams | 30/30 exact, 0 truncated |
| Images (5 cases) after the stress | 5/5 served, 5/5 correct |
| Lowest host MemAvailable | 19.7 GiB at boot, 10.7 GiB during the stress |

## 4. Round-2 export: package 160815 vs 160407 (2026-11-16 02:41)

Same binary for both (the one before section 3), cap 32, prefix cache 12G, no stack limit. 160407 = the FP8 Engram
package with the previous code; 160815 adds the round-2 switches (index and context side streams, 768-row prefill
parts for the decoder and the Engram projection, mHC PDL, BRRR `wo_a` uneven split; Engram side stream off). Two
30 s passes per cell, compared with 160407's recorded cells.

| Cell | 160815 | 160407 | Ratio |
| --- | --- | --- | --- |
| Decode c1 | 334.6 (348.4, 320.9) | 311.1 | 1.076× (steps/s 1.007×, acceptance 2.70 vs 2.53) |
| Decode c4 | 771.8 (765.1, 778.4) | 763.4 | 1.011× |
| Decode c8 | 1,044.1 (1,060.2, 1,027.9) | 1,052.4 | 0.992× (GPU ms per graph 0.950×, fewer rows per graph) |
| Decode c12 / c16 | 1,251.3 / 1,352.2 | 1,258.1 / 1,360.6 | 0.995× / 0.994× |
| Prefill 8K / 32K | 14,365 / 15,958 | 13,738 / 15,322 | 1.046× / 1.042× |
| Agent replay 8 / 16 streams | 662 / 591 turns, 393.2 / 491.7 output tok/s | 599 / 580 turns, 453.2 / 455.0 output tok/s | one replay each |
| KV, dry run / served | 4,996,096 / 5,155,456 | 4,965,248 / 5,126,272 | +0.6% |

Export wall time 34 min with a warm tuning cache (3 compiler workers): load weights 46.5 s, graph catalog 13.3 s,
tune 200.2 s, compile 767.0 s, load and warm 205.8 s, capture 402.4 s, native core export 393.0 s. Committed host RAM
in the dry run: 217.32 GiB for both packages.

## 5. Engram side stream and export KV hygiene: package 161317 vs 160815 (2026-11-16 06:55)

161317 = 160815's settings plus the Engram side stream on layer 14 and the export KV hygiene code. Same binary as
section 4, two 30 s passes per cell.

| Cell | 161317 | 160815 | Ratio |
| --- | --- | --- | --- |
| Decode c1 / c4 / c8 | 323.4 / 763.4 / 1,064 | 334.6 / 771.8 / 1,044 | 0.967× / 0.989× / 1.019× |
| Decode c12 / c16 | 1,270.3 / 1,385.9 | 1,251.3 / 1,352.2 | 1.015× / 1.025× |
| Prefill 8K / 32K | 14,413 / 15,986 | 14,365 / 15,958 | 1.003× / 1.002× |
| KV, dry run / served | 4,996,096 / 5,155,456 | 4,996,096 / 5,155,456 | identical |
| Boot | 76 s | 51 s | slower |

Verdict: level; not adopted. The hygiene check reported one unclaimed 44 MiB span per GPU in 161317's export.

## 6. Prefill merge A/B (2026-11-16 12:52)

Release binary, package 160815, cap 32, prefix cache 12G, stack limit 256, TTFT order shortest.
`FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS` 0 vs 512, arms off / on / on / off, one boot each. Replays of 300 s; the
"mixed" workload adds chat turns to the agent turns.

| Workload | off.1 | off.2 | on.1 | on.2 | on / off: turns, output, first token p50, p90 |
| --- | --- | --- | --- | --- | --- |
| agent, 8 streams | 646 / 455.8 / 0.509 / 0.903 | 640 / 402.6 / 0.532 / 0.900 | 726 / 399.6 / 0.414 / 0.719 | 671 / 471.0 / 0.404 / 0.750 | 1.086, 1.014, 0.786, 0.815 |
| mixed, 8 streams | 663 / 508.8 / 0.303 / 0.939 | 666 / 479.6 / 0.303 / 0.995 | 762 / 492.8 / 0.245 / 0.900 | 677 / 562.8 / 0.237 / 0.898 | 1.083, 1.068, 0.796, 0.930 |
| agent, 16 streams | 592 / 454.5 / 0.533 / 1.004 | 625 / 374.6 / 0.582 / 1.068 | 698 / 497.3 / 0.426 / 0.747 | 637 / 534.1 / 0.404 / 0.714 | 1.097, 1.244, 0.745, 0.705 |
| mixed, 16 streams | 687 / 551.3 / 0.312 / 0.990 | 591 / 575.0 / 0.310 / 1.015 | 735 / 606.2 / 0.254 / 0.902 | 702 / 609.0 / 0.261 / 0.952 | 1.124, 1.079, 0.827, 0.925 |

(turns / output tok/s / first token p50 s / p90 s.) Prefill launches per turn: 3.15 → 1.00 (agent 8), 3.03 → 1.35
(mixed 8), 3.23 → 1.01 (agent 16), 3.07 → 1.36 (mixed 16).

Decode 32K, one 30 s pass per arm: c1 off 313.4 / 293.7, on 303.9 / 317.2 (1.023×); c8 off 1,024.4 / 989.2, on
1,013.6 / 1,021.2 (1.011×). 0 errors in replays and decode; `gpu_check` 13/13 on the merge arm.

The adoption rule fixed before the run asked for a work-normalised gain of at least 5% above the same-flag spread at
8 and 16 streams. At 16 streams it was met (1.157×, spread 4.6%). At 8 streams the same-flag runs spread by up to
11.8% (mixed 8 on: 762 vs 677 turns) and the work fit was not defined, so the rule said HOLD. The owner adopted the
merge on the first-token times and the 16-stream result.

## 7. FP8 Engram tables: package 160407 vs the MXFP4 package 152105 (2026-11-15 22:23)

Same code and binary for both, cap 32. 160407 = DeepSeek's FP8 Engram tables, both in host RAM, prefix cache 12G;
152105 = NVIDIA's MXFP4 tables, prefix cache 32G. Two 30 s passes per cell.

| Cell | FP8 (160407) | MXFP4 (152105) | Ratio |
| --- | --- | --- | --- |
| Decode c1 / c4 / c8 | 311.1 / 763.4 / 1,052.4 | 314.6 / 749.2 / 1,024.4 | 0.989× / 1.019× / 1.027× |
| Decode c12 / c16 | 1,258.1 / 1,360.6 | 1,234.3 / 1,375.1 | 1.019× / 0.989× |
| Agent replay 8 streams | 599 turns, 453.2 output tok/s, first token p50 / p90 0.505 / 0.907 s | 561 turns, 479.3, 0.490 / 0.879 s | output 0.946× |
| Served KV | 5,126,272 | 5,220,992 | 0.982× |
| Committed host RAM (dry run) | 217.32 GiB at 12G (237.32 GiB at 32G) | 145.72 GiB | |
| Lowest MemAvailable at boot | 21.0 GiB | | |

Third arm, `ENGRAM_RAM=layer1` (layer 1's FP8 table in RAM, layer 14's on disk, 32 GiB row cache, prefix cache 32G):
committed 177.77 GiB, lowest MemAvailable 63.0 GiB, decode c1 / c4 / c8 / c12 / c16 0.988 / 0.998 / 1.010 / 0.997 /
1.013× against 152105, agent replay output 0.821× at 8 streams and 0.990× at 16, served KV 5,111,808.

The Engram precision evidence (MXFP4 vs FP8 quality scores) is in [ENGRAM-PRECISION.md](ENGRAM-PRECISION.md).

## 8. Verification cap 24 / 32 / 48 (2026-11-15 19:01, MXFP4 package 152105)

One boot per cap, two 30 s passes per cell.

| Cap | KV served | c1 | c4 | c8 | c12 | c16 | Agent 8 streams (turns, output tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 5,257,344 | 308.7 | 756.8 | 1,045 | 1,187.8 | 1,225.9 | 592, 404.4 |
| **32** | 5,220,992 | 314.6 | 749.2 | 1,024 | 1,234.3 | **1,375.1** | 561, 479.3 |
| 48 | 5,177,344 | 312.5 | 772.7 | 1,022 | 1,223.7 | 1,300.0 | 632, 382.0 |

32 adopted for the 16-slot package: c16 1.12× over 24, c1-c8 within the noise of single runs.

## 9. Changes between 20261111 and the round-2 export (2026-11-12 to 2026-11-15)

Each measured against the build before it, in the owner's records. Short form; one line per change.

| Date | Change | Measured |
| --- | --- | --- |
| 11-13 | Prefill tail merge | prefill 8K +19%, 32K +7%; agent first-token p90 0.86 → 0.76 s (4 streams), 0.99 → 0.91 s (8); agent output +10-25% |
| 11-14 | Port onto upstream `70069281` (package 142139) | served KV 5,024,384 vs 3,189,376 (+58%) at 8 slots, cap 24; c1 301 vs 318 tok/s, c4 / c8 / prefill / agent level; consistency 0.962 vs 0.963, eval 42/43, verify 0, hotel 30/30, lavd 30/30, tools 60/60, c8 stress PASS (lowest free VRAM 80 MiB), retrieval 128K / 900K pass |
| 11-14 | Fast restart (private Engram tables, fast exit) | start 62-64 → 47-48 s, stop 16-17 → 8 s, decode level |
| 11-15 | Decode glue cleanup (package 151432) | c1 steps/s 117.7 vs 113.6 (1.035×), c8 995 vs 976 tok/s, prefill 8K / 32K 13,161 / 14,684 vs 12,752 / 14,474; dry-run KV +3.4% |
| 11-15 | Two-shot pull, parked BRRR gate lifted, attention two streams, host-row embedding gather (package 151744) | c4 753-769 vs 719-736 (+4.6-4.7%), c8 1,025-1,041 vs 991-995 (+3.4-4.6%), c1 level, prefill 32K 15,403-15,418 vs 14,661-14,684 (+5%), dry-run KV +0.9% |
| 11-15 | VRAM reclaim + 16 slots (package 152105) | c1 / c4 / c8 314 / 751 / 1,043 vs 310 / 754 / 1,041 (level), c12 / c16 1,204 / 1,211 at cap 24; dry-run KV +2.0%, served 5,257,344 (+2.7%) |
