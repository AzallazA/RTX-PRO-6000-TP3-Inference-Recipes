# Measurements: Hyperflare TP3 20261111

This release's measurements come first (graph scratch, TTFT guard, BRRR parked row limit, release A/B and full
qualification). Everything from "Power and clocks" down is carried over from the 20261108 folder and was measured on
the 20261107/20261108 packages (`...070243`, 080909, 081024) before graph scratch; those settings did not change.
450 W, +75 MHz core / +6000 MHz memory offset, k5, alpha 0.8, verification row cap 24, 32G prefix cache unless noted.

| | |
| --- | --- |
| Package | `ds41-1017-engram-mxfp4@gb202-tp3-202611111227` (fork tag `hyperflare-tp3-20261111`) |
| Build | 342 hot experts per parked layer (20-33), merged VRAM/host expert bank, BRRR fused layer tail on all 40 layers (parked layers only in graphs of 12 rows or fewer), graph scratch (Engram chunk rows 1024, shared Engram residual), token embedding and vision tower in host RAM (vision staged), all-reduce sized for 1024 rows, MXFP4 Engram, prefill chunk 4096, 8 slots, 5 DSpark drafts |
| KV cache | 3,014,656 served (`flare run --dry-run`: 3,007,360; the same as graph scratch package 090700) at cap 24, 32G prefix cache |
| Serve flags | `--mtp-draft-tokens 5 --mtp-adaptive-verification --mtp-max-verification-token-budget 24 --mtp-adaptive-verification-alpha 0.8 --scheduler-policy interactive --prefix-cache-memory 32G`, `FLARE_ENGRAM_RAM_TABLES=0` |

## 20261111: graph scratch (2026-11-09, 4 alternating rounds, balanced scheduler)

Package 090700 (graph scratch) against 081024 (20261108.1), same session; raw files in
`qualification/results/graph-scratch-ab-20261109/`. Decode `llm_decode_bench.py --concurrency 1,4,8 --contexts 32k
--duration 60`; prefill = client tok/s of the bench's prefill phase.

| | 081024 (20261108.1) | 090700 (graph scratch) | ratio |
| --- | --- | --- | --- |
| KV tokens (served, cap 24, 32G) | 2,467,200 | 3,014,656 | +22.2% |
| KV tokens (`flare run --dry-run`) | 2,461,184 | 3,007,360 | the dry run reads 0.2% low for both |
| Decode c1 (r1-r4) | 306 / 306 / 299 / 307, mean 304.5 | 305 / 298 / 306 / 299, mean 302.0 | 0.992 |
| Decode c4 | 745 / 726 / 722 / 732, mean 731.3 | 733 / 724 / 741 / 743, mean 735.3 | 1.005 |
| Decode c8 | 1009 / 976 / 991 / 989, mean 991.3 | 997 / 990 / 979 / 980, mean 986.5 | 0.995 |
| Prefill 32K (4 rounds) | 14,763 (14,678-14,883) | 14,547 (14,529-14,589) | 0.985 |
| Prefill 128K (4 rounds) | 14,541 (14,398-14,713) | 14,448 (14,347-14,537) | 0.994 |
| 6,000-token prompt, greedy 16 tokens | — | same text, top-1 16/16, max abs logprob difference 0.00006 (floor 0.00002) | |

Quick checks on 090700 (balanced): corruption check 0 true (10 runs), eval 42/43, tools 60/60, lavd 30/30,
hotel-lights 29/30 (one run answered 49 instead of 48), c8 KV stress PASS (16/16, pinger 115/115).

## 20261111: TTFT guard (2026-11-10, package 090700, 3 alternating rounds per configuration)

`tools/ttft_under_load.py`: `probe` = N decoding streams (2K-token prompts, ignore EOS) plus a new prompt every 3 ± 1 s
(shared cached system prefix + a unique ~1.9K-token user text, 16 output tokens, thinking off), 120 s; `agent` = 4
streams replaying turns (system prefix + growing history + a new ~1.9K-token tool result, ≤600 output tokens), 300 s
after a 30 s warm-up. Medians over 3 rounds; raw files in `qualification/results/ttft-ab-20261110/`.

| metric (median of 3) | balanced (old default) | throughput | interactive (new default) |
| --- | --- | --- | --- |
| probe, 3 decoding streams: TTFT p50 / p90 (s) | 2.04 / 2.54 | 1.55 / 1.93 | **0.36 / 0.37** |
| probe, 7 decoding streams: TTFT p50 / p90 (s) | 4.00 / 5.11 | 2.83 / 3.86 | **0.38 / 0.38** |
| decoder tok/s between probes, 3 / 7 streams | 715 / 1,190 | 722 / 1,150 | 730 / 1,180 |
| largest decoder pause, 3 / 7 streams (s) | 0.25 / 0.26 | 0.45 / 0.26 | 0.31 / 0.30 |
| agent replay, 4 streams: turns in 300 s | 448 | 474 | **519** |
| agent replay: turn TTFT p90 (s) / output tok/s | 1.97 / 271 | 1.94 / 246 | **0.93 / 294** |
| decode bench 32K c1 / c4 / c8 (tok/s) | 302 / 713 / 942 | 291 / 696 / 985 | 300 / 708 / 965 |

The new binary with the guard off measured the same as the old binary (c1 / c4 / c8 298 / 703 / 969; probe p50 2.08 /
3.99 s). Quick checks with `interactive` (`qualification/results/ttft-quality-20261110/`): corruption check 0 true,
eval 42/43, tools 60/60, c8 KV stress PASS (16/16, pinger 67/67, 9.5M prompt tokens, min free VRAM 60 MiB).

## 20261111: BRRR parked row limit 12 vs 24 (release chain stage A, 2026-11-11)

Two exports from the same code (fork `ds41-tp3-v3` at the commit this release is tagged from, `export.sh` defaults,
HOT 342, MAXC 8), differing only in `PARKED_ROWS`: **12** = package `202611111227` (the release package), **24** =
`202611111254` (dry run 3,029,120 KV tokens against 3,007,360; both serve 3,014,656 at cap 24 and 32G prefix cache).
Served with `serve.sh` settings (interactive scheduler, alpha 0.8, cap 24, 32G), 450 W, alternating rounds, 32K decode
`llm_decode_bench.py --concurrency 1,4,8 --contexts 32k --duration 60` (ignore EOS); raw files in
`qualification/results/release-ab/`.

| round | 12: c1 | c4 | c8 | 24: c1 | c4 | c8 | c8 ratio 24/12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | 305 | 739 | 1,001 | 312 | 723 | 1,028 | 1.027 |
| r2 | 305 | loop | 994 | 301 | 731 | 1,010 | 1.016 |
| r3 | 306 | 731 | 982 | 312 | 738 | 974 | 0.992 |
| r4 | 286 | loop | 977 | 303 | 731 | 1,007 | 1.031 |
| r5 | 304 | 733 | 987 | 307 | 745 | loop | — |
| **mean of completed cells** | **301.2** | **734.3** | **988.2** | 307.0 | 733.6 | 1,004.8 | 1.017 |

| long context, c1, 1 run | 12 | 24 | ratio |
| --- | --- | --- | --- |
| 128K | 289 | 284 | 0.983 |
| 512K | 271 | 263 | 0.970 |

"loop" = the cell was stopped by `llm_decode_bench`'s exact-loop guard: with `ignore_eos` the model is forced past its
end token and can fall into a repeat (see the README's Known limitations); the natural-EOS bench of the full
qualification shows 0 loops. The adoption gate for 24 was c8 ≥ +1% in every round, c1/c4 ≥ 0.99×, 128K/512K c1 ≥ 0.98×.
The chain counted only rounds with all six cells (2 of 5) and kept 12 for that reason; on the completed cells 24 also
misses the gate itself (c8 −0.8% in r3, c4 0.98× in r1, 512K 0.97×). **Verdict: 12 stays the default**; 24 is +1.7% at
c8 on average, within the 0.5-1% export-to-export plus ~2% session variance. Context: on the 16-slot package the limit
24 measured c16 1,155 vs 1,117 tok/s (+3.4%, 4/4 rounds), c1-c8 level (0.996 / 1.002 / 1.006×).

## 20261111: full qualification (2026-11-11, package 202611111227)

Gates = the 20261108.1 results (package 081024) with the tolerances in `qualification/RESULTS.md`; served KV 3,014,656
(20261108.1: 2,467,200). Raw files in `qualification/results/run-hyperflare-c8/` (corruption check:
`release-ab/verify.log`).

| check | 20261108.1 | this release | gate |
| --- | --- | --- | --- |
| corruption check, 10 runs, true corruptions | 0 | 0 (14 natural-uncertainty flags) | PASS |
| eval-43 (thinking off) | 43/43 | 42/43 | PASS (≥ 42) |
| tool calls, 60 at c8 | 60/60 | 60/60 | PASS |
| lavd, 30 runs at c8 | 30/30 | 30/30 (0 wrong, 0 truncated) | PASS |
| hotel-lights, 30 runs at c8 | 30/30 | 30/30 (0 wrong, 0 truncated) | PASS |
| long retrieval 128K / 900K | pass / pass | pass / pass | PASS |
| long generation after 581K tokens, 4 seeds | 4/4, worst 8-gram 0.0008 | 4/4, worst 8-gram 0.0013, natural stop | PASS |
| natural-EOS decode 32K c1 / c4 / c8 (tok/s) | 290 / 717 / 963, 0 loops | 300 / 713 / 950, 0 loops | PASS (≥ 0.97× each) |
| LiveBench coding (2024-11-25) | 81.6 | 83.5 (LCB_generation 73.077, coding_completion 94.0) | PASS (≥ 80.5) |
| c8 KV stress (2 waves × 8 × ~530K + pinger) | PASS 16/16, pinger 117/117 | PASS 16/16, pinger 68/68, 9,511,557 prompt tokens, min free VRAM 82 MiB | PASS |

All gates pass.

---

# Earlier measurements (carried over from the 20261108 folder)

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
exactly. It also shows that the 2,373,760 → 2,461,184 KV change (~87K tokens) between the 16G and 32G configurations was the verification
row cap (24 drops 13 speculative graphs, ~76 MiB per rank), not the prefix cache (which costs ~10 MiB of device headroom).
The dry run allocates no GPU memory and creates no CUDA context, but needs the NVIDIA driver and the GPUs visible.

### Qualification, 2026-11-08 (c8, both packages)

See the README's "Qualification of this folder's packages" table; raw results under `qualification/results/run-20261108-c8/`
and `run-20261108.1-c8/`. Export-to-export variance measured the same day (identical code re-exported: c1/c4/c8 298/731/988
vs 302/726/983 over 4 rounds) and session drift (~2% at c8) are recorded in the fork handbook, section D.1.
