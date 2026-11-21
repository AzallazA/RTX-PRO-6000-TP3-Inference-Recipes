# Measurements: Hyperflare TP3 20261121

All runs: 3 × RTX PRO 6000 Blackwell Workstation, **450 W, +75 MHz core, +6000 MHz memory**, k5 drafts, adaptive
verification alpha 0.8, cap 32, 16 slots, prefix cache 12G, interactive scheduler, tail merge on, TTFT order shortest,
prefill merge 512, stack limit 256, both FP8 Engram tables in host RAM. Packages are named by their export timestamp
(`gb202-tp3-<timestamp>`). The release package is **`ds41-1017-engram-fp8@gb202-tp3-202611201530`** (served KV
5,687,040); the release binary is `steersync-a7c3a04a` (commit `a7c3a04a`, sha256 `af06512fa11a…`).

Most A/B runs in this series compare **GPU time per decode round** ("corrected ms/round": GPU milliseconds per
verification round at equal offered drafts, which removes the scheduler's draft-count choices) over 3 alternating
fresh-boot rounds per arm, plus raw round time, tokens per round and tok/s. Prefill cells are
`llm_decode_bench.py --prefill-only`. Earlier builds' measurements are in the
[20261116 MEASUREMENTS.md](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/MEASUREMENTS.md).

## 1. Owner speed table (release package and binary)

2026-11-20, final default (package 201530 on `steersync-a7c3a04a`, read-ahead on). `llm_decode_bench.py` from
llm-inference-bench v0.9.1 plus the steps patch in `tools/` (the result files record bench version 0.8.0). Decode:
30 s cells, `--skip-prefill`, c1 / c4 / c8 / c16 in one run, three separate runs per context; prefill:
`--prefill-only` (cold prompts), three runs. Each cell is the **median of the three runs** (steps/s and acceptance are
medians of their own); ranges are the lowest and highest run.

| Cell | Result |
| --- | --- |
| Decode 32K c1 / c4 / c8 / c16 | **319.9 / 787.2 / 1,068.2 / 1,464.7** tok/s (ranges 310.7-323.7 / 772.6-809.9 / 1,058.8-1,101.5 / 1,459.0-1,466.6); steps/s 129.8 / 320.9 / 430.9 / 503.4; acceptance 2.46 / 2.44 / 2.42 / 2.22 |
| Decode 128K c1 / c4 / c8 / c16 | **322.8 / 764.2 / 1,040.0 / 1,399.1** tok/s (ranges 320.6-323.3 / 758.4-774.0 / 1,033.3-1,063.0 / 1,395.9-1,399.9); steps/s 127.5 / 322.2 / 423.7 / 440.0; acceptance 2.53 / 2.29 / 2.31 / 2.20 |
| Decode 512K c1 | **306.5** tok/s (301.1-308.6); steps/s 120.5; acceptance 2.56 |
| Prefill 8K / 32K / 128K / 256K / 512K / 1M (cold) | **16,567 / 18,300 / 18,938 / 18,548 / 17,516 / 15,935** tok/s (ranges 16,527-16,603 / 18,269-18,336 / 18,791-19,128 / 18,412-18,612 / 17,481-17,671 / 15,840-16,023) |
| Agent replay, 8 streams, 300 s (qualification run) | 658 turns, 530.8 output tok/s, first token p50 / p90 / max 0.395 / 0.627 / 1.02 s, 0 errors |

## 2. Against the published 20261116 numbers

| | this build | 20261116 (published) | Method difference |
| --- | --- | --- | --- |
| KV tokens | 5,687,040 | 5,403,008 | +284,032 (+5.3%), same slots, cap and prefix cache |
| Decode 32K c1 / c4 / c8 / c16 | 319.9 / 787.2 / 1,068.2 / 1,464.7 | 319.5 / 781.2 / 978.8 / 1,331.5 | level / +0.8% / +9.1% / +10.0%. 20261116: single 30 s passes (c4 from its integration check); here: median of 3 runs |
| Decode 128K / 512K c1 | 322.8 / 306.5 | 320.8 / 292.7 | level / +4.7% |
| Prefill 32K | 18,300 | 15,766 | +16.1%; 20261116: one run of 4 samples, here: median of 3 runs |
| Agent replay, 8 streams | 658 turns, 530.8 output tok/s | 649 turns, 464.6 | same tool and settings |
| Boot | ~51 s | ~55 s | |

## 3. Steering: package 201530 (scale 0 and 1) vs the same export without the sidecar (2026-11-20 15:01)

A = package 192051 on the previous serving binary (upstream picks); B = 201530 on the steering binary (the same Rust
code plus steering, without the upstream picks; those are front-end only). Three rounds per arm, A B A B A B, fresh
boot each. The c8 scale-0 flag below is superseded by the release binary check in section 6 (one binary, final pair). B ran each cell at scale 0 and at scale 1. KV A 5,679,744, B 5,687,040.

| Cell | A ms/round raw / corrected | B scale 0 | B scale 1 | B0 − A corrected | B1 / B0 raw | tokens/round A / B0 / B1 |
| --- | --- | --- | --- | --- | --- | --- |
| c1 32K | 7.982 / 6.256 | 7.750 / 6.260 | 7.826 / 6.207 | +0.004 | 1.010 | 2.62 / 2.55 / 2.53 |
| c4 32K | 12.210 / 7.776 | 12.192 / 7.800 | 12.119 / 7.755 | +0.025 | 0.994 | 9.56 / 9.69 / 9.49 |
| c8 32K | 17.194 / 8.242 | 17.275 / 8.481 | 16.554 / 8.617 | +0.239 | 0.958 | 18.95 / 18.81 / 17.78 |
| c16 32K | 19.549 / 11.748 | 19.527 / 11.675 | 19.467 / 11.532 | −0.073 | 0.997 | 28.90 / 29.02 / 28.77 |

| Prefill | A tok/s | B scale 0 | B scale 1 | B0 / A | B1 / B0 |
| --- | --- | --- | --- | --- | --- |
| 32K | 17,835 | 17,811 | 18,038 | 0.999 | 1.013 |
| 128K | 18,251 | 18,222 | 18,197 | 0.998 | 0.999 |
| 512K | 17,169 | 17,163 | 17,125 | 1.000 | 0.998 |

Tokens per round, scale 1 / scale 0 (run summary): 0.994 / 0.979 / 0.946 / 0.992 at c1 / c4 / c8 / c16. Combined with
the raw round time, scale 1 delivers about 0.98-1.00× the output rate of scale 0 at 32K (derived: tokens per round ÷
raw round time = 0.984 / 0.985 / 0.987 / 0.995). The analyzer flagged scale 0 vs A at c8 (+0.239 ms corrected; raw
round time +0.5%); A and B ran on different binaries. The release binary check (section 6) measured the final pair
directly: c8 −2.4%, c1 +1.7% (level). Exactness: long-prompt top-1 agreement B vs A 1.000 at 32K and
128K, as A vs A (PASS); the short-prompt byte check was void because A differed from itself (run-to-run
nondeterminism). Proxy and server errors 0. Lowest host MemAvailable 15.0 GiB.

**Package and correctness checks of 201530** (2026-11-20 09:56; export 35 min from the 192051 recipe plus the sidecar,
same tuning plan): served KV 5,687,040 (+7,296 vs 192051; dry run +14,592), smoke, eval-43 43/43, needles 64K / 128K
pass, corruption check 0 in 10 runs, prefix isolation (switching scale 0 → 1 reuses 0 cached tokens; the same scale
again reuses all 3,429), mixed batches: c8 first tokens equal to the homogeneous run of their scale 8/8, c16 15/16 (the
one lane also differs from its own homogeneous repeat).

**Kernel tests of the in-kernel steering** (2026-11-20 09:39, test model with 24 layers): BRRR tail tests 5/5; with
deterministic ordering, scale 0 is bitwise equal to the plain kernel in 48/48 cases (repeat, scale 0, mixed rows);
steered outputs within the reference bound (worst error / bound 0.812).

## 4. Refusal set and GPQA at scale 0 (2026-11-20, package 201530)

| Check | Scale 0 (omitted) | Scale 1.0 |
| --- | --- | --- |
| 30 prompts the stock model tends to refuse: refused (+ partial) | 24/30 (+3) | 3/30 (+4) |
| 30 benign prompts: refused | 0/30 | 0/30 |
| GPQA Diamond, one sample of 198, 8 streams, 32K tokens | **91.4%** (181/198, Wilson 95% 86.7-94.6), 5 truncated without an answer, 11 hit max_tokens, 0 errors | 89.4% (177/198), 3 truncated; paired vs scale 0: 4 questions wrong only at scale 1, 0 only at scale 0, exact McNemar p 0.125 |

Refusals are scored by a keyword classifier; the prompt texts are not published.

## 5. Serving binary: upstream picks (2026-11-20 13:06)

Package 192051; A = previous binary, B = binary with the upstream picks. Three rounds each, corrected GPU ms per
round: c1 32K 6.346 vs worst A round 6.196 (+2.4%; one B round 6.774, the other two 6.129 / 6.149), c8 32K 8.442 vs
8.451 (−0.1%). Checks: KV 5,679,744 unchanged, image 5/5, 16-bit PNG (gray, RGB) and GIF served, animated GIF refused
with HTTP 400, tool calls 5/5 (OpenAI, OpenAI streaming, Anthropic `tool_use`, parallel off = one call, 1.1 MiB
history), eval-43 42/43, corruption check 0, 8-stream KV stress 16/16 long, 88/88 pings, lowest free VRAM 61 / 80 / 62
MiB (GPU 0 / 1 / 2), log scan clean.

## 6. Release binary check (steering + access gate + upstream picks) with package 201530

2026-11-20 19:51, binary `steersync-80764e80` (steering, access gate, upstream picks) with package 201530, served by
the serve script with only the package and binary changed: boot 52 s, served KV 5,687,040; smoke 2/2; images 5/5;
16-bit PNG (gray, RGB) and GIF served, animated GIF refused (4/4); tool-call checks 5/5; eval-43 43/43 with the 32K
needle; corruption check 0 in 10 runs; refusal set trigger 23/30 refused at scale 0 → 2/30 at scale 1, benign 0/30 at
both, 0 errors; prefix isolation PASS; mixed batches c8 8/8; 8-stream KV stress 16/16 long, 88/88 pings, lowest free
VRAM 78 MiB on each GPU, 0 bad lines; log scans clean, no OOM kill. Access gate (second boot with a per-run key file):
403 in 7/7 refused cases, 200 with the right key (`X-Steering-Key`, Bearer, streaming), scale 0 / omitted without a
key 200; the key in no log, argument or environment. Decode vs the previous default (192051 on the upstream-pick
binary), three fresh-boot rounds each, corrected GPU ms per round: c1 32K 6.236 vs worst A round 6.133 (+1.7%), c8
32K 8.070 vs 8.267 (−2.4%): level. After the switch: read-ahead on and inert (no thread, no metric line), steering
moves a trigger prompt.

**Final binary `steersync-a7c3a04a`** (2026-11-20 20:49) = the above without the command-line key option (keys only
from a file or the environment): boot 52 s, KV 5,687,040, smoke 2/2, corruption check 0, the old option rejected as an
unknown argument, steering moves the trigger prompt, access gate 7/7 as above, read-ahead inert, log scans clean. No
decode code changed.

## 7. Engram read-ahead (2026-11-18 / 11-19, package 182306, read-ahead build)

Cold cells start with an empty row cache: one random-token 128K prompt, one text 128K prompt, then a re-read; agent
replay 120 s; decode c1 / c4 / c8 32K, 60-90 s cells. A = both tables in RAM.

| Setup | Read-ahead off | Read-ahead on | On vs off |
| --- | --- | --- | --- |
| Both in RAM (A) | random 7.53 s, text 7.14 s; c1 / c8 330 / 1,070 | c1 / c8 326 / 1,115; no read-ahead metric line | inert (guard PASS) |
| Layer 14 on disk, 32 GiB cache | random 11.15 / 11.24 s, text 8.36 / 8.41 s | random 8.87 / 8.84 s, text 7.31 / 7.36 s | random +20.4 / +21.4%, text +12.6 / +12.5%; decode c1 −2.8%, c8 +1.6% |
| Layer 14 on disk, 2 GiB cache (filled first) | random 11.01 s, text 8.78 s | random 8.90 s, text 7.40 s | random +19.2%, text +15.7% |
| Layer 14 on disk, 8 GiB cache (filled) | random 10.48 / 10.61 s, text 8.37 / 8.41 s, 32K code 1.81 s | random 8.53 / 8.52 s, text 7.45 / 7.43 s | random +18.6%, text +11.0%, code −0.7%; decode c1 / c4 / c8 1.008 / 1.014 / 0.992× |
| Both on disk, 32 GiB cache | random 16.63 s, text 13.83 s, code 2.99 s; c1 300 | random 10.34 s, text 7.94 s, code 2.23 s; c1 318 | random +37.9%, text +42.6%, code +25.4%; c1 1.059×; agent output 1.105× |

Rows read ahead were used (useful share 1.00, extra reads 0.01, 0 errors) in every pair. On the release binary the
switch was checked on with both tables in RAM: no read-ahead thread, no metric line, KV unchanged.

## 8. PCIe microbenchmarks (issue #348; 2026-11-20 11:37, no model loaded)

| Pattern (256 MiB per transfer) | GB/s |
| --- | --- |
| Every GPU reads from both peers at once (SM loads), per GPU | 42.2 |
| Every GPU pushes to both peers at once (SM stores), per GPU | 2.6 |
| Single peer read | 50.5 |
| Copy-engine ring, per hop | 55.6 |

b12x all-reduce, best pull vs best push (µs): 1 × 5120 rows 5 / 6, 16 × 5120 12 / 41, 48 × 5120 23 / 104,
240 × 5120 86 / 160, 1024 × 5120 436 / 823; one 4096-row prefill chunk (5 × 768 + 256-row parts) 1,412 / 1,834.
All cells verified (409/409 bit-exact; 208/208 within tolerance).

## 9. Changes between 20261116 and the steering package (2026-11-17 to 11-20)

Each measured against the build before it. Short form; one line per change.

| Date | Change (package) | Measured |
| --- | --- | --- |
| 11-18 | Blended expert order, 355 hot in 20-39, vision arena host, shared operands, Markov in VRAM (180810) | KV 5,548,672 (+2.7%); tok/s c1 / c4 / c8 / c16 0.935 / 1.006 / 1.011 / 1.024× vs 160815; qualification PASS (hotel 30/30, 16-stream stress 32/32, lowest free VRAM 53 MiB) |
| 11-18 | Encoder-only prefill, pad-once, glue pass 2, Markov rows staged from host, faster export (181447) | prefill 8K / 32K / 128K 1.039 / 1.071 / 1.112×; tok/s c1 / c4 / c8 / c16 1.009 / 1.010 / 1.013 / 0.983×; 60/60 tools, consistency 0.965, hotel 30/30, stress 32/32 (80 MiB) |
| 11-18 | Exporter scratch fix (181718) | KV 5,614,208 (+65,536); stress 32/32, 78 MiB free on each GPU |
| 11-18 | Pinned q projection, rotary once, merged logprob exchange (182306) | corrected ms/round c1 32K −0.068, c1 128K −0.124, c16 32K −0.238; KV same; stress PASS (76 MiB) |
| 11-19 | Attention split rules + H24 kernel (191126) | c8 32K −0.585 ms (−6.8%), c16 32K −0.670 (−5.5%), c1 −0.008, c4 −0.103; KV 5,687,040; eval-43 42/43, consistency 0.979; stress PASS (60-78 MiB) |
| 11-19 | Long-context indexer row split (191618) | prefill 8K / 32K / 128K / 512K / 1M 1.000 / 1.000 / 1.030 / 1.154 / 1.289×; c8 −0.377 ms; KV 5,679,744; retrieval 500K / 750K / 1M pass |
| 11-19 | DCP gather pull + prefill shared rotary (192051) | prefill 128K / 512K / 1M 1.020 / 1.031 / 1.035×; append 4096 at 512K 379 → 358 ms; c4 32K +0.121 ms (+0.99% raw) in a 3-round rerun; c8 −0.336, c16 −0.196; consistency 0.965, eval-43 43/43; stress 32/32 (47-80 MiB); retrieval 500K / 750K / 1M pass |
| 11-19 | Binary with Engram read-ahead (off) | corrected ms/round c1 −0.7%, c8 −4.2% against the previous binary's worst round (3 rounds each); 8-stream stress 16/16; read-ahead inert off and on |
| 11-20 | Binary with upstream picks | section 5 |
| 11-20 | Steering package 201530 | sections 3 and 4 |
| 11-20 | Release binaries (steering + access gate + upstream picks; then without the command-line key option) | section 6 |
