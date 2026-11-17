# Qualification results — Hyperflare TP3 20261116

Package `ds41-1017-engram-fp8@gb202-tp3-202611160815`, run on the live server without restarts (the stress rerun on a new start of the same binary and settings), 450 W, served exactly as
`../serve.sh` serves it (16 slots, cap 32, prefix cache 12G, alpha 0.8, interactive scheduler, TTFT order shortest,
prefill merge 512, stack limit 256). Served KV 5,403,008 tokens. Run of 2026-11-16 from 15:02; the 16-stream KV
stress, post-run checks and log scan from 18:50 (rerun, same binary, package and settings). Raw outputs:
`results/qual-20261116/` and `results/stress-rerun-20261116/`; the prefill-merge A/B behind the serve default:
`results/prefill-merge-ab-20261116/`. Binary: sha256 `27864a1d79e4…` for every stage. A rebuild from the tree of fork
commit `51f3df1f` gave sha256 `874eb8031a8e…` (the build embeds its source path, so the file differs); the tag
`hyperflare-tp3-20261116` (`f4ce2830`) differs from `51f3df1f` by documentation only.

## Scope: minimum tier for a weights change

This release was **not** run through the full suite of earlier releases. The owner's tiered policy (2026-11-16) picks
stages by what changed:

| Tier | Adds | When |
| --- | --- | --- |
| binary | GPU kernel checks, smoke, images, tool calls (60), decode 32K c1/c8/c16, 8-stream agent replay, hotel-lights at 16 streams, 16-stream KV stress, post-run smoke and log scan | any change |
| export | consistency, corruption check, eval-43, tool-eval-bench, decode 128K c1, prefill 32K | a new export |
| weights | GPQA Diamond (one sample of 198) | a quantisation change (here: FP8 Engram tables) |
| full | 512K c1, long retrieval, long generation, lavd, EOS bench, LiveBench coding, 128K c8 | before a public release under the earlier policy |

This run: **tier weights plus long retrieval and the 512K cell**. The tier was chosen because the FP8 tables are a
quantisation change.

**Not run:** LiveBench coding, long generation (581K-token conversation), lavd, the natural-EOS decode bench, the
128K c8 decode cell, the 60-image vision sets, and the 8-stream KV stress. The 8-stream stress is in no tier; the
16-stream stress covers the same failure mode at higher load. The stages left out have not failed on this model in
our records: lavd always 30/30, long generation always 4/4, EOS 0 loops on natural EOS, LiveBench 81.3-84.2. That is
the reason for the cut, not a result for this build.

## Results

| Stage | Result | Gate | Reference |
| --- | --- | --- | --- |
| GPU kernel checks (`gpu_check`) | 13/13 | 13/13 | — |
| Smoke: thinking off / on, tool-call smoke | 2/2, 3/3 | all | — |
| Images: small, 1920×900, over-size 4160×1280, two images, four images | 5/5 served, 5/5 correct | 5/5 | — |
| Tool calls: 60 tool-requiring prompts at 8 streams | **60/60** | 60/60 | 20261111: 60/60 |
| Decode vs prefill consistency (`tools/consistency.py 96 6`) | 284 positions, top-1 agreement **0.951**, missing 0, median Δlogprob 0.0081 | ≥ 0.95, ≤ 1, ≤ 0.01 | 0.962 on the 2026-11-14 port package |
| Corruption check (`tools/verify_fix.py 10`) | **0** true corruptions (0 retried, 0 skipped) | 0 | 0 |
| eval-43 (thinking off; temperature clamped to 1.0) | **42/43** | ≥ 42 | 20261111: 42/43 |
| tool-eval-bench (92 scenarios; TC-45 excluded, 91 graded) | **90/100**, 77/91 pass | ≥ 88 | 20261111 package: 91 (78/91) |
| Decode 32K c1 / c8 / c16, 30 s, one pass | 319.5 / 978.8 / 1,331.5 tok/s, 0 errors | notes only (WARN below band) | WARN at c1 and c8 against an earlier two-pass run on the older binary (334.6 / 1,044.1); see below |
| Decode 128K c1, 30 s | 320.8 tok/s | notes | — |
| Decode 512K c1, 60 s | 292.7 tok/s | notes | 20261111: 271 |
| Prefill 32K, 4 samples | 15,766 tok/s | notes | 20261111: 14,547 |
| Agent replay, 8 streams, 300 s | 649 turns, 464.6 output tok/s, first token p50 / p90 0.42 / 0.72 s, 0 errors | 0 turn errors | — |
| Long retrieval: three markers at 128K and 900K tokens | **pass / pass** | pass / pass | 20261111: pass / pass |
| GPQA Diamond, one sample of 198, 8 streams, 32K tokens | **91.4%** (181/198, Wilson 95% 86.7-94.6), 0 errors; **WARN**: 4 truncated without an answer, 7 hit max_tokens; paired vs baseline 5 gained / 3 lost, McNemar p 0.727 | rc 0, 198 scored, 0 errors; WARN below 85.5% or > 3 truncated | 20261111 package: 90.4% (85.5-93.8) |
| hotel-lights, 30 runs at 16 streams | **30/30** exact, 0 wrong, 0 truncated | 30/30 (29 = WARN), 0 truncated | 20261111: 30/30 at 8 streams |
| KV stress: 2 waves × 16 × ~530K unique prompts + pinger, KV oversubscribed | **PASS** (rerun): 32/32 long requests, 212/212 pings, 0 bad log lines, lowest free VRAM 64 MiB (≥ 16 MiB gate, below the kit's 480 MiB margin), 19.0M prompt tokens, 25.6 min | all long requests and pings ok, 0 bad log lines, free VRAM ≥ 16 MiB | same package on the binary before the prefill merge (integration check): 32/32, 207/207, lowest free VRAM 80 MiB |
| Post-run smoke, served KV unchanged, corruption check | **PASS** (rerun): smoke 2/2, served KV 5,403,008 unchanged, 0 true corruptions in 10 runs | all | — |
| Server and kernel log scan | **PASS** (rerun): 0 fault lines in the server log, 0 in the kernel log | no fault line | — |

Overall: every stage passed or warned as listed above. During qualification, a hardware fault on one GPU (Xid 79, the card dropped off the PCIe bus) interrupted the first
16-stream KV stress attempt after 42 seconds; the stress was rerun on the same build and settings, and its result follows. The interrupted attempt is in
`results/qual-20261116/` (`GATES.tsv` and `verdict.stress16`: `stress16 FAIL`, "no result"; `CHAIN.md`: "CRASH after
stress16"; post and scan `NOTRUN`): it produced no result in either direction. The machine was power-cycled. The rerun at 18:50 on the same binary (sha256 `27864a1d79e4…`, the binary every stage of this file ran on; its Rust
source equals the tag's), package and settings passed the stress, the post-run checks and the log scan (`results/stress-rerun-20261116/`). Lowest free VRAM in the rerun: 64 MiB, above
the gate's 16 MiB floor and below the kit's 480 MiB comfort margin (the KV cache takes the VRAM the graphs leave).
2 waves, 19.0M prompt tokens.

**Decode WARN.** The decode stage compares single 30 s passes with a reference recorded on the round-2 export run
(older binary without the stack limit, KV 5,155,456, two passes each). It flagged c1 at 0.955× and c8 at 0.937×.
The same binary and package gave c1 293.7-317.2 and c8 989.2-1,024.4 tok/s over four passes earlier the same day.
We report the values as measured and claim no change at c1 or c8 against the previous local builds.

**tool-eval TC-45.** The harness probes `tool_choice: "required"` once in Hyperflare's default thinking mode. The
server now answers HTTP 400, as DeepSeek's reference does (was HTTP 500), so the harness excludes TC-45 in both cases
and grades 91 scenarios. The score compares directly with the 91-scenario baseline.

## Kit

The scripts here are the kit this run used: the 20261111 kit plus `qual_core.sh` (stage gates in `GATES.tsv`,
GPQA Diamond and tool-eval-bench stages, stage reuse) and the 16-stream KV stress (`stress_kv.py` takes a per-request
timeout, `STRESS_LONG_TIMEOUT_S`, default 1800 s). The tiered wrapper that ran the decode, prefill, agent, image and
consistency stages around this kit is local to our test machine and is not shipped. Every one of those stages is one
command:

| Stage | Command (server on port 15015) |
| --- | --- |
| Decode 32K | one 1,024-token warm-up chat request, then per cell `llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash --concurrency <1, 8, 16> --contexts 32k --duration 30 --skip-prefill` |
| Decode 128K / 512K | `... --concurrency 1 --contexts 128k --duration 30 --skip-prefill` / `... --concurrency 1 --contexts 512k --duration 60 --skip-prefill` |
| Prefill 32K | `... --prefill-only --prefill-contexts 32k --prefill-duration 10` |
| tool-eval-bench | `... --tool-eval` |
| GPQA Diamond | `... --test-profile gpqa-diamond --profile-concurrency 8 --max-tokens 32768 --display-mode plain` |
| Agent replay | `python3 ../tools/ttft_under_load.py agent --streams 8 --warmup 30 --duration 300` |
| Consistency | `<fork>/.venv/bin/python ../tools/consistency.py 96 6 <checkpoint view>` (needs `transformers`) |
| Corruption check | `python3 ../tools/verify_fix.py 10` |

Running the kit: start the server, then `FLARE_LOG=<server log> CKPT=<checkpoint view> ./qual_all.sh` (or name
stages: `./qual_all.sh eval tools`). This run called it once per stage (`eval`, `tools`, `retrieval`, `hotel`,
`stress`) with `QUAL_PROFILE=full QC=16 STRESS_LONG_TIMEOUT_S=3600`; the tool-call test runs its 60 prompts at 8 streams. `QUAL_PROFILE=core` selects the reduced stage list in `qual_all.sh`'s header.
Every output lands in a fresh `run-<date>/` directory next to the script (`QUAL_OUT` overrides it); the shipped
`results/` and this file are never written. LiveBench and `llm-inference-bench` checkouts (each with its own `.venv`)
are expected in `~/LiveBench` and `~/llm-inference-bench`; `LIVEBENCH_DIR` and `LLM_BENCH_DIR` override the paths.
