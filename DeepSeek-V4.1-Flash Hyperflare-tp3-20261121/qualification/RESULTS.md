# Qualification results — Hyperflare TP3 20261121

Package `ds41-1017-engram-fp8@gb202-tp3-202611201530` (the 192051 export recipe plus the WARMIND v1.1 steering
sidecar), binary `steersync-a7c3a04a` (commit `a7c3a04a`, sha256 `af06512fa11a…`; the tag commit `5cf7e601` = `a7c3a04a` + script defaults and comments, runtime code identical), 450 W, served exactly as `../serve.sh`
serves it (16 slots, cap 32, prefix cache 12G, alpha 0.8, interactive scheduler, TTFT order shortest, prefill merge
512, stack limit 256, encoder-only prefill on, Engram read-ahead on, no steering keys). Served KV 5,687,040 tokens.
Release qualification run of 2026-11-20 22:18-23:10 (52 min). Raw outputs: `results/qual-20261121/`; the release binary checks (seq389, seq390): `results/binary-checks-20261120/`;
the speed table's result files: `results/owner-speed-20261120/`.

## Scope: what changed and which tier ran

Against the last qualified local build (package 192051 on the binary with the upstream picks), only two things changed:
the binary (steering inside the BRRR kernel, the steering access gate) and the package (the same export recipe and
tuning plan plus the steering sidecar). Under the owner's tiered policy (2026-11-16) that is the **export tier** for the
package and the **binary tier** for the binary. The export-tier checks ran on this package in its export run
(2026-11-20 09:56); the binary-tier checks ran in the release binary checks (2026-11-20 19:51 and 20:49) and the release
qualification run (2026-11-20 22:18). GPQA (weights tier) ran at scale 0 in the export run and at scale 1 in the
qualification run, because steering changes the model's outputs.

| Tier | Stages | When |
| --- | --- | --- |
| binary | GPU kernel checks, smoke, images, tool calls (60), decode 32K c1/c8/c16, 8-stream agent replay, hotel-lights at 16 streams, 16-stream KV stress, post-run smoke and log scan | any change |
| export | consistency, corruption check, eval-43, tool-eval-bench, decode 128K c1, prefill 32K | a new export |
| weights | GPQA Diamond (one sample of 198) | a quantisation change |
| full | 512K c1, long retrieval, long generation, lavd, EOS bench, LiveBench coding, 128K c8 | earlier policy |

Stages that did not run on this exact build cite the most recent run that covers them, with the build named.

**Not run on this build:** GPU kernel checks (`gpu_check`), hotel-lights, tool calls (60) and consistency (the last three cited from earlier packages below), LiveBench coding, long generation, lavd, the natural-EOS decode bench, the vision sets. Decode and prefill are the owner speed table in MEASUREMENTS.md (it includes 128K c4-c16).
The stages left out have not failed on this model in our records (lavd always 30/30, long generation always 4/4, EOS
0 loops, LiveBench 81.3-84.2). That is the reason for the cut, not a result for this build.

## Results

| Stage | Result | Gate | Build and run |
| --- | --- | --- | --- |
| GPU kernel checks (`gpu_check`) | not run on this build | 13/13 | 20261116: 13/13 |
| Steering kernel tests (BRRR fused tail with steering) | tail 5/5; scale 0 bitwise equal to the plain kernel 48/48 (ordered build); steered rows within bound (worst 0.812 of the bound) | all | this kernel, 24-layer test model, 2026-11-20 09:39 |
| Smoke: thinking off / on | 2/2 | all | this package, steering binary; release binaries 2/2 and 2/2; release run post-check 2/2 |
| Images: 5 cases; 16-bit PNG, GIF, animated GIF refused | **5/5** served and correct; **4/4** (16-bit gray and RGB PNG and GIF read correctly; animated GIF refused with HTTP 400) | 5/5; 4/4 | this package on the release binary with the command-line key option (same image code) |
| Tool-call checks (OpenAI, streaming, Anthropic `tool_use`, parallel off, 1.1 MiB history) | **5/5** | 5/5 | this package on the release binary with the command-line key option (same tool code) |
| Tool calls: 60 prompts at 8 streams | 60/60 | 60/60 | package 181447 (2026-11-18), not re-run |
| Decode vs prefill consistency (`tools/consistency.py 96 6`) | top-1 **0.965**, median Δlogprob 0.0025 | ≥ 0.95, ≤ 0.01 | package 192051 = this export without the sidecar |
| Corruption check (`tools/verify_fix.py 10`) | **0** true corruptions (1,641 tokens) | 0 | this package |
| eval-43 (thinking off) | **43/43**, 32K needle pass | ≥ 42 | this package |
| Needles at 64K / 128K | pass / pass | pass | this package |
| tool-eval-bench (92 scenarios; TC-45 excluded, 91 graded) | 93/100, 81/91 pass | ≥ 88 | 20261116: 90 (77/91) |
| Decode 32K / 128K / 512K, prefill | owner table in [MEASUREMENTS.md](../MEASUREMENTS.md#1-owner-speed-table-release-package-and-binary) | notes | — |
| Agent replay, 8 streams, 300 s | 658 turns, 530.8 output tok/s, first token p50 / p90 / max 0.395 / 0.627 / 1.02 s, 0 errors | 0 turn errors | 20261116: 649 turns, 464.6 tok/s, 0.42 / 0.72 / 1.23 s |
| Long retrieval: three markers at 128K and 900K tokens | pass / pass | pass / pass | 20261116: pass / pass |
| Long retrieval at 500K / 750K / 1M tokens | pass / pass / pass (0 wrong, 0 errors) | 0 errors, < 2 wrong | package 192051 (2026-11-19) |
| GPQA Diamond, scale 0, one sample of 198, 8 streams, 32K tokens | **91.4%** (181/198, Wilson 95% 86.7-94.6), 0 errors; **WARN**: 5 truncated without an answer, 11 hit max_tokens | rc 0, 198 scored, 0 errors; WARN below 85.5% or > 4 truncated | this package; 20261116: 91.4% (181/198), 4 truncated |
| GPQA Diamond, scale 1 | **89.4%** (177/198, Wilson 95% 84.3-93.0), 3 truncated without an answer, 6 hit max_tokens, 0 errors; paired vs scale 0: 4 questions wrong only at scale 1, 0 only at scale 0, exact McNemar p 0.125 | completion only (rc 0, 198 scored, 0 errors); reported, no score gate | this package and binary |
| Steering refusal set (30 trigger + 30 benign prompts, keyword classifier plus a private deflection phrase list; texts not published) | trigger refused 24/30 at scale 0 → 3/30 at scale 1; benign 0/30 at both; 0 errors | scale 1 < scale 0 | this package; release binary: 23/30 → 2/30, benign 0/30, 0 errors |
| Steering prefix isolation | scale 0 → 1 reuses 0 cached tokens; same scale or omitted-after-0 reuses all 3,429 | as designed | this package |
| Steering mixed batches (scale 0 and 1 lanes incl. two ~33K prompts) | c8: first token equal to the homogeneous run 8/8 (PASS); c16: 15/16 (WARN; that lane also differs from its own homogeneous repeat) | first tokens equal | this package |
| Scale-0 exactness vs the non-steering package | long prompts (32K, 128K): top-1 agreement 1.000, as A vs A (PASS); short-prompt byte check void (the reference differs from itself run to run) | long stats inside A/A spread | this package |
| Steering access gate (keys from a file) | HTTP 403 in 7/7 cases (no key, wrong `X-Steering-Key`, wrong Bearer, scale in `chat_template_kwargs`, completions, messages, streaming); 200 with the right key via `X-Steering-Key`, Bearer and streaming; scale 0 / omitted without a key 200 with prefix reuse; key in no log, argument or environment; command-line key option rejected | all | both release binaries |
| hotel-lights, 30 runs at 16 streams | 30/30 exact, 0 truncated | 30/30 | package 181447 (2026-11-18), not re-run |
| KV stress: 2 waves × 16 × ~530K unique prompts + pinger, KV oversubscribed | 32/32 long requests, 168/168 pings, 0 bad log lines, lowest free VRAM 78 / 78 / 78 MiB (GPU 0 / 1 / 2), 19,025,719 prompt tokens, 20.3 min | all long requests and pings ok, 0 bad log lines, free VRAM ≥ 16 MiB per GPU | package 192051: 32/32, 176/176, 47-80 MiB |
| KV stress, 8 streams (release binary check) | 16/16 long, 88/88 pings, 0 bad lines, lowest free VRAM 78 / 78 / 78 MiB | as above | this package on the release binary with the command-line key option |
| Post-run smoke, served KV unchanged | smoke 2/2, served KV 5,687,040 unchanged, corruption check 0 in 10 runs | all | — |
| Server and kernel log scan | 0 fault lines in the server log, 0 in the kernel log | no fault line | — |
| Lowest host MemAvailable during the run | 10.6 GiB | — | steering A/B: 15.0 GiB |

Overall: **all gates pass.** Release run: retrieval, tool-eval, 16-stream stress with the per-GPU VRAM floor, post-check and log scan PASS; agent replay and GPQA at scale 1 reported; the scale-0 GPQA truncation WARN is explained below.

**GPQA truncation WARN.** At scale 0 the run had 5 answers cut off at the 32K-token limit without a final answer
(the kit warns above 4); they count as wrong. 20261116 had 4 on the same harness. The score itself equals 20261116's.

**Scale-0 identity.** Neither the steering package nor the plain package is bit-reproducible from run to run (FP32
atomics in the fused kernel, BF16 atomics in prefill MoE), so exact byte checks on short prompts are void when the
reference differs from itself. The long-prompt statistics decide; they passed.

**tool-eval TC-45.** As in 20261116: the harness probes `tool_choice: "required"` in thinking mode; the server answers
HTTP 400 as DeepSeek's reference does, and the harness grades 91 scenarios.

## Kit

As in 20261116 (`qual_all.sh`, `qual_core.sh`, `stress_kv.py`, `tool_calls.py`, `long_retrieval.py`; the one-command
stage list there). GPQA at scale 1 ran through a small local proxy that adds `"steering_scale": 1.0` to every request (the bench cannot
add a request field); the proxy logged one rewrite per question (198, all at 1.0) and 0 errors. The proxy ships as `../tools/steer_proxy.py`, with the refusal-set runner `../tools/refusal_set.py` (no prompts
ship; bring a set in the documented format) and the access check `../tools/steering_access_check.py` (README "Steering tools").
