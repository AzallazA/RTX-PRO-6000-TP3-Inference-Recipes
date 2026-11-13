# Qualification results — GLM-5.3-Flash Hyperflare TP3 / TP2 20261112

Four packages, all at 450 W/GPU, served with the folder's scripts' defaults (`balanced` scheduler, prefix cache 32G,
5 drafts; TP3 alpha 0.8, TP2 alpha 1.0), thinking on, temperature 1.0, top-p 0.95. Effort: eval-43 and image questions
`low`, tool calls `high`, everything else `max`.

| Package | What it is | Results |
| --- | --- | --- |
| `gb202-tp3-202611130000` | **TP3 release package** (images, 4 image slots, 32 slots) | `results/tp3-image-202611130000/` |
| `gb202-tp2-202611130307` | **TP2 release package** (images, 2 image slots, 8 slots) | `results/tp2-image-202611130307/` |
| `gb202-tp3-202611120938` | TP3 text-only package of the same text path (`VISION=0`) | `results/tp3-text-202611120938/` |
| `gb202-tp2-202611121033` | TP2 text-only package of the same text path (`VISION=0`) | `results/tp2-text-202611121033/` |

The text checks (eval-43, tool calls, corruption check, long retrieval, long generation, determinism; GPQA, tool-eval
and hotel-lights at TP3) ran on the text-only packages and carry to the image packages: the image commits only touch
the vision tower (weight staging and working memory), so the text path, kernels and serve flags are the same. The
image packages were checked for what the image commits change (KV, full-concurrency stress, speed, images), plus TP2's
GPQA, tool-eval and hotel-lights. See the README's [Quality](../README.md#quality) section.

Gate names below are the kit's (`GATES.tsv` / `SUMMARY.md` in each run). Two kit gates read stricter than the release
criteria: hotel-lights is FAIL below 30/30 (GLM's baseline is 26-30/30 on every GLM build we ran, vLLM included), and
the stress gate is WARN when every request passed but free VRAM dipped below 480 MiB. GPQA and tool-eval are INFO
(no stored GLM baseline: these runs are the baselines). Times in `QUAL.md`/`GATES.tsv` are UTC, in `CHAIN.md` local
(UTC−5).

## TP3 image package `gb202-tp3-202611130000` (release)

| Check | Result | File |
| --- | --- | --- |
| KV, dry run (PCM 32G) / served | 13,393,152 / **13,130,496**; text-only 0938: 13,737,984 (−2.5%); upstream-dataflow image package 2211: 13,011,456 (−5.3%) | `lean-chain/K3-dryrun-*.log`, `lean-chain/CHAIN.md` |
| Memory per GPU (`Plan cache`) | 64.88-64.92 GiB loaded, 29.35-29.39 GiB KV budget; operator scratch 578 MiB | `lean-chain/S3-serve.log`, `budget-ab/LOG`, `lean-chain/E3-export.log` |
| Image latency | first 0.15 s, next small 0.13 s, 4,096-token 0.70 s, 8,000-token 1.51 s, 4 small in parallel 0.54 s | `lean-chain/S3-latency.{json,log}` |
| Image smoke (small, near-max, over-max, 2 and 4 images) | 5/5 correct | `lean-chain/S3-smoke.{json,log}` |
| Image questions (DS41 sets, 60 + 60 harder), effort low | **60/60, 51/60**, 0 errors, 0 truncated | `lean-chain/S3-eval.{json,log}` |
| Same sets on the upstream-dataflow image package 2211 (earlier chain, same day) | 60/60, 49/60 | `upstream-dataflow-ref/S3-eval.{json,log}` |
| Vision features vs upstream dataflow (one GPU, real weights, 8 chart images + 8,000-token composite, graph replay) | **bit-identical** (9/9 images), and lean run-to-run bit-identical | `vision-features/SUMMARY.txt`, `cmp-*.json` |
| First-token probe (12 image prompts, top-5 logprobs) | 24/24 ok, same-server repeat identical 12/12; vs the 2211 package: top-1 same 9/12, exact 5/12 (different exports: kernel tuning differs per export) | `lean-chain/S3-probe-*.{json,log}` |
| Text decode vs text-only 0938, 32K, one same-session round | c1 279 vs 281 tok/s (0.995×), c8 870 vs 866 (1.004×) | `lean-chain/S3-new.*`, `S3-old.*` |
| Verification row budget A/B, 32K, 60 s cells | c32: default 1,718 / 1,689, `BUDGET=32` 1,365 / 1,407, `BUDGET=48` 1,239 tok/s; c1-c16 level or lower | `budget-ab/glm-budget-ab-1112-2116b.out`, `*.json` |
| KV stress: 2 waves × 32 × ~530K unique prompts + pinger | **PASS** 64/64, pinger 246/246, 37.5M prompt tokens, 0 bad log lines, min free 44 MiB (gate WARN: below the 480 MiB margin) | `lean-chain/S3-qual/` |
| CPU tests / staging CUDA test (code 72548a42) | 19 passed, 4 skipped / 16 passed | `lean-chain/T-cpu-lean.log`, `T-cuda-test.log` |

## TP2 image package `gb202-tp2-202611130307` (release)

| Check | Result | File |
| --- | --- | --- |
| KV, dry run (PCM 32G) / served | 2,414,592 / **2,231,808**; text-only 1033: 2,589,184 (−6.7%) | `lean-chain/K2-dryrun-*.log` |
| Memory per GPU (`Plan cache`) | 86.56 GiB loaded, 7.71 GiB KV budget | `lean-chain/S2-serve.log` |
| Image latency | first 0.15 s, next small 0.14 s, 4,096-token 0.71 s, 8,000-token 1.46 / 1.53 s, 4 small in parallel 0.60 s | `lean-chain/S2-latency.{json,log}` |
| Image smoke | 5/5 correct | `lean-chain/S2-smoke.{json,log}` |
| Image questions, effort low | **59/60, 49/60**, 0 errors, 0 truncated | `lean-chain/S2-eval.{json,log}` |
| First-token probe | 24/24 ok, same-server repeat identical 12/12 | `lean-chain/S2-probe-lean.{json,log}` |
| Text decode, 32K, 2 rounds, vs the stored 1033 rounds (246 / 514 / 717) | c1 252 / 246, c4 507 / 509, c8 699 / 705 tok/s (1.012 / 0.988 / 0.979×) | `lean-chain/S2-new-r*.{json,log}` |
| KV stress: 2 waves × 8 × ~530K + pinger | **PASS** 16/16, pinger 139/139, 9.4M prompt tokens, 0 bad log lines, min free 96 MiB (gate WARN) | `lean-chain/S2-qual/` |
| hotel-lights, 30 runs at c8 | 26/30: 2 unparseable answers, 1 wrong (49 for 48), 1 truncated | `qual-hotel-gpqa-tooleval/hotel-lights-c8.*` |
| GPQA Diamond (198, effort max) | **180/198 = 90.9%** (Wilson 95% 86.1-94.2), 1 hit max tokens, 0 errors | `qual-hotel-gpqa-tooleval/gpqa.{json,log}` |
| tool-eval-bench | **92**/100 | `qual-hotel-gpqa-tooleval/tooleval.{json,log}` |

## TP3 text-only package `gb202-tp3-202611120938` (text checks carried to the TP3 image package)

| Check | Result | File |
| --- | --- | --- |
| eval-43 (effort low) | **42/43**, refused 0/37, 0 truncated, 0 markup leaks | `quick-checks/eval_glm-c32-sync-1112.json`, `eval.log` |
| Tool calls, 60 prompts at c32 | **60/60** | `quick-checks/tool_calls.json`, `tools.log` |
| Corruption check, 20 runs, 6,000 tokens | **0** true corruptions, 0 natural-uncertainty flags | `quick-checks/verify_fix.log` |
| hotel-lights, 30 runs at c32 | **30/30** (run 1148); first run (1125): 26/30, 2 wrong (one 49 for 48, one unparseable), 2 truncated | `fullqual-1148/qual/hotel-lights-c32.*`, `fullqual-1125/qual/hotel-lights-c32.*` |
| Long retrieval, 3 markers at 128K / 900K | pass / pass | `fullqual-1125/qual/long-retrieval*` |
| Long generation after the captured conversation (530,058 prompt tokens under the GLM template), seeds 201-202 | pass: 10,086 and 20,363 tokens, natural stop, worst 8-gram repeat 0.0005 | `fullqual-1148/qual/long-generation.json`, `longgen-*` |
| GPQA Diamond (198, effort max) | **177/198 = 89.4%** (Wilson 95% 84.3-93.0), 1 hit max tokens, 0 errors | `fullqual-1148/qual/gpqa.{json,log}` |
| tool-eval-bench | **92**/100 | `fullqual-1148/qual/tooleval.{json,log}` |
| KV stress at c32, 1,800 s per-request timeout | FAIL 44/64: 10 requests per wave hit the timeout (each wave pushes 32 × 530K = 17M tokens into the 13.5M pool); pinger 197/197, 0 bad log lines, server healthy | `fullqual-1148/qual/stress*` |
| KV stress at c32, re-run with 3,600 s timeout | **PASS** 64/64, pinger 237/237, 37.5M prompt tokens, min free 78 MiB (gate WARN) | `stress-rerun/` |
| Prefill 8K / 32K / 128K | 10,270 / 11,606 / 11,409 tok/s | `fullqual-1125/prefill.{json,log}` |
| Determinism: same requests, fresh prefill, two servers (restart between) | first-token top-5 logprobs 24/24 identical; seeded multi-token answers 2/9 identical | `fullqual-1125/det-a.json`, `det-b.json` |
| Startup / host RSS / KV served | 34 s / 92 GiB / 13,523,712 | `fullqual-1125/SERVE.txt` |
| Same-session A/B vs the previous default 112036, 3 rounds, 32K | tok/s c1 0.994×, c4 0.985×, c8 1.050×, c16 0.998×, c32 1.014×; KV 13,523,712 vs 13,416,960 (+0.8%) | `ab-vs-112036/SUMMARY.md`, `LOG` |

eval, tools and verify ran once (`quick-checks/`, from the upstream-sync validation run) and were reused by both full
runs on the same fingerprint (package, server binary checksum, serve arguments; `fingerprint.txt`). Run 1125 stopped
after long retrieval (the kit had no `CAPTURE` set); run 1148 reused its passed stages and ran the rest.

## TP2 text-only package `gb202-tp2-202611121033` (text checks carried to the TP2 image package)

| Check | Result | File |
| --- | --- | --- |
| eval-43 (effort low) | **43/43**, refused 1/37, 0 truncated, 0 markup leaks | `quick-checks/eval_glm-tp2-sync-1112.json`, `eval.log` |
| Tool calls, 60 prompts at c8 | **60/60** | `quick-checks/tool_calls.json`, `tools.log` |
| Corruption check, 20 runs, 6,000 tokens | **0** true corruptions (35 natural-uncertainty flags) | `quick-checks/verify_fix.log` |
| hotel-lights, 30 runs at c8 | 26/30: 2 unparseable answers, 2 wrong (49 and 47 for 48), 0 truncated | `fullqual-1148/qual/hotel-lights-c8.*` |
| Long retrieval, 128K / 900K | pass / pass | `fullqual-1148/qual/long-retrieval*` |
| Long generation (530,058 prompt tokens), seeds 201-202 | pass: 11,164 and 17,340 tokens, natural stop, worst 8-gram repeat 0.0004 | `fullqual-1148/qual/long-generation.json`, `longgen-*` |
| GPQA / tool-eval / KV stress | **INVALID** (server killed mid-GPQA by a queue incident; see `fullqual-1148/INVALID.md`). Valid TP2 runs: the image package above | `fullqual-1148/qual/gpqa.*`, `tooleval.*` |
| Prefill 8K / 32K / 128K (right after the incident, informational) | 10,702 / 12,026 / 11,720 tok/s | `fullqual-1148/prefill.{json,log}` |
| Determinism across a restart | first token 24/24 identical; multi-token answers 2/9 identical | `fullqual-1148/det-a.json`, `det-b.json` |
| Startup / host RSS / KV served | 28 s / 75 GiB / 2,446,336 | `fullqual-1148/SERVE.txt` |

The first TP2 full run directory (1125) is empty; 1148 is the TP2 run.

## What was removed from the raw outputs

Paths, user and host names are replaced by tokens (`<fork>`, `<disk>`, `<results>`, `<tmp>`, `~`) and `hostname`/`uname`
fields dropped. hotel-lights JSONs keep `reasoning_text` and `content_text` per run but not `output_text` (their
concatenation). Long-generation `result.json` files keep metrics and usage but not the generated text or token ids (the
continuation quotes the capture's example paths). The GPQA/tool-eval bench cache paths read `<llm_decode_bench cache>`.
Not shipped: the vision feature tensors (`*.pt`, 85 MB each), the long-generation SSE streams, the tool-eval per-run
conversation reports, export logs other than `E3-export.log`/`E2-export.log`.

## Running it

The kit is `qual_all.sh` (stages, gates in `qual_core.sh`) with the Python tools here and in `../tools/`
(`glm_common.py`, `tool_calls.py`, `verify_fix.py`; `TOOLS_DIR` overrides). Start the server on :15015 with
`--served-model-name GLM-5.3-Flash` (the serve scripts do), then:

```bash
QUAL_PROFILE=core QC=32 LABEL=glm-tp3 CKPT=<checkpoint dir> FLARE_LOG=<server log> ./qual_all.sh   # TP3 (QC=8 for TP2)
./qual_all.sh eval tools retrieval                                                                 # smoke, nothing but the server
```

Outputs go to a fresh `run-<date>/` next to the script (`QUAL_OUT` overrides it); `results/` and this file are never
written. `QUAL_PROFILE=core` runs eval, tools, verify, hotel, retrieval, longgen (2 seeds), GPQA Diamond, tool-eval-bench
and stress (the runs above); `full` (default) is the DeepSeek suite's list (eval tools lavd hotel retrieval longgen eos
livebench stress). `REUSE_FROM=<earlier run dir>` skips stages that passed there on the same fingerprint.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CKPT` | — | checkpoint directory (tokenizer for `longgen` and `consistency`) |
| `FLARE_LOG` | — | server log, scanned by `stress` |
| `CAPTURE` | `../../DeepSeek-V4.1-Flash Hyperflare-tp3-20261111/qualification/captured-request.san.json` | the sanitized captured conversation for `longgen` |
| `LLM_BENCH_DIR` / `LIVEBENCH_DIR` | `$HOME/llm-inference-bench` / `$HOME/LiveBench` | checkouts, each with its `.venv` (hotel, lavd, eos, GPQA, tool-eval / LiveBench) |
| `PY` | `$FLARE_DIR/.venv/bin/python` if `FLARE_DIR` is set, else `python3` | `consistency` needs `transformers` |
| `QC` | 8 | concurrency of profiles, tool calls, GPQA and stress |
| `MODEL`, `FLARE_URL` | `GLM-5.3-Flash`, `http://127.0.0.1:15015` | server |
| `QUAL_TOP_P`, `QUAL_TEMPERATURE`, `QUAL_EFFORT`, `EVAL_EFFORT`, `TOOLS_EFFORT` | 0.95, 1.0, max, low, high | GLM recipe ranges; effort must be low, high or max (the template renders any other value as max) |
| `STRESS_LONG_TIMEOUT_S` | 1800 | per long request in `stress`; use 3600 at TP3 c32 (32 × 530K oversubscribes the pool, see above) |
| `BASELINE_DIR`, `GPQA_BASE`, `TOOLEVAL_BASE` | `$HOME/bench-results/baselines`, auto, auto | paired GPQA (McNemar) and tool-eval gates; INFO without a baseline |

Measured stage times (TP3 at c32): hotel-lights ~16 min, retrieval ~2, longgen ×2 ~3, GPQA ~30, tool-eval ~3, stress
~70 (3,600 s timeout); TP2 at c8: hotel-lights ~33, GPQA ~60, stress ~16.

Image checks: `../tools/glm_images.py latency|smoke|eval LABEL OUT.json` against the live server (the DeepSeek V4.1
vision sets; `VISION_SETS=` points at them, default the `DeepSeek-V4.1-Flash Kraken-tp3-20261019` folder of this repo).
`vision/` holds the two vision-tower equivalence checks:

- `vision/lean_probe.py run LABEL OUT.json` / `cmp A.json B.json`: 12 image prompts, first-token top-5 logprobs, sent
  twice, against a live server (`PORT`, default 15015); compares two packages prompt by prompt.
- `vision/vision-feature-check.sh RESULT_DIR` with `OLD=<worktree> LEAN=<worktree> FLARE_GLM53_SNAPSHOT=<checkpoint>`:
  dumps the vision tower's output features with `vision_feature_dump.py` from each worktree on one idle GPU (`GPU`,
  default 0; the worktree's venv, `PY` overrides) and compares them with `vision_feature_cmp.py` (old vs lean:
  rounding or better; lean vs lean again: bit-identical). `vision_feature_dump.py --check-cpu` checks the inputs
  against the Flare frontend without a GPU.

The prefill numbers come from `llm_decode_bench.py --prefill-only --prefill-contexts 8k,32k,128k`; the determinism probe
(`det-*.json`) from a separate tool that is not part of this kit.
