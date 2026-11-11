# Qualification results — Hyperflare TP3 20261111 (graph scratch + TTFT guard)

Run on the running package without restarts, 450 W, KV 3,014,656 tokens (package 202611111227), serve flags as in `../serve.sh` (alpha 0.8,
**interactive** scheduler, cap 24, 32G prefix cache), concurrency 8 where the protocol allows. Raw outputs:
`results/run-hyperflare-c8/` (corruption check: `results/release-ab/verify.log`). Gates: the 20261108.1 release (package 081024) with the tolerances in the table.

| Check | Result | 20261108.1 (081024) | Gate |
| --- | --- | --- | --- |
| Corruption check, 10 runs | 0 true corruptions (14 natural-uncertainty flags) | 0 true corruptions | 0 |
| eval-43 (thinking off; temperature clamped to 1.0) | 42/43 | 43/43 | ≥ 42/43 |
| Tool calls, 60 at c8, argument validity + second turn | 60/60 | 60/60 | 60/60 |
| lavd, 30 runs at c8, effort max, natural EOS | 30/30 exact, 0 truncated | 30/30 exact | ≥ 29/30 exact, 0 truncated |
| hotel-lights, 30 runs at c8 | 30/30 exact, 0 truncated | 30/30 exact | ≥ 29/30 exact, 0 truncated |
| Long retrieval, 3 markers at 128K / 900K tokens | pass / pass | pass / pass | pass / pass |
| Long generation after a 581,537-token conversation, seeds 201-204 | 4/4, worst 8-gram repeat 0.0013, natural stop | 4/4, worst 8-gram repeat 0.0008 | 4/4, natural stop |
| Decode bench with natural EOS, 32K c1/c4/c8 | 299.5 / 712.7 / 949.8 tok/s, 0 loops | 290.1 / 716.9 / 963.3 tok/s, 0 loops | ≥ 0.97× each, 0 loops |
| LiveBench coding (2024-11-25 release, 128 questions) | 83.5 (LCB_generation 73.077, coding_completion 94.0) | 81.6 | ≥ 80.5 |
| KV stress: 2 waves × 8 × ~530K unique prompts + pinger | PASS: 16/16, pinger 68/68, 9.5M prompt tokens | PASS: 16/16, pinger 117/117, 9.5M prompt tokens | PASS |

All gates pass. The LiveBench line in `results/run-hyperflare-c8/QUAL.md` is empty: pandas truncates the long display
name in `show_livebench_result.py`'s table, and the name-based grep then matched nothing. The score above is read from
`livebench-score.log` ("All Groups" row); `qual_all.sh` in this folder now reads that row directly.

Earlier quick checks of the same package family: `results/graph-scratch-ab-20261109/quick-checks/` (balanced scheduler)
and `results/ttft-quality-20261110/` (interactive scheduler).

LiveBench runs at temperature 1.0, so scores vary by about ±1 between runs. Raw outputs are in `results/` (profile
runs keep their full texts).

Running it: start the server, then `FLARE_LOG=<server log> CKPT=<checkpoint view> ./qual_all.sh` (or name stages:
`./qual_all.sh eval tools`). Every output lands in a fresh `run-<date>/` directory next to the script (`QUAL_OUT` overrides
it); the shipped `results/` and this file are never written. The long-generation stage replays `captured-request.san.json`
(the sanitized 581K-token conversation, shipped here). LiveBench and `llm-inference-bench` checkouts (each with its own
`.venv`) are expected in `~/LiveBench` and `~/llm-inference-bench`; `LIVEBENCH_DIR` and `LLM_BENCH_DIR` override the paths.

Stages in `qual_all.sh` order, with the durations of the shipped run (450 W, concurrency 8) and what each one needs
beyond the running server:

| Stage | Minutes | Needs |
| --- | --- | --- |
| `eval` | ~1 | — |
| `tools` | ~1 | — |
| `lavd` | ~8 | `llm-inference-bench` (`LLM_BENCH_DIR`) |
| `hotel` | ~10 | `llm-inference-bench` |
| `retrieval` | ~2 | — |
| `longgen` | ~7 | `CKPT=<checkpoint view>` (its tokenizer) |
| `eos` | ~4 | `llm-inference-bench` |
| `livebench` | ~8 | LiveBench checkout with its `.venv` (`LIVEBENCH_DIR`) |
| `stress` | ~15 | `FLARE_LOG=<server log>` |

Smoke test, about 5 minutes and nothing but the server: `./qual_all.sh eval tools retrieval`. The full suite is about
1 hour. Environment: `FLARE_LOG`, `CKPT`, `LLM_BENCH_DIR` (default `~/llm-inference-bench`), `LIVEBENCH_DIR` (default
`~/LiveBench`), `QUAL_OUT` (output directory), `QC` (profile concurrency, default 8).
