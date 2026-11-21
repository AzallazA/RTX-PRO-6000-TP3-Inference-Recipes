# Speed table runs, 2026-11-20

Raw `llm_decode_bench.py` result files behind MEASUREMENTS.md section 1 (package 202611201530 on the release binary,
`serve.sh` defaults, read-ahead on). llm-inference-bench v0.9.1 with `tools/llm-bench-flare-steps.patch` (the files
record bench version 0.8.0). Host names and hardware temperature fields are removed; everything else is as written.

| Files | Run |
| --- | --- |
| `ds41prefillrun{1,2,3}.json` | `--prefill-only`, cold prompts 8K-1M, three runs |
| `ds41C1-C16-32krun{1,2,3}.json` | decode 32K, c1 / c4 / c8 / c16, 30 s cells, `--skip-prefill`, three runs |
| `ds41C1-C16-128krun{1,2,3}.json` | decode 128K, same cells |
| `ds41C1-C16-512krun{1,2,3}.json` | decode 512K, c1 |

The table reports the median of the three runs per cell; `tools/bench_steps.py <file>` prints steps/s and tokens per
step.
