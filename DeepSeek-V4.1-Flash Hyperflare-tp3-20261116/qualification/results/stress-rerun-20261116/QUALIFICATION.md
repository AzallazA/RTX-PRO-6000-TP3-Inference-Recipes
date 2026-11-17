# Qualification results — DS41 Flare TP3 local default (weights tier, preset B), merged

Two runs make up this qualification. Each row names the run it comes from.

- **seq235** (`~/bench-results/ds41-qual-1116-1502`): every stage up to hotel16. Cited here because the live server has the same identity (exe sha, argv, environment).
- **Rerun**: the 16-stream stress, the post checks and the log scan. The first stress attempt in seq235 was cut off by a hardware fault.

Preset B (rerun of seq235's stress): pid 843676, exe 27864a1d79e4c5acf802e2d564a732b97f59d8478517b5f3501a1632eb491f75 (~/<fork-worktree>/flare-cli/target/release/flare), package gb202-tp3-202611160815,
served KV 5403008, argv: --gpus 0,1,2 --port 15015 --host 127.0.0.1 --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory 12G --mtp-draft-tokens 5 --mtp-adaptive-verification --mtp-max-verification-token-budget 32 --mtp-adaptive-verification-alpha 0.8 --scheduler-policy interactive --scheduler-prefill-tail-merge true --scheduler-ttft-order shortest 
env: FLARE_FAST_EXIT=1 FLARE_ENGRAM_RAM_TABLES=0 FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_CUDA_STACK_LIMIT=256 FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512 ; server log ~/<fork-worktree>/serve-best.log (from byte 1943); serve-best.sh sha f7e5831b290c
identity vs seq235 id/ (sha, argv, env, fingerprint): MATCH; server started Mon Nov 16 18:47:34 2026

| Check | Result | Reference (source) | Gate | Source run | Minutes |
| --- | --- | --- | --- | --- | --- |
| Request checks (gpu_check.py, incl. thinking-mode tool_choice 400) | 13/13 passed | 13/13 (seq214 ds41-integration-1116-0614/gpu_check.txt (same build)) | PASS | seq235 ds41-qual-1116-1502 | 0.1 |
| Smoke 2/2 + tool call 3/3 | smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s; tool call 3/3 Paris:ok Tokyo:ok Lima:ok | smoke 2/2 + tool call 3/3 (seq214 S/smoke-Sq.txt, S/tools.txt) | PASS | seq235 ds41-qual-1116-1502 | 0.1 |
| Image smoke (5 cases) | IMAGE_SMOKE qB served 5/5 correct 5/5 / small: ok 222p 0.1s / large 1920x900 (frontend cap 381 tok): ok 998p 0.2s / over-max 4160x1280 (downscaled): ok 974p 0.3s / two images: ok 422p 0.1s / four images: ok 828p 0.2s | served 5/5 correct 5/5 (seq214 S/image-smoke.log) | PASS | seq235 ds41-qual-1116-1502 | 0.1 |
| Tool calls, 60 at c8, argument validity + second turn | tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 18 s | 60/60 OK at c8 (hyperflare-tp3-20261111 RESULTS.md (1227)) | PASS | seq235 ds41-qual-1116-1502 | 0.3 |
| Decode-vs-prefill consistency (96 tokens x 6 prompts) | 284 positions, top-1 agree 0.951, missing 0, median 0.0081 (gate >= 0.95 / <= 1 / <= 0.01) | top-1 0.962, missing 0, median 0.0055 (gate >= 0.95 / <= 1 / <= 0.01) (port-g2q-1114-1811/consistency-port.txt (142139); 1241 0.963) | PASS | seq235 ds41-qual-1116-1502 | 0.2 |
| Corruption check (verify_fix, 10 runs) | verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0) | 0 TRUE corruptions in 10 runs (hyperflare-tp3-20261111 RESULTS.md; seq214 S/verify-*.log) | PASS | seq235 ds41-qual-1116-1502 | 0.2 |
| eval-43 (thinking off; temperature clamped to 1.0) | 42/43 (thinking off, temperature clamped to 1.0) | 42/43 (gate >= 42; 41 WARN) (hyperflare-tp3-20261111 RESULTS.md (1227); 081024 43/43) | PASS | seq235 ds41-qual-1116-1502 | 0.5 |
| tool-eval-bench (bench --tool-eval, leaderboard settings) | score 90/100, pass 77/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88 | 91/100, 78/91 graded pass, TC-45 excluded (gate >= baseline - 3) (baselines/flare-ds41-1111-2124/tooleval.json (1227); vLLM .12 90/100 79/92) | PASS | seq235 ds41-qual-1116-1502 | 1.9 |
| Decode 32K c1 / c8 / c16 (notes) | c1 WORSE: 319.5 tok/s vs ref 334.6 (0.955x, band +-4%); steps/s 122.82986915431576; accept 2.6; errors 0/c8 WORSE: 978.8 tok/s vs ref 1044.1 (0.937x, band +-2.5%); steps/s 350.7253299449249; accept 2.35; errors 0/c16 LEVEL: 1331.5 tok/s vs ref 1352.2 (0.985x, band +-3%); steps/s 301.0074432519567; accept 2.21; errors 0 | c1 334.6 / c8 1044.1 / c16 1352.2 tok/s (bands 4/2.5/3 %); seq214 same build 322.7/1078.8/1391.6 (ds41-exp-fp8-1116-0241/B (160815, 2 passes)) | WARN | seq235 ds41-qual-1116-1502 | 2.7 |
| Long-context decode 128K | 128K-c1 BETTER: 320.8 tok/s vs ref 292.5 (1.097x, band +-4%); steps/s 119.23316991231852; accept 2.69; errors 0 | c1 290-295, c8 990-1,025 tok/s (owner runs 11-15 ~/llm-inference-bench/run1.json + run2.json (152105, budget 32)) | PASS | seq235 ds41-qual-1116-1502 | 1.1 |
| Long-context decode 512K c1 | 512K-c1 BETTER: 292.7 tok/s vs ref 271.0 (1.080x, band +-5%); steps/s 112.76629784947252; accept 2.59; errors 0 | c1 271 tok/s (records 263-282) (hyperflare-tp3-20261111 release-ab/base-longctx.json (1227); ds41-sync-1112-0230/A) | PASS | seq235 ds41-qual-1116-1502 | 2.7 |
| Prefill 32K (notes) | 32768 prefill 15766 tok/s (4 samples) vs ref 15958 (0.988x, band 4%) | 15,958 tok/s 32K cold (band 4 %) (ds41-exp-fp8-1116-0241 Bq-prefill-32K (160815 on 67c691d29041)) | PASS | seq235 ds41-qual-1116-1502 | 0.2 |
| Agent replay, 8 streams, 300 s (notes) | agent c8: 649 turns in 300 s, TTFT p50/p90/max 0.41666698455810547/0.7224347591400146/1.2345826625823975 s, 464.6 output tok/s; errors 0; ref 646 turns, 455.8 out tok/s, p50/p90 0.51/0.90 s | 646 turns, 455.8 out tok/s, TTFT p50/p90 0.51/0.90 s (ds41-pmerge-c8fb-1116-1252/off.1.std8.json (merge flag 0); 160407 agent8 0.505/0.907) | INFO | seq235 ds41-qual-1116-1502 | 5.6 |
| Long retrieval 128K / 900K | DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass | 128K pass / 900K pass (hyperflare-tp3-20261111 RESULTS.md) | PASS | seq235 ds41-qual-1116-1502 | 2.1 |
| Long generation after the 581,537-token conversation, seeds 201-204 | skipped by tier | 4/4, worst 8-gram 0.0013, natural stop (hyperflare-tp3-20261111 RESULTS.md) | - | seq235 (tier weights) | - |
| lavd, 30 runs at c16 | skipped by tier | 30/30 exact, 0 truncated (gate 30 PASS, 29 WARN) (hyperflare-tp3-20261111 RESULTS.md (c8)) | - | seq235 (tier weights) | - |
| Decode bench with natural EOS, 32K c1/c4/c8 | skipped by tier | 299.5 / 712.7 / 949.8 tok/s, 0 loops (gate >= 0.97x each, 0 loops) (hyperflare-tp3-20261111 RESULTS.md (1227)) | - | seq235 (tier weights) | - |
| LiveBench coding (2024-11-25 release) | skipped by tier | 83.5 (gate >= 80.5; 081024 81.6, vLLM .12 81.9) (hyperflare-tp3-20261111 RESULTS.md) | - | seq235 (tier weights) | - |
| GPQA Diamond (198 items, one sample, c8, max_tokens 32768) | 181/198 = 91.4% (Wilson95 86.7-94.6), truncated without answer 4, hit max_tokens 7, errors 0, max_tokens 32768, c8; ref 90.4% (85.5-93.8); paired vs baseline: 198 items, 5 gained / 3 lost, McNemar p 0.727; WARN: truncated without answer > 3; one 198-item sample resolves about +-5 points | 90.4% (179/198, Wilson95 85.5-93.8), truncated-no-answer 1, hit max_tokens 8 (baselines/flare-ds41-1111-2124/gpqa.json (1227, c8, max_tokens 32768); vLLM .12 89.4% (8 truncated)) | WARN | seq235 ds41-qual-1116-1502 | 23.9 |
| hotel-lights, 30 runs at c16 | DONE hotel-lights-c16 rc=0 / exact 30 / wrong 0 / truncated 0 / [] | 30/30 exact, 0 truncated (gate 30 PASS, 29 WARN) (seq214 Q/hotel (c16, same build); published c8 30/30) | PASS | seq235 ds41-qual-1116-1502 | 8.2 |
| KV stress, 16 streams (2+ waves x 16 x ~530K unique prompts + pinger) | PASS 32/32 long, ping 212/212, min free 64 MiB, bad lines 0, wall 25.6 min (min free < 480 MiB kit margin, >= 16 MiB floor) | PASS 32/32 long, ping 207/207, min free 80 MiB, 0 bad lines (gate: all ok, min free >= 16 MiB) (seq214 Q/stress (c16, same build); published c8 16/16, 68/68, 82 MiB) | PASS | rerun (run) | 25.6 |
| Post checks (smoke, KV unchanged) | smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s / served KV 5403008 = launch / verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0) | smoke 2/2, served KV unchanged (-) | PASS | rerun (run) | 0.1 |
| Serve log + kernel log scan | serve log 0 fault lines, kernel log 0 NVRM fault lines | no CUDA fault / OOM / panic / Xid (-) | PASS | rerun (run) | 0.0 |

## Notes

- served KV 5403008 = seq235's 5403008 (same as every cited stage).
- First stress attempt (seq235, 15:52:09): GPU 2 (PCI e1:00, top card) fell off the bus (Xid 79) at 15:52:51 at ~430-450 W, 42 s into wave 1. Every long request failed with CUDA_ERROR_LAUNCH_FAILED (719), and the server exited. The crash log was lost to the restore. That attempt is superseded by the rerun below and is not a software verdict.
- Stress volume (rerun): capacity 5,403,008 tokens (live KV) -> 2 waves x 16 x ~530,000; 19,027,513 prompt tokens pushed = 3.52x the KV pool; 32/32 long, ping 212/212, min free 64 MiB, wall 25.6 min (seq214: 19,021,130 tokens = 3.52x of 5,403,008).
- No power cap (GPU_PL empty).
- Restore: skipped (RESTORE=0): server stopped by PID, GPUs free
- Lowest MemAvailable MiB: stress16 12716, post 20230, restore 20250.

GPU minutes: seq235 stages cited 50, rerun stages 26.

QUALIFICATION DONE preset B tier weights: all gates pass (seq235 cited + stress rerun) (WARN cited: decode32 gpqa); KV 5403008; restore skipped (RESTORE=0): server stopped by PID, GPUs free; 25 min | ~/bench-results/ds41-qual-rerun-1116-1850

