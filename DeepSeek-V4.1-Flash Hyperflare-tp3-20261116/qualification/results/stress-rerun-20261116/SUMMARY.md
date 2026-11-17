# DS41 qualification — weights tier, preset B (seq235 + stress rerun)

Preset B (rerun of seq235's stress): pid 843676, exe 27864a1d79e4c5acf802e2d564a732b97f59d8478517b5f3501a1632eb491f75 (~/<fork-worktree>/flare-cli/target/release/flare), package gb202-tp3-202611160815,

- gpu_check: CITED PASS (seq235 ds41-qual-1116-1502, 0.1 min) — 13/13 passed
- smoke: CITED PASS (seq235 ds41-qual-1116-1502, 0.1 min) — smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s; tool call 3/3 Paris:ok T
- image: CITED PASS (seq235 ds41-qual-1116-1502, 0.1 min) — IMAGE_SMOKE qB served 5/5 correct 5/5 | small: ok 222p 0.1s | large 1920x900 (frontend cap 381 tok):
- tools60: CITED PASS (seq235 ds41-qual-1116-1502, 0.3 min) — tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 18 s
- consistency: CITED PASS (seq235 ds41-qual-1116-1502, 0.2 min) — 284 positions, top-1 agree 0.951, missing 0, median 0.0081 (gate >= 0.95 / <= 1 / <= 0.01)
- verify: CITED PASS (seq235 ds41-qual-1116-1502, 0.2 min) — verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0)
- eval: CITED PASS (seq235 ds41-qual-1116-1502, 0.5 min) — 42/43 (thinking off, temperature clamped to 1.0)
- tooleval: CITED PASS (seq235 ds41-qual-1116-1502, 1.9 min) — score 90/100, pass 77/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88
- decode32: CITED WARN (seq235 ds41-qual-1116-1502, 2.7 min) — c1 WORSE: 319.5 tok/s vs ref 334.6 (0.955x, band +-4%); steps/s 122.82986915431576; accept 2.6; erro
- long128: CITED PASS (seq235 ds41-qual-1116-1502, 1.1 min) — 128K-c1 BETTER: 320.8 tok/s vs ref 292.5 (1.097x, band +-4%); steps/s 119.23316991231852; accept 2.6
- long512: CITED PASS (seq235 ds41-qual-1116-1502, 2.7 min) — 512K-c1 BETTER: 292.7 tok/s vs ref 271.0 (1.080x, band +-5%); steps/s 112.76629784947252; accept 2.5
- prefill32: CITED PASS (seq235 ds41-qual-1116-1502, 0.2 min) — 32768 prefill 15766 tok/s (4 samples) vs ref 15958 (0.988x, band 4%)
- agent8: CITED INFO (seq235 ds41-qual-1116-1502, 5.6 min) — agent c8: 649 turns in 300 s, TTFT p50/p90/max 0.41666698455810547/0.7224347591400146/1.234582662582
- retrieval: CITED PASS (seq235 ds41-qual-1116-1502, 2.1 min) — DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
- longgen: skipped by tier (seq235 (tier weights))
- lavd: skipped by tier (seq235 (tier weights))
- eos: skipped by tier (seq235 (tier weights))
- livebench: skipped by tier (seq235 (tier weights))
- gpqa: CITED WARN (seq235 ds41-qual-1116-1502, 23.9 min) — 181/198 = 91.4% (Wilson95 86.7-94.6), truncated without answer 4, hit max_tokens 7, errors 0, max_to
- hotel16: CITED PASS (seq235 ds41-qual-1116-1502, 8.2 min) — DONE hotel-lights-c16 rc=0 | exact 30 / wrong 0 / truncated 0 | []
- stress16: RUN PASS (rerun, 25.6 min) — PASS 32/32 long, ping 212/212, min free 64 MiB, bad lines 0, wall 25.6 min (min free < 480 MiB kit margin, >= 16 MiB flo
- post: RUN PASS (rerun, 0.1 min) — smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s | served KV 5403008 = launch | verify_fix 10:
- scan: RUN PASS (rerun, 0.0 min) — serve log 0 fault lines, kernel log 0 NVRM fault lines

- served KV 5403008 = seq235's 5403008 (same as every cited stage).
- First stress attempt (seq235, 15:52:09): GPU 2 (PCI e1:00, top card) fell off the bus (Xid 79) at 15:52:51 at ~430-450 W, 42 s into wave 1. Every long request failed with CUDA_ERROR_LAUNCH_FAILED (719), and the server exited. The crash log was lost to the restore. That attempt is superseded by the rerun below and is not a software verdict.
- Stress volume (rerun): capacity 5,403,008 tokens (live KV) -> 2 waves x 16 x ~530,000; 19,027,513 prompt tokens pushed = 3.52x the KV pool; 32/32 long, ping 212/212, min free 64 MiB, wall 25.6 min (seq214: 19,021,130 tokens = 3.52x of 5,403,008).
- No power cap (GPU_PL empty).
- Restore: skipped (RESTORE=0): server stopped by PID, GPUs free
- Lowest MemAvailable MiB: stress16 12716, post 20230, restore 20250.

GPU minutes: seq235 50 cited + rerun 26
QUALIFICATION DONE preset B tier weights: all gates pass (seq235 cited + stress rerun) (WARN cited: decode32 gpqa); KV 5403008; restore skipped (RESTORE=0): server stopped by PID, GPUs free; 25 min | ~/bench-results/ds41-qual-rerun-1116-1850
