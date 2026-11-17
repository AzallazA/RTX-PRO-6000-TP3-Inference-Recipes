15:02:12 start: preset B tier weights on 27864a1d79e4, gb202-tp3-202611160815, KV 5403008, reuse none, seq seq235, MemAvailable 22840 MiB -> ~/bench-results/ds41-qual-1116-1502
15:02:17 gpu_check run PASS (0.1 min): 13/13 passed
15:02:17 gpu_check PASS (0.1 min): 13/13 passed
15:02:22 smoke run PASS (0.1 min): smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s; tool call 3/3 Paris:ok Tokyo:ok Lima:ok
15:02:22 smoke PASS (0.1 min): smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s; tool call 3/3 Paris:ok Tokyo:ok Lima:ok
15:02:27 image run PASS (0.1 min): IMAGE_SMOKE qB served 5/5 correct 5/5 | small: ok 222p 0.1s | large 1920x900 (frontend cap 381 tok): ok 998p 0.2s | over-max 4160x1280 (downscaled): ok 974p 0.3s | two images: ok 422p 0.1s | four images: ok 828p 0.2s
15:02:27 image PASS (0.1 min): IMAGE_SMOKE qB served 5/5 correct 5/5 | small: ok 222p 0.1s | large 1920x900 (frontend cap 381 tok): ok 998p 0.2s | over-max 4160x1280 (downscaled): ok 974p 0.3
15:02:47 tools60 run PASS (0.3 min): tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 18 s
15:02:47 tools60 PASS (0.3 min): tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 18 s
15:03:02 consistency run PASS (0.2 min): 284 positions, top-1 agree 0.951, missing 0, median 0.0081 (gate >= 0.95 / <= 1 / <= 0.01)
15:03:02 consistency PASS (0.2 min): 284 positions, top-1 agree 0.951, missing 0, median 0.0081 (gate >= 0.95 / <= 1 / <= 0.01)
15:03:12 verify run PASS (0.2 min): verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0)
15:03:12 verify PASS (0.2 min): verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0)
15:03:42 eval run PASS (0.5 min): 42/43 (thinking off, temperature clamped to 1.0)
15:03:42 eval PASS (0.5 min): 42/43 (thinking off, temperature clamped to 1.0)
15:05:37 tooleval run PASS (1.9 min): score 90/100, pass 77/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88
15:05:37 tooleval PASS (1.9 min): score 90/100, pass 77/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88
15:08:17 decode32 run WARN (2.7 min): c1 WORSE: 319.5 tok/s vs ref 334.6 (0.955x, band +-4%); steps/s 122.82986915431576; accept 2.6; errors 0|c8 WORSE: 978.8 tok/s vs ref 1044.1 (0.937x, band +-2.5%); steps/s 350.7253299449249; accept 2.35; errors 0|c16 LEVEL: 1331.5 tok/s vs ref 1352.2 (0.985x, band +-3%); steps/s 301.0074432519567; accept 2.21; errors 0
15:08:17 decode32 WARN (2.7 min): c1 WORSE: 319.5 tok/s vs ref 334.6 (0.955x, band +-4%); steps/s 122.82986915431576; accept 2.6; errors 0|c8 WORSE: 978.8 tok/s vs ref 1044.1 (0.937x, band +-2.5
15:09:22 long128 run PASS (1.1 min): 128K-c1 BETTER: 320.8 tok/s vs ref 292.5 (1.097x, band +-4%); steps/s 119.23316991231852; accept 2.69; errors 0
15:09:22 long128 PASS (1.1 min): 128K-c1 BETTER: 320.8 tok/s vs ref 292.5 (1.097x, band +-4%); steps/s 119.23316991231852; accept 2.69; errors 0
15:12:02 long512 run PASS (2.7 min): 512K-c1 BETTER: 292.7 tok/s vs ref 271.0 (1.080x, band +-5%); steps/s 112.76629784947252; accept 2.59; errors 0
15:12:02 long512 PASS (2.7 min): 512K-c1 BETTER: 292.7 tok/s vs ref 271.0 (1.080x, band +-5%); steps/s 112.76629784947252; accept 2.59; errors 0
15:12:17 prefill32 run PASS (0.2 min): 32768 prefill 15766 tok/s (4 samples) vs ref 15958 (0.988x, band 4%)
15:12:17 prefill32 PASS (0.2 min): 32768 prefill 15766 tok/s (4 samples) vs ref 15958 (0.988x, band 4%)
15:17:53 agent8 run INFO (5.6 min): agent c8: 649 turns in 300 s, TTFT p50/p90/max 0.41666698455810547/0.7224347591400146/1.2345826625823975 s, 464.6 output tok/s; errors 0; ref 646 turns, 455.8 out tok/s, p50/p90 0.51/0.90 s
15:17:53 agent8 INFO (5.6 min): agent c8: 649 turns in 300 s, TTFT p50/p90/max 0.41666698455810547/0.7224347591400146/1.2345826625823975 s, 464.6 output tok/s; errors 0; ref 646 turns, 455.8 o
15:19:58 retrieval run PASS (2.1 min): DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
15:19:58 retrieval PASS (2.1 min): DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
15:19:58 longgen skipped - (- min): tier weights
15:19:58 lavd skipped - (- min): tier weights
15:19:58 eos skipped - (- min): tier weights
15:19:58 livebench skipped - (- min): tier weights
15:43:53 gpqa run WARN (23.9 min): 181/198 = 91.4% (Wilson95 86.7-94.6), truncated without answer 4, hit max_tokens 7, errors 0, max_tokens 32768, c8; ref 90.4% (85.5-93.8); paired vs baseline: 198 items, 5 gained / 3 lost, McNemar p 0.727; WARN: truncated without answer > 3; one 198-item sample resolves about +-5 points
15:43:53 gpqa WARN (23.9 min): 181/198 = 91.4% (Wilson95 86.7-94.6), truncated without answer 4, hit max_tokens 7, errors 0, max_tokens 32768, c8; ref 90.4% (85.5-93.8); paired vs baseline: 1
15:52:09 hotel16 run PASS (8.2 min): DONE hotel-lights-c16 rc=0 | exact 30 / wrong 0 / truncated 0 | []
15:52:09 hotel16 PASS (8.2 min): DONE hotel-lights-c16 rc=0 | exact 30 / wrong 0 / truncated 0 | []
15:52:59 stress16 run FAIL (0.8 min): cannot read ~/bench-results/ds41-qual-1116-1502/Q/stress/stress_qB-11161502-c16.json: FileNotFoundError(2, 'No such file or directory')
15:52:59 stress16 FAIL (0.8 min): cannot read ~/bench-results/ds41-qual-1116-1502/Q/stress/stress_qB-11161502-c16.json: FileNotFoundError(2, 'No such file or directory')
15:53:04 server-after-stress16 run FAIL (- min): server pid 2777117 gone after stress16
15:53:04 CRASH after stress16: server pid 2777117 gone after stress16
15:53:04 post notrun NOTRUN (- min): not run: server pid 2777117 gone after stress16
15:53:04 scan notrun NOTRUN (- min): not run: server pid 2777117 gone after stress16
