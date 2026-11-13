2026-11-12T16:49:28 — outputs: qual
2026-11-12T16:49:28 — qualification start: packages/models/glm53-fp4-local/gb202-tp3-202611120938 stages: eval tools verify hotel retrieval longgen gpqa tooleval stress
2026-11-12T16:49:28 — model GLM-5.3-Flash, effort max (eval low, tools high), temperature 1.0, top-p 0.95
2026-11-12T16:49:28 — KV 13523712, power 450.00 W 450.00 W 450.00 W 
2026-11-12T16:49:28 — profile: core; longgen seeds 201 202; REUSE_FROM qual (same fingerprint)
2026-11-12T16:49:28 — GATE eval PASS: REUSED from qual: REUSED from qual: 42/43
2026-11-12T16:49:28 — GATE tools PASS: REUSED from qual: REUSED from qual: tool calls: 60/60 OK (0 chose a different tool), concurrency 32, 13 s
2026-11-12T16:49:28 — GATE verify PASS: REUSED from qual: REUSED from qual: 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 0
2026-11-12T17:05:28 —  DONE hotel-lights-c32 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-12T17:05:28 — GATE hotel-lights PASS: DONE hotel-lights-c32 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-12T17:05:28 — GATE retrieval PASS: REUSED from qual: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
2026-11-12T17:08:36 — long generation: 2 runs, fails 0 | prompt [530058] | completions [10086, 20363] | repeat_8gram [0.0005, 0.0]
2026-11-12T17:08:36 — GATE longgen PASS: 2/2 runs, fails [], completions [10086, 20363], worst 8-gram 0.0005
2026-11-12T17:08:36 — gpqa: GPQA Diamond, model GLM-5.3-Flash, concurrency 32, max_tokens bench default, baseline none
2026-11-12T17:37:49 — GATE gpqa INFO: 177/198 = 89.4% (Wilson95 84.3-93.0), truncated 0, hit max_tokens 1, errors 0; no baseline: this run can serve as one
2026-11-12T17:40:15 — GATE tooleval INFO: score 92/100 ★★★★★ Excellent; no baseline: this run can serve as one
2026-11-12T18:40:24 — stress: [glm-hyperflare-tp3-c32] FAIL: 44/64 long requests ok, pinger 197/197 ok, 25,766,822 prompt tok pushed, min free VRAM 98 MiB (1750 samples), health=200, bad log lines=0, wall 60.2 min
2026-11-12T18:40:24 — GATE stress FAIL: FAIL 44/64 long, ping 197/197, min free 98 MiB, bad lines 0, wall 60.2 min
2026-11-12T18:40:24 — gates: 6 PASS, 0 WARN, 1 FAIL (profile core; reused: eval tools verify retrieval; SUMMARY.md)
2026-11-12T18:40:24 — qualification done
