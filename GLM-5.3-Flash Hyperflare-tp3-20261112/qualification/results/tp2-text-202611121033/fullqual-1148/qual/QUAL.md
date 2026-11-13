2026-11-12T18:41:31 — outputs: qual
2026-11-12T18:41:31 — qualification start: packages/models/glm53-fp4-local/gb202-tp2-202611121033 stages: eval tools verify hotel retrieval longgen gpqa tooleval stress
2026-11-12T18:41:31 — model GLM-5.3-Flash, effort max (eval low, tools high), temperature 1.0, top-p 0.95
2026-11-12T18:41:31 — KV 2446336, power 450.00 W 450.00 W 450.00 W 
2026-11-12T18:41:32 — profile: core; longgen seeds 201 202; REUSE_FROM qual (same fingerprint)
2026-11-12T18:41:32 — GATE eval PASS: REUSED from qual: 43/43
2026-11-12T18:41:32 — GATE tools PASS: REUSED from qual: tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 20 s
2026-11-12T18:41:32 — GATE verify PASS: REUSED from qual: 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 35
2026-11-12T19:12:17 —  DONE hotel-lights-c8 rc=0 | exact 26 / wrong 4 / truncated 0 | [(5, 'unparseable: multiple numbers without answer anchor', 33815), (14, 'unparseable: multiple numbers without answer anchor', 33221), (19, 'expected 48, got 49', 9301), (29, 'expected 48, got 47', 78831)]
2026-11-12T19:12:17 — GATE hotel-lights FAIL: DONE hotel-lights-c8 rc=0 | exact 26 / wrong 4 / truncated 0 | [(5, 'unparseable: multiple numbers without answer anchor', 33815), (14, 'unparseable: multiple numbers without answer anchor', 33221), (19, 'expected 48, got 49', 9301), (29, 'expected 48, got 47', 78831)]
2026-11-12T19:14:26 —  DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
2026-11-12T19:14:26 — GATE retrieval PASS: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
2026-11-12T19:17:40 — long generation: 2 runs, fails 0 | prompt [530058] | completions [11164, 17340] | repeat_8gram [0.0004, 0.0]
2026-11-12T19:17:40 — GATE longgen PASS: 2/2 runs, fails [], completions [11164, 17340], worst 8-gram 0.0004
2026-11-12T19:17:40 — gpqa: GPQA Diamond, model GLM-5.3-Flash, concurrency 8, max_tokens bench default, baseline none
2026-11-12T20:12:19 — GATE gpqa INFO: 175/189 = 92.6% (Wilson95 88.0-95.5), truncated 2, hit max_tokens 2, errors 9; no baseline: this run can serve as one
2026-11-12T20:12:20 — GATE tooleval FAIL: exit 1, {'kind': 'server', 'message': 'Cannot connect to the OpenAI-compatible server at http://localhost:15015: ConnectError: [Errno 111] Connection refused', 'hint': 
2026-11-12T20:12:20 — stress: FAIL (no flare_total_kv_tokens metric)
2026-11-12T20:12:20 — GATE stress FAIL: no flare_total_kv_tokens metric
2026-11-12T20:12:20 — gates: 5 PASS, 0 WARN, 3 FAIL (profile core; reused: eval tools verify; SUMMARY.md)
2026-11-12T20:12:20 — qualification done
