# qualification qual

profile: core  
stages: eval tools verify hotel retrieval longgen gpqa tooleval stress  
reused:  eval tools verify retrieval (from qual)  
gates: 6 PASS, 0 WARN, 1 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 16:49:28 | eval | PASS | REUSED from qual: REUSED from qual: 42/43 |
| 16:49:28 | tools | PASS | REUSED from qual: REUSED from qual: tool calls: 60/60 OK (0 chose a different tool), concurrency 32, 13 s |
| 16:49:28 | verify | PASS | REUSED from qual: REUSED from qual: 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 0 |
| 17:05:28 | hotel-lights | PASS | DONE hotel-lights-c32 rc=0 / exact 30 / wrong 0 / truncated 0 / [] |
| 17:05:28 | retrieval | PASS | REUSED from qual: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass |
| 17:08:36 | longgen | PASS | 2/2 runs, fails [], completions [10086, 20363], worst 8-gram 0.0005 |
| 17:37:49 | gpqa | INFO | 177/198 = 89.4% (Wilson95 84.3-93.0), truncated 0, hit max_tokens 1, errors 0; no baseline: this run can serve as one |
| 17:40:15 | tooleval | INFO | score 92/100 ★★★★★ Excellent; no baseline: this run can serve as one |
| 18:40:24 | stress | FAIL | FAIL 44/64 long, ping 197/197, min free 98 MiB, bad lines 0, wall 60.2 min |
