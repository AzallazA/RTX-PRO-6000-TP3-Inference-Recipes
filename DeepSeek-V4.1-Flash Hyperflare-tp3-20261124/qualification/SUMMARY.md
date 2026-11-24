# qualification minqual-tp3-warmind-0502

profile: full  
stages: eval tools hotel retrieval stress  
reused: none  
gates: 4 PASS, 1 WARN, 0 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 07:32:06 | eval | PASS | 43 / 43 |
| 07:32:26 | tools | PASS | tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 20 s |
| 07:39:21 | hotel-lights | PASS | DONE hotel-lights-c16 rc=0 / exact 30 / wrong 0 / truncated 0 / [] |
| 07:40:43 | retrieval | PASS | DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass |
| 07:57:03 | stress | WARN | PASS 32/32 long, ping 142/142, min free 78 MiB, bad lines 0, wall 16.3 min (every request passed; min free below the 480 MiB margin) |
