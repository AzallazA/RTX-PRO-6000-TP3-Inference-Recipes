# qualification Q

profile: core  
stages: stress  
reused: none  
gates: 2 PASS, 2 WARN, 0 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 20:14:51 | consistency | INFO | 192 positions (0 sequences cut at ambiguous tokens): top-1 agree 0.958, decode top-1 missing from prefill top-5 0, /dlogprob/ median 0.0088 p90 0.1335 p99 0.3294 max 0.6894 |
| 20:15:31 | eval | PASS | 42/43 |
| 20:17:36 | retrieval | PASS | DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass |
| 20:46:27 | hotel-lights | WARN | DONE hotel-lights-c8 rc=0 / exact 29 / wrong 1 / truncated 0 / [(25, 'expected 48, got 49', 7768)] |
| 21:01:43 | stress | WARN | PASS 16/16 long, ping 92/92, min free 84 MiB, bad lines 0, wall 15.3 min (every request passed; min free below the 480 MiB margin) |
