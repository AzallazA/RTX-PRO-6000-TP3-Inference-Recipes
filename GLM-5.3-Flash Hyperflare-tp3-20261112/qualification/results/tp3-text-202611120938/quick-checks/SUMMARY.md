# qualification qual

profile: full  
stages: verify eval tools  
reused: none  
gates: 3 PASS, 0 WARN, 0 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 10:03:10 | eval | PASS | 42/43 |
| 10:03:23 | tools | PASS | tool calls: 60/60 OK (0 chose a different tool), concurrency 32, 13 s |
| 10:03:41 | verify | PASS | 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 0 |
