# qualification qual

profile: full  
stages: verify eval tools  
reused: none  
gates: 3 PASS, 0 WARN, 0 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 10:47:04 | eval | PASS | 43/43 |
| 10:47:25 | tools | PASS | tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 20 s |
| 10:47:50 | verify | PASS | 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 35 |
