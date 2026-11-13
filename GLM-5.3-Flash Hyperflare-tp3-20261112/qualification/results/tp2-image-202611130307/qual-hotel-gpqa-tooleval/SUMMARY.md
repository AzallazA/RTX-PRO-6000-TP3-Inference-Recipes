# qualification qual

profile: core  
stages: hotel gpqa tooleval  
reused: none  
gates: 0 PASS, 0 WARN, 1 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 08:22:33 | hotel-lights | FAIL | DONE hotel-lights-c8 rc=0 / exact 26 / wrong 3 / truncated 1 / [(8, 'unparseable: multiple numbers without answer anchor', 35781), (11, 'unparseable: multiple numbers without answer anchor', 49855), (26, 'expected 48, got 49', 10255)] |
| 09:21:52 | gpqa | INFO | 180/198 = 90.9% (Wilson95 86.1-94.2), truncated 1, hit max_tokens 1, errors 0; no baseline: this run can serve as one |
| 09:24:57 | tooleval | INFO | score 92/100 ★★★★★ Excellent; no baseline: this run can serve as one |
