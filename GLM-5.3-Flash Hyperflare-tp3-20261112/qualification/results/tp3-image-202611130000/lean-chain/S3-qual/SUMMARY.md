# qualification S3-qual

profile: core  
stages: stress  
reused: none  
gates: 0 PASS, 1 WARN, 0 FAIL

| time | gate | result | detail |
|---|---|---|---|
| 01:18:40 | stress | WARN | PASS 64/64 long, ping 246/246, min free 44 MiB, bad lines 0, wall 69.3 min (every request passed; min free below the 480 MiB margin) |
