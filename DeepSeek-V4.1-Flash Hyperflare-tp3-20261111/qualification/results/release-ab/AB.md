# Stage A: base vs rl24 (32K decode, 60 s cells, serve-best.sh flags)

| round | base c1 | base c4 | base c8 | rl24 c1 | rl24 c4 | rl24 c8 | c8 ratio |
|---|---|---|---|---|---|---|---|
| r1 | 305 | 739 | 1001 | 312 | 723 | 1028 | 1.0269 |
| r2 (invalid) | 305 | ERR | 994 | 301 | 731 | 1010 | - |
| r3 | 306 | 731 | 982 | 312 | 738 | 974 | 0.9923 |
| r4 (invalid) | 286 | ERR | 977 | 303 | 731 | 1007 | - |
| r5 (invalid) | 304 | 733 | 987 | 307 | 745 | ERR | - |

PICK base: only 2 valid rounds (need 3)
