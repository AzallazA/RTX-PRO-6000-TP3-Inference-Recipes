# Final-binary decode check

FINAL BINDEC LEVEL: c1-32k new 6.236 vs current worst 6.133 (A-r2) +1.7% [D-r1 6.412, D-r2 6.155, D-r3 6.147; info -]; c8-32k new 8.070 vs current worst 8.267 (A-r1) -2.4% [D-r1 8.158, D-r2 8.087, D-r3 7.967; info -] (corrected GPU ms/round at equal offered drafts; final binary vs current binary ds41-steersync-1120-1951 A rounds; FAIL if > current worst x 1.03)

| cell | label | new pooled ms/round corrected @ offered (rounds) | current worst (round); all current rounds | new vs worst | new per round | info |
|---|---|---|---|---|---|---|
| c1-32k | 1x6 | 6.236 @ 4.03 (20940) | 6.133 (A-r2); A-r1 6.132, A-r2 6.133, A-r3 6.132 | +1.7% | D-r1 6.412, D-r2 6.155, D-r3 6.147 |  |
| c8-32k | 8x6 | 8.070 @ 17.73 (8531) | 8.267 (A-r1); A-r1 8.267, A-r2 8.073, A-r3 8.189 | -2.4% | D-r1 8.158, D-r2 8.087, D-r3 7.967 |  |

Reference: ~/bench-results/ds41-steersync-1120-1951 arm A (default package on the current binary). Corrected = ms/round minus slope x offered drafts (e2_analyze SLOPE).

