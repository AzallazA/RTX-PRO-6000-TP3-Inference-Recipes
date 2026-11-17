PMERGE VERDICT: HOLD (std8:WNT>=1.05; std8:WNT-gain>band; mixed8:WNT-no-regression) | std8 turns x1.086 out x1.014 WNT x- (band 2.0%) TTFT p50 x0.786 p90 x0.815 launches 3.15->1.00; mixed8 turns x1.083 out x1.068 WNT x- (band 2.0%) TTFT p50 x0.796 p90 x0.930 launches 3.03->1.35; std16 turns x1.097 out x1.244 WNT x1.157 (band 4.6%) TTFT p50 x0.745 p90 x0.705 launches 3.23->1.01; mixed16 turns x1.124 out x1.079 WNT x1.099 (band 4.6%) TTFT p50 x0.827 p90 x0.925 launches 3.07->1.36; decode c1 x1.023; decode c8 x1.011; errors replay 0 decode 0; gpu_check 13/13 passed

Arms: 2 off, 2 on. Rule: see the module docstring of pmerge_analyze.py (and RULE.txt in this folder).

## std8 (agent replay ~3,460 new tokens/turn, 8 streams)

| run | turns | out tok/s | WNT | TTFT p50 | p90 | max | gap p90 | launches/turn | snap/turn | msg ckpt/turn | prefill tok/turn | reused/lookup | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off#1 | 646 | 455.8 | - | 0.509 | 0.903 | 1.451 | 0.887 | 3.19 | 2.60 | 1.00 | 3462 | 111927 | 0 |
| off#2 | 640 | 402.6 | - | 0.532 | 0.900 | 1.415 | 0.918 | 3.11 | 2.61 | 0.99 | 3455 | 111573 | 0 |
| on#1 | 726 | 399.6 | - | 0.414 | 0.719 | 1.095 | 0.946 | 1.00 | 1.64 | 0.00 | 3480 | 114273 | 0 |
| on#2 | 671 | 471.0 | - | 0.404 | 0.750 | 1.174 | 0.966 | 1.00 | 1.61 | 0.00 | 3448 | 111370 | 0 |

ON/OFF (arm means, 2 on / 2 off): turns 1.086, out 1.014, WNT - (band 2.0%), TTFT p50 0.786 p90 0.815 max 0.792 mean 0.812, gap p90 1.059; launches/turn 3.15 -> 1.00, msg ckpt/turn 0.99 -> 0.00, prefill tok/turn x1.001, reused/lookup x1.010
WNT unavailable at 8 streams (needs >= 2 OFF runs per kind and a positive fit)
checks: turns-or-output>=1.05 OK (turns x1.086 out x1.014), WNT>=1.05 NOT MET (WNT x-), WNT-gain>band NOT MET (band 2.0%), TTFT-p50 OK (p50 x0.786 (band +5%)), TTFT-p90 OK (p90 x0.815 (band +5%))

## mixed8 (mixed sizes, 8 streams)

| run | turns | out tok/s | WNT | TTFT p50 | p90 | max | gap p90 | launches/turn | snap/turn | msg ckpt/turn | prefill tok/turn | reused/lookup | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off#1 | 663 | 508.8 | - | 0.303 | 0.939 | 1.800 | 0.988 | 3.03 | 2.67 | 0.97 | 2534 | 64497 | 0 |
| off#2 | 666 | 479.6 | - | 0.303 | 0.995 | 2.089 | 1.023 | 3.04 | 2.61 | 0.96 | 2583 | 62129 | 0 |
| on#1 | 762 | 492.8 | - | 0.245 | 0.900 | 2.547 | 0.530 | 1.34 | 1.68 | 0.00 | 2531 | 61680 | 0 |
| on#2 | 677 | 562.8 | - | 0.237 | 0.898 | 3.404 | 0.532 | 1.37 | 1.70 | 0.00 | 2653 | 61589 | 0 |

ON/OFF (arm means, 2 on / 2 off): turns 1.083, out 1.068, WNT - (band 2.0%), TTFT p50 0.796 p90 0.930 max 1.530 mean 0.836, gap p90 0.528; launches/turn 3.03 -> 1.35, msg ckpt/turn 0.97 -> 0.00, prefill tok/turn x1.013, reused/lookup x0.973
WNT unavailable at 8 streams (needs >= 2 OFF runs per kind and a positive fit)
checks: WNT-no-regression NOT MET (WNT x- band 2.0%), TTFT-p50 OK (p50 x0.796 (band +5%)), TTFT-p90 OK (p90 x0.930 (band +6%))

| class | turns off / on | TTFT p50 off / on | p90 off / on (ratio) |
|---|---|---|---|
| small | 347.5 / 373.0 | 0.225 / 0.167 | 0.397 / 0.295 (0.744) |
| medium | 223.5 / 238.5 | 0.327 / 0.269 | 0.515 / 0.398 (0.773) |
| large | 93.5 / 108.0 | 1.024 / 0.955 | 1.436 / 1.235 (0.860) |

## std16 (agent replay ~3,460 new tokens/turn, 16 streams)

| run | turns | out tok/s | WNT | TTFT p50 | p90 | max | gap p90 | launches/turn | snap/turn | msg ckpt/turn | prefill tok/turn | reused/lookup | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off#1 | 592 | 454.5 | 1.024 | 0.533 | 1.004 | 2.941 | 1.216 | 3.30 | 2.62 | 1.01 | 3742 | 103456 | 0 |
| off#2 | 625 | 374.6 | 0.978 | 0.582 | 1.068 | 2.820 | 1.301 | 3.15 | 2.54 | 1.01 | 3823 | 102035 | 0 |
| on#1 | 698 | 497.3 | 1.169 | 0.426 | 0.747 | 1.259 | 1.057 | 1.01 | 1.66 | 0.00 | 3432 | 103752 | 0 |
| on#2 | 637 | 534.1 | 1.146 | 0.404 | 0.714 | 2.419 | 1.220 | 1.02 | 1.69 | 0.00 | 3509 | 101925 | 0 |

ON/OFF (arm means, 2 on / 2 off): turns 1.097, out 1.244, WNT 1.157 (band 4.6%), TTFT p50 0.745 p90 0.705 max 0.638 mean 0.742, gap p90 0.905; launches/turn 3.23 -> 1.01, msg ckpt/turn 1.01 -> 0.00, prefill tok/turn x0.918, reused/lookup x1.001
WNT fit on OFF 16-stream runs: P std16 0.293 s/turn, mixed16 0.209 s/turn, D 0.982 ms/token (~1019 tok/s)
checks: turns-or-output>=1.05 OK (turns x1.097 out x1.244), WNT>=1.05 OK (WNT x1.157), WNT-gain>band OK (band 4.6%), TTFT-p50 OK (p50 x0.745 (band +9%)), TTFT-p90 OK (p90 x0.705 (band +6%))

## mixed16 (mixed sizes, 16 streams)

| run | turns | out tok/s | WNT | TTFT p50 | p90 | max | gap p90 | launches/turn | snap/turn | msg ckpt/turn | prefill tok/turn | reused/lookup | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off#1 | 687 | 551.3 | 1.020 | 0.312 | 0.990 | 3.508 | 1.148 | 3.06 | 2.55 | 0.98 | 2508 | 60997 | 0 |
| off#2 | 591 | 575.0 | 0.977 | 0.310 | 1.015 | 2.756 | 1.258 | 3.08 | 2.57 | 0.97 | 2741 | 59990 | 0 |
| on#1 | 735 | 606.2 | 1.108 | 0.254 | 0.902 | 1.811 | 0.748 | 1.36 | 1.72 | 0.01 | 2585 | 60487 | 0 |
| on#2 | 702 | 609.0 | 1.087 | 0.261 | 0.952 | 1.987 | 0.737 | 1.37 | 1.70 | 0.00 | 2734 | 61259 | 0 |

ON/OFF (arm means, 2 on / 2 off): turns 1.124, out 1.079, WNT 1.099 (band 4.6%), TTFT p50 0.827 p90 0.925 max 0.606 mean 0.817, gap p90 0.617; launches/turn 3.07 -> 1.36, msg ckpt/turn 0.98 -> 0.01, prefill tok/turn x1.013, reused/lookup x1.006
WNT fit on OFF 16-stream runs: P std16 0.293 s/turn, mixed16 0.209 s/turn, D 0.982 ms/token (~1019 tok/s)
checks: WNT-no-regression OK (WNT x1.099 band 4.6%), TTFT-p50 OK (p50 x0.827 (band +5%)), TTFT-p90 OK (p90 x0.925 (band +5%))

| class | turns off / on | TTFT p50 off / on | p90 off / on (ratio) |
|---|---|---|---|
| small | 328.0 / 362.5 | 0.228 / 0.175 | 0.446 / 0.320 (0.716) |
| medium | 221.5 / 252.0 | 0.334 / 0.279 | 0.585 / 0.421 (0.719) |
| large | 89.5 / 104.0 | 1.027 / 0.993 | 1.343 / 1.319 (0.982) |

## decode (llm_decode_bench 32K, 30 s, one pass per arm)

| cell | OFF tok/s (runs) | ON tok/s (runs) | ON/OFF | floor | recorded integration pass | errors |
|---|---|---|---|---|---|---|
| c1 | 303.6 (313.4, 293.7) | 310.6 (303.9, 317.2) | 1.023 | 0.960 | 322.7 (OFF x0.941) | 0 |
| c8 | 1006.8 (1024.4, 989.2) | 1017.4 (1013.6, 1021.2) | 1.011 | 0.975 | 1078.8 (OFF x0.933) | 0 |
