18:30:34 start: <fork> 72548a42 (0 dirty) | ignition from <fork>/ignition/__init__.py | TP3 bin <fork>/flare-cli/target/release/flare | TP2 bin <fork>/flare-cli/target/release/flare
18:30:34 chain start (<results>/glm-lean-1112-1830): T, E3 K3 S3, E2 K2 S2; code 72548a42, expected ~5.5-6.5 h
18:35:32 T lean CPU tests: 19 passed, 4 skipped, 16 warnings in 9.14s
18:35:32 T staging CUDA test exit 0 after 4 min: 16 passed, 61 warnings in 266.71s (0:04:26)
18:35:32 overlay: overlay ready: <fork>/.b12x-overlay
18:35:32 <disk> free 637 GB
19:00:41 E3 export exit 0 after 25 min: <disk> Graph catalog: 74 graphs; staging:  1 vision staging slots 22 MiB (tail aliased: True, parked 373 MiB);
19:00:44 K3 dry run (PCM 32G): gb202-tp3-202611120938: 13,737,984 KV tokens fixed 63.71 GiB mapped-host 0 | gb202-tp3-202611122211: 13,011,456 KV tokens fixed 65.19 GiB mapped-host 373.3 MiB | gb202-tp3-202611130000: 13,393,152 KV tokens fixed 64.37 GiB mapped-host 373.3 MiB | expected TP3 lean ~13.39-13.46M (old vision 13.01M, text 13.74M)
19:01:25 S3 latency (KV served 13130496): cold 0.15 s warm 0.13 s seq8_med 0.13 s large8000 first 1.51 s second 1.51 s (8021 prompt tok) mid4096 0.70 s par4_wall 0.54 s text 0.00 s ok True
19:01:30 S3 LEAN_PROBE lean-tp3: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
19:01:34 S3 smoke: served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.7s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
19:01:59 S3 image eval: T=1.0 effort low: set 60/60 (errors 0, truncated 0) | hard 51/60 (errors 0, truncated 0) | DS41 Flare ref (T=0+1, set): results-vision-staged-real-set.json
19:05:15 old KV served 13523712
19:07:46 S3 decode 32K c1/c8 (1 round, new then old): c1 tok/s 279 vs 281 (0.995x), req steps/s 113.7 vs 111.0; c8 tok/s 870 vs 866 (1.004x), req steps/s 391.3 vs 458.3
19:08:35 S3 probe old vision gb202-tp3-202611122211:  ok 24/24, with logprobs 24/24, same-server repeat identical 12/12 | LEAN_PROBE cmp lean-tp3 vs oldvision-tp3: compared 12/12, exact 5, top-1 same 9, max |dlogprob| 1.434
21:44:50 S3 stress c32 exit 0: stress WARN PASS 64/64 long, ping 246/246, min free 44 MiB, bad lines 0, wall 69.3 min (every request passed; min free below the 480 MiB margin)| | server error lines 0
21:44:51 <disk> free 625 GB
22:07:45 E2 export exit 0 after 22 min: <disk> Graph catalog: 56 graphs; staging:  1 vision staging slots 22 MiB (tail aliased: True, parked 373 MiB);
22:07:46 K2 dry run (PCM 32G): gb202-tp2-202611121033: 2,589,184 KV tokens fixed 85.56 GiB mapped-host 1.18 GiB | gb202-tp2-202611130307: 2,414,592 KV tokens fixed 86.04 GiB mapped-host 1.55 GiB | expected TP3 lean ~13.39-13.46M (old vision 13.01M, text 13.74M)
22:08:23 S2 latency (KV served 2231808): cold 0.15 s warm 0.14 s seq8_med 0.14 s large8000 first 1.46 s second 1.53 s (8021 prompt tok) mid4096 0.71 s par4_wall 0.60 s text 0.00 s ok True
22:08:27 S2 LEAN_PROBE lean-tp2: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
22:08:31 S2 smoke: served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.3s | over-max 4480x3040 (downscaled): ok 8021p 1.7s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
22:09:02 S2 image eval: T=1.0 effort low: set 59/60 (errors 0, truncated 0) | hard 49/60 (errors 0, truncated 0) | DS41 Flare ref (T=0+1, set): results-vision-staged-real-set.json
22:17:21 S2 decode 32K c1/c4/c8 x2 (new only; stored 1033 = 246/514/717 tok/s): c1 252/246 tok/s (mean 249, 1.012x); c4 507/509 tok/s (mean 508, 0.988x); c8 699/705 tok/s (mean 702, 0.979x)
22:34:19 S2 stress c8 exit 0: stress WARN PASS 16/16 long, ping 139/139, min free 96 MiB, bad lines 0, wall 16.3 min (every request passed; min free below the 480 MiB margin)| | server error lines 0
22:34:22 chain finished (<results>/glm-lean-1112-1830); packages: gb202-tp2-202611130307 gb202-tp3-202611130000 
22:35:23 DS41 default restored
