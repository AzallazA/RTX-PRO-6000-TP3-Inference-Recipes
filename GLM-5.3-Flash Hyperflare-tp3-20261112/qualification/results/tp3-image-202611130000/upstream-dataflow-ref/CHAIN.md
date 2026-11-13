16:27:38 start: <fork> be55368c (0 dirty) | ignition from <fork>/ignition/__init__.py | TP3 bin <fork>/flare-cli/target/release/flare | TP2 bin <fork>/flare-cli/target/release/flare
16:27:38 chain start (<results>/glm-vision-1112-1627): T, E3 K3 S3, E2 K2 S2; code be55368c, expected ~5.5-6.5 h
16:29:35 T staging CUDA test exit 0 after 1 min: 16 passed, 477 warnings in 96.61s (0:01:36)
16:29:35 overlay: overlay ready: <fork>/.b12x-overlay
16:29:35 <disk> free 659 GB
17:11:49 E3 export exit 0 after 42 min: <disk> Graph catalog: 74 graphs; staging:  1 vision staging slots 22 MiB (tail aliased: True, parked 373 MiB);
17:11:51 K3 dry run (PCM 32G): gb202-tp3-202611120938: 13,737,984 KV tokens fixed 63.71 GiB mapped-host 0 | gb202-tp3-202611122211: 13,011,456 KV tokens fixed 65.19 GiB mapped-host 373.3 MiB | expected TP3 ~13.22M vs 13.49M
17:12:32 S3 latency (KV served 12784896): cold 0.14 s warm 0.13 s seq8_med 0.12 s large8000 first 1.48 s second 1.50 s (8021 prompt tok) mid4096 0.69 s par4_wall 0.52 s text 0.00 s ok True
17:12:36 S3 smoke: served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.7s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
17:12:58 S3 image eval: T=1.0 effort low: set 60/60 (errors 0, truncated 0) | hard 49/60 (errors 0, truncated 0) | DS41 Flare ref (T=0+1, set): results-vision-staged-real-set.json
17:16:15 old KV served 13523712
17:18:46 S3 decode 32K c1/c8 (1 round, new then old): c1 tok/s 277 vs 278 (0.994x), req steps/s 115.2 vs 114.2; c8 tok/s 855 vs 867 (0.985x), req steps/s 390.9 vs 458.7
18:29:14 S3 stress c32 exit 0: stress WARN PASS 64/64 long, ping 248/248, min free 86 MiB, bad lines 0, wall 69.7 min (every request passed; min free below the 480 MiB margin)| | server error lines 0
18:29:16 signal received
18:30:33 chain interrupted or failed early (see CHAIN.md); DS41 default restored
18:30:33 DS41 default restored
