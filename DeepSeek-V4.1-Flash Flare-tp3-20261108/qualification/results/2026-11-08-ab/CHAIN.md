00:55 chain start, results in ~/bench-results/bisect-1108-0055
01:20 export variantA exit 0 after 24 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080612 [code: ~/flare-up-base/ignition/__init__.py; newest kernel: decoder_6eff858545]
01:46 export vision exit 0 after 26 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080639 [code: ~/flare-up-vision/ignition/__init__.py; newest kernel: decoder_9cf2e32122]
02:10 export s2gated exit 0 after 24 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080703 [code: ~/flare-up-stage2/ignition/__init__.py; newest kernel: decoder_1bad0832d4]
02:35 export s2ungated exit 0 after 25 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080728
  vision weight staging (memcpy): 190 MiB of slots, max over vision graphs 
[    OK    ] Construct graph catalog.  vision weight staging (m…  129/129  17.1s
02:36 default KV 2461184 verify: 10 runs, 1958 tokens: TRUE corruptions 0, natural-uncertainty flags 20
02:40 bench default r1: c1=301 c4=725 c8=919
02:41 variantA KV 2461184 verify: 10 runs, 1654 tokens: TRUE corruptions 0, natural-uncertainty flags 29
02:45 bench variantA r1: c1=303 c4=732 c8=980
02:46 s2gated KV 2461184 verify: 10 runs, 1718 tokens: TRUE corruptions 0, natural-uncertainty flags 20
02:50 bench s2gated r1: c1=286 c4=673 c8=938
02:52 s2ungated KV 2461184 verify: 10 runs, 1584 tokens: TRUE corruptions 0, natural-uncertainty flags 7
02:56 bench s2ungated r1: c1=291 c4=706 c8=942
03:01 bench default r2: c1=305 c4=723 c8=981
03:06 bench variantA r2: c1=337 c4=717 c8=982
03:11 bench s2gated r2: c1=299 c4=704 c8=938
03:15 bench s2ungated r2: c1=285 c4=696 c8=-3
03:21 bench default r3: c1=304 c4=729 c8=972
03:26 bench variantA r3: c1=294 c4=728 c8=977
03:31 bench s2gated r3: c1=298 c4=695 c8=953
03:36 bench s2ungated r3: c1=297 c4=691 c8=954
03:37 vision KV 2461184 verify: 10 runs, 2144 tokens: TRUE corruptions 0, natural-uncertainty flags 28
03:37 vision-staged: cold 0.14 0.11 0.11 0.11 0.11 0.11 | median 0.11 s ; warm median 0.00 s
[    OK    ] Prepare host prefix cache.  memory 28.2 GiB (74637 slots, split by rank) · CPU inputs 4.0 GiB limit, 0.0 GiB staging  2.1s
03:39 default: cold 1.72 1.70 1.70 1.69 1.69 1.68 | median 1.70 s ; warm median 0.00 s
03:40 default restored: alpha 0.8
03:40 chain done
03:42 vision staged real: T=0: total 49/60 | chart 12/12 count 12/12 grid 1/12 lines 12/12 ocr 12/12 T=1: total 49/60 | chart 12/12 count 12/12 grid 1/12 lines 12/12 ocr 12/12 
T=0 median secs 1.60 ok 49/60
T=1 median secs 0.60 ok 49/60
03:44 default restored
