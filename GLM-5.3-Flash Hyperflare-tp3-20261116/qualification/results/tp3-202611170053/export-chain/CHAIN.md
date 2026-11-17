19:17:56 chain start: stages 'X K S', ~/<fork-worktree> perf/glm-vram-reclaim 7a4c4b21 GLM export-layout VRAM switches: trial graph pool, slab segment, fitted one-shot  (contains 99ec7a9f), bin 67c691d29041, config ~/.config/flare-port-glm-careful/config.toml, FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_SLAB_SEGMENT=1 FLARE_BF16_BMM_CUBLAS=0, S single QUAL 0; <results>/port-g3reclaim-1116-1917
19:17:56 X export start (230 GB free on <store>; overrides: none)
19:53:26 gate X-export PASS: gb202-tp3-202611170053 in 35 min (2130 s)
19:53:26 gate X-published-untouched PASS: gb202-tp3-202611151109 file list/sizes/mtimes/inodes unchanged (867de7c53de9b100)
19:53:26 info X-wall: export wall 2130 s; [ OK ] Prepare and tune kernels. 117.6s;[ OK ] Export native core. <store>/models/glm53-fp4… 313.8s;
19:53:26 gate X-vision-staging PASS: vision blocks and tail parked in device-visible host memory and staged per image (FLARE_GLM53_VISION_STAGE=1) | vision staging slots 22 MiB (parked 373 MiB, tail copied per image 108 MiB): free=29507 MiB, allocated=4590 MiB, reserved=4672 MiB, rss=3727 MiB
19:53:26 gate X-vision-pool PASS: largest vision pool vision32768 976.0 MiB vs largest text pool prefill4096 1008.4 MiB (margin 32.4 MiB; 6 vision graphs, 68 text graphs)
19:53:26 info X-pools: Graph pool requests (MiB, 74 compiled forwards, largest first; max 1008.4 in prefill4096); text max prefill4096 1008.4 MiB; vision: vision32768 976.0, vision16384 488.0, vision4096 194.5, vision1024 136.0, vision256 120.1, vision64 110.2
19:53:26 info X-graphs: Graph catalog: 74 graphs (before compilation/capture) (130000: 74)
19:53:26 info X-moe: 
19:53:26 info X-tuning-mode: no explicit sharded/reconciled line in the log (see EXTRACT.md: last b12x ready lines)
19:53:26 gate X-reclaim-lines PASS: trial graphs in their own pool, released before the serving capture (FLARE_GLM53_TRIAL_POOL=separate) | trial graph pool released: 3574 of 3574 MiB returned before the serving capture | prefill slab 528.26 MiB in 530 MiB of its own segments (FLARE_GLM53_SLAB_SEGMENT=1) | 
19:53:26 info X-reclaim-inventory: rank 0 MiB: core 62448 (151109 64,554); retained graph pool 1110 (151109 3216; T1 floor 2106, upper ~1010); other owned 826.13 (151109 826.13; T2 -528.26); prep-only graph pool 0 (151109 1368)
19:53:26 stage X done (35 min)
19:53:27 gate K-dry-KV PASS: 15025152 tokens vs 14035968 (ratio 1.0705, gate >= 0.995)
19:53:27 info K-dry-KV-130000: 151109 on the same binary: 14035968 (recorded 14035968); new/same-binary ratio 1.0705
19:53:27 gate K-reclaim-gain PASS: +989184 tokens vs 151109 on the same binary (floor-to-upper; expectation T1+T2 14810880..15310848)
19:53:27 info K-fixed: new: 60.98 GiB, mapped host 373.3 MiB | 130000: 63.04 GiB, mapped host 373.3 MiB
19:53:27 gate K-rank2-memory PASS: ranks 0/1/2 MiB: used 66,321/66,319/66,315 (r2-max -6); unattributed 1,333/1,331/1,333 (r2-max +0); retained:Allocator slack 19/19/18 (r2-max -1); core 62,448/62,448/62,442 (r2-max -6); retained 62,449/62,449/62,443 (r2-max -6) (tol +64 MiB)
19:53:27 info K-kv_b: no kv_b owner among the listed top groups
19:53:27 info K-tail-example: retained: 22.06 MiB resources[28][*]<VisionStaging>.blocks<WeightStaging>.slots[*]['block'] — ignition/components/vision_staging.py:80
19:53:28 info K-rank2-151109: ranks 0/1/2 MiB: used 69,827/69,795/69,789 (r2-max -38); unattributed 1,365/1,333/1,333 (r2-max -32); retained:Allocator slack 21/21/20 (r2-max -1); core 64,554/64,554/64,548 (r2-max -6); retained 64,555/64,555/64,549 (r2-max -6) (reference export log)
19:53:28 info K-tail-provenance: model.visual.tail mentions in the package inspect: 0; blocks: 0
19:53:28 stage K done (0 min)
19:53:58 serve s-new: gb202-tp3-202611170053 KV 14834688, boot 30 s
19:53:58 gate S-serve PASS: gb202-tp3-202611170053 via serve-glm.sh (interactive, alpha 0.8), boot 30 s
19:53:58 gate S-kv PASS: served KV 14834688 vs 13833984 (ratio 1.0723, gate >= 0.995)
19:53:58 info S-kv-gain: served KV 14834688 vs 151109 13833984: 1000704 tokens (1.0723x)
19:54:04 gate S-image-latency PASS: cold 0.14 s (<= 0.20; 130000 11-12: 0.15 s): cold 0.14 s warm 0.12 s seq8_med 0.13 s large8000 first 1.46 s second 1.47 s (8021 prompt tok) mid4096 0.68 s par4_wall 0.45 s text 0.00 s ok True
19:54:09 gate S-probe PASS: LEAN_PROBE portg3-new: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
19:54:09 info S-probe-cmp-stored:  compared 12/12, exact 12, top-1 same 12, max |dlogprob| 0 (vs 130000 11-12 probe, report only, C.27)
19:54:13 gate S-smoke PASS: IMAGE_SMOKE portg3-new served 5/5 correct 5/5 | small: ok 309p 0.1s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.6s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
19:55:49 prefill probe new-tp3-202611170053 rc=0 (1 min)
19:55:49 gate S-prefill-probe PASS: new-tp3-202611170053: PASS (marginal: needle_33K) | new-tp3-202611151109: PASS (comparator: recorded 151109, glm-prefill-probe-1115-0621)
19:58:27 gate S-single-decode PASS: decode vs 151109 recorded (same binary, median of 2): c1 279 vs 274 tok/s (1.019x), steps/s 117.6 vs 113.0 (1.041x); c8 887 vs 873 tok/s (1.016x), steps/s 465.2 vs 427.6 (1.088x); c32 1812 vs 1806 tok/s (1.004x), steps/s 753.3 vs 746.5 (1.009x)
19:58:27 gate S-single-prefill PASS: prefill-only vs 151109 recorded: 8K 10408 vs 10362 tok/s (1.004x); 32K 11712 vs 11574 tok/s (1.012x)
19:58:27 gate S-alive PASS: server answering after the S checks
19:58:27 stage S done (4 min)
19:58:34 restore skipped (RESTORE=0): GPUs idle (no compute process, < 1000 MiB used)
19:58:34 chain finished, exit 0 after 40 min; results <results>/port-g3reclaim-1116-1917
19:58:34 GLM RECLAIM TP3 VERDICT: PASS
