20:44:51 start: stages 'X K Sq Bq', perf/glm-vram-reclaim 7a4c4b21 GLM export-layout VRAM switches: trial graph pool, slab segment, fitted one-shot  (contains 99ec7a9f), bin 67c691d29041a6453b5a847c0d544b7372b9772b3f99d691ecc414baf70b49a2, config ~/.config/flare-port-glm-careful/config.toml, overlay ~/<serial-overlay>, REF_ARM 0 REF_PROBE 0 PROBE 1 PASSES 1 SERIAL moe\.experts$; tuning DB: PRESENT (1226 race checkpoints; the export resumes from it)
20:45:04 X export start (220 GB free on <store>; 8 existing packages fingerprinted; tuning DB PRESENT (1226 race checkpoints; the export resumes from it))
20:45:05 gate X-env PASS: export pid 2264655 environment: FLARE_BF16_BMM_CUBLAS=0 FLARE_GLM53_EXPERT_ACTIVATION=a16 FLARE_GLM53_KDA_SIDE_CHAIN=1 FLARE_GLM53_EMBED_HOST=1 FLARE_GLM53_VISION_STAGE=1 FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_ONESHOT_FIT=1 B12X_SERIAL_RACES=moe\.experts$, overlay ~/<serial-overlay>, port kernel cache; command = 130307's flags
21:02:04 gate X-export PASS: gb202-tp2-202611170201 in 17 min (1021 s)
21:02:05 gate X-existing-packages-untouched PASS: 8 packages incl. gb202-tp2-202611151621: file list/sizes/mtimes/inodes unchanged
21:02:05 gate X-plan PASS: GPUs 0-1 · concurrency 8 · images 2 · up to 5 (= 130307's)
21:02:05 gate X-embed-host PASS: token embedding in device-visible host memory (FLARE_GLM53_EMBED_HOST=1)
21:02:05 gate X-vision-staging PASS: vision blocks and tail parked in device-visible host memory and staged per image (FLARE_GLM53_VISION_STAGE=1) | vision staging slots 22 MiB (parked 373 MiB, tail copied per image 108 MiB): free=6877 MiB, allocated=4418 MiB, reserved=4546 MiB, rss=4870 MiB
21:02:05 gate X-vision-pool PASS: largest vision pool vision32768 976.0 MiB vs largest text pool prefill4096 1072.1 MiB (margin 96.1 MiB; 6 vision graphs, 50 text graphs)
21:02:05 info X-graphs: Graph catalog: 56 graphs (before compilation/capture) (130307: 56)
21:02:05 info X-tuning: preparation id 5d60dc65a6a1: last progress line: 846/1176 ready, 69 measured, 603 cached, 0 restored, 0 compilations at 1:08 | [ OK ] Prepare and tune kernels. 270.4s | selected kernels: {'cached': 21384, 'tuned': 2616, 'fixed': 11986} | [ OK ] Compile model forwards. 3 compiler workers; per ra… 28/28 311.1s (130307 11-12: 1176 items, 0 restored, tune 664 s at 8 compilers)
21:02:05 info X-serial: ledger rows 1432, with serial_ranks true 58
21:02:05 info X-inventory: ranks 0/1 MiB: used 89,929/89,929; unattributed 1,243/1,243; retained:Allocator slack 17/17; core 85,696/85,696; retained 85,697/85,697; mapped_host 1,583/1,583
21:02:05 gate X-reclaim-lines PASS: trial graphs in their own pool, released before the serving capture (FLARE_GLM53_TRIAL_POOL=separate) | trial graph pool released: 2990 of 2990 MiB returned before the serving capture | one-shot all-reduce sized for 512 rows (4 MiB per call) instead of 4096 (FLARE_GLM53_ONESHOT_FIT=1; prefill of 1024+ rows uses the copy-engine ring) | 
21:02:05 info X-reclaim-inventory: rank 0 MiB: core 85,696; retained graph pool 1,110; TP communication 138; other owned 147.41; prep-only graph pool 0 (151621: graph pool 2442, TP communication 502 (W2 -280), prep-only pool 1622; W1 floor pool 1914, upper ~1074)
21:02:05 stage X done (17 min)
21:02:06 gate K-dry-KV PASS: new 3177472 vs 130307 2653184 on the same binary: 1.1976x, +19.76%, +524288 tokens (gate >= 0.995)
21:02:06 gate K-reclaim-gain PASS: +524288 tokens vs 151621 on the same binary (above-upper; expectation W1+W2 2907136..3169280)
21:02:06 info K-fixed: new: 83.69 GiB, mapped host 1.55 GiB | 130307: 85.34 GiB, mapped host 1.55 GiB per GPU
21:02:06 KV dry-run (same binary 67c691d29041): new 3177472 vs 151621 2653184 (1.1976x, +19.76%, +524288 tokens) | served: new ? vs 151621 recorded 2518016
21:02:07 gate K-metadata PASS: metadata new vs 130307: equal: identity=16ae6cd8459b8fab2e755392a7cc6774333b63ac, max_active=8, prefill_chunk_size=4096, page_tokens=512, feature_slots=2, widths=[1, 2, 3, 4, 5, 6], proposal=mtp, deployment=[('gb202-tp2', 2, [0, 1])], graphs=56 graphs, decode_graphs (0 = --mtp-only)=0
21:02:07 info K-memory: mapped host MiB per rank new 1,583/1,583 vs 130307 1,583/1,583; device GiB per rank new 83.69/83.69 vs 85.34/85.34; payloads new 102 (167.3 GiB) vs 102 (167.3 GiB)
21:02:07 stage K done (0 min)
21:02:35 serve new: gb202-tp2-202611170201 pid 2437530 KV 3049984 boot 28 s; args: ~/<fork-worktree>/flare-cli/target/release/flare run <store>/models/glm53-fp4-local/gb202-tp2-202611170201 --gpus 0,1 --port 15015 --host 127.0.0.1 --served-model-name GLM-5.3-Flash --prefix-cache-memory 32G --mtp-draft-tokens 5 --scheduler-policy interactive
21:02:35 gate S-serve PASS: gb202-tp2-202611170201 via serve-glm.sh GPUS=0,1 ALPHA= (package alpha, interactive, 32G, K 5), sha 67c691d29041, boot 28 s
21:02:35 gate S-kv PASS: served KV new 3049984 vs 130307 recorded 2518016: 1.2113x, +21.13%, +531968 tokens (gate >= 0.995)
21:02:35 KV dry-run (same binary 67c691d29041): new 3177472 vs 151621 2653184 (1.1976x, +19.76%, +524288 tokens) | served: new 3049984 vs 151621 recorded 2518016 (1.2113x, +21.13%, +531968 tokens)
21:02:40 gate S-smoke PASS: LEAN_PROBE portg3tp2-new: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
21:02:40 info S-smoke-cmp:  compared 12/12, exact 12, top-1 same 12, max |dlogprob| 0 (vs 130307 11-12 stored probe, report only)
21:02:44 gate S-image-smoke PASS: IMAGE_SMOKE portg3tp2-new served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.6s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
21:03:25 gate S-verify PASS: QC 8: 20 runs, 6000 tokens: TRUE corruptions 0, natural-uncertainty flags 5
21:03:25 gate S-tools PASS: QC 8: tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 17 s
21:04:52 prefill probe new-tp2-202611170201 rc=0 (186 lines, 1 min)
21:04:52 gate S-alive PASS: server answering after the Sq checks
21:04:52 stage Sq done (2 min)
21:07:32 gate B-decode PASS: decode vs 151621 recorded on the same binary (port-g3tp2-1115-0942, 2 passes): c1 247 vs 245 tok/s (1.004x), steps/s 105.0 vs 103.9 (1.011x); c4 519 vs 517 tok/s (1.005x), steps/s 276.5 vs 280.2 (0.987x); c8 763 vs 748 tok/s (1.020x), steps/s 314.2 vs 310.0 (1.014x)
21:07:32 info B-plan-rule: (steps >= 0.98x and tok/s >= 0.96x per cell, report only): c1 met, c4 met, c8 met
21:07:32 info B-band: (recorded 130307 min-max, widened by the 1033 session spread): c1 247 vs recorded 246-252 (band 236-262): inside; c4 519 vs recorded 507-509 (band 497-519): ABOVE; c8 763 vs recorded 699-705 (band 692-712): ABOVE
21:07:32 info B-accept: (tokens per verify, new, median): c1 2.35, c4 1.88, c8 2.43
21:07:32 info B-recorded-compare: (new vs recorded 130307, old binary): c1 0.991x tok, 1.010x steps; c4 1.022x tok, 1.004x steps; c8 1.087x tok, 1.056x steps
21:07:32 info B-prefill-only: tok/s: 8K new 11559 vs 130307 same binary 11403/11385 (1.014x) | proxy 1033 recorded 10702 (1.080x); 32K new 12271 vs 130307 same binary 12086/12140 (1.013x) | proxy 1033 recorded 12026 (1.020x)
21:07:32 stage Bq done (2 min)
21:07:33 gate S-prefill-probe PASS: new-tp2-202611170201: PASS | new-tp2-202611151621: PASS (comparator: ref-tp2-202611151621<results>/port-g3tp2reclaim-1116-2044/S/probe/ref-tp2-202611151621.json)
21:07:33 restore skipped (RESTORE=0): GPUs idle (no compute process, < 1000 MiB used)
21:07:33 power logger stopped (pid 2262494): 40822 lines in <results>/gpu-power/20261116-204450.csv; peak power.draw.instant (100 ms): gpu0 638 W gpu1 584 W gpu2 108 W; gpu0+gpu1 1090 W at 2026/11/16 21:02:39.6
21:07:33 KV dry-run (same binary 67c691d29041): new 3177472 vs 151621 2653184 (1.1976x, +19.76%, +524288 tokens) | served: new 3049984 vs 151621 recorded 2518016 (1.2113x, +21.13%, +531968 tokens)
21:07:33 chain finished, exit 0 after 22 min; results <results>/port-g3tp2reclaim-1116-2044
