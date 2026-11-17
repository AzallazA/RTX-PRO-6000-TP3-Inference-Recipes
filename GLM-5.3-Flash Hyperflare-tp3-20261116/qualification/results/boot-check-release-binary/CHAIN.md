21:11:40 tp3 boot PASS: gb202-tp3-202611170053 on 27864a1d79e4, up in 30 s, served KV 14929920
21:11:40 tp3 stack-line PASS: CUDA stack limit 256 B per thread (recorded 1024 B, largest kernel stack 32 B)
21:11:40 tp3 text-smoke PASS: ok '391' finish stop, 47 tokens
21:11:44 tp3 image-smoke PASS: IMAGE_SMOKE bootcheck-tp3 served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.7s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
21:11:49 tp3 lean PASS: LEAN_PROBE bootcheck-tp3: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
21:14:27 tp3 temps INFO: gpu0 max 60 mean 54.5 s>=90 0; gpu1 max 69 mean 62.1 s>=90 0; gpu2 max 76 mean 66.8 s>=90 0; 
21:14:27 tp3 decode PASS: c1 32K 290.3 tok/s steps/s 114.2 c8 32K 878.8 tok/s steps/s 459.8 c32 32K 1827.6 tok/s steps/s 759.8
21:14:27 tp3 prefill PASS: 8K 10180 tok/s 32K 11701 tok/s
21:14:27 tp3 speed-vs-export-chain INFO: c1 BETTER 290 vs 279 tok/s (1.040x), steps/s 114.2 vs 117.6 (0.971x); plan rule NOT met;c8 LEVEL 879 vs 887 tok/s (0.991x), steps/s 459.8 vs 465.2 (0.988x); plan rule met;c32 LEVEL 1828 vs 1812 tok/s (1.008x), steps/s 759.8 vs 753.3 (1.009x); plan rule met;8K LEVEL 10180 vs 10408 tok/s (0.978x);32K LEVEL 11701 vs 11712 tok/s (0.999x); (same package on 67c691d29041)
21:14:27 tp3 scan-serve-log PASS: no CUDA fault / OOM / panic line
21:14:27 tp3 scan-kernel-log PASS: no NVRM Xid / fault
21:14:27 tp3 alive PASS: answering at the end, KV 14929920 unchanged
21:14:32 tp3 done in 3 min
21:15:01 tp2 boot PASS: gb202-tp2-202611170201 on 27864a1d79e4, up in 29 s, served KV 3113472
21:15:01 tp2 stack-line PASS: CUDA stack limit 256 B per thread (recorded 1024 B, largest kernel stack 48 B)
21:15:01 tp2 text-smoke PASS: ok '391' finish stop, 12 tokens
21:15:05 tp2 image-smoke PASS: IMAGE_SMOKE bootcheck-tp2 served 5/5 correct 5/5 | small: ok 309p 0.2s | near-max 2800x1876 (6700 tok): ok 6729p 1.2s | over-max 4480x3040 (downscaled): ok 8021p 1.6s | two images: ok 588p 0.2s | four images: ok 1152p 0.3s
21:15:10 tp2 lean PASS: LEAN_PROBE bootcheck-tp2: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
21:17:08 tp2 temps INFO: gpu0 max 62 mean 56.9 s>=90 0; gpu1 max 71 mean 65.4 s>=90 0; gpu2 max 50 mean 48.7 s>=90 0; 
21:17:08 tp2 decode PASS: c1 32K 245.4 tok/s steps/s 104.8 c8 32K 750.2 tok/s steps/s 312.3
21:17:08 tp2 prefill PASS: 8K 11583 tok/s 32K 12275 tok/s
21:17:08 tp2 scan-serve-log PASS: no CUDA fault / OOM / panic line
21:17:08 tp2 scan-kernel-log PASS: no NVRM Xid / fault
21:17:08 tp2 alive PASS: answering at the end, KV 3113472 unchanged
21:17:12 tp2 done in 2 min
21:17:12 restore skipped (RESTORE=0): GPUs idle (no compute process, < 1000 MiB used)
