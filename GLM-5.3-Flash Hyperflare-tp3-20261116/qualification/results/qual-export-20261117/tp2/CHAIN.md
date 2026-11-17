15:13:40 start: profile export, stages: consistency eval long128 tooleval retrieval hotel stress post scan; DS41 at start  KV ; MemAvailable 248000 MiB
15:13:40 tp2 start: gb202-tp2-202611170201
15:14:09 tp2/nostack boot PASS: gb202-tp2-202611170201 on 1bdcce7378e4 (ttft shortest, no stack override), up in 28 s, KV 3049984 (export-chain nostack reference 3049984, other binary, INFO: +0.00%)
15:14:14 tp2/nostack lean PASS: LEAN_PROBE glmchk-tp2-nostack: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
15:14:14 tp2/nostack lean-cmp INFO:  compared 12/12, exact 9, top-1 same 10, max |dlogprob| 1.81 (vs the recorded reference boot; report only, C.27)
15:14:14 tp2/nostack stage lean done (0 min)
15:14:46 tp2/main stack-line PASS: CUDA stack limit 256 B per thread (recorded 1024 B, largest kernel stack 48 B)
15:14:46 tp2/main kv-stack PASS: KV 3113472 with stack 256 vs 3049984 without: +63488 tokens (expected ~+63488)
15:14:46 tp2/main boot PASS: up in 28 s; FLARE_CUDA_STACK_LIMIT=256; served KV 3113472
15:14:46 tp2/main kv-ref PASS: KV 3113472 vs 3113472 (seq307, same binary + stack 256): +0.00%
15:14:51 tp2/main consistency INFO: QC 8: 192 positions (0 sequences cut at ambiguous tokens): top-1 agree 0.958, decode top-1 missing from prefill top-5 0, |dlogprob| median 0.0088 p90 0.1335 p99 0.3294 max 0.6894
15:14:51 tp2/main stage consistency done (0 min)
15:15:31 tp2/main eval PASS: QC 8: 42/43
15:15:31 tp2/main stage eval done (0 min)
15:15:31 tp2/main long128 INFO: trimmed (INFO-only stage, TP3 only)
15:15:31 tp2/main tooleval INFO: trimmed (INFO-only stage, TP3 only)
15:17:36 tp2/main stage retrieval done (2 min)
15:17:38 tp2/main retrieval PASS: QC 8: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
15:46:27 tp2/main stage hotel done (28 min)
15:46:27 tp2/main hotel-lights WARN: QC 8: DONE hotel-lights-c8 rc=0 | exact 29 / wrong 1 / truncated 0 | [(25, 'expected 48, got 49', 7768)]
16:01:43 tp2/main stage stress done (15 min)
16:01:45 tp2/main stress WARN: QC 8: PASS 16/16 long, ping 92/92, min free 84 MiB, bad lines 0, wall 15.3 min (every request passed; min free below the 480 MiB margin)
16:01:50 tp2/main post-lean PASS: LEAN_PROBE glmchk-tp2-main-post: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
16:01:50 tp2/main post-kv PASS: served KV 3113472 = boot value
16:01:50 tp2/main stage post done (0 min)
16:01:50 tp2/main scan-serve-log PASS: no CUDA fault / OOM / panic line
16:01:50 tp2/main scan-kernel-log PASS: no NVRM Xid / fault since chain start
16:01:50 tp2/main stage scan done (0 min)
16:01:53 tp2 done (48 min, new FAIL gates 0)
