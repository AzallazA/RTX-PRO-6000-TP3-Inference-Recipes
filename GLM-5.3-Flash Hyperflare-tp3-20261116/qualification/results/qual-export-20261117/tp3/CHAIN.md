14:43:30 start: profile export, stages: consistency eval long128 tooleval retrieval hotel stress post scan; DS41 at start gb202-tp3-202611160815 KV 5403008; MemAvailable 21406 MiB
14:43:30 tp3 start: gb202-tp3-202611170053
14:44:12 tp3/nostack boot PASS: gb202-tp3-202611170053 on 1bdcce7378e4 (ttft shortest, no stack override), up in 29 s, KV 14834688 (export-chain nostack reference 14834688, other binary, INFO: +0.00%)
14:44:16 tp3/nostack lean PASS: LEAN_PROBE glmchk-tp3-nostack: ok 24/24, with logprobs 24/24, same-server repeat identical 12/12
14:44:16 tp3/nostack lean-cmp INFO:  compared 12/12, exact 8, top-1 same 10, max |dlogprob| 1.645 (vs the recorded reference boot; report only, C.27)
14:44:16 tp3/nostack stage lean done (0 min)
14:44:50 tp3/main stack-line PASS: CUDA stack limit 256 B per thread (recorded 1024 B, largest kernel stack 32 B)
14:44:50 tp3/main kv-stack PASS: KV 14929920 with stack 256 vs 14834688 without: +95232 tokens (expected ~+95232)
14:44:50 tp3/main boot PASS: up in 28 s; FLARE_CUDA_STACK_LIMIT=256; served KV 14929920
14:44:50 tp3/main kv-ref PASS: KV 14929920 vs 14929920 (seq307, same binary + stack 256): +0.00%
14:44:56 tp3/main consistency INFO: QC 32: 192 positions (0 sequences cut at ambiguous tokens): top-1 agree 0.953, decode top-1 missing from prefill top-5 0, |dlogprob| median 0.0239 p90 0.1751 p99 0.5951 max 0.8574
14:44:56 tp3/main stage consistency done (0 min)
14:45:29 tp3/main eval PASS: QC 32: 43/43
14:45:29 tp3/main stage eval done (0 min)
14:46:39 tp3/main stage long128 done (1 min)
14:46:39 tp3/main long128 INFO: c1 128K 287.3 tok/s steps/s 112.2
14:49:36 tp3/main stage tooleval done (2 min)
14:49:37 tp3/main tooleval INFO: QC 32: score 93/100 ★★★★★ Excellent; no baseline: this run can serve as one
14:51:49 tp3/main stage retrieval done (2 min)
14:51:49 tp3/main retrieval PASS: QC 32: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
15:12:29 tp3/main stage hotel done (20 min)
15:12:31 tp3/main hotel-lights FAIL: 20:12:29 hotel-lights FAIL DONE hotel-lights-c32 rc=0 | exact 26 / wrong 2 / truncated 2 | [(28, 'unparseable: multiple numbers without answer anchor', 40321), (30, 'expected 48, got 49', 10946)]
