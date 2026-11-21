22:18:43 GATE identity PASS: pid 3013272: gb202-tp3-202611201530 on steersync-a7c3a04a af06512fa11a (commit a7c3a04a) = ds41-tp3.sh lines 73/89 (sha 25f90967604b), argv = the script's, FLARE_* = FLARE_CUDA_STACK_LIMIT=256 FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_ENGRAM_RAM_TABLES=0 FLARE_FAST_EXIT=1 FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512 FLARE_ENGRAM_READ_AHEAD=1, KV 5687040, smoke '391', Engram RAM, steering 0/1.0 HTTP 200 200
22:18:43 GATE adopt-record INFO: no queue-log SWITCHED record for gb202-tp3-202611201530 + steersync-a7c3a04a (manual switch?)
22:18:43 stage retrieval start (MemAvailable 15059 MiB, host cache 10.50 GiB)
22:20:18 stage retrieval body exit 0 after 1.6 min; host cache 9.27 GiB; lowest MemAvailable 14581 MiB
22:20:18 GATE retrieval PASS: DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
22:20:19 stage tooleval start (MemAvailable 15014 MiB, host cache 9.27 GiB)
22:21:54 stage tooleval body exit 0 after 1.6 min; host cache 9.04 GiB; lowest MemAvailable 14936 MiB
22:21:54 GATE tooleval PASS: score 93/100, pass 81/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88; 20261116 release ref 90/100 (77/91 graded, seq235)
22:21:54 stage agent8 start (MemAvailable 15405 MiB, host cache 9.04 GiB)
22:27:29 stage agent8 body exit 0 after 5.6 min; host cache 10.48 GiB; lowest MemAvailable 15152 MiB
22:27:29 GATE agent8 INFO: agent c8: 658 turns in 300 s, TTFT p50/p90/max 0.39452052116394043/0.627225399017334/1.020204782485962 s, 530.8 output tok/s; errors 0; ref 646 turns, 455.8 out tok/s, p50/p90 0.51/0.90 s
22:27:29 stage stress16 start (MemAvailable 15258 MiB, host cache 10.48 GiB)
22:47:50 stage stress16 body exit 0 after 20.4 min; host cache 9.01 GiB; lowest MemAvailable 10876 MiB
22:47:50 GATE stress16 PASS: PASS 32/32 long, ping 168/168, min free 78 MiB, bad lines 0, wall 20.3 min (min free < 480 MiB kit margin, >= 16 MiB floor)
22:47:50 GATE stress16-vram-per-gpu PASS: stress16: GPU 0 min free 78 MiB (used 97264, 1101 samples, max gap 1 s); GPU 1 min free 78 MiB (used 97264, 1101 samples, max gap 1 s); GPU 2 min free 78 MiB (used 97264, 1101 samples, max gap 1 s) (floor 16 MiB per GPU)
22:47:50 stage gpqa1 start (MemAvailable 14748 MiB, host cache 9.01 GiB)
23:10:41 stage gpqa1 body exit 0 after 22.9 min; host cache 8.96 GiB; lowest MemAvailable 14611 MiB
23:10:41 GATE gpqa1-rewrite PASS: proxy REWRITE lines: 198 chat requests sent with steering_scale 1.0 for 198 attempted items, 0 other values; STOP rewritten 198 passthrough 2 errors 0 disconnects 0 idle 0 completed 200
23:10:41 GATE gpqa1 INFO: steering_scale 1.0: 177/198 = 89.4% (Wilson95 84.3-93.0), truncated without answer 3, hit max_tokens 6, errors 0, max_tokens 32768, c8; ref 90.4% (85.5-93.8); paired vs baseline: 198 items, 4 gained / 6 lost, McNemar p 0.754; one 198-item sample resolves about +-5 points | vs seq382 s=0 181/198 (5 truncated): GPQA PAIRED PASS: A 181/198, B 177/198; wrong only in B 4, only in A 0; exact McNemar p 0.125
23:10:47 GATE post PASS: smoke 2/2 thinking-off:ok '391' stop 0.05s thinking-on:ok '391' stop 0.11s | served KV 5687040 unchanged | verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0)
23:10:47 GATE scan PASS: serve log 0 fault lines, kernel log 0 NVRM fault lines
23:10:47 PASS: RELQUAL gb202-tp3-202611201530 on steersync-a7c3a04a all gates PASS; retrieval DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass; tooleval score 93/100, pass 81/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 8; agent8 agent c8: 658 turns in 300 s, TTFT p50/p90/max 0.39452052116394043/0.627225399017334/1.020; stress16 PASS 32/32 long, ping 168/168, min free 78 MiB, bad lines 0, wall 20.3 min (min free < 480; gpqa1 steering_scale 1.0: 177/198 = 89.4% (Wilson95 84.3-93.0), truncated without answer 3, hit ; 52.1 min | ~/bench-results/ds41-relqual-1120-2218
