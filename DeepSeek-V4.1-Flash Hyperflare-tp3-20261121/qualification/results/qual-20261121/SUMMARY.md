# DS41 release qualification of the live default (ds41-relqual-1120-2218)

build: gb202-tp3-202611201530 (~/.cache/flare/packages/models/ds41-1017-engram-fp8/gb202-tp3-202611201530) on <fork-worktree>/flare-cli/target/release/flare sha af06512fa11a297889526caf4cfe1fa2c673290bd8d4f9730818c3718ea72dd8 commit a7c3a04a6a9972195fac780153cc5809f9444fb1; served KV 5687040; Engram read-ahead on
server: pid 3013272, log <fork-worktree>/serve-best.log (from byte 1935); argv <fork-worktree>/flare-cli/target/release/flare run ~/.cache/flare/packages/models/ds41-1017-engram-fp8/gb202-tp3-202611201530 --gpus 0,1,2 --port 15015 --host 127.0.0.1 --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory 12G --mtp-draft-tokens 5 --mtp-adaptive-verification --mtp-max-verification-token-budget 32 --mtp-adaptive-verification-alpha 0.8 --scheduler-policy int
env: FLARE_CUDA_STACK_LIMIT=256 FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_ENGRAM_RAM_TABLES=0 FLARE_ENGRAM_READ_AHEAD=1 FLARE_FAST_EXIT=1 FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512 ; ds41-tp3.sh sha 25f90967604bdc1ecb9fc1d2db8d4fecf0337d315cc17027d862ea1c99f660b6; adopt record: no queue-log SWITCHED record for gb202-tp3-202611201530 + steersync-a7c3a04a (manual switch?)

| kind | gate | result | detail |
|---|---|---|---|
| GATE | identity | PASS | pid 3013272: gb202-tp3-202611201530 on steersync-a7c3a04a af06512fa11a (commit a7c3a04a) = ds41-tp3.sh lines 73/89 (sha 25f90967604b), argv = the script's, FLARE_* = FLARE_CUDA_STACK_LIMIT=256 FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_ENGRAM_RAM_TABLES=0 FLARE_FAST_EXIT=1 FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512 FLARE_ENGRAM_READ_AHEAD=1, KV 5687040, smoke '391', Engram RAM, steering 0/1.0 HTTP 200 200 |
| GATE | adopt-record | INFO | no queue-log SWITCHED record for gb202-tp3-202611201530 + steersync-a7c3a04a (manual switch?) |
| GATE | retrieval | PASS | DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass |
| GATE | tooleval | PASS | score 93/100, pass 81/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 88; 20261116 release ref 90/100 (77/91 graded, seq235) |
| GATE | agent8 | INFO | agent c8: 658 turns in 300 s, TTFT p50/p90/max 0.39452052116394043/0.627225399017334/1.020204782485962 s, 530.8 output tok/s; errors 0; ref 646 turns, 455.8 out tok/s, p50/p90 0.51/0.90 s |
| GATE | stress16 | PASS | PASS 32/32 long, ping 168/168, min free 78 MiB, bad lines 0, wall 20.3 min (min free < 480 MiB kit margin, >= 16 MiB floor) |
| GATE | stress16-vram-per-gpu | PASS | stress16: GPU 0 min free 78 MiB (used 97264, 1101 samples, max gap 1 s); GPU 1 min free 78 MiB (used 97264, 1101 samples, max gap 1 s); GPU 2 min free 78 MiB (used 97264, 1101 samples, max gap 1 s) (floor 16 MiB per GPU) |
| GATE | gpqa1-rewrite | PASS | proxy REWRITE lines: 198 chat requests sent with steering_scale 1.0 for 198 attempted items, 0 other values; STOP rewritten 198 passthrough 2 errors 0 disconnects 0 idle 0 completed 200 |
| GATE | gpqa1 | INFO | steering_scale 1.0: 177/198 = 89.4% (Wilson95 84.3-93.0), truncated without answer 3, hit max_tokens 6, errors 0, max_tokens 32768, c8; ref 90.4% (85.5-93.8); paired vs baseline: 198 items, 4 gained / 6 lost, McNemar p 0.754; one 198-item sample resolves about +-5 points / vs seq382 s=0 181/198 (5 truncated): GPQA PAIRED PASS: A 181/198, B 177/198; wrong only in B 4, only in A 0; exact McNemar p 0.125 |
| GATE | post | PASS | smoke 2/2 thinking-off:ok '391' stop 0.05s thinking-on:ok '391' stop 0.11s / served KV 5687040 unchanged / verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0) |
| GATE | scan | PASS | serve log 0 fault lines, kernel log 0 NVRM fault lines |

Stage minutes / lowest MemAvailable MiB / host cache GiB: retrieval 1.6 14581 10.50->9.27;tooleval 1.6 14936 9.27->9.04;agent8 5.6 15152 9.04->10.48;stress16 20.4 10876 10.48->9.01;gpqa1 22.9 14611 9.01->8.96;

PASS: RELQUAL gb202-tp3-202611201530 on steersync-a7c3a04a all gates PASS; retrieval DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass; tooleval score 93/100, pass 81/91 graded of 92 (excluded ['TC-45']); baseline 91 (78/91), gate >= 8; agent8 agent c8: 658 turns in 300 s, TTFT p50/p90/max 0.39452052116394043/0.627225399017334/1.020; stress16 PASS 32/32 long, ping 168/168, min free 78 MiB, bad lines 0, wall 20.3 min (min free < 480; gpqa1 steering_scale 1.0: 177/198 = 89.4% (Wilson95 84.3-93.0), truncated without answer 3, hit ; 52.1 min | ~/bench-results/ds41-relqual-1120-2218
