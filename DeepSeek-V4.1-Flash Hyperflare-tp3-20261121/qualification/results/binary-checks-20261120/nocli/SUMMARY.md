# DS41 steersync nocli binary (steersync-a7c3a04a) check (ds41-nocli-1120-2049)

binary <fork-worktree> (sha af06512fa11a); package gb202-tp3-202611201530 (the live default, KV gate 5687040); served by <serve-dir>/ds41-tp3.sh with FLARE_DIR= only; boot KV 5687040; seq seq390 after seq389

| kind | gate | result | detail |
|---|---|---|---|
| GATE | def | PASS | gb202-tp3-202611201530 (line 73) + steersync-80764e80 (line 89) + read-ahead (line 91): record seq389 ds41-steersync-1120-1951 ADOPTED, KV 5687040; restore target = this default |
| GATE | live-before | PASS | pid 2802630 = gb202-tp3-202611201530 on steersync-80764e80, KV 5687040 == record |
| GATE | boot | PASS | gb202-tp3-202611201530 on steersync-a7c3a04a af06512fa11a via ds41-tp3.sh FLARE_DIR= only, up in 52 s, flags = live default's (encoder prefill true), FLARE_* = ds41-tp3.sh's, Engram both in RAM, access open |
| GATE | boot-kv | PASS | served KV 5687040 == the live default's KV 5687040 (seq389 kv-boot) |
| GATE | boot-log | PASS | 15 [ OK ] lines = seq389's boot of this package on steersync-80764e80 |
| GATE | smoke | PASS | smoke 2/2 thinking-off:ok '391' stop 0.04s thinking-on:ok '391' stop 0.11s |
| GATE | verify | PASS | verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0) |
| GATE | clireject | PASS | CLIREJECT PASS: 'flare run --steering-api-key ...' -> exit 2 unknown argument; --help lists --steering-api-keys-file only |
| GATE | steer | PASS | STEER SANITY PASS: trigger #8 moved (s=0 vs s=1 text differs; 1 tried, errors 0) |
| GATE | boot-access | PASS | gb202-tp3-202611201530 on steersync-a7c3a04a af06512fa11a via ds41-tp3.sh FLARE_DIR= + the key-file variable only, up in 51 s, flags = live default's (encoder prefill true), FLARE_* = ds41-tp3.sh's + the key-file variable, Engram both in RAM, access restricted (1 key) |
| GATE | boot-access-kv | PASS | served KV 5687040 == the live default's KV 5687040 (seq389 kv-boot) |
| GATE | boot-access-log | PASS | 15 [ OK ] lines = seq389's boot of this package on steersync-80764e80 |
| GATE | access-smoke | PASS | smoke 2/2 thinking-off:ok '391' stop 0.04s thinking-on:ok '391' stop 0.11s |
| GATE | access | PASS | ACCESS PASS: 403 x7/7 (no key, wrong X-Steering-Key, wrong Bearer, chat_template_kwargs, completions, messages (Anthropic), stream); right key X-Steering-Key 200 cached 0, Bearer 200 cached 3423, stream 200 deltas 16; s=0 no key 200 cached 3423, omitted 200 cached 3423; INFO steered text moved vs s=0 |
| GATE | access-key-scan | PASS | key not in the serve log, server argv / environ, run files or queue log (environ carries the file path only) |
| GATE | access-key-removed | PASS | per-run key file + folder deleted |
| GATE | scan-serve | PASS | serve log 0 fault lines, kernel log 0 NVRM fault lines |
| GATE | scan-serve-access | PASS | serve log 0 fault lines, kernel log 0 NVRM fault lines |
| GATE | scan-oom | PASS | no OOM kill in the kernel log since the chain start |
| GATE | ready | PASS | <work-dir>/steersync-nocli-20261120/NOCLI-READY written (line 89 -> steersync-a7c3a04a) |
| GATE | switch | PASS | ds41-tp3.sh line 89 -> steersync-a7c3a04a (line 73 gb202-tp3-202611201530 + read-ahead unchanged; inode 9343960 -> 9345472, sha 25f90967604b); backup <serve-dir>/history/ds41-tp3.sh.pre-nocli  |
| GATE | adopted | PASS | OK pid 3013272 gb202-tp3-202611201530 KV 5687040 smoke 391 (exe af06512fa11a, encoder prefill true, Engram both in RAM) |
| GATE | adopted-ra-inert | PASS | FLARE_ENGRAM_READ_AHEAD=1 in environ; no read-ahead metric lines, no read-ahead threads (both tables in RAM) |
| GATE | adopted-steer | PASS | STEER SANITY PASS: trigger #8 moved (s=0 vs s=1 text differs; 1 tried, errors 0) |

PASS: NOCLI ADOPTED all gates PASS; default = gb202-tp3-202611201530 on steersync-a7c3a04a + read-ahead on (KV 5687040; PASS: ACCESS PASS: 403 x7/7 (no key, wrong X-Steering-Key, wrong Bearer, chat_template_kwargs, completions, messages (Anthropic), stream); right key X-Steering-Key 200 cached 0, Bearer 200 cached 3423, stream 200 deltas 16; s=0 no key 200 cached 3423, omitted ; script SWITCHED (ds41-tp3.sh line 89 -> steersync-a7c3a04a (line 73 gb202-tp3-202611201530 + read-ahead unchanged; inode 9343960 -> 9345472, sha 25f90967604b); backup <serve-dir>/history/ds41-tp3.sh.pre-nocli); adopted OK pid 3013272 gb202-tp3-202611201530 KV 5687040 smoke 391; 3 min) | ~/bench-results/ds41-nocli-1120-2049
