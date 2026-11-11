# DS41 Hyperflare TP3 release chain: summary

Winner package: `gb202-tp3-202611111227` (row limit 12); base `gb202-tp3-202611111227`, rl24 `gb202-tp3-202611111254`.

## Release table (fill the README headline + quality tables from this)

| | 20261108.1 (pkg 081024) | this release |
|---|---|---|
| KV tokens (cap 24, 32G prefix cache) | 2,467,200 | 3,014,656 served (dry run 3,007,360) |
| Decode 32K c1 / c4 / c8, ignore-EOS, mean of 2 rounds | 305 / 734 / 986 (8-round A/B, balanced) | 305 / 735 / 991 (interactive) |
| Decode 128K / 512K c1 (1 run) | 296 / 275 (20261107 package) | 289 / 271 |

## Stage A

# Stage A: base vs rl24 (32K decode, 60 s cells, serve-best.sh flags)

| round | base c1 | base c4 | base c8 | rl24 c1 | rl24 c4 | rl24 c8 | c8 ratio |
|---|---|---|---|---|---|---|---|
| r1 | 305 | 739 | 1001 | 312 | 723 | 1028 | 1.0269 |
| r2 (invalid) | 305 | ERR | 994 | 301 | 731 | 1010 | - |
| r3 | 306 | 731 | 982 | 312 | 738 | 974 | 0.9923 |
| r4 (invalid) | 286 | ERR | 977 | 303 | 731 | 1007 | - |
| r5 (invalid) | 304 | 733 | 987 | 307 | 745 | ERR | - |

PICK base: only 2 valid rounds (need 3)


## Stage Q

# Stage Q: full qualification gates (reference = 20261108.1 release, package 081024)

Served KV: 3014656 (20261108.1: 2,467,200)

| check | 20261108.1 | this run | gate |
|---|---|---|---|
| verify_fix 10 runs, TRUE corruptions | 0 | 0 | PASS |
| eval-43 (thinking off) | 43/43 | 42/43 | PASS (gate >= 42) |
| tool calls, 60 at c8 | 60/60 | 60/60 | PASS |
| lavd-c8, 30 runs | 30/30 | 30/30 (wrong 0, truncated 0) | PASS (gate 30/30, 0 truncated; one miss -> one re-run of 30 that must be 30/30) |
| hotel-lights-c8, 30 runs | 30/30 | 30/30 (wrong 0, truncated 0) | PASS (gate 30/30, 0 truncated; one miss -> one re-run of 30 that must be 30/30) |
| long retrieval 128K / 900K | pass / pass | DONE 128000 pass DONE 900000 pass | PASS |
| long generation after 581K tokens, 4 seeds | 4/4, worst 8-gram 0.0008 | 4/4, worst 8-gram 0.0013, natural stop True | PASS |
| natural-EOS decode 32K c1/c4/c8 (tok/s) | 290/717/963, 0 loops | 300/713/950, 0 loops | PASS (gate >= 0.97x each, 0 loops) |
| LiveBench coding (2024-11-25) | 81.6 | 83.5 | PASS (gate >= 80.5) |
| c8 KV stress (2 waves x 8 x ~530K + pinger) | PASS 16/16, pinger 117/117 | [hyperflare-202611111227-11110858-c8] PASS: 16/16 long requests ok, pinger 68/68 ok, 9,511,557 prompt tok pushed, min fr | PASS |

PASS all gates


## Chain log

```
07:06 release chain start: stages 'E12 E A Q'; v3 eb05dd8a Merge branch 'ds41-tp3-sched' into ds41-tp3-v3; base gb202-tp3-202611090700; PARKED=24; SCHED=interactive; default pkg gb202-tp3-202611090700
07:06 power: 450.00 W 450.00 W 450.00 W | free RAM 92 GiB | store free 239 GB | signing key 2919fdff03a69f86
07:34 E export head-rl12 (HOT=342) exit 0 after 28 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611111227
07:34 E head-rl12 sidecar: ok brrr_parked_max_rows=12 (ref 12)
07:34 E ignition import path: <fork>/ignition/__init__.py
08:02 E export rl24 (HOT=342) exit 0 after 27 min: ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611111254
08:02 E sidecar: ok brrr_parked_max_rows=24 (ref 12)
08:02 E dry run gb202-tp3-202611111227: 3,007,360 KV tokens
08:02 E dry run gb202-tp3-202611111254: 3,029,120 KV tokens
08:03 serve base-r1: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:03 A base r1 smoke: 391
08:06 A base r1 32K: c1 305/118.0s/2.58t/0.39a c4 739/79.4s/9.32t/0.60a c8 1001/57.9s/17.29t/0.70a
08:07 serve cand-r1: gb202-tp3-202611111254 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:07 A cand r1 smoke: 391
08:11 A cand r1 32K: c1 312/116.8s/2.67t/0.40a c4 723/75.9s/9.52t/0.53a c8 1028/57.4s/17.91t/0.72a
08:12 serve base-r2: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:12 A base r2 smoke: 391
08:16 A base r2 32K: c1 305/117.3s/2.60t/0.39a c4 ERR c8 994/57.7s/17.22t/0.75a
08:17 serve cand-r2: gb202-tp3-202611111254 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:17 A cand r2 smoke: 391
08:21 A cand r2 32K: c1 301/117.1s/2.57t/0.38a c4 731/74.4s/9.82t/0.52a c8 1010/57.9s/17.44t/0.68a
08:22 serve base-r3: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:22 A base r3 smoke: 391
08:26 A base r3 32K: c1 306/118.0s/2.59t/0.41a c4 731/79.3s/9.22t/0.62a c8 982/63.0s/15.57t/0.79a
08:30 A base long context c1: 128K 289 512K 271
08:31 serve cand-r3: gb202-tp3-202611111254 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:31 A cand r3 smoke: 391
08:35 A cand r3 32K: c1 312/117.1s/2.67t/0.41a c4 738/76.5s/9.65t/0.57a c8 974/57.1s/17.06t/0.73a
08:39 A cand long context c1: 128K 284 512K 263
08:39 A only 2 valid rounds after 3: one more round
08:41 serve base-r4: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:41 A base r4 smoke: 391
08:44 A base r4 32K: c1 286/117.1s/2.44t/0.38a c4 ERR c8 977/59.1s/16.52t/0.71a
08:45 serve cand-r4: gb202-tp3-202611111254 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:45 A cand r4 smoke: 391
08:49 A cand r4 32K: c1 303/118.5s/2.56t/0.40a c4 731/76.7s/9.52t/0.56a c8 1007/58.7s/17.14t/0.73a
08:49 A only 2 valid rounds after 4: one more round
08:50 serve base-r5: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:50 A base r5 smoke: 391
08:54 A base r5 32K: c1 304/117.3s/2.59t/0.39a c4 733/77.6s/9.44t/0.61a c8 987/58.8s/16.80t/0.75a
08:55 serve cand-r5: gb202-tp3-202611111254 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:55 A cand r5 smoke: 391
08:58 A cand r5 32K: c1 307/117.4s/2.61t/0.40a c4 745/74.9s/9.93t/0.55a c8 ERR
08:58 A gate (c8 >= +1% every round, c1/c4 >= 0.99x, 128K/512K >= 0.98x) after 56 min: PICK base: only 2 valid rounds (need 3)
08:58 winner: gb202-tp3-202611111227
08:59 serve qual: gb202-tp3-202611111227 budget 24 alpha 0.8 --scheduler-policy interactive KV 3014656
08:59 Q smoke: 391
08:59 Q verify_fix (exit 0): 10 runs, 1941 tokens: TRUE corruptions 0, natural-uncertainty flags 14
09:53 Q qual_all (default stages) exit 0: KV 3014656, power 450.00 W 450.00 W 450.00 W |eval-43: 42 / 43 (temperature clamped to 1.0)|tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 26 s| DONE lavd-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []| DONE hotel-lights-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []| DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass|long generation: 4 runs, fails 0 | completions [19381, 32566, 11925, 15834] | repeat_8gram [0.0, 0.0013, 0.0, 0.0003]|respect-eos bench: │ 32k │ 299.5 28/3 │ 712.7 47/6 │ 949.8 58/8 │ loops: 0|LiveBench coding: |stress: [hyperflare-202611111227-11110858-c8] PASS: 16/16 long requests ok, pinger 68/68 ok, 9,511,557 prompt tok pushed, min free VRAM 82 M|
09:53 Q gates after 55 min (QUALGATES.md): FAIL: LiveBench coding (2024-11-25)
09:54 default restored: gb202-tp3-202611090700 --scheduler-policy interactive  KV 3014656
09:54 next (not automatic): read SUMMARY.md; if the winner is rl24 and Q passes, ask the user before switching serve-best.sh P= and before any public push
09:54 SUMMARY.md written
09:54 release chain done after 168 min (exit 0)
```

Note (added when the release docs were filled): the chain log's `09:53 Q gates ... FAIL: LiveBench coding` line is the
first gate pass, which could not read the score: the qualification note in `run-hyperflare-c8/QUAL.md` is empty because
pandas truncates the long LiveBench display name, so the name-based grep in `qual_all.sh` matched nothing (fixed in this
folder's `qual_all.sh`). The gates were re-evaluated from `run-hyperflare-c8/livebench-score.log` ("All Groups" coding
83.5; LCB_generation 73.077, coding_completion 94.0): PASS, as in the Stage Q table above and `QUALGATES.md`.
Stage A: "only 2 valid rounds" means the gate counted a round only if all six cells completed; three rounds lost one cell
each to `llm_decode_bench`'s loop guard (base r2 and r4 c4, rl24 r5 c8), the known `ignore_eos` artifact (the model
emits its end token, then the bench forces it on). On the completed cells rl24 also missed the gate on its own terms
(c8 −0.8% in r3, c4 0.98× in r1, 512K c1 0.97×), so row limit 12 stays the default.
