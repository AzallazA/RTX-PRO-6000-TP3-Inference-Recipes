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
