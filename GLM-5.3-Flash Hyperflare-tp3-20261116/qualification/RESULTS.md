# Check results — GLM-5.3-Flash Hyperflare TP3 / TP2 20261116

This file lists what ran on these packages: the two export chains (quick checks, binary `67c691d29041`), one boot of
each package on the release binary (`27864a1d79e4`), and a minimal tier "export" qualification on 2026-11-17 (binary
`1bdcce7378e4`, built from fork `main` `036cde24`). **The TP3 KV stress at 32 streams was not run.** Numbers with their
files: [../MEASUREMENTS.md](../MEASUREMENTS.md).

| Package | What it is | Results |
| --- | --- | --- |
| `gb202-tp3-202611170053` | TP3 release package (32 slots, images, 4 image slots) | `results/tp3-202611170053/export-chain/` |
| `gb202-tp2-202611170201` | TP2 release package (8 slots, images, 2 image slots) | `results/tp2-202611170201/export-chain/` |
| both | boot check on the release binary | `results/boot-check-release-binary/` |
| both | tier "export" qualification, 2026-11-17 | `results/qual-export-20261117/tp3/`, `results/qual-export-20261117/tp2/` |

Gate names are the chains' (`GATES.tsv`, `SUMMARY.md`, `CHAIN.md`). Reference packages in the chains: `151109` (TP3)
and `151621` (TP2) = the same export settings without the reclaim switches, exported from the port on 2026-11-15 and
not published. Some TP2 gate texts label the 151621 values "130307"; the values are 151621's.

## What ran

| Check | TP3 | TP2 | Binary |
| --- | --- | --- | --- |
| Export, reclaim log lines, graph pools | PASS | PASS | export (Python) |
| KV dry run and served | PASS (15,025,152 / 14,834,688) | PASS (3,177,472 / 3,049,984) | `67c691d29041` |
| First-token probe 24 + repeat 12 | PASS | PASS | `67c691d29041` |
| Image smoke 5 | PASS | PASS | `67c691d29041` |
| Image latency | PASS (first 0.14 s) | not run | `67c691d29041` |
| Prefill probe | PASS (marginal 33K needle, 30/32 vs 32/32, p = 0.25) | PASS | `67c691d29041` |
| Corruption check (20 runs) | not run | PASS (0 true) | `67c691d29041` |
| Tool calls 60 | not run | PASS (60/60 at c8) | `67c691d29041` |
| Decode / prefill pass | PASS against 151109 | PASS against 151621 | `67c691d29041` |
| Boot check (KV, stack line, text and image smoke, probe, one speed pass, log scan) | PASS (14,929,920 KV) | PASS (3,113,472 KV) | `27864a1d79e4` |

## Tier "export" qualification (2026-11-17)

Binary `1bdcce7378e4`, built from fork `main` `036cde24` (tag `hyperflare-tp2-exl3-20261117`). Its Rust changes since
this release's tag are DeepSeek Engram code and a KV-layout fix for rows that lie wholly in granule padding; the GLM
packages serve the same KV as on the release binary. Settings of `serve-glm-tp3.sh` / `serve-glm-tp2.sh` (TTFT order
shortest, stack limit 256, prefix cache 32G, 5 drafts). Sampling: temperature 1.0, top-p 0.95. QC = concurrency.

| Check | TP3 `gb202-tp3-202611170053` (QC 32) | TP2 `gb202-tp2-202611170201` (QC 8) |
| --- | --- | --- |
| Boot without stack limit, served KV | PASS, 14,834,688 | PASS, 3,049,984 |
| Boot with stack limit 256, served KV | PASS, **14,929,920** (+95,232) | PASS, **3,113,472** (+63,488) |
| First-token probe 24 + same-server repeat 12 | 24/24, 12/12 identical | 24/24, 12/12 identical (also after the stress) |
| Decode-vs-prefill consistency, 192 positions (INFO) | top-1 agree 0.953; decode top-1 never outside prefill top-5 | top-1 agree 0.958; same |
| eval-43 | **43/43** | **42/43** |
| 128K decode, c1 (INFO) | 287.3 tok/s (steps/s 112.2) | not run (TP3 only) |
| tool-eval-bench (INFO) | 93/100 (no earlier GLM run to compare with) | not run (TP3 only) |
| Long retrieval 128K / 900K | pass / pass | pass / pass |
| Hotel-lights, 30 runs | at 32 streams: 26 exact, 2 wrong, 2 truncated | at 8 streams: 29 exact, 1 wrong |
| KV stress at full concurrency | **not run** (owner decision; the client was stopped within a minute of its start) | PASS: 16/16 long requests, 92/92 pings, 0 bad lines, 15.3 min; lowest free VRAM 84 MiB |
| Served KV after the run, server and kernel log scans | not run (stopped after hotel-lights) | KV unchanged; no CUDA fault, OOM, panic or Xid line |

Hotel-lights asks for one number (48). Earlier GLM runs ranged from 26 to 30 exact out of 30 (20261112 TP2: 26/30).
One wrong answer in both runs here is "expected 48, got 49", an item an earlier GLM run also missed; the other TP3
miss is an answer with several numbers and no final one. The chain marks the TP3 run FAIL and the TP2 run WARN by its
own thresholds; the results are reported as numbers.

In `tp3/tp3-main/Q/` the kit's `SUMMARY.md` and `GATES.tsv` carry a `stress` row "no result": that is the stopped
stress client, not a failed stress. The TP2 chain marks the stress WARN only for the lowest free VRAM (84 MiB, below
the chain's 480 MiB margin); every request passed.

Not run in this tier on these packages: the TP3 KV stress at 32 streams, request checks (13 requests), tool calls 60
at TP3, corruption check at TP3, image latency at TP2, the prefill-merge arm at TP3, a same-session speed A/B. Not part
of the tier: GPQA Diamond (same weights and quantisation), long generation, the image question sets.

The kit in this folder (`qual_all.sh`, `qual_core.sh` and the Python files) is the 20261112 kit, unchanged; the
20261112 folder documents its stages and variables.
