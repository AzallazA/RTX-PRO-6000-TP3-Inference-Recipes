# Check results — GLM-5.3-Flash Hyperflare TP3 / TP2 20261116

**The tiered qualification of these packages has not run yet.** This file lists what did run: the two export chains
(quick checks, binary `67c691d29041`) and one boot of each package on the release binary (`27864a1d79e4`). Numbers
with their files: [../MEASUREMENTS.md](../MEASUREMENTS.md).

| Package | What it is | Results |
| --- | --- | --- |
| `gb202-tp3-202611170053` | TP3 release package (32 slots, images, 4 image slots) | `results/tp3-202611170053/export-chain/` |
| `gb202-tp2-202611170201` | TP2 release package (8 slots, images, 2 image slots) | `results/tp2-202611170201/export-chain/` |
| both | boot check on the release binary | `results/boot-check-release-binary/` |

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

## Not run yet

Tier "export" of the release policy (same weights, new export), on the release binary: boots with and without the
stack limit (KV), request checks (13 requests), first-token probe, image latency and smoke, tool calls 60, corruption
check, prefill probe, one decode and prefill pass, a prefill-merge arm (TP3), hotel-lights (TP3 at 32 streams, TP2 at
8), KV stress at full concurrency (TP3 32, TP2 8), decode-vs-prefill consistency, eval-43, tool-eval-bench (TP3), 128K
decode (TP3), long retrieval 128K / 900K. Not part of that tier: GPQA Diamond (same weights and quantisation), long
generation, the image question sets. Results will be added to this folder when they exist.

The kit in this folder (`qual_all.sh`, `qual_core.sh` and the Python files) is the 20261112 kit, unchanged; the
20261112 folder documents its stages and variables.
