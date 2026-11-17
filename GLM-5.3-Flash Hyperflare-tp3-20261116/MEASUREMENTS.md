# Measurements — GLM-5.3-Flash Hyperflare TP3 / TP2 20261116

Every number in the README, with the run it comes from. Paths are relative to `qualification/results/`. All runs at
450 W/GPU, +75 MHz core, +6000 MHz memory offset, on the machine in the README's Test system.

## Runs

| Run | What | Binary | Results |
| --- | --- | --- | --- |
| TP3 export chain (2026-11-16 19:17-19:58) | export of `gb202-tp3-202611170053` with `FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_SLAB_SEGMENT=1`; dry-run KV against the package without reclaim (`gb202-tp3-202611151109`, same settings, exported on 2026-11-15 from the port); one boot with KV, image, probe and speed checks | `67c691d29041` (port build without the 20261116 scheduler and runtime options) | `tp3-202611170053/export-chain/` |
| TP2 export chain (2026-11-16 20:44-21:07) | export of `gb202-tp2-202611170201` with `FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_ONESHOT_FIT=1`; reference `gb202-tp2-202611151621` (same settings without reclaim, 2026-11-15) | `67c691d29041` | `tp2-202611170201/export-chain/` |
| Boot check on the release binary | one boot of each package, TTFT order shortest, stack limit 256 | `27864a1d79e4` (tested build of the release tree) | `boot-check-release-binary/` |
| Tier "export" qualification (2026-11-17 14:43-16:02) | boots without and with the stack limit, probe, consistency, eval-43, 128K decode and tool-eval-bench (TP3), long retrieval, hotel-lights, KV stress (TP2 only), post-run checks (TP2 only) | `1bdcce7378e4` (build of fork `main` `036cde24`) | `qual-export-20261117/tp3/`, `qual-export-20261117/tp2/` |

Both chains served with the `serve-glm` defaults of the time: `--scheduler-policy interactive`, prefix cache 32G,
5 drafts, TP3 alpha 0.8, TP2 the package alpha (1.0); no `--scheduler-ttft-order`, no `FLARE_CUDA_STACK_LIMIT`.
Decode: `llm_decode_bench.py` (`llm-inference-bench` v0.9.1), 32K context, ignore-EOS, one 30 s pass per cell, 3 s
warm-up. Prefill: the same tool's prefill-only pass, 8K (10 samples) and 32K (4 samples). Times in `CHAIN.md` are
local (UTC−5); package timestamps and the kit's `SUMMARY.md` are UTC.

## TP3 `gb202-tp3-202611170053`

| Number | Value | File |
| --- | --- | --- |
| Export wall time | 35 min (2,130 s); Load weights 52.6 s, Prepare and tune kernels 117.6 s, Compile model forwards 765.0 s (3 workers) | `CHAIN.md`, `X/export.log` |
| Graphs | 74 (68 text, 6 vision); largest vision pool 976.0 MiB below the largest text pool 1,008.4 MiB | `GATES.tsv` (X-graphs, X-vision-pool) |
| Reclaim | trial graph pool 3,574 of 3,574 MiB returned before the serving capture; prefill slab 528.26 MiB in 530 MiB of its own segments; kept graph pool 1,110 MiB per GPU (151109: 3,216); core 62,448 MiB per rank (151109: 64,554) | `GATES.tsv` (X-reclaim-lines, X-reclaim-inventory), `X/export.log` |
| KV dry run (97,281 MiB free per GPU, PCM 32G) | **15,025,152** (151109 on the same binary: 14,035,968; +989,184) | `K/dryrun-new.log`, `K/dryrun-151109.log` |
| KV served | **14,834,688** (151109: 13,833,984; +1,000,704, 1.072×) | `serve-s-new.log` (Allocate cache), `GATES.tsv` (S-kv) |
| Memory per GPU | 61.48-61.49 GiB loaded, 32.92 GiB cache budget (98.7 GiB over 3 GPUs); prefix cache 39.7 GiB (8,031 slots) | `serve-s-new.log` |
| Startup | 30 s | `GATES.tsv` (S-serve) |
| Decode 32K c1 / c8 / c32 | **279.1 / 886.6 / 1,812.5** tok/s; steps/s 117.6 / 465.2 / 753.3; tokens per step 2.37 / 1.90 / 2.40 | `S/single/new-1.{json,log}` |
| Same cells, 151109, same binary (median of 2 passes, 2026-11-15) | 274 / 873 / 1,806 tok/s; steps/s 113.0 / 427.6 / 746.5 | `S/single/gates.txt` |
| Prefill 8K / 32K | 10,408 / 11,712 tok/s (151109: 10,362 / 11,574) | `S/single/new-1-prefill.{json,log}`, `S/single/gates.txt` |
| Image latency | first 0.14 s, next 0.12 s, 8 in sequence median 0.13 s, 8,000-token image 1.46 / 1.47 s, 4,096-token 0.68 s, 4 small images at once 0.45 s | `S/latency.{json,log}` |
| Image smoke | 5/5 correct (small, near-max 6,729 tokens, over-max downscaled 8,021 tokens, 2 and 4 images) | `S/smoke.{json,log}` |
| First-token probe | 24/24, with logprobs 24/24, same-server repeat identical 12/12; against the 20261112 package's stored probe: 12/12 exact | `S/probe.{json,log}`, `S/probe-cmp-stored.log` |
| Prefill probe | PASS; marginal: 33K needle 30/32 found (2 omissions) against 32/32 for 151109, p = 0.25; pooled needles 246/248 against 248/248; images 36/36 | `S/probe/RESULTS.md` |

## TP2 `gb202-tp2-202611170201`

| Number | Value | File |
| --- | --- | --- |
| Export wall time | 17 min (1,021 s; resumed the tuning database of the 2026-11-15 TP2 export); Prepare and tune kernels 270.4 s | `CHAIN.md`, `GATES.tsv` (X-export, X-tuning) |
| Graphs | 56 (50 text, 6 vision); largest vision pool 976.0 MiB below the largest text pool 1,072.1 MiB | `GATES.tsv` |
| Reclaim | trial graph pool 2,990 of 2,990 MiB returned; one-shot all-reduce sized for 512 rows (4 MiB per call) instead of 4,096; kept graph pool 1,110 MiB (151621: 2,442), TP communication 138 MiB (151621: 502) | `GATES.tsv` (X-reclaim-lines, X-reclaim-inventory) |
| Serial MoE expert races | 58 of 1,432 tuning ledger rows timed one GPU at a time (`B12X_SERIAL_RACES`, local patch) | `GATES.tsv` (X-serial) |
| KV dry run | **3,177,472** (151621 on the same binary: 2,653,184; +524,288). In `GATES.tsv` the reference is labelled "130307"; the value is 151621's | `K/*.log`, `GATES.tsv` (K-dry-KV) |
| KV served | **3,049,984** (151621: 2,518,016; +531,968, 1.211×; same labelling note) | `serve-new.log`, `GATES.tsv` (S-kv) |
| Memory per GPU | 84.14 GiB loaded, 10.26 GiB cache budget (20.5 GiB over 2 GPUs); 1,583 MiB mapped host memory per GPU (embedding, vision); prefix cache 30.9 GiB (9,375 slots) | `serve-new.log`, `GATES.tsv` (K-memory) |
| Startup | 28 s | `GATES.tsv` (S-serve) |
| Decode 32K c1 / c4 / c8 | **246.5 / 519.2 / 762.8** tok/s; steps/s 105.0 / 276.5 / 314.2; tokens per step 2.35 / 1.88 / 2.43 | `B/new-p1.{json,log}`, `B/BQ.md` |
| Same cells, 151621, same binary (2 passes, 2026-11-15) | 245.5 / 516.7 / 748.0 tok/s; steps/s 103.9 / 280.2 / 310.0 | `B/BQ.md` |
| Prefill 8K / 32K | 11,559 / 12,271 tok/s (151621 same binary: 11,403 and 11,385 / 12,086 and 12,140) | `B/new-p1-prefill.{json,log}`, `B/BQ.md` |
| First-token probe | 24/24, repeat 12/12; against the 20261112 package's stored probe: 12/12 exact | `S/smoke-probe*.{json,log}` |
| Image smoke | 5/5 correct | `S/image-smoke.{json,log}` |
| Corruption check | 20 runs, 6,000 tokens: 0 true corruptions (5 natural-uncertainty flags) | `S/qual/verify_fix.log`, `S/qual/SUMMARY.md` |
| Tool calls | 60/60 at concurrency 8 | `S/qual/tools.log`, `S/qual/tool_calls.json` |
| Prefill probe | PASS; needles 248/248 (151621: 247/248), images 36/36 | `S/probe/RESULTS.md` |

## Release binary boot check (`boot-check-release-binary/`, 2026-11-16 21:11-21:17)

Binary `27864a1d79e4`, the `serve-glm` defaults plus `--scheduler-ttft-order shortest` and `FLARE_CUDA_STACK_LIMIT=256`
(the settings of `serve-glm-tp3.sh` / `serve-glm-tp2.sh`). Same tools and cell lengths as the export chains.

| Number | TP3 | TP2 | File |
| --- | --- | --- | --- |
| Startup / served KV | 30 s / **14,929,920** | 29 s / **3,113,472** | `CHAIN.md`, `tp*/serve.log` |
| Stack line | 256 B per thread (recorded 1,024 B, largest kernel stack 32 B) | 256 B (largest 48 B) | `tp*/serve.log` (Prepare kernels) |
| Memory per GPU | 61.49 GiB loaded, 33.12-33.13 GiB cache budget (99.3 GiB) | 84.15 GiB loaded, 10.47 GiB (20.9 GiB) | `tp*/serve.log` |
| Text smoke / image smoke / first-token probe | 391 / 5/5 / 24/24 + 12/12 | 391 / 5/5 / 24/24 + 12/12 | `SUMMARY.md`, `tp*/image-smoke.*`, `tp*/lean.*` |
| Decode 32K, one 30 s pass | c1 / c8 / c32 **290.3 / 878.8 / 1,827.6** tok/s; steps/s 114.2 / 459.8 / 759.8 | c1 / c8 **245.4 / 750.2**; steps/s 104.8 / 312.3 | `tp*/decode.{json,log}` |
| Prefill 8K / 32K | 10,180 / 11,701 | 11,583 / 12,275 | `tp*/prefill.{json,log}` |
| Against the export chain's pass (other binary) | c1 1.040× tok/s (steps/s 0.971×), c8 0.991×, c32 1.008×; prefill 0.978× / 0.999× | — | `tp3/cmp.md` |
| Log scans | no CUDA fault, OOM or panic line; no Xid | same | `SUMMARY.md` |

## Tier "export" qualification (`qual-export-20261117/`, 2026-11-17 14:43-16:02)

Binary `1bdcce7378e4`, built from fork `main` `036cde24` (Rust changes since this release's tag: DeepSeek Engram code
and a KV-layout fix for rows wholly in granule padding). Settings of `serve-glm-tp3.sh` / `serve-glm-tp2.sh`; each
package booted first without `FLARE_CUDA_STACK_LIMIT` (probe only), then with 256 for the checks. Sampling temperature
1.0, top-p 0.95, reasoning effort max (eval-43 low, tool-eval high). The TP3 run was stopped by hand after hotel-lights
on the owner's decision: the KV stress at 32 streams and the post-run checks did not run. File paths below are under
`qual-export-20261117/`.

| Number | TP3 (QC 32) | TP2 (QC 8) | File |
| --- | --- | --- | --- |
| Served KV, no stack limit / stack 256 | 14,834,688 / **14,929,920** (+95,232) | 3,049,984 / **3,113,472** (+63,488) | `tp*/CHAIN.md`, `tp*/tp*-nostack/serve.log`, `tp*/tp*-main/serve.log` |
| Startup | 29 s / 28 s | 28 s / 28 s | `tp*/CHAIN.md` |
| Stack line | 256 B per thread (largest kernel stack 32 B) | 256 B (largest 48 B) | `tp*/GATES.tsv` (stack-line) |
| First-token probe | 24/24 with logprobs, repeat 12/12 identical | same; again after the stress: 24/24, 12/12 | `tp*/tp*-nostack/lean.*`, `tp2/tp2-main/lean-post.json` |
| Decode-vs-prefill consistency (192 positions, 0 cut) | top-1 agree 0.953; decode top-1 outside prefill top-5: 0; abs. logprob difference median 0.0239, p90 0.1751, p99 0.5951, max 0.8574 | 0.958; 0; median 0.0088, p90 0.1335, p99 0.3294, max 0.6894 | `tp*/tp*-main/Q/consistency.log` |
| eval-43 | 43/43 | 42/43 | `tp*/tp*-main/Q/eval.log`, `eval_*.json` |
| 128K decode, c1, one 30 s pass | 287.3 tok/s, steps/s 112.2 | not run | `tp3/tp3-main/long128.{json,log}` |
| tool-eval-bench | 93/100 | not run | `tp3/tp3-main/Q/tooleval.{json,log}` |
| Long retrieval 128K / 900K | pass / pass | pass / pass | `tp*/tp*-main/Q/long-retrieval*` |
| Hotel-lights, 30 runs, max tokens 100,000 | 26 exact / 2 wrong / 2 truncated; misses listed: run 28 several numbers and no final answer, run 30 "expected 48, got 49" | 29 exact / 1 wrong (run 25: "expected 48, got 49") | `tp*/tp*-main/Q/hotel-lights-c*.log` |
| KV stress at full concurrency | not run | PASS: 2 waves × 8 requests of ~530,000 prompt tokens, 16/16 ok, 9,367,312 prompt tokens, pinger 92/92, 0 bad log lines, lowest free VRAM 84 MiB, 15.3 min | `tp2/tp2-main/Q/stress.log`, `stress_*.json` |
| Served KV after the run; server and kernel log scans | not run | 3,113,472 = boot value; no CUDA fault, OOM, panic or Xid line | `tp2/GATES.tsv`, `tp2/SUMMARY.md` |

Times in `CHAIN.md` are local (UTC−5); the kit's `Q/` files are UTC. In `tp3/tp3-main/Q/` the `stress` row "no result"
is the stopped stress client.

## Comparison values from earlier folders

| Value | Source |
| --- | --- |
| 20261112 TP3 `gb202-tp3-202611130000`: KV 13,130,496 served / 13,393,152 dry; decode c1 / c8 / c32 279 / 866 / 1,790 (averages of repeated 60 s runs, `balanced`); prefill 32K 11,763; images 0.15 / 0.13 / 1.51 / 0.54 s; startup 35 s; 64.88-64.92 GiB loaded, 29.35-29.39 GiB budget | [20261112 README](../GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md) |
| 20261112 TP2 `gb202-tp2-202611130307`: KV 2,231,808 / 2,414,592; decode c1 / c4 / c8 248 / 512 / 727; prefill 32K 11,980; 86.56 GiB loaded, 7.71 GiB budget; hotel-lights 26/30; KV stress 16/16 | same |
| Port without reclaim (local packages, not published): TP3 151109 13,833,984 served (2026-11-15; decode level with 20261112 on the same binary: c1 274 vs 276, c8 873 vs 880, c32 1,806 vs 1,797; prefill +0.5-1%), TP2 151621 2,518,016 served | the two chains' reference lines above; 2026-11-15 port runs |
