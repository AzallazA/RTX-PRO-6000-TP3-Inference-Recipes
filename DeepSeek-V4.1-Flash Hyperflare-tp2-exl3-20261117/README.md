# DeepSeek-V4.1-Flash-1017 (EXL3 3-bit) on NVIDIA Hyperflare at TP2 — 2026-11-17

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 2 on two 96 GB RTX PRO 6000 Blackwell cards**,
from Diffbot's EXL3 3-bit expert pack. This is the first TP2 build of DeepSeek V4.1 in this repository. It is a
sibling of the TP3 build ([Hyperflare-tp3-20261116](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md)), not
a replacement: TP3 runs DeepSeek's own weights on three cards; this build runs a 3-bit quantisation of the routed
experts on two.

- **Weights:** [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
  @ `6110453b901e4b51da66f539f3802b5c7f18e8e8`, used as published. Routed experts in EXL3 3-bit (trellis), the rest DeepSeek's FP8, Diffbot's int4
  Engram tables, an EXL3 drafter. The weights keep DeepSeek's MIT license.
- **Engine:** the public fork [AzallazA/hyperflare-inference](https://github.com/AzallazA/hyperflare-inference) (formerly
  `spark-local-inference`) at tag `hyperflare-tp2-exl3-20261117` (`036cde24`), on upstream Hyperflare `70069281`, the base of the TP3
  20261116 release. EXL3 routed experts run on Diffbot's MIT `xmoe` kernels (their sm_120 rewrite of ExLlamaV3's MoE
  kernels), ported into the fork. **Nothing from the pack's AGPL `patches/vllm-exl3/` (its vLLM glue) is used.**
- **8 request slots, KV cache 2,140,800 tokens** (two full 1M-token sessions), 5 DSpark drafts with adaptive
  verification (cap 32, alpha 0.8), 4096-token prefill chunks, 8G host prefix cache, image input kept.
- **Resident and host experts:** layers 0, 1 and 5 keep all 384 routed experts in VRAM. The other 37 layers keep
  **344 (layers 2-4, 6-19) or 316 (layers 20-39)** of 384 experts resident in VRAM, chosen from our own route counts,
  and read the rest from host RAM over PCIe. Fewer resident experts buy more KV for less speed:
  [KV-TRADEOFF.md](KV-TRADEOFF.md).

**Qualification:** hotel-lights 30/30, 8-stream long KV stress passed (16/16 long requests, 137/137 pings), GPQA Diamond 92.4% (183/198); see [Quality](#quality).

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) covers the pack
download (~322 GiB), building the fork and the export (~41 min on our machine with a warm kernel cache). Then, from
the fork checkout:

```bash
FLARE=flare-cli/target/release/flare /path/to/this-folder/serve.sh   # GPUs 0,1, port 15015, model DeepSeek-V4.1-Flash, ~1 min
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.
Running it on another machine: [NOTES-OTHER-MACHINES.md](NOTES-OTHER-MACHINES.md).

## Contents

1. [Headline numbers](#headline-numbers) · [Speed](#speed) · [Test system](#test-system)
2. [What this build contains](#what-this-build-contains) · [Memory](#memory)
3. [Quality](#quality) · [Known limitations](#known-limitations)
4. [Files](#files) · [Licenses](#licenses) · [Credits](#credits)
5. Guides: [EXPORT.md](EXPORT.md) (build the package) · [KV-TRADEOFF.md](KV-TRADEOFF.md) (more or less KV vs speed) · [EXPERT-ORDER.md](EXPERT-ORDER.md) (measure your own traffic for the expert order)

## Headline numbers

| | this build (TP2, EXL3 3-bit experts, 450 W) |
| --- | --- |
| Request slots / verification cap / prefix cache | 8 / 32 / 8G |
| KV cache | **2,140,800** tokens |
| Decode 32K c1 / c4 / c8 (tok/s) | **243 / 553 / 757** (same-session A/B) · 236-245 / 556-577 / 759 (`llm-inference-bench`, 3 runs) |
| Decode 128K c1 / c4 / c8 (tok/s) | 236-242 / 523-550 / 743 (`llm-inference-bench`, 3 runs; c8 one run) |
| Prefill 32K (tok/s) | **11.0-11.7K** |
| Host RAM, server RSS | 151 GiB with both Engram tables in RAM (default); 110 GiB with one on disk; 64 GiB with both on disk ([Memory](#memory)) |
| Startup | ~1 min (warm page cache) |
| GPQA Diamond | 92.4% (183/198) |
| hotel-lights, 30 runs | 30/30 exact |
| KV stress (8 streams, long prompts, KV oversubscribed) | PASS: 16/16 long requests, 137/137 pings |

For orientation, the TP3 build of 20261116 (three cards, DeepSeek's own weights) does 32K c1 / c8 320 / 979 tok/s
with 5.40M KV.

## Speed

All numbers are single passes (no averaging) on the release package and binary with the `serve.sh` defaults,
thinking off. Decode cells use `ignore_eos`.

**Same-session A/B against the previous internal TP2 EXL3 package** (same binary and serve settings; the previous
package predates this release's prefill, merge and VRAM changes and kept 320 resident experts on every split layer).
Decode cells: `llm_decode_bench.py` from `llm-inference-bench`, 45 s per cell. Fixed-prompt and four-stream cells:
one fixed code prompt, the method of Diffbot's speed page. Prefill: one ~32K-token code prompt, fresh (no prefix
reuse).

| | this build | previous internal package |
| --- | --- | --- |
| Decode 32K c1 / c4 / c8 (tok/s) | **243 / 553 / 757** (c1 again later in the session: 241) | 170 / 380 / 563 |
| Verification steps/s at 32K c1 / c4 / c8 | 98.3 / 62.1 / 50.8 | 69.3 / 44.1 / 35.1 |
| Decode, 1 stream, fixed 2K / 46K prompt (tok/s) | 275-295 / 287 | 216 / 225 |
| Decode, 4 streams × 46K prompt (aggregate tok/s) | 580-596 | — |
| Prefill 32K (tok/s) | 11,013-11,457 (4 samples) | 9,526-9,791 (2 samples) |
| KV cache (tokens) | 2,140,800 | 3,684,480 |

**`llm-inference-bench` v0.8.0**, three separate runs on the same server (sustained decode, 30 s per cell; prefill
measured inside the decode runs):

| Context | c1 (tok/s) | c4 (tok/s) | c8 (tok/s) | Prefill (tok/s) |
| --- | --- | --- | --- | --- |
| 8K | — | — | — | 8.1-10.5K |
| 32K | 244.8 / 238.8 / 235.7 | 577.1 / 556.3 / 565.2 | 759.2 (run 3 only) | 10.9-11.7K |
| 64K | — | — | — | 11.5-11.7K |
| 128K | 238.1 / 241.8 / 236.1 | 534.9 / 549.9 / 522.6 | 743.1 (run 3 only) | 11.3-11.4K |

**Diffbot's own numbers for the same pack, for reference only.** These come from the pack's model card: **vLLM** with
their recipe, **on their machine** (also 2 × RTX PRO 6000). Engine, machine, power settings and benchmark scripts all
differ from ours, so read this as context, not as a like-for-like comparison. Our fixed-prompt and four-stream cells
follow their method; our prefill sample is ~32K tokens, theirs ~40K.

| | this build (Hyperflare, our machine) | Diffbot 2x default, GPU-only (vLLM, their machine) | Diffbot 2x CPU runtime (`CPU_EXPERTS=1`) |
| --- | --- | --- | --- |
| Code, 1 stream, 46K prompt (tok/s) | 287 | 172.8 | 193-223 |
| Fresh prefill (tok/s) | 11,013-11,457 (32K) | 11,157-11,467 (40K) | 13,442-14,389 (40K) |
| Four 46K streams (aggregate tok/s) | 580-596 | 362.9 | 396-452 |
| KV cache | 2.14M tokens | 1.07M tokens (fp8) | same |

## Test system

| Part | Spec |
| --- | --- |
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), each on PCIe Gen5 x16 (two of the TP3 machine's three cards) |
| GPU settings | **450 W** power limit, **+75 MHz core / +6000 MHz memory offset** (memory clock 16,365 MHz); ECC off; persistence mode on; peer-to-peer enabled |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, 1 socket, 1 NUMA node |
| RAM | 256 GB DDR5 (251 GiB visible to Linux), 8 channels at 6000 MT/s |
| Package drive | PCIe Gen5 x4 NVMe, ~12 GB/s direct read |
| Software | Ubuntu, NVIDIA driver 615.71.09, CUDA 13.1 |

Host-to-GPU copy measured ~37 GB/s per card. Experts that are not resident in VRAM are read over that link, so the
PCIe link, the power limit and the memory clock all show up in decode speed. See
[NOTES-OTHER-MACHINES.md](NOTES-OTHER-MACHINES.md).

## What this build contains

The fork's TP3 DeepSeek work (see the [TP3 20261116 README](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#what-this-build-contains))
carries over where it is format-independent: host-row embedding gather, vision tower in host RAM with weight
staging, VRAM reclaim, indexer and context side streams, PDL in the mHC norm kernels, vocabulary-parallel sampling,
TTFT guard (`shortest` order), prefill tail merge, fast start and stop. BRRR (MXFP4-only) is off. New for EXL3 at TP2:

| Change | Effect when it landed |
| --- | --- |
| EXL3 routed experts on the `xmoe` kernels for decode and verification; Diffbot's decode-once kernel (`xmoe_do`) for prefill, with a device-built job table so it is CUDA-graph capturable | the pack runs inside Hyperflare's captured graphs |
| Resident/host expert split per layer from our own blended code/prose route counts (`expert-order/`), two resident counts (344 on layers 2-4, 6-19; 316 on 20-39); host experts on a side stream with 16 SMs, 16-wide host-expert launches; shared expert on a side stream beside the router; fused routing prep | one host-expert decode launch 241 vs 292 µs at width 16 vs 8 |
| Single `wo_a` copy (no unused tile-major duplicate) | −688 MiB/GPU of VRAM, returned to the KV cache and resident experts |
| Prefill MoE scratch in 2 groups and 4,096-row passes, host experts on a side stream in prefill | −294 MiB/GPU; layer-10 prefill MoE 12.5 → 8.5 ms |
| Merged top-k lookup and merged MoE epilogue (bit-exact with the paths they replace) | fewer kernels per MoE layer |
| CUDA stack limit 128 B per thread (`FLARE_CUDA_STACK_LIMIT=128`; largest registered kernel 120 B) | ~+41K KV tokens vs 256 B |
| **All of the above together** vs the previous internal package | 32K c1 / c4 / c8 170 / 380 / 563 → 243 / 553 / 757 tok/s; prefill 32K ~9.7K → ~11.2K tok/s ([Speed](#speed)) |

Measured and not adopted: Diffbot's bulk-copy hot-expert loop (`xmoe_bulk`, ported, correct, no gain here; off), a
merged inverse-RoPE + quantisation kernel (no gain; off).

## Memory

**GPU (per GPU, served):** 92.75 GiB loaded (weights, resident experts, graphs, scratch), 1.92 GiB KV cache budget;
3.8 GiB of KV over the 2 GPUs = 2,140,800 tokens (890 bytes per token per GPU).

**Host RAM:** with both Engram tables in RAM (the default) Hyperflare plans ~140.5 GiB of host RAM and the server
process measured **151 GiB RSS** (peak 153.3 GiB during the qualification run). The plan, by item (both tables in RAM):

| Item | GiB |
| --- | --- |
| Engram int4 tables (2 x 51.5, `ENGRAM=ram`, default) | 103 |
| Host-resident expert banks (the experts not resident in VRAM) | ~25 |
| Prefix cache (`PREFIX_CACHE=8G`) | ~8.2 |
| Mapped host buffers | ~4.3 |

**Engram tables: RAM or disk.** The same package, KV 2,140,800 tokens in all three modes, measured in one session:

| | Both tables in RAM (default) | One table on disk (layer 14; `ENGRAM=one` = `--embedding-table-storage disk` + `FLARE_ENGRAM_RAM_LAYERS=0`) | Both tables on disk (`ENGRAM=disk` = `--embedding-table-storage disk`) |
| --- | --- | --- | --- |
| Server RAM (measured RSS) | 151 GiB | 110 GiB | 64 GiB |
| Flare's planned host RAM | 140.5 GiB | 98.7 GiB | 52.9 GiB |
| Decode 32K, 1 / 4 / 8 streams (tok/s) | 247 / 541 / 733 | 239 / 541 / 745 | 239 / 528 / 729 |
| Decode, 1 stream, fixed 2K / 46K prompt (tok/s) | 262 / 290 | 299 / 298 | 288 / 277 |
| Prefill 32K (tok/s) | 11.1-11.6K | 11.1-11.2K | 8.9-10.4K |
| Four 46K streams (aggregate tok/s) | 562-592 | 575-588 | 551-563 |

Disk modes use a 4 GiB row cache (`--embedding-table-cache-size 4GiB`; `ENGRAM_CACHE=4GiB`); the hit rate was ~56-57%
on the random-prompt benchmark. **One table on disk saves ~41 GiB with no measurable speed cost** (differences are
within run-to-run noise). **Both on disk saves ~87 GiB** at -1 to -3% decode and -10 to -20% prefill, because cold
Engram rows are read from NVMe. Disk reads bypass the OS page cache (O_DIRECT), so RAM use is predictable. The tables
need a fast NVMe (each is 51.5 GiB; ours reads ~12 GB/s).

Total machine RAM to plan for (leaving ~15 GiB for the OS):

| Mode | Minimum | Comfortable |
| --- | --- | --- |
| Both tables in RAM | ~168 GiB | 256 GB |
| One table on disk | ~125 GiB | 160-192 GB |
| Both tables on disk | ~80 GiB | 96-128 GB |

**Disk:** pack ~322 GiB, package ~314 GiB, kernel cache ~7 GiB. Exporting needs the pack and the package side by
side.

## Quality

Qualification: hotel-lights (30 runs), a long KV stress at 8 streams, and GPQA Diamond, on the package served exactly as
`serve.sh` serves it, at 450 W. Post-run: smoke with thinking off and on correct, KV unchanged, no faults. The tool-call
and consistency rows are from an earlier run of the same build.

| Check | This build | TP3 20261116 (for reference; different weights, 3 GPUs) |
| --- | --- | --- |
| Smoke (thinking on and off) and tool-call smoke | correct; 2/2 tool calls | correct; 2/2, 3/3 |
| Decode vs prefill consistency (top-1 agreement) | 0.979 over 427 positions (gate ≥ 0.95) | 0.951 |
| GPQA Diamond (198 questions, one sample each) | 92.4% (183/198; Wilson 95% interval 87.9-95.4%); 32,768 max tokens, temperature 0, thinking on; 5 answers hit the token limit without finishing | 91.4% (same set; the difference is not significant) |
| hotel-lights, 30 runs (8 streams) | 30/30 exact | 30/30 |
| KV stress: 8 streams, long unique prompts, KV oversubscribed, plus a short-request pinger | PASS: 2 waves, 16 long requests (~530K-token unique prompts, KV oversubscribed about 2x) 16/16 and 137/137 pings; no fault lines; lowest free VRAM 84 MiB; post-run smoke OK, KV unchanged | PASS (16 streams) |
| LiveBench, eval suites, long retrieval, long generation | not run | partly run |

Diffbot's model card reports the pack's own quality measurements against the original weights (KL divergence,
GSM8K, HumanEval); see the pack's README.

## Known limitations

- **3-bit routed experts.** This is a quantised model; quality differs from DeepSeek's weights (see GPQA above and
  Diffbot's card).
- **8 request slots.** 16 would exceed the `xmoe` flat-kernel row limit; not built.
- **Minimal qualification:** hotel-lights, KV stress and GPQA Diamond only.
- **Host RAM:** 151 GiB RSS with both Engram tables in RAM; `ENGRAM=one` / `ENGRAM=disk` cut that to 110 / 64 GiB ([Memory](#memory)).
- **Speed depends on the machine** more than at TP3: experts not resident in VRAM stream over PCIe from host RAM.
  The resident set comes from our code/prose route counts; a very different workload routes more often to host
  experts.
- **top_p is fixed at 1.0** (vocabulary-parallel sampling); temperature is fixed at 1.0 by Hyperflare's DeepSeek
  recipe.
- **The `--mtp-only` package requires speculation** at serve time (`DRAFTS` > 0, the default 5).
- **Thinking mode and forced tool calls:** `tool_choice: "required"` or a named function with thinking on returns
  HTTP 400, as DeepSeek's reference does. Turn thinking off (`chat_template_kwargs: {"enable_thinking": false}`) to
  force a call.
- **Bench loops:** `llm_decode_bench` decode cells use `ignore_eos`; a rare repetition loop there is a bench artifact.
- **Hardware-specific package:** 2 × 96 GB SM120 (RTX PRO 6000) only, tuned on the exporting machine. Other setups
  need a fresh export.
- Personal fork of Hyperflare; not upstreamable as written.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `KV-TRADEOFF.md` | trade decode speed for more KV, or KV for more speed (resident expert counts) |
| `EXPERT-ORDER.md` | measure routes on your own traffic and build your own expert order |
| `NOTES-OTHER-MACHINES.md` | what to check before expecting these numbers elsewhere |
| `serve.sh` / `export.sh` | serve the package / export it from the pack (wraps the fork's `tp3-tools/ds41-exl3-export-tp2.sh`) |
| `expert-order/` | per-layer route counts (blended 0.6 code / 0.4 prose, EXL3 TP2) that choose the resident experts |
| `source.patch` | the fork changes on top of upstream-synced public main, as patches (`git am`) |

## Licenses

DeepSeek weights: MIT. Diffbot EXL3 pack: weights MIT (DeepSeek's), recipe code MIT; its `patches/vllm-exl3/`
(AGPL-3.0-only) is not used, copied or linked. `xmoe` kernels in the fork
(`ignition/recipes/deepseek41/exl3/kernels/xmoe/`): MIT, Copyright (c) 2025 Turboderp and Copyright (c) 2026 Diffbot
(`LICENSE.exllamav3`, `LICENSE.diffbot` next to them; changes listed in `ignition/recipes/deepseek41/exl3/THIRD_PARTY.md`).
ExLlamaV3: MIT. Hyperflare: Apache-2.0. FlashInfer (patched by the fork's `tp3-patches/`): Apache-2.0.
`llm-inference-bench`: MIT.

## Credits

NVIDIA Hyperflare and FlashInfer b12x; DeepSeek for the model; Diffbot for the EXL3 3-bit pack, the int4 Engram
tables and the `xmoe` kernels; turboderp for ExLlamaV3 and EXL3; Local Inference Lab for `llm-inference-bench`.
