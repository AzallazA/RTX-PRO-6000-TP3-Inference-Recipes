# DeepSeek-V4.1-Flash-1017 (EXL3 3-bit) on NVIDIA Hyperflare at TP2 — 20261124

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, CLI (`flare run`, `flare models`), and cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 2 on two 96 GB RTX PRO 6000 Blackwell cards** from Diffbot's EXL3 3-bit expert pack. This release supersedes [Hyperflare-tp2-exl3-20261117](../DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261117/README.md) (`hyperflare-tp2-exl3-20261117`, package `gb202-tp2-202611171416`), unifying TP2 EXL3 onto the **post-split Hyperflare stack** shared verbatim with [Hyperflare-tp3-20261124](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261124/README.md):

- Runtime / Ignition: [AzallazA/hyperflare-inference@hyperflare-tp2-exl3-20261124](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp2-exl3-20261124) (`5ffe4d54`, identical serving binary to TP3 `hyperflare-tp3-20261124`)
- Recipes / EXL3 kernels: upstream [`local-inference-lab/hyperflare-recipes`](https://github.com/local-inference-lab/hyperflare-recipes) @ `fee34e1` (PR `#9`) + [tp3-patches/hyperflare-recipes-tp2-exl3-20261124.patch](tp3-patches/hyperflare-recipes-tp2-exl3-20261124.patch)
- Kernels (`b12x`): upstream [`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer) `main` @ `0aa01681` (PR `#132`, `#136`) + [tp3-patches/b12x-tp2-exl3-20261124.patch](tp3-patches/b12x-tp2-exl3-20261124.patch)

**How to get it:** export the package from Diffbot's EXL3 pack using [EXPORT.md](EXPORT.md), then serve with [serve.sh](serve.sh):
```bash
FLARE=/path/to/hyperflare-inference/target/release/flare ./serve.sh
```

Exact tool versions, SHAs, and checksums are in [versions.txt](versions.txt); full pre-publish minimum qualification summary is in [qualification/SUMMARY.md](qualification/SUMMARY.md).

---

## Headline Numbers

Measured on `2 × RTX PRO 6000 Blackwell 96 GB` (SM120, PCIe Gen5 x16, **450 W/GPU PL**, `MAXC=8`, `PREFILL=4096`, `DRAFTS=5`, `ALPHA=0.8`, `BUDGET=32`, `PREFIX_CACHE=8G`, `ENGRAM=ram`, `344/316` resident experts, `STACK_LIMIT=144`).

| Metric | This Release (`gb202-tp2-202611241005`, R1–R2 range) | Last Published (`2026-11-17`, 3-run range) | Gain vs Published |
|---|---:|---:|---:|
| **KV Cache Capacity (plain / WARMIND)** | **2,839,808** (`269` snaps) / **2,868,992** (`272` snaps) | 2,140,800 tokens | **+699,008 to +728,192 (+32.7% to +34.0%)** |
| **Max Kernel Stack / `STACK_LIMIT`** | **`136 B` / `144 B`** (`102` graphs, `1.7s` prep) | `120 B` / `128 B` | — |
| **Prefill 8K (`client tok/s`)** | **11,119–13,348** | 8,100–10,500 | **+27.1% to +37.3%** |
| **Prefill 32K (`client tok/s`)** | **13,428–13,510** | 10,900–11,700 | **+15.5% to +23.2%** |
| **Prefill 64K (`client tok/s`)** | **12,831–13,566** | 11,500–11,700 | **+11.6% to +15.9%** |
| **Prefill 128K (`client tok/s`)** | **13,266–13,469** | 11,300–11,400 | **+17.4% to +18.1%** |
| **Decode 32K `c1` (`tok/s` / `verif/s`)** | **246.3–253.6** (`104.2–106.4 verif/s`) | 235.7–244.8 (`98.3 verif/s`) | **+3.6% to +4.5%** (`+6.0% to +8.2%` verif/s) |
| **Decode 32K `c4` (`tok/s` / `verif/s`)** | **569.6–575.4** (`71.3–74.5 verif/s`) | 556.3–577.1 (`62.1 verif/s`) | **`-0.3% to +2.4%`** (**`+14.8% to +20.0%` verif/s**) |
| **Decode 32K `c8` (`tok/s` / `verif/s`)** | **804.9–813.2** (`53.7–54.2 verif/s`) | 759.2 (`50.8 verif/s`) | **+6.0% to +7.1%** (`+5.7% to +6.7%` verif/s) |
| **Decode 128K `c1` (`tok/s` / `verif/s`)** | **255.0–260.9** (`102.7–103.7 verif/s`) | 236.1–241.8 | **+7.9% to +8.0%** |
| **Decode 128K `c4` (`tok/s` / `verif/s`)** | **563.3–594.2** (`71.0–72.9 verif/s`) | 522.6–549.9 | **+7.8% to +8.1%** |
| **Decode 128K `c8` (`tok/s` / `verif/s`)** | **806.1–807.2** (`55.1–56.3 verif/s`) | 743.1 | **+8.5% to +8.6%** |
| **Real-World Coding (`tetristest.py`)** | **296.1–296.9 avg / 413.6–416.4 MAX** | — | — |

*(Last Published numbers are the 3-run ranges from `Hyperflare-tp2-exl3-20261117` (`30s` decode cells; `c8` 1 run; `verif/s` from its same-session A/B); this release's numbers are the exact R1–R2 ranges from two paired runs (`llm_decode_bench.py`, `15s` decode cells, `ignore_eos=true`). Gains compare low-to-low and high-to-high. Telemetry across all 12 decode cells: active GPU `100.0%`, PCIe RX `8.9–16.0 GB/s`, `0` repetition loops.)*

---

## What Changed Since `2026-11-17`

### Shared Post-Split Stack & TP2 EXL3 Speed Pass (100% Lossless)

| Component | Upstream Base + Merged PRs | Release Tag / Patch | Key TP2 EXL3 Additions |
|---|---|---|---|
| **[`AzallazA/hyperflare-inference`](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp2-exl3-20261124)** | `NVIDIA/hyperflare-inference` `main` + PR `#416`, `#473`, `#479`, `#491` | **`hyperflare-tp2-exl3-20261124`** (`5ffe4d54`) | Shared verbatim with TP3: `INT4_FP16_BLOCK32 = 16` Engram support, `FLARE_FLAT_MEMORY_POOL`, **Helios disjoint physical CPU pinning**, native static `has_addend`/`has_gate`, host prefix-cache recency, KV plane granule padding & `O(N log N)` snapshot selection |
| **[`local-inference-lab/hyperflare-recipes`](https://github.com/local-inference-lab/hyperflare-recipes)** | `main` @ `fee34e1` (PR `#9`) | **`fee34e1` + [patch](tp3-patches/hyperflare-recipes-tp2-exl3-20261124.patch)** | Unified `deepseek41` + `exl3/` recipe: **`xmoe_bulk` hot loop across `M=1..64` (MUL1 + MCG draft, `0 B` stack)**, `SplitLaggedMHC` `M<=96` + **fused `attention_hc → MXFP8 dense` (`-43` quant launches/step)**, TP2 H32 1-wave MLA decode splits (`9:10,18:5,23:4,31:3,47:2,94:1`), prefill `shared`-before-`experts` `MERGE_EPILOGUE=1` (WARMIND-safe), hoisted `pass_counts`, vectorized `uint4/float4` `xmoe_decode_prep_mapped_kernel`, `register_tc_gemv_rope` `.state` fix (`-51` Triton launches/step), **batched Markov vocabulary finish**, EXL3 `int16` routing sort keys, pooled EXL3 scratch + flat 1D Inductor pool (`+699K–728K` KV), rank-local & L14 side-stream int4 Engram, 26× faster zero-scale Triton steering, **dedicated non-blocking capture & EXL3 side streams** |
| **[`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer)** | `main` @ `0aa01681` (PR `#132`, `#136`) | **`0aa01681` + [patch](tp3-patches/b12x-tp2-exl3-20261124.patch)** | `SplitLaggedMHC` extended to `M <= 96` + fused `emit_mxfp8_dense` (bit-exact swizzled MMA scales for `qkv_a`), MXFP8 TC-GEMV `split_k=5,10` & BF16 TC-GEMV `split_k=20,40,80`, `B12X_PCIE_TWOSHOT_PULL=1` (`addend` + TP2 `H=5120`), static `has_gate` DCP a2a, `B12X_PCIE_DCP_GATHER_PULL=1`, **4-slot stream-isolated split-K TC-GEMV pools**, **dedicated non-blocking DMA streams** |

### Performance & Memory Highlights
- **800+ tok/s at BOTH 32K and 128K on 2 GPUs (`+6.0% to +8.6%` at `c8`, `+7.8% to +8.1%` at `128K c1/c4`):**
  - **`xmoe_bulk` (`FLARE_DS41_EXL3_BULK_MIN_ROWS=1`, MUL1 + MCG draft):** Cold-L2 microbench measured `-5.3% to -9.5%` hot MoE across `M=4..32` at `0 B` kernel stack.
  - **Fused `SplitLaggedMHC` (`M <= 96`) → MXFP8 Dense Quantizer:** Eliminates all `43` `MXFP8RowsQuantLaunch` kernels/step across `c1..c8` (`M=1..48`), bit-exact vs `mxfp8.quantize_dense_rows`, saving `~0.75 ms/step` while overlapping `fn @ residual_out + Sinkhorn` behind `att_reduce` and `mlp_reduce`.
  - **1-Wave TP2 H32 MLA Decode Splits (`9:10,18:5,23:4,31:3,47:2,94:1`), MXFP8/BF16 TC-GEMV `split_k`, `B12X_PCIE_TWOSHOT_PULL=1`, Fused RoPE & Batched Markov Finish:** Eliminates 2-wave SM spills at `M=24` and `M=32..37`, removes `-51` standalone RoPE launches/step, and pushes `c4`/`c8` verification rates to **`71.3–74.5 / 53.7–54.2 verif/s` (32K)** and **`71.0–72.9 / 55.1–56.3 verif/s` (128K)**.
- **+699,008 to +728,192 KV Tokens (+32.7% to +34.0% vs `2026-11-17`, ~2.85M KV on 2 cards):** Flat 1D Inductor pool (`FLAT_POOL=1`), pooled EXL3 scratch (`POOL_SCRATCH=1`), single `wo_a` layout (`WOA_SINGLE=1`, `-688 MiB/GPU`), host vision arena (`VISION_ARENA_HOST=1`), `IMAGE_CONCURRENCY=1`, host Markov embeddings (`MARKOV_EMBED_HOST=1`), static collective tuning (`102` graphs, `1.7s` prep), and `ENGRAM_FULL_PROJECTION=0`.
- **+11.6% to +37.3% Faster Prefill (`13.3–13.6K tok/s` at 32K–128K):** Encoder-only prefill (`--scheduler-encoder-prefill true`), `PREFILL_SKIP_SUFFIX=3072,3584,4096`, `IndexSideChain` prefill overlap, `ENGRAM_CHUNK_ROWS=768`, `B12X_PCIE_DCP_GATHER_PULL=1`, prefill `shared`-before-`experts` `MERGE_EPILOGUE=1`, hoisted `pass_counts`, EXL3 `int16` routing sort keys, and rank-local L14 side-stream int4 Engram.

### Correctness & Stability Fixes
- **Split-K TC-GEMV Concurrent-Stream Data Race Fix:** Eliminates the concurrent side-stream vs main-stream split-K `atomicAdd(tc_flags)` race on compressed global KV (`compressor.project_gates` vs `project_values`). Verified **`0 / 102` racy graphs**, normal PCIe RX (`8.9–16.0 GB/s`), and `0` repetition loops at `128K c8`.
- **Dedicated Non-Blocking Capture, DMA & EXL3 Side Streams:** Allocates capture, DMA helper, and EXL3 `_SIDE`/`_PREFILL_SIDE` streams via `cuStreamCreate(CU_STREAM_NON_BLOCKING)` outside PyTorch's 32-slot pool so streams never alias.
- **Helios Disjoint Physical CPU Pinning:** Moves rank 0 off CPU 0 → CPU 1, pins peer ranks after `GraphExecutor::new_unstarted`, and assigns Engram reader threads monotonically starting at `1 + world_size`.
- **`ENGRAM=one` Lossless on Disk:** Layer-14 int4 Engram table on disk (`4 GiB` cache, `~110 GiB` RSS vs `~151 GiB` for `ENGRAM=ram`) verified at **`0.969–0.979` top-1 agreement (`0` miss)** with rank-local L14 side-stream overlap.

---

## Qualification Summary

Full pre-publish minimum qualification **PASSED** (see [qualification/SUMMARY.md](qualification/SUMMARY.md)):

- **Static CUDA-Graph Race & Overlap Check:** **PASS** — `0 / 102` racy graphs on both plain `1005` (`encoder_prefill4096` critical path `3,717`) and WARMIND twin `1209` (`3,709`).
- **Greedy Corruption Check (`verify_fix.py 10`):** **PASS** — `0` TRUE corruptions (`1,950` tok on `1005`; `1,697` tok on `1209`).
- **Prefill vs Decode Consistency (`consistency.py 48 4`):** **PASS** —
  - Plain `1005`: **`0.979`** (RAM, `0` miss, med `0.0034`) / **`0.969`** (`ENGRAM=one`, `0` miss, med `0.0016`).
  - WARMIND `1209`: **`0.984`** (RAM, `0` miss, med `0.0042`) / **`0.979`** (`ENGRAM=one`, `0` miss, med `0.0055`).
- **Long-Context Global-KV NIAH & Coherence:** **PASS** — `53,819`-token prompt retrieved both needles verbatim (`KESTREL-9472-QUARTZ`, `FALCON-3189-OBSIDIAN`) in `4.72s`, `0` loops; `46,365`-token coding prompt coherent (`has_def=True`, `1,024` tok in `6.34s`, `0` loops).
- **Pre-Publish Minimum Qualification Suite (run on WARMIND v1.1 twin `gb202-tp2-202611241209`, `2,868,992` KV / `272` snaps, `scale=0`, `450 W` PL):**
  - **Live WARMIND v1.1 Steering:** **PASS** — `scale=None` vs `0.0` exact match, monotonic shift `0.0 → 0.5 → 1.0` (`steer_dlp=2.27`), `256`-tok coding `319.1 → 309.5 tok/s`, HTTP `400` on mismatch, `1.5` clamped to `1.0`. *(Note: because EXL3 has no BRRR in-kernel steering, the 66 Triton steering launches add ~`0.65 ms/step` at `c1` vs plain `1005`, while matching `1005` at `c8`: `32K 806.4` / `128K 798.5 tok/s`, Tetris `293.6 avg / 417.5 MAX`. Export without `STEERING=` for max `c1` speed, or with `STEERING=` for live per-request steering.)*
  - **Smoke (thinking off / on):** **PASS** — `2/2` (`'391'`, `'391'`).
  - **Images / Vision Smoke (`5` cases):** **PASS** — **`5/5` served, `5/5` correct** (`small` `0.2s`, `1920×900` `0.3s`, `4160×1280` downscaled `0.3s`, `2 images` `0.2s`, `4 images` `0.3s`).
  - **`eval-43` (thinking off):** **PASS** — **`42 / 43`**.
  - **Tool calls (`60` prompts @ `8` streams):** **PASS** — **`60 / 60` OK** (`0` wrong tool, `38s`).
  - **`hotel-lights` (`30` runs @ `8` streams):** **PASS** — **`30 / 30` exact**, `0` wrong, `0` truncated.
  - **Long retrieval (`128K` & `900K`):** **PASS** — `128000 pass`, `900000 pass`.
  - **KV Stress (`8` streams, 2 waves × 8 × ~530K unique prompts + pinger = `1.5×` KV oversubscription):** **PASS** — **`16/16` long requests ok, `109/109` pings ok**, `9,511,722` prompt tokens pushed, `0` bad log lines, min free VRAM `82 MiB` (`393` samples), `13.3 min` wall; post-stress smoke `'391'` OK and KV unchanged at `2,868,992`.
- **Sustained Decode Loop Detection:** **PASS** — `0` loops across all LDB cells and Tetris runs on both `1005` and `1209`.

---

## Test System & Recommended Host Tuning (Linux)

- **Test System:** 2 × RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120) on PCIe Gen5 x16, **450 W PL**, `3165/17001 MHz` max clocks; AMD Ryzen Threadripper PRO 7965WX (24-core); 256 GB DDR5-6000 (8-ch); PCIe Gen5 x4 NVMe; NVIDIA driver `615.71.09`, CUDA `13.1`.
- **Kernel boot params:** `amd_iommu=off iommu=off pcie_aspm=off pcie_port_pm=off processor.max_cstate=1 irqaffinity=0,7-24,31-47` *(optional `mitigations=off` on dedicated private hosts)*.
- **PCIe MRRS = 4096 B (`+1.0%` to `+3.3%` across prefill & decode):**
  ```bash
  for bus in 01:00.0 21:00.0; do sudo setpci -s "$bus" CAP_EXP+0x08.w=5000:7000; done
  ```
- **CPU / Memory:** `performance` governor, C2 disabled (`state2/disable = 1`), THP `always` (`defer+madvise`), KSM off (`ksm/run = 0`), movable IRQs steered off Helios's pinned cores & SMT siblings (`1-6,25-30`).
- **Warning — do NOT lock GPU memory clocks (`nvidia-smi -lmc`) under a `450 W` PL:** Locking `-lmc 17001,17001` on RTX PRO 6000 Blackwell regressed decode `ms/graph` by `-19%` to `-26%` under `450 W`; keep dynamic clocks unlocked (`nvidia-smi -rmc`).

---

## Licenses

- **DeepSeek-V4.1-Flash-1017 weights:** MIT ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
- **Diffbot EXL3 3-bit pack & `xmoe` kernels:** Weights MIT (DeepSeek's), recipe code MIT; pack's `patches/vllm-exl3/` (AGPL-3.0-only) is **not** used, copied, or linked. `xmoe` kernels (`exl3/kernels/xmoe/`): MIT, Copyright (c) 2025 Turboderp and Copyright (c) 2026 Diffbot.
- **ExLlamaV3:** MIT.
- **Hyperflare (`NVIDIA/hyperflare-inference`), `hyperflare-recipes`, and `flashinfer`:** Apache-2.0.
- **Optional WARMIND v1.1 steering sidecar:** MIT ([jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`).
- **`llm-inference-bench`:** MIT.

## Credits

- **NVIDIA Hyperflare** and **FlashInfer / b12x** teams for the core runtime, ignition compiler, and SM120 Blackwell kernels.
- **Local Inference Lab** (`local-inference-lab/flashinfer`, `local-inference-lab/hyperflare-recipes`, `llm-inference-bench`) for PR `#132`/`#136`, PR `#9`/`#19`, and the benchmarking suite.
- **DeepSeek** for DeepSeek-V4.1-Flash-1017.
- **Diffbot** for the [EXL3 3-bit pack](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), int4 Engram tables, and `xmoe` kernels.
- **turboderp** for ExLlamaV3 and EXL3.
- **jcartu** for the steering feature (upstream PR `#404`) and the [WARMIND v1.1 steering sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar).
- The upstream authors of the included `hyperflare-inference` PRs (`#416`, `#473`, `#479`, `#491`).
