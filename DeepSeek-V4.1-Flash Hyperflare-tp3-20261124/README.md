# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261124

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, CLI (`flare run`, `flare models`), and cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000 Blackwell cards. This release supersedes [Hyperflare-tp3-20261121](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261121/README.md) (`hyperflare-tp3-20261121.1`, package `gb202-tp3-202611201530`), moving onto the **post-split 3-repository Hyperflare stack** with tag **`hyperflare-tp3-20261124`** across all three forks:

- Runtime / Ignition: [AzallazA/hyperflare-inference@hyperflare-tp3-20261124](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp3-20261124) (`5ffe4d54`, rebased on `NVIDIA/hyperflare-inference` `main` + PR `#416`, `#473`, `#479`, `#491`)
- Recipes / BRRR: [AzallazA/hyperflare-recipes@hyperflare-tp3-20261124](https://github.com/AzallazA/hyperflare-recipes/releases/tag/hyperflare-tp3-20261124) (`63f9e4b`, rebased on `local-inference-lab/hyperflare-recipes` `main` + PR `#9`, `#19`)
- Kernels (`b12x`): [AzallazA/flashinfer@hyperflare-tp3-20261124](https://github.com/AzallazA/flashinfer/releases/tag/hyperflare-tp3-20261124) (`14aa5e64`, rebased on `local-inference-lab/flashinfer` `main` + PR `#132`, `#136`)

**Default package (`gb202-tp3-202611240502`) includes the optional WARMIND v1.1 steering sidecar** (`steering_scale=0` by default = unsteered model; `steering_scale` in `[0, 1]` runs inside the BRRR fused kernel; `STEERING_KEYS_FILE` optional).

**How to get it:** export the package from your checkpoint using [EXPORT.md](EXPORT.md), then serve with [serve.sh](serve.sh):
```bash
FLARE=/path/to/hyperflare-inference/target/release/flare ./serve.sh
```

Exact tool versions, SHAs, and checksums are in [versions.txt](versions.txt); minimum pre-publish qualification summary is in [qualification/SUMMARY.md](qualification/SUMMARY.md).

---

## Headline Numbers

Measured on `3 × RTX PRO 6000 Blackwell 96 GB` (SM120, PCIe Gen5 x16, **450 W/GPU PL**, `MAXC=16`, `HOT=355`, `PREFILL=4096`, `DRAFTS=5`, `ALPHA=0.8`, `BUDGET=32`, `PREFIX_CACHE=12G`, `ENGRAM_RAM=1`, `STACK_LIMIT=48`).

| Metric | This Release (`2026-11-24`, R1–R2 range) | Last Published (`2026-11-21`, 3-run range) | Gain vs Published |
|---|---:|---:|---:|
| **KV Cache Capacity (WARMIND / plain)** | **5,810,816** (`549` snaps) / **5,825,408** (`551` snaps) | 5,687,040 tokens | **+123,776 to +138,368 (+2.2% to +2.4%)** |
| **Max Kernel Stack / `STACK_LIMIT`** | **`40 B` (`32 B` plain) / `48 B`** (`107` graphs, `1.3–1.4s` prep) | `176 B` / `256 B` | **-77.3% to -81.8%** |
| **Prefill 8K (`client tok/s`, cold)** | **21,054–23,347** | 16,527–16,603 | **+27.4% to +40.6%** |
| **Prefill 32K (`client tok/s`, cold)** | **23,668–24,462** (`24,444` WARMIND `scale=0`) | 18,269–18,336 | **+29.6% to +33.4%** |
| **Prefill 64K (`client tok/s`, cold)** | **22,922–24,903** | — | — |
| **Prefill 128K (`client tok/s`, cold)** | **23,530–24,227** (`24,142` WARMIND `scale=0`) | 18,791–19,128 | **+25.2% to +26.7%** |
| **Decode 32K `c1` (`tok/s`)** | **366.3–388.9** (`374.6` WARMIND `scale=0`) | 311–324 | **+17.8% to +20.0%** |
| **Decode 32K `c4` (`tok/s`)** | **848.1–850.1** (`827.4` WARMIND `scale=0`) | 773–810 | **+5.0% to +9.7%** |
| **Decode 32K `c8` (`tok/s`)** | **1,174.0–1,182.1** (`1,165.4` WARMIND `scale=0`) | 1,059–1,102 | **+7.3% to +10.9%** |
| **Decode 32K `c16` (`tok/s`)** | **1,611.4–1,626.0** (`1,658.9` WARMIND `scale=0`) | 1,459–1,467 | **+10.4% to +10.8%** |
| **Decode 128K `c1` (`tok/s`)** | **342.0–354.2** | 321–323 | **+6.5% to +9.7%** |
| **Decode 128K `c4` (`tok/s`)** | **818.6–827.4** | 758–774 | **+6.9% to +8.0%** |
| **Decode 128K `c8` (`tok/s`)** | **1,115.7–1,131.2** | 1,033–1,063 | **+6.4% to +8.0%** |
| **Decode 128K `c16` (`tok/s`)** | **1,579.5–1,587.9** | 1,396–1,400 | **+13.1% to +13.4%** |
| **Real-World Coding (`tetristest.py`)** | **561.9–566.0 avg / 735.8–747.3 MAX** | — | — |

*(Last Published numbers are the 3-run ranges from `Hyperflare-tp3-20261121` (`30s` decode cells); this release's R1–R2 ranges are two paired runs (`llm_decode_bench.py`, `15s` decode cells, `ignore_eos=true`, scout cold prefill), with WARMIND v1.1 `scale=0` verification shown in parentheses (`STEERING_PLAIN_ID=1` selects identical plans). Gains compare low-to-low and high-to-high. Telemetry across all 16 decode cells: CPU `11.5–11.6%`, GPU `99.9–100.0%`, PCIe RX `6.3–17.0 GB/s`, `0` repetition loops.)*

---

## What Changed Since `2026-11-21`

### 3-Repository Upstream Rebase & Architecture

| Repository | Upstream Base + Merged PRs | Release Tag (`commit`) | TP3 Additions |
|---|---|---|---|
| **[`AzallazA/flashinfer`](https://github.com/AzallazA/flashinfer/releases/tag/hyperflare-tp3-20261124)** | `local-inference-lab/flashinfer` `main` @ `0aa01681` (PR `#132`, `#136`) | **`hyperflare-tp3-20261124`** (`14aa5e64`) | World-3 DCP/tuning/gather-pull, two-shot pull, Engram MXFP4, scratch preference, RO=1 acquire (`7fd84f83`); unconditional 8-SM cooperative MoE reserve (`3e317f0b`); **4-slot stream-isolated split-K TC-GEMV pools** (`05cf561a`); **dedicated non-blocking DMA streams** (`14aa5e64`) |
| **[`AzallazA/hyperflare-inference`](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp3-20261124)** | `NVIDIA/hyperflare-inference` `main` + PR `#416`, `#473`, `#479`, `#491` | **`hyperflare-tp3-20261124`** (`5ffe4d54`) | Static gated `PreparedAllGather` & `ConcurrentAllReduce(own_source=True)` (`d7eaffbd`); **Helios disjoint physical CPU pinning** (`9c895738`); native static `has_addend`/`has_gate` + `pcie_all_reduce_addend_dep` (`6e596d41`); host prefix-cache completion recency (`3142ec53`); KV plane granule padding & `O(N log N)` two-row snapshot selection (`5ffe4d54`) |
| **[`AzallazA/hyperflare-recipes`](https://github.com/AzallazA/hyperflare-recipes/releases/tag/hyperflare-tp3-20261124)** | `local-inference-lab/hyperflare-recipes` `main` + PR `#9`, `#19` (`b52a19a`) | **`hyperflare-tp3-20261124`** (`63f9e4b`) | Pad-free 2×2048 DMA prefill WO/MLP + `IndexSideChain` side-stream `dsa.quantize_q_mxfp4` + `ENGRAM_CHUNK_ROWS=768` (`20fd67d`); `SplitLaggedMHC` on BRRR `attention_hc` + **`register_tc_gemv_rope` `.state` fix (`-51` Triton launches/step)** (`dc15fb1`); **batched Markov vocabulary finish + staged embedding reuse + tiled FP8 bias** (`bd2a8d1`); **dedicated non-blocking capture & side streams** (`63f9e4b`) |

### Performance & Memory Highlights (100% Lossless)
- **Split-lagged mHC + fused RoPE + BF16 TC-GEMV (`flashinfer` PR `#136`, `hyperflare-recipes` PR `#19`):**
  - `SplitLaggedMHC` (`ClusterMhcPostYNorm` + overlapped `ReadOnlyFnPartialM8` + `_SplitFinalize`) overlaps `attention_hc` Sinkhorn directly on BRRR layers at `M <= 8`.
  - Fixed missing `.state` unwrap in `register_tc_gemv_rope`, eliminating 51 standalone `_rotate_grouped` Triton launches per step.
  - Added BF16 TC-GEMV `split_k = 5, 10` for `K = 5120`, `M <= 8` router and `index_weight` projections.
- **Batched Markov Vocabulary Finish (`hyperflare-recipes bd2a8d1`):**
  - Batches Markov vocabulary finalization across draft steps (`_finish_distribution_batched`), reuses staged Markov embeddings in `PreparedDraftConfidence`, and tiles multi-row FP8 Markov bias.
- **2×2048 Pad-Free Pipelined DMA Prefill + Static Collectives (`PREFILL=4096`):**
  - Eliminated standalone `constant_pad_nd` and clone buffers in pipelined prefill WO/MLP (`encoder_prefill4096` critical path `2,134`).
  - Static collective tuning (`107` graphs, `1.3–1.4s` boot prep) + static gated generation logprobs + `FLAT_POOL=1` + `ENGRAM_CHUNK_ROWS=768` (`24 → 14` collectives/chunk) + `STACK_LIMIT=48` (`40 B` max registered kernel stack) push served KV to **`5,810,816` tokens** (`549` shared snapshots with WARMIND; `5,825,408` plain).

### Correctness & Stability Fixes
- **Split-K TC-GEMV Concurrent-Stream Data Race Fix (`flashinfer 05cf561a`):** Replaced the process-global `(tc_scratch, tc_flags)` buffer in `bf16_gemv` and `blockscaled` split-K TC-GEMV with a 4-slot capture/stream-isolated pool keyed via `cuStreamGetCaptureInfo`. Prevents concurrent side-stream `compressor.project_gates` and main-stream `compressor.project_values` (layers 2, 8, 14) from racing on `atomicAdd(tc_flags)`. Verified **`0 / 107` racy graphs**.
- **Dedicated Non-Blocking DMA & Capture Streams (`flashinfer 14aa5e64`, `hyperflare-recipes 63f9e4b`):** Allocates `PCIeDmaAllReduce` helper streams and the graph capture stream via `cuStreamCreate(CU_STREAM_NON_BLOCKING)` outside PyTorch's 32-slot pool, eliminating stream-handle aliasing.
- **Helios Disjoint Physical CPU Pinning (`hyperflare-inference 9c895738`):** Moves rank 0 off CPU 0 → CPU 1, pins peer ranks after `GraphExecutor::new_unstarted`, and assigns Engram reader threads monotonically starting at `1 + world_size`.

---

## Qualification Summary

All pre-publish qualification gates **PASSED** on **`gb202-tp3-202611240502`** (WARMIND v1.1 default at `scale=0`, `450 W` PL; see [qualification/SUMMARY.md](qualification/SUMMARY.md)):

- **Static CUDA-Graph Race & Overlap Check:** **PASS** — `0 / 107` racy graphs; `encoder_prefill4096` critical path `2,134`.
- **Greedy Corruption Check (`verify_fix.py`):** **PASS** — `0` TRUE corruptions (`1,021` tok on WARMIND `0502`; `1,733` tok on plain `0119`).
- **Prefill vs Decode Consistency (`consistency.py 48 4`):** **PASS** — **`0.978` top-1 agree** (`0` missing from prefill top-5; plain `0119`: `0.960`, `0` missing).
- **Live WARMIND v1.1 Steering Verification:** **PASS** — `scale=None` vs `0.0` exact match (`top1_dlp=0.005`), monotonic distribution shift `0.0 → 0.5 → 1.0` (`steer_dlp=1.47`), `256`-tok coding `483.5 → 541.1 tok/s`, mismatched top-level vs `chat_template_kwargs` → HTTP `400`, `1.5` clamped to `1.0`.
- **Smoke (thinking off / on):** **PASS** — `2/2` (`'391'`, `'391'`).
- **Images / Vision Smoke (`5` cases):** **PASS** — **`5/5` served, `5/5` correct** (`small` `0.1s`, `1920×900` `0.2s`, `4160×1280` downscaled `0.3s`, `2 images` `0.1s`, `4 images` `0.2s`).
- **`eval-43` (thinking off):** **PASS** — **`43 / 43`** (`25s`).
- **Tool calls (`60` prompts @ `8` streams):** **PASS** — **`60 / 60` OK** (`0` chose a different tool, `20s`).
- **`hotel-lights` (`30` runs @ `16` streams):** **PASS** — **`30 / 30` exact**, `0` wrong, `0` truncated.
- **Long retrieval (`128K` & `900K`):** **PASS** — `128000 pass`, `900000 pass` (plus `53,819`-tok NIAH both needles verbatim & `46,412`-tok coding coherent, `0` loops).
- **KV Stress (`16` streams, 2 waves × 16 × ~530K unique prompts + pinger):** **PASS** — **`32/32` long requests ok, `142/142` pings ok**, `19,029,127` prompt tokens pushed, `0` bad log lines, min free VRAM `78 MiB` (`482` samples, matching `2026-11-21`'s `78 MiB`), `16.3 min` wall; post-stress smoke `'391'` OK and KV unchanged at `5,810,816`.
- **Pipelined Prefill Deadlock Sweep (`431..7,831` tokens) & Sustained Decode Loop Gate:** **PASS** — `5/5` clean, `0` loops across all 16 LDB cells + Tetris.

---

## Test System & Recommended Host Tuning (Linux)

- **Test System:** 3 × RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120) on PCIe Gen5 x16 (each on its own CPU root port), **450 W PL**, `3165/17001 MHz` max clocks; AMD Ryzen Threadripper PRO 7965WX (24-core); 256 GB DDR5-6000 (8-ch); NVIDIA driver `615.71.09`, CUDA `13.1`.
- **Kernel boot params:** `amd_iommu=off iommu=off pcie_aspm=off pcie_port_pm=off processor.max_cstate=1 irqaffinity=0,7-24,31-47` *(optional `mitigations=off` on dedicated private hosts)*.
- **PCIe MRRS = 4096 B (`+1.0%` to `+3.3%` across prefill & decode):**
  ```bash
  for bus in 01:00.0 21:00.0 e1:00.0; do sudo setpci -s "$bus" CAP_EXP+0x08.w=5000:7000; done
  ```
- **CPU / Memory:** `performance` governor, C2 disabled (`state2/disable = 1`), THP `always` (`defer+madvise`), KSM off (`ksm/run = 0`), movable IRQs steered off Helios's pinned cores & SMT siblings (`1-6,25-30`).
- **Warning — do NOT lock GPU memory clocks (`nvidia-smi -lmc`) under a `450 W` PL:** Locking `-lmc 17001,17001` on RTX PRO 6000 Blackwell regressed decode `ms/graph` by `-19%` to `-26%` under `450 W`; keep dynamic clocks unlocked (`nvidia-smi -rmc`).

---

## Licenses

- **DeepSeek-V4.1-Flash-1017 weights & FP8 Engram tables:** MIT ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
- **Hyperflare (`NVIDIA/hyperflare-inference`), `hyperflare-recipes`, `flashinfer`, and BRRR fused layer tail:** Apache-2.0 (see [licenses/](licenses/) and respective repository `LICENSE` / `NOTICE` files).
- **WARMIND v1.1 steering sidecar:** MIT ([jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`).

## Credits

- **NVIDIA Hyperflare** and **FlashInfer / b12x** teams for the core runtime, ignition compiler, and SM120 Blackwell kernels.
- **Local Inference Lab** (`local-inference-lab/flashinfer`, `local-inference-lab/hyperflare-recipes`, `llm-inference-bench`, `tool-eval-bench`) for PR `#132`/`#136`, PR `#9`/`#19`, and the benchmarking/evaluation suite.
- **DeepSeek** for DeepSeek-V4.1-Flash-1017 and the FP8 Engram tables.
- **Fraser Price** for [BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (fused decoder layer tail).
- **jcartu** for the steering feature (upstream PR `#404`) and the [WARMIND v1.1 steering sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar).
- The upstream authors of the included `hyperflare-inference` PRs (`#416`, `#473`, `#479`, `#491`).
