# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261125

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, CLI (`flare run`, `flare models`), and cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000 Blackwell cards. This release supersedes [Hyperflare-tp3-20261124](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261124/README.md) (`hyperflare-tp3-20261124`, package `gb202-tp3-202611240502`), rebasing onto the latest **3-repository upstream Hyperflare stack** (PRs **`#23`/`#27`**, **`#138`/`#139`/`#140`**, **`#424`/`#426`/`#498`/`#499`/`#510`**, and `TERMINAL_PREFILL = 14`), extending **PCIe row-sharded encoder prefill (`PREFILL=3840`)** + **bit-exact MXFP8 attention-input AllGather (`#27`)** to TP3 with **sharded Engram DMA `reduce_scatter` (`-525 MiB` VRAM)**, landing **Wave 5–6 lossless decode optimizations** (`BRRR_PUSH_V2=2` zero-fence push, fused `H24` Q-RoPE, `M <= 96` fast paths), and baking in **21 P0/P1/P2 audit bug fixes** across all three repositories:

- Runtime / Ignition: [AzallazA/hyperflare-inference@hyperflare-tp3-20261125](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp3-20261125) (`ba6a4c11`, rebased on `NVIDIA/hyperflare-inference` `main` @ `84ac035c` + `#424`, `#426`, `#498`, `#499`, `#510`, `TERMINAL_PREFILL = 14`, static-only collectives)
- Recipes / BRRR: upstream [`local-inference-lab/hyperflare-recipes`](https://github.com/local-inference-lab/hyperflare-recipes) @ `39c157a4` (PRs `#23`, `#27`) + [tp3-patches/hyperflare-recipes-tp3-20261125.patch](tp3-patches/hyperflare-recipes-tp3-20261125.patch)
- Kernels (`b12x`): upstream [`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer) `main` @ `0d48af53` (PRs `#138`, `#139`, `#140`) + [tp3-patches/b12x-tp3-20261125.patch](tp3-patches/b12x-tp3-20261125.patch)

**Default package (`gb202-tp3-202611251350`) includes the WARMIND v1.1 steering sidecar** (`steering_scale=0` by default = unsteered model; `steering_scale` in `[0, 1]` runs inside the BRRR fused kernel at **`24 B` stack**; `STEERING_KEYS_FILE` optional).

**How to get it:** export the package from your checkpoint using [EXPORT.md](EXPORT.md), then serve with [serve.sh](serve.sh):
```bash
FLARE=/path/to/hyperflare-inference/flare-cli/target/release/flare ./serve.sh
```

Exact tool versions, SHAs, and checksums are in [versions.txt](versions.txt); qualification summary is in [qualification/SUMMARY.md](qualification/SUMMARY.md).

---

## Headline Numbers

Measured on `3 × RTX PRO 6000 Blackwell 96 GB` (SM120, PCIe Gen5 x16, `MAXC=16`, `HOT=355`, `PREFILL=3840`, `ROW_SHARD_PREFILL=1`, `DRAFTS=5`, `ALPHA=0.8`, `BUDGET=32`, `PREFIX_CACHE=12G`, `ENGRAM_RAM=1`, `STACK_LIMIT=48`).

### 1. Standardized `lil-bench 1.7.1 standard/1` & `tetristest.py` (450 W PL vs Stock 600 W TDP)

| Workload | `2026-11-25` @ **450 W PL** | `2026-11-25` @ **600 W TDP** | Notes |
|---|---:|---:|---|
| **KV Cache Capacity** | **5,883,648 tokens** (`556` snaps) | **5,883,648 tokens** (`556` snaps) | **`+72,832`** vs `2026-11-24`, **`+196,608`** vs `2026-11-21` |
| **Max Kernel Stack / `STACK_LIMIT`** | **`32 B` / `48 B`** (`12,669` kernels, `0.6s`) | **`32 B` / `48 B`** (`12,669` kernels, `0.6s`) | Steered BRRR stack shrunk `40 B → 24 B` |
| **Prefill 32K** | **27,624 tok/s** (`1.168s`) | **30,553 tok/s** (`1.056s`) | **`+10.6%`** @ 600 W |
| **Prefill 128K** | **27,688 tok/s** (`4.642s`) | **31,564 tok/s** (`4.072s`) | **`10,521 tok/s/GPU`** on 3 GPUs (`+14.0%` @ 600 W) |
| **Decode C1 @ `0 / 64k / 128k`** | **371.6 / 370.7 / 363.6 tok/s** | **384.3 / 386.9 / 376.2 tok/s** | `0` PCIe replays/errors |
| **Decode C8 @ `0 / 64k / 128k`** | **1,242.5 / 1,252.1 / 1,231.0 tok/s** | **1,339.6 / 1,307.5 / 1,283.0 tok/s** | All 9 decode cells uncapped `ok` @ 600 W |
| **Decode C16 @ `0 / 64k / 128k`** | **1,733.1 / 1,709.6 / 1,686.4 tok/s** | **1,825.2 / 1,795.2 / 1,769.0 tok/s** | `2.67–2.76 GHz` @ 600 W |
| **Real-World Coding (`tetristest.py`)** | **613.2 tok/s avg / 768.2 MAX** (`38.3K` tok) | **602.9 tok/s avg / 773.8 MAX** (`41.9K` tok) | Peak `788.5–798.7 tok/s` across runs |

### 2. Matched 450 W PL Gains vs Last Published (`2026-11-24`, `0502`)

| Metric (450 W PL) | This Release (`2026-11-25`) | Last Published (`2026-11-24`) | Gain vs `2026-11-24` |
|---|---:|---:|---:|
| **KV Cache Capacity (WARMIND)** | **5,883,648** (`556` snaps) | 5,810,816 (`549` snaps) | **+72,832 (+1.3%)** |
| **Max Kernel Stack / Prep** | **`32 B`** (`0.6s` prep) | `40 B` (`1.3–1.4s` prep) | **-20% stack / 2× faster prep** |
| **Prefill 32K (`lil-bench`)** | **27,624 tok/s** | 23,668–24,462 | **+12.9% to +16.7%** |
| **Prefill 64K (`llm_decode_bench`)** | **28,008–28,453 tok/s** | 22,922–24,903 | **+14.3% to +22.2%** |
| **Prefill 128K (`lil-bench` / LDB)** | **27,688** (`27,129–27,652` LDB) | 23,530–24,227 | **+14.3% to +17.7%** |
| **Decode 64K–128K `c1`** | **363.6–398.8** (`156–160 v/s`) | 342.0–354.2 | **+6% to +13%** |
| **Decode 64K–128K `c8`** | **1,231.0–1,253.2** (`72.1–72.4 v/s`) | 1,115.7–1,131.2 | **+8.8% to +10.8%** |
| **Decode 64K–128K `c16`** | **1,686.4–1,712.5** (`59.8–60.3 v/s`) | 1,579.5–1,587.9 | **+6.8% to +7.8%** |
| **Real-World Coding (`tetristest.py`)** | **613.2 avg / 768.2–788.5 MAX** | 561.9–566.0 avg / 735.8–747.3 MAX | **+8.3% to +9.1% avg** |

---

## What Changed Since `2026-11-24`

### 1. Upstream Rebase + TP3 Row-Sharded Encoder Prefill (`#23` + `#27`, `PREFILL=3840`)
- **Upstream PRs `#23`, `#139`, `#140`, `#499`:** Partitions each encoder prefill microbatch half into per-rank contiguous row shards (`half_tokens // world_size`, aligned to 128 rows → multiples of `768` on TP3, so **`PREFILL=3840`** = `5 × 768` with `half=1920`, `shard=640`). Eliminates 3× redundant mHC, routing, and FFN MXFP8 quantization across the 20 encoder layers and replaces full all-reduces with overlapped CE ring `reduce_scatter` + shard gathers.
- **Upstream PR `#27` (Bit-Exact MXFP8 Attention-Input AllGather):** Quantizes each rank's row shard to MXFP8 (`min_amax=1e-4`) *before* the attention-input AllGather instead of gathering BF16 and redundantly quantizing on every rank — **100% bit-exact (`torch.equal`)**, cuts attention AllGather traffic by **48.4%**, drops **414 kernels** (`13,083 → 12,669`), gains **`+14,592` KV tokens** (`5,883,648`), and adds **`+3.6%` to `+4.4%`** prefill throughput (`27.7K` @ 450 W, **`31.6K`** @ 600 W).
- **Sharded Engram DMA `reduce_scatter` (`-525 MiB` VRAM):** Avoids replicating full Engram projections on TP3, preserving `HOT=355` resident experts/layer and **`5,883,648` KV tokens**.
- **Upstream `#424`, `#426`, `#498`, `#510` & `TERMINAL_PREFILL = 14`:** Parallel long-prompt tokenization (`#424`, 5s faster on `900K` retrieval), deferred `TERMINAL_PREFILL` state snapshot guard (`#426`), Heliopause `RO=1` D2H export fence (`#498`), packed speculative duration zero-poisoning fix (`#510`), and static-only collective cleanup.
- **Upstream `#23` CSF Fixture Bug Fix:** Fixed `scale_expansion_benchmark_call` poisoning `("moe_inputs", name)` without `a_scales` on CSF-compressed experts.

### 2. Wave 5–6 Lossless Decode Optimizations (`BRRR`, `H24 MLA`, `b12x`, `M <= 96`)
- **`BRRR_PUSH_V2=2` Zero-Fence Payload-Ready Push:** Replaces fence-based two-shot/flag push with `st.relaxed.sys.v4` + `-0.0` BF16 sentinel polling and chunk-major peer inboxes (`LOCAL = 0 B`, `32 B` plain / `24 B` steered stack). Verified 100% sound under **both `RO=0` and `RO=1`**.
- **Lifted All Fast Paths from `M <= 32` to `M <= 96`:** `SplitLaggedMHC`, `ready_small_mxfp8`, `PDL`, and `PIN_MID_QKV_NORM` cover full `8×6` (`M=48`) and `16×6` (`M=96`) verification batches without regressing.
- **Fused `H24` Q-RoPE (`H24_Q_ROPE=1`) & Tuned TP3 MLA Splits:** `MLA_SPLITS="18:10,37:5,47:4,75:3,94:2,96:1"`.
- **Fixed `TWOSHOT_PULL` `**addend_call` Regression:** Restored missing `**addend_call` forwarding on `TWOSHOT_PULL` in `b12x`.

### 3. Pre-Publish Audit: 21 P0 / P1 / P2 Bug & Crash Fixes
- **`hyperflare-inference` (`ba6a4c11`):** Fixed upstream `#426` partial-`terminal_prefill_catalog` bug where small tails under `deferred_prefill` + `prefill_merge`/`prefill_tail_merge` could bypass `TerminalPrefill` and snapshot incomplete state; fixed interactive TTFT guard dropping `GraphKind::TerminalPrefill`; fixed `GatherPool` spin-deadlock on spawn failure or worker panic; replaced `FLARE_ENGINE_HEADROOM_MIB` panic with clean `Result::Err`; upgraded `os.rs` pinning to deduplicate SMT siblings via sysfs and preserve/restore initial container/cgroup `sched_getaffinity`; made `MADV_NOHUGEPAGE` non-fatal on restricted kernels; added `PreparedEngramLookupLocal` scale validation; restored fixed-alignment padding check in `cache_memory.rs`; folded `min_measured_duration` in `service.rs`.
- **`hyperflare-recipes` (`536d2ba`):** Fixed `BRRR_PUSH_V2=1` two-shot `gather()` `pub_off` offset corruption; clamped `-1` padding sentinels before `index_select(hot_map)` in `_run_ffn_gathered`; guarded `ENGRAM_FULL_PROJECTION=1` with `PREFILL_DMA=0` and raised `partial_rows` cap `256 → 512`; fixed hardcoded `PreparedRoutedExperts` under `row_shard_dma` (EXL3 compatibility); replaced `mla_h24_kernel.py` `attn_sink` LSE merge with overflow-safe branchless `fmax` log-sum-exp; eliminated `NO_ENGRAM=1` double lookup/collectives; added `brrr/kernels/*` and `exl3/kernels/**/*` to `pyproject.toml` package-data; fixed `pending_merge` stale state and `_commit_window` `int64` indexing; defaulted `export.sh` to `PREFILL=3840 ROW_SHARD_PREFILL=1` and hardened `serve.sh`/`check_env.sh`.
- **`flashinfer` (`964b31ec`):** Split `mla_norm_mxfp8` into 4 exact `mutates_args` custom ops; added explicit `ValueError` guards preventing silent `None` output aliasing in `wo_mxfp8`, `mxfp8_rows`, and `_split_lagged`; primed `dma_sm_copy` pre-capture and fixed raw `cuStreamCreate` teardown under device context; clamped `mla_norm_mxfp8` exponent to `[-127, 127]`.

---

## Qualification Summary

All qualification, stress, and portability gates **PASSED** on **`gb202-tp3-202611251350`** (`hyperflare-tp3-20261125` binary SHA-256 `7aa439a25794`, `STACK_LIMIT=48`, `BUDGET=32`; see [qualification/SUMMARY.md](qualification/SUMMARY.md)):

- **Static CUDA-Graph Race Check:** **PASS** — `0 / 107` racy graphs (`12,669` kernels, `0.6s` prep, max stack `32 B`, `LOCAL = 0 B`).
- **Greedy Corruption Check (`verify_fix.py 10`):** **PASS** — **`0` TRUE corruptions** (`1,667` qual + `943` live smoke tokens on `v10c`; `0` TRUE across `5,336` tokens on `v10b`).
- **Prefill vs Decode Consistency (`consistency.py 48 6`):** **PASS** — Det **`0.972`** (`0` miss, med `0.0040`); 3× Stoch **`0.972 / 0.972 / 0.957`** (**`0` miss**).
- **WARMIND v1.1 Steering Verification:** **PASS** — `scale=None` vs `0.0` exact match (`top1_dlp=0.000000`), verified shift at `0.5/1.0`, OOB clamping & HTTP 400 mismatch checks PASS.
- **Images / Vision Smoke (`5` cases):** **PASS** — **`5/5` served, `5/5` correct** (`small` `0.1s`, `1920×900` `0.2s`, `4160×1280` downscaled `0.3s`, `2 images` `0.1s`, `4 images` `0.2s`).
- **`eval-43` (thinking off):** **PASS** — **`42 / 43`** (`v10b`: `43/43`, temp=1.0).
- **Tool calls (`60` prompts @ `8` streams):** **`59 / 60` OK** (`0` chose a different tool, `16s`; `v10b`: `60/60`, temp=1.0).
- **`hotel-lights` (`30` runs @ `16` streams):** **PASS** — **`30 / 30` exact**, `0` wrong, `0` truncated.
- **Long retrieval (`128K` & `900K`):** **PASS** — **`128000 pass`, `900000 pass`** (`73s`).
- **Long-Context NIAH & Partial-Tail Row-Sharding:** **PASS** — `53,819`-tok NIAH both needles verbatim (`2.40s`), `46,365`-tok `code32k` coherent (`3.84s`, `0` loops), `3/3` + **`22/22` bucket-boundary partial-tail NIAH (`127..7681+` tok)** 100% verbatim PASS.
- **Live Crash Sweep & 5-Phase Stress Test:** **PASS** — 12/12 API/4xx/steering OOB, 8-stream abrupt disconnects, 16-way mixed burst (`16/16` HTTP 200) PASS; 5-phase sustained stress test (>2.5M prompt tokens, `48/48` prefix-cache eviction recalls, `112.5K` NIAH + 4× concurrent decode, `0` failed requests, `0` leaked slots) PASS.
- **Alternate Modes & Clean-Checkout Packaging:** **PASS** — `ENGRAM_RAM=0` NVMe (`0` TRUE, `0.969` cons / `0` miss), `ENGRAM_RAM=layer1` hybrid (`0` TRUE, `0.969` cons / `0` miss), `BUDGET=96` (`0` TRUE), `git apply --check` clean on fresh `origin/main` across all 3 repos, `pyproject.toml` kernel packaging & 14/14 CLI flag parity PASS, `flare doctor` `7/7` non-root checks PASS.

---

## Troubleshooting & Host Notes

- **`PREFILL=3840` on TP3:** Row-sharded encoder prefill requires `(PREFILL / 2 / world_size) % 128 == 0` → multiples of `768` on TP3 (`3840` = `5 × 768`; `4096` falls back to non-row-sharded prefill).
- **450 W PL vs 600 W TDP:** At `450 W` PL, RTX PRO 6000 Blackwell cards throttle SM clocks to `~2.09 GHz` during dense `128K` prefill (`27,688 tok/s`) and `~2.24–2.30 GHz` during `C8/C16` decode (`1,733 tok/s` C16). At stock `600 W` TDP, the exact same package reaches **`30,553 / 31,564 tok/s`** prefill and **`1,825 tok/s`** `C16` decode.
- **PCIe `RO=0` vs `RO=1` (Zen 5 Note — upstream `#25`):** `BRRR_PUSH_V2=2`, `DCP`, `TwoShot`, and `pcie_dma` are verified sound under both `RO=0` and `RO=1`. Recommended on Zen 4 (`iommu=off`, x16): `NVreg_EnablePCIERelaxedOrderingMode=1` + `MRRS=4096 B`. If you hit a request hang on **Zen 5 (Threadripper PRO 9965WX)** with `iommu=pt` or mixed x4/x16 links (upstream `hyperflare-recipes#25`), set `iommu=off` or disable PCIe Relaxed Ordering (`NVreg_EnablePCIERelaxedOrderingMode=0` or `setpci -s <bus> CAP_EXP+8.w=0x0000:0x0010`).
- **Linux Kernels `< 6.7` (upstream `#481`):** Keep the default `FAST=1` (`FLARE_ENGRAM_PRIVATE_TABLES=1`), which uses anonymous `MAP_PRIVATE` memory and avoids the Linux `< 6.7` `F_SEAL_WRITE` + `MAP_SHARED` `EPERM` bug (do not pass `FAST=0` on kernels `< 6.7` like Rocky 9 / Ubuntu 22.04 GA / Debian 12).
- **Older `nvcc` (`STACK_LIMIT`):** Default `STACK_LIMIT=48` fits this build (`max kernel stack 32 B`). If an older CUDA toolkit emits a larger stack frame, set `STACK_LIMIT=64` (or `STACK_LIMIT=""`).
- **Warning — do NOT lock GPU memory clocks (`nvidia-smi -lmc`) under a `450 W` PL:** Locking `-lmc` under 450 W starves SM power headroom and regresses decode `ms/graph`; keep dynamic clocks unlocked (`nvidia-smi -rmc`).

---

## Licenses & Credits

- **DeepSeek-V4.1-Flash-1017 weights & FP8 Engram tables:** MIT ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
- **Hyperflare (`NVIDIA/hyperflare-inference`), `hyperflare-recipes`, `flashinfer`, and BRRR:** Apache-2.0 (see [licenses/](licenses/) and respective repository `LICENSE` / `NOTICE` files).
- **WARMIND v1.1 steering sidecar:** MIT ([jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`).
- **Credits:** NVIDIA Hyperflare & FlashInfer/b12x teams; Local Inference Lab (`flashinfer`, `hyperflare-recipes`, `lil-bench`, `llm-inference-bench`, `tool-eval-bench`); **Dooner / FujitsuPolycom** for upstream PCIe row-sharded prefill (`#23`, `#27`, `#139`, `#140`, `#499`) and Heliopause `RO=1` export fence (`#498`); **Jason Cook** for `#424` and `#426`; **Luke Alonso** for static-only collectives, `#510`, and `flare doctor`; **Fraser Price** for BRRR; **jcartu** for steering & WARMIND v1.1; **DeepSeek** for DeepSeek-V4.1-Flash-1017.
