# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261128 (Wave 15.1)

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, CLI (`flare run`, `flare models`), and cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000 Blackwell cards. This release supersedes [Hyperflare-tp3-20261125](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261125/README.md) (`gb202-tp3-202611251350`) and consolidates **Waves 8 through 15.1** across all three repositories — jumping **prefill from `27.7K` to `39.6K–40.9K tok/s` (+43–48% at 450 W)**, **C1 decode to `395–404 tok/s`** (`586–622 tok/s` avg / **`1,034–1,074 tok/s` MAX** in real-world coding), adding **zero-replication encoder attention (`-2.26 GiB/GPU`)**, **global cross-layer hot-expert allocation (`HOT=375` default / `HOT=355` 5.06M KV)**, **wide SAM copy-verify (`1x16,2x16,4x16`)**, **AMD Zen 4 `MONITORX`/`MWAITX` low-latency host waits**, **SHA-256 `CUmodule` cubin deduplication**, **`io_uring` fixed-file Engram disk refill + `#566/#70` graph-driven disk lookahead prefetch**, and the **v0.4.1 `DispatchRing` deadlock fix (`da752e07`) + `#568` prefix-cache eviction protection**.

For the first time, this release also ships a **prebuilt Linux x86_64 `flare` tarball (`flare-linux-amd64.tgz`) + `SHA256SUMS.txt`** matching upstream NVIDIA Hyperflare's release packaging format.

- **Runtime / Ignition (with prebuilt `flare-linux-amd64.tgz`):** [AzallazA/hyperflare-inference@v0.4.1-tp3-20261128](https://github.com/AzallazA/hyperflare-inference/releases/tag/v0.4.1-tp3-20261128) (`7eecfda6`, branch `tp3-20261128`)
- **Kernels (`b12x`):** [AzallazA/flashinfer@tp3-20261128](https://github.com/AzallazA/flashinfer/releases/tag/tp3-20261128) (`9b87317d`) — or apply [tp3-patches/b12x-tp3-20261128.patch](tp3-patches/b12x-tp3-20261128.patch) cleanly to `local-inference-lab/flashinfer @ 0d48af53`
- **Recipes / BRRR:** `local-inference-lab/hyperflare-recipes @ 39c157a4` + [tp3-patches/hyperflare-recipes-tp3-20261128.patch](tp3-patches/hyperflare-recipes-tp3-20261128.patch)

**Default package (`gb202-tp3-202611281539`) includes the WARMIND v1.1 steering sidecar** (`steering_scale=0` by default = unsteered model; `steering_scale` in `[0, 1]` runs inside the BRRR fused kernel at **`24 B` stack**; `STEERING_KEYS_FILE` optional).

**How to get it:** download the prebuilt `flare-linux-amd64.tgz` (or build from source), export the package from your checkpoint using [EXPORT.md](EXPORT.md), then serve with [serve.sh](serve.sh):
```bash
FLARE=/path/to/flare ./serve.sh
```

Exact tool versions, SHAs, and checksums are in [versions.txt](versions.txt); qualification summary is in [qualification/SUMMARY.md](qualification/SUMMARY.md).

---

## Headline Numbers

Measured on `3 × RTX PRO 6000 Blackwell Workstation Edition 96 GB` (SM120, PCIe Gen5 x16, **450 W PL**, headless, `MAXC=16`, `PREFILL=7680`, `ROW_SHARD_PREFILL=1`, `ROW_SHARD_ATTENTION=1`, `ZERO_REP_ATTENTION=1`, `MOE_RESERVE_SMS=4`, `GLOBAL_HOT_BUDGET=1`, `DRAFTS=5`, `BUDGET=64`, `ALPHA=1.0`, `--history-drafts`, `PREFIX_CACHE=12G`, `ENGRAM_RAM=1`, `STACK_LIMIT=48`).

### 1. Wave 15.1 (`HOT=375` Default) vs Wave 15 (`HOT=355`, 5.06M KV) vs `2026-11-25` (450 W PL)

| Metric (450 W PL) | Wave 15.1 `HOT=375` (`1539`, +800/2850 MHz) | Wave 15 `HOT=355` (`0459`, stock clocks) | Last Published (`2026-11-25`) | Gain vs `2026-11-25` |
|---|---:|---:|---:|---:|
| **KV Cache Capacity** | **2,432,000 tok** (`230` snaps) | **5,060,736 tok** (`478` snaps) | 5,883,648 tok (`556` snaps) | Choose `HOT=375` (speed) or `HOT=355` (5M KV) |
| **Resident Experts (L20–39)** | **7,500 / 7,680 (97.7%)** global | **7,100 / 7,680 (92.4%)** global | 7,100 / 7,680 uniform | Global cross-layer allocation cuts cold hits |
| **Max Kernel Stack / Prep** | **`32 B`** (`0.5s` prep, `4,042` deduped modules) | **`32 B`** (`0.5s` prep) | `32 B` (`0.6s` prep) | `255 MiB` deduped cubins |
| **Prefill 8K (`LDB`)** | **31,531 tok/s** (`0.26s`) | 30,560 tok/s (`0.27s`) | — | `prefill768` closes sub-3K cliff |
| **Prefill 32K (`LDB`)** | **38,089 tok/s** (`0.85s`) | 36,898 tok/s (`0.87s`) | 27,624 tok/s | **+37.9%** |
| **Prefill 64K (`LDB`)** | **39,978 tok/s** (`1.61s`, `40.6K` cool) | 38,498 tok/s (`1.67s`) | 28,008–28,453 tok/s | **+40.5% to +42.7%** |
| **Prefill 128K (`LDB`)** | **38,628 tok/s** (`3.33s`, `40.9K` cool) | 37,590 tok/s (`3.42s`) | 27,129–27,688 tok/s | **+39.5% to +47.6%** (`13.6K/GPU`) |
| **Prefill 256K (`LDB`)** | **39,619 tok/s** (`6.48s`) | 38,211 tok/s (`6.72s`) | — | Sustained ~40K @ 256K context |
| **Decode 32K `C1`** | **403.6 tok/s** (`155.7 v/s`, `6.41 ms/g`) | 390.9 tok/s (`155.7 v/s`) | 371.6 tok/s | **+8.6%** |
| **Decode 128K `C1`** | **395.3 tok/s** (`155.8 v/s`, `6.41 ms/g`) | 374.8 tok/s (`161.7 v/s`) | 363.6 tok/s | **+8.7%** |
| **Decode 32K `C4 / C6 / C8`** | **881.1 / 1,082.2 / 1,255.7 tok/s** | 864.6 / 1,070.5 / 1,274.9 tok/s | — / — / 1,242.5 tok/s | Up to **+2.6%** |
| **Decode 32K `C12 / C16`** | **1,529.5 / 1,738.3 tok/s** | 1,506.7 / 1,727.7 tok/s | — / 1,733.1 tok/s | Matches @ `PREFILL=7680` |
| **Decode 128K `C8 / C16`** | **1,229.4 / 1,723.4 tok/s** | 1,197.7 / 1,701.7 tok/s | 1,231.0 / 1,686.4 tok/s | **+2.2%** `128K C16` |
| **Real-World Coding (`tetristest.py`)** | **586–622 tok/s avg / 1,034–1,074 MAX** | 577.7 avg / 922.3 MAX | 613.2 avg / 768–789 MAX | **+35% peak burst** via `1x16,2x16,4x16` copy-verify |
| **4×2048 Greedy Coding (`temp=0`)** | **506.6 tok/s** mean (`6.890 ms/rd`) | 458.4 tok/s mean (`6.935 ms/rd`) | — | **+10.5%** from `HOT=375` + `+800/2850 MHz` |
| **SAM Verbatim Copy-Draft (`C1 / C4`)** | — (`1,034–1,074` burst in Tetris) | **1,044 C1 short / 917 C1 long / 1,373 C4** | — | Lossless `4×16` (`P=64`) copy-verify |

### 2. Disk-Backed Engram Mode (`#566 + #70` Lookahead Prefetch + `#561` `io_uring`, `ENGRAM_RAM=0`)

Measured on `gb202-tp3-202611281539` with both Engram tables on NVMe (`450 W`, `+800 / 2850 MHz`):

| Disk Mode Configuration | 64K Prefill | 128K Prefill | 256K Prefill | Notes |
|---|---:|---:|---:|---|
| **Lookahead Prefetch ON** (`16 GiB` cache) | **40,292 tok/s** | **40,552 tok/s** | **39,422 tok/s** | `0` prefetch errors; `40K` disk NIAH PASS (`QUARTZ-91847-COBALT`) |
| **Lookahead Prefetch OFF** (`--no-embedding-table-prefetch`, `16 GiB`) | 39,685 tok/s | 39,838 tok/s | 38,863 tok/s | ON is `+1.5–1.8%` faster even on warm NVMe |
| **Cold-Cache Verification (`4 GiB` cache)** | **39,300 tok/s** | **39,400 tok/s** | **38,400 tok/s** | `flare_engram_prefetch_rows_loaded_total = 1,448`, `0` errors |

### 3. Power Efficiency (`450 W/GPU` Sweet Spot)

At `450 W/GPU` (`~1,095–1,350 W` total GPU draw across 3 cards), Wave 15.1 TP3 delivers **`~0.37 tok/s/W`** at single-user `C1` decode, **`~0.55–0.58 tok/s/W`** on real-world coding (`tetristest.py`), **`~1.29 tok/s/W`** at `C16` decode, and **`~29.5 tok/s/W`** on sustained `64K–256K` prefill.

---

## What Changed Since `2026-11-25` (Waves 8 → 15.1)

### 1. Prefill Leap: `27.7K → 40.9K tok/s` (+43–48% at 450 W)
- **Full 64-Head Row-Sharded Encoder Attention (`ROW_SHARD_ATTENTION=1`) + `PREFILL=7680`:** Scales row-sharded encoder prefill from `3840` to **`7680`** with ladder `128,256,512,768,1536,3072,4608,6144,7680` (`prefill768` closes the sub-3K prefill cliff), `H24_PREFILL=1` one-launch 24-head prefill MLA with streaming `.cs` Q/O, and split sequential `ScratchPool` materialization so `PREFILL=7680` fits comfortably inside 96 GB GPUs.
- **Zero-Replication Encoder Attention (`ZERO_REP_ATTENTION=1`) + `MOE_RESERVE_SMS=4`:** Eliminates full replicated MXFP8 copies of encoder `wq_b`/`wo_a`/`wo_b` across layers `0–19`, keeping only `5 KiB` of FP32 `attn_sink` resident and streaming weight AllGathers in a 2-window overlapped schedule beside MoE execution with `_flare_instantiation_flags=9` high-priority streams and `4` reserved SMs (`MOE_RESERVE_SMS=4`). Reclaims **`~2.26 GiB/GPU`** while keeping full `40K tok/s` prefill speed.
- **Inline CSF Scales for `M64/M128 N128` `w4a8_mx` (`b12x`):** Eliminates separate CSF scale expansion on encoder prefill GEMMs and adds configurable dynamic `reserve_sms`.

### 2. Decode, Copy-Verify, and Expert-Residency Wins
- **Global Cross-Layer Hot-Expert Allocation (`GLOBAL_HOT_BUDGET=1`):** Allocates `20 × HOT` VRAM-resident expert slots across parked layers `20–39` globally by normalized route frequency (`1 <= hot_l <= 384`) instead of a flat per-layer count. Combined with **`ZERO_REP_ATTENTION=1`** VRAM savings, the default build raises residency to **`HOT=375`** (`7,500 / 7,680` = **97.7% of parked decoder experts in VRAM**, `2.43M` KV tokens), while **`HOT=355`** offers **`5.06M` KV tokens** (`478` snaps).
- **Compact Wide SAM Copy-Verify (`--history-drafts`, `COPY_VERIFY=1x16,2x16,4x16`):** Async background Suffix-Automaton (SAM) prompt/history copy-drafting without bumping `request.epoch`, paired with width-matched sampling AllGathers and fixed `#540` scheduler accounting (`age refresh`, `Pareto pruning`, `copy isolation`, `debt cap`), pushing real-world coding peak speeds past **`1,034–1,074 tok/s`**.
- **Lossless Decode Kernel & BRRR Upgrades:**
  - `b12x` PR `#148` two-hop RSAG fused Lamport mHC all-reduce (`M <= 128`) + PR `#149` lossless decode GEMM/WO/KV-RoPE/compact MoE opts + `B12X_PCIE_ONESHOT_NANOSLEEP_NS=0` + lazy `PCIeTwoShotBF16` allocation + shared per-device split-K `tc_scratch` + `64 KiB` one-shot rank data buffer + `PR #37/#63/#67` determinism/NaN/memo fixes.
  - Tuned BRRR compile-time knobs (`BRRR_R1_M_THRESH=17`, `BRRR_PARKED_HOT_R1P=2`, `BRRR_SPIN_NS=32`), adaptive draft `r1_parts`, consolidated `CoreStorage` scratch, `H24DecodeKernel` cold I-cache `.text` shrink, and dense multi-query MXFP4 request-mapping gated to `P >= 16` (Layer 20 at `P >= 2`).

### 3. Upstream Hyperflare Runtime & Storage Ports (`v0.4.1-tp3-20261128`)
- **`v0.4.1` `da752e07` (`DispatchRing` Deadlock Fix):** Commit-before-launch ordering fix and stronger TP verification, preserving AMD Zen 4 `MONITORX`/`MWAITX` (`host_wait` & Engram) and `cleared_pages`.
- **`#568` (`5deacda9`):** Age-discounted marginal gain eviction protection for optional `Endpoint` prefix-cache checkpoints.
- **`#566` (`7745deea`) + `hyperflare-recipes#70` (`4a5346e`):** Graph-driven advisory lookahead prefetch for disk-backed Engram tables, preserving `#561` continuous `io_uring` fixed-file (`COOP_TASKRUN`) disk lane refill.
- **Additional Upstream Ports:** `#542` lock-free `DispatchRing` + `PreparedAllToAll`, `#556` TLA+ `DispatchRing` fixes, `#504`/`#523`–`#527` pure-CUDA GPU/CPU lossless KV prefix-cache compression (`Latent576` page-split), `#539` Engram FD lease preflight & compact rank-local staging, `#540` scheduler cost model + 4 bug fixes, `#546` collective prep deadlock detector, `#559` deferred cyclic GC after capture retirement, `#484` `state_transfer._copy_rows` `gridDim.y` stride fix, and **SHA-256 `CUmodule` cubin deduplication** at export (`6,934` kernels in `4,042` modules, `255.10 MiB` cubins).

---

## Qualification Summary

All lossless correctness, long-context, disk-lookahead, vision, steering, and thermal soak gates **PASSED** on **`gb202-tp3-202611281539`** (`v0.4.1-tp3-20261128` binary SHA-256 `70d03647d8a9...`; see [qualification/SUMMARY.md](qualification/SUMMARY.md)):

- **Rust & Python Unit Suites (`flare-core` + `hyperflare-recipes`):** **PASS** — `execution::threaded (21)`, `engram (58)`, `cache (129)`, `runtime_core (195)`, `core (498)`, `model (62)`, `mwaitx (2)`, `heliopause (216)`, `flare-cli (225)`, Ignition `456/456` pytest, DS41 recipes `98/98` pytest (`_supports_advisory_prefetch() == True`).
- **Greedy Corruption Check (`verify_fix.py 10`):** **PASS** — **`0` TRUE corruptions** (`1,617` tok on Wave 15.1 `1539`; `0` TRUE pre-soak `1,483` tok and post-soak `1,631` tok across `307s` OC soak).
- **Prefill vs Decode Consistency (`consistency.py 48 4`):** **PASS** — **`0` miss** both `--no-history-drafts` (`top-1 0.969`, med `0.0042`) and `--history-drafts` (`top-1 0.964`, med `0.0020`, `max 0.8702`).
- **6-Case Long-Context NIAH (up to `270K`):** **PASS** — **`6/6` verbatim** (`1.8K`, `5.2K`, `9.7K`, `64K`, `125K`, `270K`).
- **Images / Vision Smoke (`5` cases):** **PASS** — **`5/5`** (`Red`, `Blue`, `Circle`, `Square`, `stripes`).
- **WARMIND v1.1 Steering Verification:** **PASS** — `scale=0` unsteered vs `scale=1` verified shift.
- **Disk Lookahead Prefetch (`#566 + #70`) & NVMe NIAH:** **PASS** — `flare_engram_prefetch_rows_loaded_total = 1,448`, `0` errors, `40K` disk NIAH verbatim (`QUARTZ-91847-COBALT`).
- **GPU Thermal & Clock Soak (`450 W`, `+800 / 2850 MHz`, `307s`):** **PASS** — **`0` Xids**, `0.0%` SW/HW thermal throttle, pre & post `verify_fix 10` = `0` TRUE.
- **Clean-Checkout Patch & Packaging Hygiene:** **PASS** — `hyperflare-recipes-tp3-20261128.patch` & `b12x-tp3-20261128.patch` pass `git apply --check` on fresh upstream `39c157a4` & `0d48af53`; `flare-linux-amd64.tgz` built via upstream `packaging/archive.py` and smoke-tested cleanly.

---

## Troubleshooting & Host Notes

- **`HOT=375` vs `HOT=355`:** `HOT=375` (`2.43M` KV tokens) is the default for **headless** 96 GB GPUs (`1,159 MiB` free before tuning; `2.30 GiB/GPU` KV budget). If GPU 0 also drives a desktop compositor (`gdm`/Xorg/Wayland) or you want **`5.06M` KV tokens** (`478` shared snapshots), export with `HOT=355` (the only change needed).
- **`PREFILL=7680` on TP3:** Row-sharded encoder prefill requires `(PREFILL / 2 / world_size) % 128 == 0` → multiples of `768` on TP3 (`7680` = `10 × 768`). `export.sh` automatically captures the `128,256,512,768,1536,3072,4608,6144,7680` ladder with `ROW_SHARD_ATTENTION=1` and `ZERO_REP_ATTENTION=1`.
- **Optional 450 W Pseudo-Undervolt (`nvoc`):** On `3 × RTX PRO 6000 Blackwell Workstation Edition` at `450 W` PL, applying `nvoc -d all -c 210,2850 -o 800 -m 6000` lifts sustained `450 W` prefill clocks by `+150–250 MHz` (`38.5K → 40.0K–40.9K tok/s`) while capping peak boost at `2850 MHz` for stability (`0` Xids across full 307s soak; our GPU 0 hit an Xid 109/154 stability wall at `+950 / 2950 MHz`, so leave generous margin). Do **not** lock `-lmc` under a `450 W` PL.
  - *Systemd boot-ordering tip:* If you persist power limit (`nvidia-pl.service`) and clocks (`nvoc-oc.service`) via systemd, make sure `nvidia-pl.service` uses `After=nvidia-persistenced.service` (NOT `After=multi-user.target` while also `WantedBy=multi-user.target`, which creates an ordering cycle that causes systemd to drop `nvoc-oc.service` on boot).
- **AMD Zen 4 `MONITORX`/`MWAITX`:** Detected automatically via CPUID on Zen 4/5 hosts (`[flare] engram request wait: MWAITX`, `[flare] frame completion wait: MWAITX`), cutting spin-wait heat and jitter with no configuration required.
- **Disk Mode (`ENGRAM_RAM=0` or `ENGRAM_RAM=layer1`):** `serve.sh` leaves `FLARE_ENGRAM_READ_AHEAD` unset by default so upstream `#566 + #70` graph-driven advisory lookahead prefetch runs automatically. Set `ENGRAM_PREFETCH=0` to pass `--no-embedding-table-prefetch`, or `READ_AHEAD=1` to use the legacy CPU pre-chunk reader instead.
- **PCIe `RO=0` vs `RO=1` (Zen 5 Note — upstream `#25`):** Recommended on Zen 4 (`iommu=off`, x16): `NVreg_EnablePCIERelaxedOrderingMode=1` + `MRRS=4096 B`. On Zen 5 with `iommu=pt` or mixed x4/x16 links, use `iommu=off` or `NVreg_EnablePCIERelaxedOrderingMode=0`.
- **Linux Kernels `< 6.7` (`#481`):** Keep `FAST=1` (`FLARE_ENGRAM_PRIVATE_TABLES=1`) so anonymous `MAP_PRIVATE` avoids the Linux `< 6.7` `F_SEAL_WRITE` + `MAP_SHARED` `EPERM` bug.
- **Older `nvcc` (`STACK_LIMIT`):** Default `STACK_LIMIT=48` fits this build (`largest kernel stack 32 B`). Set `STACK_LIMIT=64` (or `STACK_LIMIT=""`) if an older toolkit emits larger stack frames.

---

## Licenses & Credits

- **DeepSeek-V4.1-Flash-1017 weights & FP8 Engram tables:** MIT ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
- **Hyperflare (`NVIDIA/hyperflare-inference`), `hyperflare-recipes`, `flashinfer`, and BRRR:** Apache-2.0 (see [licenses/](licenses/) and the third-party license notices bundled inside `flare-linux-amd64.tgz`).
- **WARMIND v1.1 steering sidecar:** MIT ([jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`).
- **Credits:** NVIDIA Hyperflare & FlashInfer/b12x teams; Local Inference Lab (`flashinfer`, `hyperflare-recipes`, `lil-bench`, `llm-inference-bench`, `tool-eval-bench`); upstream contributors for `#23`/`#27`/`#46`/`#47`/`#49`/`#50`/`#70`, `#37`/`#63`/`#67`/`#138`–`#140`/`#147`–`#149`, and `#484`/`#504`/`#520`/`#523`–`#527`/`#539`/`#540`/`#542`/`#546`/`#556`/`#559`/`#561`/`#566`/`#568`/`da752e07`; Fraser Price for BRRR; jcartu for steering & WARMIND v1.1; DeepSeek for DeepSeek-V4.1-Flash-1017.
