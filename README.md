#  Native TP3

Inference recipes for 3x RTX PRO 6000 Blackwell (TP3) with vLLM and NVIDIA Hyperflare (formerly Flare). Each folder is one build:
image or export recipe, launch files, measurements and the source changes.

All builds run on 3x RTX PRO 6000 Blackwell 96 GB (PCIe), tensor parallel 3 unless a row says otherwise. Power is the
per-GPU limit the numbers were measured at. Newest first in each table.

## NVIDIA Flare / Hyperflare: DeepSeek-V4.1-Flash

| Release | Date | Hardware / TP | Key numbers | Notes | Get it |
| --- | --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash Hyperflare-tp2-exl3-20261117](DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261117/README.md) | 2026-11-17 | **TP2 on 2 GPUs**, 450 W | 8 slots, 2.14M KV; 32K c1/c4/c8 243/553/757 tok/s, 128K c1/c8 ~240/743; 32K prefill ~11.3K tok/s | diffbot's EXL3 3-bit expert pack on NVIDIA Hyperflare (binary still `flare`), fork tag `hyperflare-tp2-exl3-20261117`: ~85% of experts in VRAM, rest in host RAM; Engram tables in RAM (~151 GiB server RAM) or on disk (one table: ~110 GiB, no speed cost; both: ~64 GiB); minimum qualification (hotel 30/30, 8-stream stress PASS, GPQA 92.4%); guides for KV vs speed and for an expert order from your own traffic | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261117/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Hyperflare-tp3-20261116](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md) | 2026-11-16 | TP3, 450 W | 16 slots, 5.40M KV (cap 32, 12G prefix cache); 32K c1/c8/c16 320/979/1,332 tok/s (single 30 s passes; c1/c8 level with 20261111); prefill 32K 15.8K tok/s; 8-stream agent replay first token p50 0.42 s | DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare (binary still `flare`), fork tag `hyperflare-tp3-20261116`: new upstream base `70069281`, DeepSeek's FP8 Engram tables in RAM (256 GB host RAM minimum, ~217 GiB committed), round-2 export switches, prefill merge and shortest-first TTFT guard; minimum qualification (no LiveBench / long generation), 16-stream KV stress PASS | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Hyperflare-tp3-20261111](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md) | 2026-11-11 | TP3, 450 W | 3.01M KV (2.47M before graph scratch, same settings); 32K c1/c4/c8 ~301/734/988 tok/s; first token 0.36-0.38 s instead of 2-4 s while 3-7 streams decode (TTFT guard); LiveBench coding 83.5 | DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare (formerly Flare; binary still `flare`), fork tag `hyperflare-tp3-20261111`: graph scratch, TTFT guard (`--scheduler-policy interactive`); same settings; full c8 qualification | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261108](DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md) | 2026-11-08 | TP3, 450 W | 2.46M KV; 32K c1/c4/c8 ~306/719/973 tok/s; 0.11 s per image instead of 1.70 s | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare, fork tag `flare-tp3-20261108`: 20261107 plus BRRR kernel hygiene, vision weight staging, Engram per-table storage, `flare run --dry-run`; same settings | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261108/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261107](DeepSeek-V4.1-Flash%20Flare-tp3-20261107/README.md) | 2026-11-07 | TP3, 450 W | 1M context, 8 slots, 2.37M KV (6.25M variant); 32K c1/c4/c8 ~305/738/975 tok/s | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare, rebased on upstream, BRRR fused layer tail; Engram in RAM or on disk. Supersedes 20261105 | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261107/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261105](DeepSeek-V4.1-Flash%20Flare-tp3-20261105/README.md) | 2026-11-05 | TP3, 350 W | 1M context, 8 slots, 2.99M KV | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare (original weights + NVIDIA MXFP4 Engram tables in RAM) | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261105/EXPORT.md)) |

## NVIDIA Flare / Hyperflare: GLM-5.3-Flash

| Release | Date | Hardware / TP | Key numbers | Notes | Get it |
| --- | --- | --- | --- | --- | --- |
| [GLM-5.3-Flash Hyperflare-tp3-20261116](GLM-5.3-Flash%20Hyperflare-tp3-20261116/README.md) | 2026-11-16 | TP3 on 3 GPUs (DCP3) and TP2 on 2 GPUs (DCP2), 450 W | TP3: 32 slots, 14.93M KV, 32K c1/c8/c32 290/879/1828 tok/s (one 30 s pass); TP2: 8 slots, 3.11M KV, c1/c8 245/750 tok/s | GLM-5.3-Flash (NVIDIA FP4-Local) on Hyperflare 70069281, fork tag `hyperflare-glm-20261116`: export-time VRAM reclaim (+14% KV TP3, +40% TP2 vs 20261112), text + images; tier "export" qualification 2026-11-17 (eval-43 43/43 TP3, 42/43 TP2; retrieval 128K/900K pass; TP2 8-stream KV stress PASS; TP3 32-stream KV stress not run) | export it yourself ([EXPORT.md](GLM-5.3-Flash%20Hyperflare-tp3-20261116/EXPORT.md)) |
| [GLM-5.3-Flash Hyperflare-tp3-20261112](GLM-5.3-Flash%20Hyperflare-tp3-20261112/README.md) | 2026-11-12 | TP3 on 3 GPUs (DCP3) and TP2 on 2 GPUs (DCP2), 450 W | TP3: 32 slots, 13.13M KV, 32K c1/c4/c8/c16/c32 279/630/866/1290/1790 tok/s, prefill ~11.7K; TP2: 8 slots, 2.23M KV, 248/512/727 tok/s | GLM-5.3-Flash (NVIDIA FP4-Local) on NVIDIA Hyperflare (binary still `flare`), fork tag `hyperflare-glm-20261112`: text + images; TTFT guard available, off by default | export it yourself ([EXPORT.md](GLM-5.3-Flash%20Hyperflare-tp3-20261112/EXPORT.md)) |

## vLLM: DeepSeek-V4.1-Flash

| Release | Date | Hardware / TP | Key numbers | Notes | Get it |
| --- | --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash Kraken-tp3-20261019](DeepSeek-V4.1-Flash%20Kraken-tp3-20261019/README.md) | 2026-10-19, updated to `.12.2` on 2026-11-14 | TP3, 450 W | 1M context, 8 slots, KV 2,185,581; 32K c1/c4/c8 290/610/995 tok/s (`.12.1`); `.12.2`: agent 8-stream TTFT p50 3.93 → 1.25 s | DeepSeek-V4.1-Flash-1017 (original weights + NVIDIA 4-bit Engram tables) on Local Inference Lab's vLLM "karmic-kraken" image; `.12.2` = prefill compute share 0.6 by default | `azallaza/ds41-kraken-tp3:20261019.12.2` |

## vLLM: GLM-5.3-Flash

| Release | Date | Hardware / TP | Key numbers | Notes | Get it |
| --- | --- | --- | --- | --- | --- |
| [GLM-5.3-Flash Kraken-tp3-CSF](GLM-5.3-Flash%20Kraken-tp3-CSF/README.md) | 2026-10-15 | TP3, 350 W | 32 slots; KV 3.96x 1M (MTP3/DCP1), 10.0x 1M (DCP3) | GLM-5.3-Flash NVFP4-MXFP8 CSF-QAD | `azallaza/glm53-kraken-tp3:20261015` |
| [GLM-5.3-Flash Kraken-tp3-20260927](GLM-5.3-Flash%20Kraken-tp3-20260927/README.md) | 2026-09-27 | TP3, 350 W | 8 slots, MTP3, DCP1 | GLM-5.3-Flash | `azallaza/glm53-kraken-tp3:20260927` |
| [GLM-5.3-Flash Kraken tp3 experimental](GLM-5.3-Flash%20Kraken%20tp3%20experimental/glm-5.3-flash-tp3-kraken.md) | 2026-09-23 | TP3, 350 W | against 20260921: C1 decode +11.4% (88.2 verifier steps/s), C8 +8.1% (283), prefill +5.5-6.5% | GLM-5.3-Flash (experiments) | see folder |
| [GLM-5.3-Flash Kraken-tp3](GLM-5.3-Flash%20Kraken-tp3/glm-5.3-flash-tp3-kraken.md) | 2026-09-21 | TP3, 350 W | 8 slots; KV 3,410,670 tokens (MTP3/DCP1) | GLM-5.3-Flash | see folder |
| [GLM-5.3-Flash Jovian r34-v3](GLM-5.3-Flash%20Jovian%20r34-v3/glm-5.3-flash-tp3-r34.md) | 2026-09-16 | TP3, 350 W | 8 slots; KV 2,091,238 tokens | GLM-5.3-Flash on the Jovian image | see folder |
| [GLM-5.3-Flash Jovian r34-v2](GLM-5.3-Flash%20Jovian%20r34-v2/glm-5.3-flash-tp3-r34.md) | 2026-09-15 | TP3, 350 W | 8 slots; KV 2,091,238 tokens | GLM-5.3-Flash on the Jovian image | see folder |
| [GLM-5.3-Flash Jovian r34](GLM-5.3-Flash%20Jovian%20r34/glm-5.3-flash-tp3-r34.md) | 2026-09-10 | TP3, 300 W | 8 slots; KV 2,091,238 (DCP1) / 7,510,219 (DCP3) tokens | GLM-5.3-Flash on the Jovian image | see folder |

## Tools

- [tools/sqlbench](tools/sqlbench/README.md): local replay of Nick Lothian's SQL benchmark (same requests, tool loop and grader as the site, fetched at run time) with per-answer truncation / tool-call / SQL-error accounting. Used for the DeepSeek V4.1 Flash TP3 investigation in [issue 1](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/issues/1).
