#  Native TP3

Inference recipes for 3x RTX PRO 6000 Blackwell (TP3) with vLLM and NVIDIA Hyperflare (formerly Flare). Each folder is one build:
image or export recipe, launch files, measurements and the source changes.

| Folder | Model | Get it |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash Hyperflare-tp3-20261111](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/README.md) | DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare (formerly Flare; binary still `flare`), fork tag `hyperflare-tp3-20261111`: graph scratch (KV 2.47M -> 3.01M at the same settings, decode level), TTFT guard (`--scheduler-policy interactive`: first token 0.36-0.38 s instead of 2-4 s while 3-7 streams decode); same settings, 3.01M KV, 32K c1/c4/c8 ~301/734/988 tok/s at 450 W, full c8 qualification (LiveBench coding 83.5) | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261111/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261108](DeepSeek-V4.1-Flash%20Flare-tp3-20261108/README.md) | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare, fork tag `flare-tp3-20261108`: 20261107 plus BRRR kernel hygiene, vision weight staging (0.11 s per image instead of 1.70 s), Engram per-table storage, `flare run --dry-run`; same settings, 2.46M KV, 32K c1/c4/c8 ~306/719/973 tok/s at 450 W | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261108/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261107](DeepSeek-V4.1-Flash%20Flare-tp3-20261107/README.md) | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare, rebased on upstream, BRRR fused layer tail; 1M context, 8 slots, 2.37M KV (6.25M variant), Engram in RAM or on disk. Supersedes 20261105: 32K c1/c4/c8 ~305/738/975 tok/s at 450 W | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261107/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Flare-tp3-20261105](DeepSeek-V4.1-Flash%20Flare-tp3-20261105/README.md) | DeepSeek-V4.1-Flash-1017 on NVIDIA Flare (original weights + NVIDIA MXFP4 Engram tables in RAM), 1M context, 8 slots, 2.99M KV | export it yourself ([EXPORT.md](DeepSeek-V4.1-Flash%20Flare-tp3-20261105/EXPORT.md)) |
| [DeepSeek-V4.1-Flash Kraken-tp3-20261019](DeepSeek-V4.1-Flash%20Kraken-tp3-20261019/README.md) | DeepSeek-V4.1-Flash-1017 (original weights + NVIDIA 4-bit Engram tables), 1M context, 8 slots | `azallaza/ds41-kraken-tp3:20261019.11` |
| [GLM-5.3-Flash Kraken-tp3-CSF](GLM-5.3-Flash%20Kraken-tp3-CSF/README.md) | GLM-5.3-Flash NVFP4-MXFP8 CSF-QAD | `azallaza/glm53-kraken-tp3:20261015` |
| [GLM-5.3-Flash Kraken-tp3-20260927](GLM-5.3-Flash%20Kraken-tp3-20260927/README.md) | GLM-5.3-Flash | `azallaza/glm53-kraken-tp3:20260927` |
| [GLM-5.3-Flash Kraken-tp3](GLM-5.3-Flash%20Kraken-tp3/glm-5.3-flash-tp3-kraken.md) | GLM-5.3-Flash | see folder |
| [GLM-5.3-Flash Kraken tp3 experimental](GLM-5.3-Flash%20Kraken%20tp3%20experimental/glm-5.3-flash-tp3-kraken.md) | GLM-5.3-Flash (experiments) | see folder |
| GLM-5.3-Flash Jovian r34, r34-v2, r34-v3 | GLM-5.3-Flash on the Jovian image | see folders |

## Tools

- [tools/sqlbench](tools/sqlbench/README.md): local replay of Nick Lothian's SQL benchmark (same requests, tool loop and grader as the site, fetched at run time) with per-answer truncation / tool-call / SQL-error accounting. Used for the DeepSeek V4.1 Flash TP3 investigation in [issue 1](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/issues/1).
