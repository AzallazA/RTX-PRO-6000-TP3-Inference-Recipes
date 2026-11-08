# DeepSeek V4.1 Flash on vLLM Kraken TP3: future plan (speed first, then VRAM, then fixes)

State on 2026-11-08 (image `20261019.11`, local-only `.12` = `.11` + the b12x `tiny_decode` SwiGLU clamp): base
`karmic-kraken-beta-20261016.2` = Local Inference Lab vLLM `6d66eed35` (fork merge-base with upstream `0f8fa53`,
2026-09-16), FlashInfer b12x at `5b065abe` (2026-10-18), TP3, 8 slots, DSpark 5 drafts, cap-only adaptive verification
`DS41_VERIFY_ROW_CAP=24`, 40 BRRR fused layers, sparse decode on, `BRRR_RADIX_TOPK=blocks`, parked layers 20-33 with
VMM promotion (19/24 + 20/24 chunks at 2x), both MXFP4 Engram tables in RAM. 32K at 350 W: c1 97-100 steps/s
(229-265 tok/s), c4 467-494, c8 808-820 tok/s; 128K/512K c1 96/90 steps/s; prefill 14.5-15.0K tok/s; eval 42/43,
LiveBench 81.9-83.2, c8 KV stress PASS. 2026-11-08: SQL tool-calling benchmark 16-21/25 with reasoning on vs 22 stable
with reasoning off (behavioural, not numerical). Everything below is unmeasured until stated; TP3-GUIDELINES D.1 applies
(alternating A/B in one session, 3+ rounds, steps/s and tokens/step next to tok/s, trace estimates are upper bounds).

## Where the time goes (measured, `.11` torch profiler, rank 0, 2026-11-03)

- **c1 step ≈ 10-11 ms, GPU idle < 1 %.** Fused layer kernel 69-75 % (163-169 µs per resident layer: all-reduces, mHC,
  router, experts inside); drafter + target LM head between steps 1.1 ms (10 %: draft MoE 3 x 51 µs, 22 dense calls
  0.51 ms, BF16 LM head 228 µs); attention side ≈ 1 ms; 128K adds only 0.2 ms over 32K.
- **c4/c8 share one cost** (24 verify rows under the cap): fused kernel 81 % (345 µs resident, 413 avg); drafter 2.0 ms
  (9 %). At m = 24 a layer reads ~100 distinct experts (~5.9 MB each): 410 µs with DRAM at 76 % of peak over the whole
  call = the weight-read floor. Only fewer bytes (placement, routing, rows) help above c4.
- **Parked layers 20-33** cost +1.57 ms/step (14 %) at c1 and +3.0 ms (14 %) at c4/c8 on a coding prompt, ~0 on the
  32K/128K bench prompts: cold-expert cost is traffic-dependent (blend order from 10-30 traffic; real traffic re-simulated
  11-07 at 300 resident experts: 1.84 % cold routes, oracle 1.23 %).
- **Speculation**: 2.6-2.85 tokens/step at c1 (position acceptance 70/44/26/15/8 %); the cap trims only above 4 decoders.
- **Memory**: KV 2.25 GiB = 2,185,581 tokens (1 GiB ≈ 0.97M tokens); each KV preset step comes out of promotion
  (2x → 4x: c1 -3.5 %, c8 -8 %); reserve 1.5 GiB, min free 490-524 MiB under c8 stress; host ~150 GB used.

## Speed ideas, ranked

| # | Idea | Why | Expected | Effort | How to measure |
| --- | --- | --- | --- | --- | --- |
| 1 | **Draft acceptance at c1** (same lever as the Flare plan): per-position study from the serve log, draft sampling vs target temperature 1.0, upstream drafter work (#64442 inverse-CDF Markov sampling, #64476 Gumbel fusion, #63063 stacked WKV) | 2.6 tokens/step is the whole c1 story; +0.5 tokens/step ≈ +19 % | +2-4 % from drafter kernels alone; +10-20 % if acceptance moves | medium-large (upstream code is under `models/deepseek_v41/nvidia/`, the fork's is `deepseek_v4_1/`: a port, and it must coexist with BRRR's vocab-split sampling) | A/B per change, tokens/step next to steps/s |
| 2 | **Row cap scaled by concurrency** for the 12/16-slot overlays: cap 24 up to 8 decoders, 32 above (`compose.rowcapscale.yaml` dev overlay, `DS41_VERIFY_ROW_CAP_HIGH/REQS`) | measured 11-03: c12 1016 / c16 1106 tok/s vs 957 / 1009 at cap 24; c1-c8 untouched by construction | +6-10 % at c12-c16 | small; c16 KV stress before any default | seqs16 overlay A/B, 3 rounds |
| 3 | **FlashInfer b12x pin bump** (`5b065abe` → head, 2026-11-07/08 batch: native MXFP4 MoE decode, packed compressed-MLA decode, SM12x dense decode + split-K through 16 rows, FC1 CSF metadata prefetch, V4.1 window snapshots fused with cache appends, route-reuse autotuning) | fused layers cover decode MoE, so this touches prefill, the drafter MoE, wide dense rows and attention | prefill +2-5 %, c1 ≤ 2 % | medium: Dockerfile `ARG`, re-apply the `tiny_decode` patch, re-check the b12x API used by `src/` (`b12x.py`, `uva_p4.py`, `expert_residency.py`, Engram patches); cold tuning cache on first boot | prefill 32K/128K + c1/c8 A/B; `verify_fix` |
| 4 | **Adaptive-verification manager overhead** (~2 %: confidence D2H copy + TP broadcast every step, also at c1-c4 where the cap never trims): skip the copy while total rows ≤ cap, or rebase the cap patch onto #63421 (Triton budget kernel, host 76-90 → 15 µs) + #63819 (JIT warmup gaps) | pure overhead at c1-c4 | +1-2 % c1-c4 | small-medium | c1/c4 A/B steps/s |
| 5 | **Expert-order refresh from a week of real traffic** (procedure in README-tuning; un-permute slot-space counts) | +1.57 ms/step parked excess on coding prompts; 1.84 → 1.23 % cold is the ceiling | +1-3 % c1 coding, 0 on prose | small | `B12X_EXPERT_ROUTES_ALL=1`, re-blend, A/B on coding + prose |
| 6 | **Power 450 W for prefill** (decode flat on this image: clock-pinned ~2820 MHz) | measured 10-24: 32K prefill 14.0K → 15.4K tok/s | prefill +10 % | none (host setting) | record the power with every README number (D.4.5) |
| 7 | **Drafter fusion** (3 draft layers through the fused kernel, LM head left as is) | drafter is 1.1 ms at c1 | ≤ 2-3 % (re-estimated 11-03) | large | trace the inter-step window first |

Not a speed lever here: #63956 token-sharded sparse-MLA prefill (gain "small and within noise" at 4K steps; our chunk
is 4096), #61868/#63921 `latency_specialized` decode (GB300 CuTe DSL kernels, not SM120).

## KV/VRAM ideas (1 GiB ≈ 0.97M KV tokens or ~3 promoted expert chunks)

| # | Idea | Saves per GPU | Cost / risk | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Reserve 1.5 → 1.25 GiB after taking #63047** (budgets the indexer `expanded_block_table_buffer` and two more prefill allocations the startup profile misses; the 768 MiB OOM on a 31K prompt and the 1 GiB OOM-mapping retries are this gap) | ~256 MiB → promotion | stress must stay clean (min free ≥ ~480 MiB under 4 x 600K) | medium (model + indexer port) |
| 2 | **Embedding + vision tower in pinned host RAM** (`compose.hosttables.yaml` dev overlay, `DS41_EMBED_HOST/VISION_HOST`) | ~1.2 GiB → promotion | ~3 s per image request; needs the image qualification set | small (exists) |
| 3 | **Draft-layer KV dtype**: GLM-5.3 saved 0.96-5.4 GB per rank with FP8 draft KV (#63841, #65553); check whether the V4.1 DSpark layers keep their own group and dtype (#63795 adds windowed draft groups) | unknown until checked | none if the group exists | small to check |
| 4 | **Engram host tables shared across the three workers** (#62359 does it across DP ranks with host-NUMA cuMem): if the TP ranks' tables overlap, one copy frees tens of GB of host RAM, enough for lossless FP8 tables in RAM (2 x 31.5 GiB today) or a 16 GiB prefix tier | host RAM, not VRAM | the LMCache tier also needed ~80 GB regardless of its size | medium-large; verify the sharding first |
| 5 | **KV presets** 2x/4x/6x/8x already trade promotion for KV (c1 100 → 82.5 steps/s); offer, do not optimise | user choice | speed | none |

Not worth it: FP8/MXFP8 LM head (freed ~1 GiB headroom, no speed, logit precision), DCP3 (89 µs/layer at c1 over PCIe),
lossless CSF-style shrink of experts (≤ 2.6 %), NVFP4 experts for VRAM (same bytes, 73.8 vs 84 steps/s).

## Upstream changes worth taking (merged after the base; the image is a Dockerfile over a pinned base)

Taking anything means either a new base image (none exists: `karmic-kraken-beta-20261020` is the same vLLM `6d66eed35`;
the fork's own merged PRs since 10-15 are GLM KDA, CSF loader and docs; its `main` is stale) or cherry-picked `src/`
overlays. Python-only frontend files overlay cleanly; model-side PRs are ports (path and b12x divergence).

| PR | What | Claimed effect | Applies |
| --- | --- | --- | --- |
| [#59468](https://github.com/vllm-project/vllm/pull/59468) (10-15) | DFlash/DSpark draft slots no longer charged to the token budget; admission capped at `max_num_batched_tokens // num_lookahead_tokens` | 52 → 64 running requests at a 512-token budget | **no, already in effect**: the fork's #605 (= upstream #52996 + DFlash) already sets `draft_slots = 0` for DSpark and bounds admission with a separate draft budget (`scheduler.py:858-870`, `:1278-1280`, `:1358-1363` in the image); at 4096 tokens and K = 5 the cap is 819 requests, 8-16 slots use 48-96 tokens per step |
| [#62199](https://github.com/vllm-project/vllm/pull/62199) / [#62200](https://github.com/vllm-project/vllm/pull/62200) (10-19) | V4.1 encoder matches the 1017 reference: tool results alone no longer open a user turn, reasoning before a tool call is kept | correctness | **yes**, first suspect for the reasoning-on tool-calling drop; the image carries BRRR's tool-message reorder patch on the same two files: reconcile |
| [#62517](https://github.com/vllm-project/vllm/pull/62517) (11-06) | parser accepts `<｜DSML｜ tool_calls>` as an opener instead of leaking it into content | correctness | yes, one file (`parser/deepseek_v41.py`) |
| [#62729](https://github.com/vllm-project/vllm/pull/62729) (11-04), [#64971](https://github.com/vllm-project/vllm/pull/64971) (11-02) | `add_generation_prompt`/`continue_final_message` honoured; `tool_reference` results on V4.1 and `/v1/messages` | API correctness (Claude Code tool search) | yes, overlay |
| [#64442](https://github.com/vllm-project/vllm/pull/64442) (11-03), [#64476](https://github.com/vllm-project/vllm/pull/64476) (11-07) | fused Markov draft sampling (opt-in inverse CDF; Gumbel bias + combine fusion) | sampling loop 1.6x at 1 request, -42-50 % device time per Markov step; step -1.7..-3.2 % at 64-128 requests (GB200) | maybe: conflicts with BRRR vocab-split sampling; speed idea 1 |
| [#63063](https://github.com/vllm-project/vllm/pull/63063) (10-27) | one stacked KV-only context projection for the V4.1 drafter | drafter dense time | maybe (port) |
| [#63421](https://github.com/vllm-project/vllm/pull/63421) (10-26), [#63819](https://github.com/vllm-project/vllm/pull/63819) (10-28) | Triton AV budget kernel; DFlash/DSpark/AV kernels JIT at warmup | host 76-90 → 15 µs per trimmed step; no mid-serving compiles | maybe: rebase the cap-only patch onto them (speed idea 4) |
| [#63047](https://github.com/vllm-project/vllm/pull/63047) (10-29) | budget the prefill allocations the startup profile misses | no prefill OOM at 32K budgets | maybe (VRAM idea 1) |
| [#60762](https://github.com/vllm-project/vllm/pull/60762) (11-05) | SM120 sparse backends declare 64-token blocks; validated with FP8 KV + FP8 indexer KV on 8 x RTX PRO 6000 | enables `--block-size 64` + FP8 caches on SM120 | maybe: compare with the live KV/indexer dtypes before assuming a gain; not verified here |
| FlashInfer b12x `5b065abe` → head (`e1c655d4`, `8adb4d4b`, `deea4e4a`, `ce1e93b0`, `6aa22fd6`, `18af302a`, `e16f9495`, `3294fd7c`) | MoE/MLA/dense decode kernels, CSF prefetch, window snapshots, route-reuse autotuning, io_uring bounce slots (the disk-Engram path behind Flare issues 78/230) | unquantified for TP3 | maybe (speed idea 3); route reuse was level on Flare at 450 W |

Base-rebase notes: upstream removed Model Runner V1 on 11-07 (#63373) and split prefill/decode graph sizes (#63406);
a future base will be MRV2-only, which the launcher already assumes. #61994/#62183 (PCP, APC with SWA replay) and
#62054 (UVA experts) target other topologies.

## Fixes and hygiene

1. **Tool-calling triage** with `tools/sqlbench` (replays the site's questions; separates truncation, tool-call and SQL
   errors): run reasoning on/off on `.11`, then with #62199/#62200/#62517/#62729 as overlays, then on Flare, to separate
   frontend from model behaviour before touching sampling.
2. **`.12`** is local-only: publish it or fold the `tiny_decode` patch into the next tag; the same 45-line patch is the
   FlashInfer PR the Flare plan lists (one PR serves both engines).
3. **Power on every number** (D.4.5): the README tables say 350 W; the host limit has since moved. Re-measure or label.
4. **c16 KV stress** before any 12/16-slot default (pending since 11-03); the 1.5 GiB reserve, not KV size, governs it.
5. **Patches mounted over site-packages are not contract-checked**: bake what ships (`patch-rowcap` is baked since `.5`;
   `patch-ep-allreduce` is dormant and should not travel), keep dev overlays out of the public copy (`.publish-exclude`).
6. **Expert-order refresh** (speed idea 5) and a note in README-tuning that counts are in slot space for layers 20-33.
7. **Qualification**: eval 42/43 loses bat-and-ball on sampled runs; LiveBench not re-run on `.11` (only the top-k
   changed): run it once for the record before the next tag.

## Closed, do not reopen without a new angle (measured on this image)

EP all-reduce over b12x (neutral), power/clock for decode (flat; +250 core regressed), MOET kernel port, CPU-side
experts (10x bandwidth gap), DCP3 (3.6 ms/step at c1), dynamic expert cache (0.40 % cold at 342 on real traffic),
route reuse (level at 450 W on Flare), FP8/MXFP8 LM head (72.3 → 71.8), L2 prefetch (-6 %), 6-aligned graph sizes,
draft TP1, k = 3/4/6/7 defaults (k7 stays the opt-in coding preset), stock adaptive verification as default (c1 -11 %),
8192 prefill chunk (OOM at 31K), 4080-token DMA chunk, indexer chunk 256 (-3-7 %), offload range above 34 (host OOM at
load), cold fraction < 1 (under-promotes), LMCache host tier (~80 GB host regardless of size), fused prologue above 8
rows, BRRR sorted radix top-k (-6/-15 % at 128K/512K), lossless expert shrink, NVFP4 experts for speed, 2048 batched
tokens (-40 % long prefill for +8 % decode).

## What maps to Flare

| Idea | Flare status |
| --- | --- |
| Draft acceptance at c1 | same problem (2.6 tokens/step); one study serves both; Flare plan item 1 |
| Row cap by concurrency | Flare has adaptive alpha, not a scaled cap: portable (Flare plan item 3) |
| b12x pin bump | same library, different pin; Flare's upstream sync (B.6) carries a FlashInfer bump: share the API-drift notes |
| AV manager overhead | no analogue: Flare allocates prefixes on the device inside the captured program (`scheduling-runtime-design.md` "Speculative graph coverage") |
| Expert-order refresh | same blend order and route logs; one recount serves both |
| Reserve / #63047 | does not apply: Flare sizes scratch from the plan (`flare run --dry-run`) |
| Embedding + vision in host RAM | done on Flare (11-08 default package); the vLLM overlay is the port back |
| Engram host-table sharing | Flare's answer is Engram all-disk (-3-4 % decode, -90 GiB RAM); sharing is a portable idea for both |
| FP8 KV / 64-token blocks | does not apply: Flare's KV is already a compressed 288-byte latent |
| Encoder/parser fixes | Flare renders in Helios (Rust): check its V4.1 encoder against the 10-19 reference change; run `sqlbench` on both |
| **PR #59468** | **no analogue needed, already free**: Flare never has draft rows outside the target batch. A speculative program is priced whole (B x W target rows, P ≤ R, `scheduling-runtime-design.md:30-33`, `:48-52` "neither path schedules a separate proposal completion"); DSpark "requires no separate draft-page pool" (`mtp-design.md:163`); the only per-request speculative charges are KV pages (`helios/src/scheduler.rs:1217-1225` `draft_pages`), the execution limit of widest width - 1 (`:1375-1380`) and the two-round decode horizon (`:1237-1240`); mixed graphs pack `prefill_capacity + (B - 1) x token_capacity` with no draft rows (`scheduling-runtime-design.md`, "Mixed execution contract") |
