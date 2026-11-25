#!/bin/bash
# exp/ds41-exl3: export DeepSeek-V4.1-Flash-1017 from diffbot's EXL3 3-bit pack at TP2 (two 96 GB RTX PRO 6000).
# The pack is the checkpoint as published (no view): EXL3 MUL1 routed experts (xmoe kernel, recipe-local), EXL3 MCG
# drafter experts, DeepSeek's FP8 elsewhere, int4 Engram tables (encoding INT4_FP16_BLOCK32 = 16; RAM or disk + row
# cache is chosen when serving, the package is the same). BRRR is off (it runs MXFP4 experts only).
#
#   CKPT   pack directory (required); a truncated view of it for the mini export (MINI=1)
#   GPUS   default 0,1
#   PARK   parked MoE layers (FLARE_DS41_HOST_EXPERT_LAYERS, default 2-39): hot experts in VRAM, the rest in host memory
#   HOT    hot experts per parked layer (default 300): each expert fewer frees ~6.6 MB x parked layers per rank
#   EXPORT_HOT  VRAM rows mapped per parked bank DURING the export (FLARE_DS41_EXL3_EXPORT_HOT; default = HOT): the
#          package still declares HOT VRAM rows and the kernels use HOT (exl3/experts.py export_hot); keeps the export's
#          tuning headroom of a lower hot count
#   MOE_PREFILL  prefill MoE path, group (default) or do (FLARE_DS41_EXL3_PREFILL, decode-once kernel)
#          (38 layers: ~252 MB, ~ KV) and moves ~0.3 % more routes onto the host path
#   MAXC   request slots (default 8)   PREFILL  prefill chunk (default 4096)   DRAFTS  DSpark drafts (default 5 = the
#          released head at k=5, the pack's own 2x default; their k=7 code head is the optional drafter-code-k7/)
#   MINI   1 = truncated view (FLARE_DS41_MINI)
#   MTP_ONLY  1 (default, as TP3's 160815) = --mtp-only: no "-no-mtp" graphs (~25 fewer graphs, shorter export); the
#          package then serves with drafts only (every serve here uses --mtp-draft-tokens 5); 0 = both catalogs
#   SIDE_STREAMS  1 (default) = FLARE_DS41_INDEX_SIDE_STREAM=1 + FLARE_DS41_CONTEXT_SIDE_STREAM=1 (TP3's values): the
#          index chain forks after the q/KV norm and joins before the selectors (inside attention); the DSpark context
#          projections fork after the target decoder and join before the ring writes; neither is open while an xmoe
#          kernel runs. 0 = off. (Engram side layers stay off.)
#   EXL3_SPEED  extra exp/ds41-exl3-speed switches, each default off: FLARE_DS41_EXL3_HOT_LAYERS (per-layer hot counts),
#          FLARE_DS41_EXL3_COLD_SIDE (cold launch on a side stream with k SMs), FLARE_DS41_EXL3_SHARED_SIDE (shared expert
#          beside the router, joined before xmoe) are passed through from the environment as set
#   FLARE_DS41_EXL3_WOA_SINGLE  1 = keep one WO-A layout (exp/ds41-exl3-final-woa; default off until GPU-confirmed,
#          passed through from the environment, export-baked): at TP2 b12x also packs a tile-major WO-A copy
#          (values_tiled, 16 MiB per attention layer = 688 MiB per GPU over 40 + 3 drafter layers) that the WO plan
#          never reads (it declares no wo_a_tiled, so the decode WO-A GEMM reads `values` with or without it); on = the
#          copy is not kept. --plan-only prints "WO tile-major copies kept N of 43" and fails if any remain when on.
#   PREFILL SCRATCH (exp/ds41-exl3-final-scratch, default off, passed through as set; --plan-only prints the xmoe
#          scratch per GPU against the defaults): FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS=<n> routes per decode-once pass
#          (multiple of 128; unset = the chunk's 24,576 routes = 588 MiB at TP2), FLARE_DS41_EXL3_PREFILL_GROUPS=<g>
#          prefill group-kernel groups (default 4 = 392 MiB of group views at chunk 4096; each group keeps the chunk's
#          rows), FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 (needs MOE_PREFILL=do) runs a parked layer's cold prefill launch
#          on a side stream beside the decode-once kernels (scratch = group views + one pass, side by side). Scratch per
#          GPU: groups x 4096 x 25,088 B + pass routes x 25,088 B with the side stream, the max of the two without.
#          E.g. G=2 + 8192 routes = 196 MiB (-392); G=2 + 4096 routes + side = 294 MiB (-294). The export's KV
#          grows by the saving.
#   MERGES exp/ds41-exl3-final-merges (hyperflare_recipes/recipes/deepseek41/exl3/merges.py), each default off until GPU-confirmed
#          (GPU microtest: bit-exact against the paths they replace), passed through as set:
#          FLARE_DS41_EXL3_MERGE_TOPK=1 (needs FUSED_PREP=1; graphs within FUSED_PREP_MAX_ROWS): hot_map lookup folded
#            into the fused routing prep, which runs right after top-k, before the shared-expert join (the b12x top-k
#            itself is not merged: off the critical path); FLARE_DS41_EXL3_MERGE_ROPEQ=1: inverse RoPE + WO-A input
#            quantization in one Triton kernel (decode/verify attention, b12x WO GEMMs unchanged);
#            FLARE_DS41_EXL3_MERGE_EPILOGUE=1 (needs SHARED_SIDE=1): the MoE op writes routed + shared (its acc->bf16
#            copy and the graph's add in one kernel). All bit-exact vs the paths they replace; the merged CUDA kernels
#            build as their own extension (ds41_exl3_merges_<digest>, no --use_fast_math), the xmoe extension is unchanged
#   IMAGE_CONCURRENCY  --image-concurrency n image feature slots (unset = recipe default min(4, MAXC) = 4 x 40 MiB
#          VisionState.arena per GPU; 1 = 40 MiB, -120 MiB per GPU). A second concurrent image request then waits in the
#          scheduler (Capacity -> back to pending, text requests still admitted) until the first one's slot retires
#   COLD WIDTH  FLARE_DS41_EXL3_COLD_WIDTH defaults to 16 here (exp/ds41-exl3-final-bulk; s1 cw2: one decode cold launch
#          241 us at width 16 vs 292 us at the kernel default 8, k16 side stream); set FLARE_DS41_EXL3_COLD_WIDTH=0 for
#          170303's value
#   BULK   FLARE_DS41_EXL3_BULK_MIN_ROWS (xmoe_bulk hot loop for decode/verify graphs with >= N rows; 0/unset = off) is
#          passed through from the environment as set
#   PREFILL_GRAPHS  FLARE_DS41_PREFILL_GRAPHS: comma list of prefill graph capacities, a subset of the ladder
#          128,256,512,...,PREFILL that keeps 128 and PREFILL (unset = the full ladder, as 170303)
#   CONFIG optional Flare config (--config)
# Serving these packages: FLARE_CUDA_STACK_LIMIT=128 (bytes per thread; 170303 used 256, the recorded default is 1024).
#   Every EXL3 TP2 serve so far logs "largest kernel stack 120 B" for the registered kernels (m3..s1 logs, incl.
#   package 170303); the xmoe_bulk kernel needs 0 B (cuobjdump: STACK 0). 170303's kernel modules also hold functions
#   with 128-384 B stacks (xmoe n256 group kernels, torch radixSortKVInPlace variants) that no graph registers; Helios refuses a
#   limit below the largest registered kernel at load ("CUDA stack limit ... is below rank N's largest kernel stack"),
#   so a wrong value fails the boot instead of corrupting a run (fall back to 256). 256 -> 128 B: ~+41K KV tokens.
# Format-independent switches of the TP3 default package 160815, on here: embed_rows (host-row embedding gather),
# VRAM reclaim (shared Engram token map, trimmed vocabulary gather, no draft-vocabulary channel, shared vocabulary
# columns), B12X_MHC_PDL=1, index + context side streams (SIDE_STREAMS), scratch preference 0.05 / 16 MiB.
# Deliberately off: Engram side layers (beside the MoE: xmoe's grid barriers need all their blocks resident), AR_ROWS / AR_PART_ROWS / ENGRAM_PART_ROWS (the TP3 shared PCIe
# all-reduce; at TP2 Flare uses its decode all-reduce + DMA prefill all-reduce), BRRR switches, 16-slot packed shapes.
#   DEBUG_SERVE  1 = `ignition debug-serve` (prepare in this process and serve HTTP on HOST:PORT, default
#          127.0.0.1:15015) with exactly the export's environment and preparation flags, instead of `ignition export`;
#          needs heliopause's qualification build on HELIOPAUSE_PATH (a directory placed after this tree on PYTHONPATH).
#          The use is FLARE_DS41_ROUTE_COUNTS=<dir> (routecount.py, expert-order calibration on your own traffic;
#          snapshots and the order builder: tp3-tools/ds41_route_order.py --help), which an export
#          refuses.
#   --dry-run  print the environment and the exact export argv, run nothing (CPU only)
#   --plan-only  CPU plan check (tp3-tools/ds41_exl3_plan_check.py) with the same environment; stops before any GPU work
# Run from this checkout with its .venv (heliopause from this tree knows
# encoding 16). Logs go to stdout; the package path is printed at the end by ignition.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
: "${CKPT:?set CKPT to the EXL3 pack directory}"
python3 -I -c "import json,sys; q=json.load(open(sys.argv[1]))['quantization_config']; sys.exit(q.get('quant_method') != 'exl3')" \
  "$CKPT/config.json" || { echo "$CKPT is not an EXL3 pack" >&2; exit 2; }
for v in $(compgen -e | grep -E '^FLARE_DS41_BRRR' || true); do echo "refusing: $v is set (BRRR is MXFP4-only)" >&2; exit 2; done
if [ "${FLARE_DS41_EXL3_PREFILL_COLD_SIDE:-0}" = 1 ] && [ "${MOE_PREFILL:-${FLARE_DS41_EXL3_PREFILL:-group}}" != do ]; then
  echo "refusing: FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 needs MOE_PREFILL=do" >&2; exit 2; fi
if [ "${DEBUG_SERVE:-0}" != 1 ]; then   # s1: FLARE_DS41_EXL3_ROUTE_CAPTURE (routecapture.py) and the debug-serve
  # server overrides (FLARE_DEBUG_SERVE_SERVER_JSON, merged into debug-serve's native server options) are debug-serve only too
  for v in FLARE_DS41_ROUTE_COUNTS FLARE_DS41_EXL3_ROUTE_CAPTURE FLARE_DEBUG_SERVE_SERVER_JSON FLARE_DS41_DEBUG_KV_PAGES; do
    [ -n "${!v:-}" ] && { echo "refusing: $v is set (debug-serve only, never export with it)" >&2; exit 2; }; done
fi
export PYTHONPATH="${B12X_OVERLAY:+$B12X_OVERLAY:}$ROOT${PYTHONPATH:+:$PYTHONPATH}${HELIOPAUSE_PATH:+:$HELIOPAUSE_PATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" HF_HUB_OFFLINE=1 \
       TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=${MAX_JOBS:-4} \
       FLARE_DS41_EXL3=1 FLARE_DS41_TP3_GRAPH_GIB=1 \
       FLARE_DS41_HOST_EXPERT_LAYERS="${PARK-2-39}" FLARE_DS41_HOT_EXPERTS="${HOT:-300}" FLARE_DS41_MERGED_BANK=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR="${COUNTS:-$ROOT/tp3-tools/release/expert-order}" \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_EMBED_ROWS="${FLARE_DS41_EMBED_ROWS:-1}" \
       FLARE_DS41_VRAM_RECLAIM="${FLARE_DS41_VRAM_RECLAIM:-1}" B12X_MHC_PDL="${B12X_MHC_PDL:-1}" \
       FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" \
       FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_ENGRAM_CHUNK_ROWS="${FLARE_DS41_ENGRAM_CHUNK_ROWS:-768}" \
       FLARE_DS41_SHARE_ENGRAM_RESIDUAL="${FLARE_DS41_SHARE_ENGRAM_RESIDUAL:-1}" \
       FLARE_DS41_ENGRAM_FULL_PROJECTION="${ENGRAM_FULL_PROJECTION:-0}" \
       FLARE_DS41_ENGRAM_NO_READINESS_BARRIER="${ENGRAM_NO_READINESS_BARRIER:-1}" \
       FLARE_DS41_A16_QKV="${A16_QKV:-0}" \
       FLARE_FLAT_MEMORY_POOL="${FLAT_POOL:-1}" \
       B12X_PCIE_DCP_GATHER_PULL="${GATHER_PULL:-1}" \
       FLARE_DS41_EXL3_COLD_WIDTH="${FLARE_DS41_EXL3_COLD_WIDTH:-16}"
[ -n "${PREFILL_GRAPHS:-}" ] && export FLARE_DS41_PREFILL_GRAPHS="$PREFILL_GRAPHS"
[ "${MINI:-0}" = 1 ] && export FLARE_DS41_MINI=1
if [ "${SIDE_STREAMS:-1}" = 1 ]; then export FLARE_DS41_INDEX_SIDE_STREAM=1 FLARE_DS41_CONTEXT_SIDE_STREAM=1; fi
export FLARE_TUNE_SCRATCH_TOLERANCE="${FLARE_TUNE_SCRATCH_TOLERANCE:-0.05}" \
       FLARE_TUNE_SCRATCH_MIN_SAVING_MIB="${FLARE_TUNE_SCRATCH_MIN_SAVING_MIB:-16}"
[ -n "${EXPORT_HOT:-}" ] && export FLARE_DS41_EXL3_EXPORT_HOT="$EXPORT_HOT"
[ -n "${MOE_PREFILL:-}" ] && export FLARE_DS41_EXL3_PREFILL="$MOE_PREFILL"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${IGNITION_PYTHON:-.venv/bin/python}"
"$PY" -c 'import hyperflare_recipes, ignition, flare_heliopause; print("recipes", hyperflare_recipes.__file__, "ignition", ignition.__file__, "heliopause", flare_heliopause.__file__)'
/usr/bin/env | grep -E '^(FLARE_DS41_|FLARE_TUNE_|B12X_)' | sort || true   # /usr/bin/env: another `env` (e.g. uv's) can come first on PATH
VERB=export; [ "${DEBUG_SERVE:-0}" = 1 ] && VERB=debug-serve
if [ -z "${IGNITION_PYTHON:-}" ] && [ -x .venv/bin/ignition ]; then
  IGNITION_CMD=(.venv/bin/ignition)
else
  IGNITION_CMD=("$PY" -m hyperflare_recipes.cli)
fi
ARGV=("${IGNITION_CMD[@]}" "$VERB" "$CKPT" --gpus "${GPUS:-0,1}" --max-concurrency "${MAXC:-8}"
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose
  ${COMPILE_WORKERS:+--compile-workers "$COMPILE_WORKERS"}
  ${STEERING:+--steering-sidecar "$STEERING"}
  ${CONFIG:+--config "$CONFIG"})
[ "${MTP_ONLY:-1}" = 1 ] && ARGV+=(--mtp-only)
[ -n "${IMAGE_CONCURRENCY:-}" ] && ARGV+=(--image-concurrency "$IMAGE_CONCURRENCY")
[ "${DEBUG_SERVE:-0}" = 1 ] && ARGV+=(--host "${HOST:-127.0.0.1}" --port "${PORT:-15015}")
if [ "${1:-}" = --plan-only ] && [ "${DEBUG_SERVE:-0}" = 1 ]; then echo "--plan-only checks exports only" >&2; exit 2; fi
if [ "${1:-}" = --plan-only ]; then
  PLAN_ARGS=("${ARGV[@]:${#IGNITION_CMD[@]}}")
  CUDA_VISIBLE_DEVICES= exec "$PY" tp3-tools/ds41_exl3_plan_check.py "$CKPT" \
    --world "$(tr ',' '\n' <<< "${GPUS:-0,1}" | grep -c .)" -- "${PLAN_ARGS[@]}"
fi
if [ "${1:-}" = --dry-run ]; then printf 'export argv:'; printf ' %q' "${ARGV[@]}"; echo; exit 0; fi
exec "${ARGV[@]}"
