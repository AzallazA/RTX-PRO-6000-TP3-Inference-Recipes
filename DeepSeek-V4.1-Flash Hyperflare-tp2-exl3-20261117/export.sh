#!/bin/bash
# Export the DeepSeek-V4.1-Flash EXL3 TP2 package with this release's settings. Run from the fork checkout (its
# .venv), GPUs idle. Wraps the fork's tp3-tools/ds41-exl3-export-tp2.sh, which documents every switch.
#   export.sh               the export (~41 min on the test system with a warm kernel cache)
#   export.sh --plan-only   CPU-only plan check: prints the declared VRAM per GPU, stops before any GPU work
#   export.sh --dry-run     print the environment and the exact `ignition export` argv, run nothing
# Inputs (environment):
#   CKPT       the pack directory (required); its directory name becomes the model name
#   COUNTS     expert-order directory (default: expert-order/ next to this script)
#   HOT_LOW    resident (hot) experts per layer on layers 2-4 and 6-19 (default 344 of 384)
#   HOT_HIGH   resident experts per layer on layers 20-39 (default 316 of 384)
#              Layers 0, 1 and 5 keep all 384 experts in VRAM. Fewer resident experts = more KV, slower decode
#              (KV-TRADEOFF.md).
#   EXPORT_HOT VRAM rows mapped per split layer during the export only, for tuning headroom (default 308, or the
#              lowest of HOT_LOW/HOT_HIGH if that is lower); the package still uses HOT_LOW/HOT_HIGH
#   GPUS       default 0,1      CONFIG  optional Flare config file (--config), e.g. to choose the package cache
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
: "${CKPT:?set CKPT to the pack directory}"
[ -x tp3-tools/ds41-exl3-export-tp2.sh ] || { echo "run this from the fork checkout" >&2; exit 2; }
HOT_LOW=${HOT_LOW:-344}; HOT_HIGH=${HOT_HIGH:-316}
LOWEST=$(( HOT_LOW < HOT_HIGH ? HOT_LOW : HOT_HIGH ))
EH=${EXPORT_HOT:-308}; [ "$EH" -gt "$LOWEST" ] && EH=$LOWEST
HOT_LAYERS=$(python3 -I -c "
import sys
lo, hi = sys.argv[1], sys.argv[2]
print(','.join([f'{l}:{lo}' for l in (2, 3, 4, *range(6, 20))] + [f'{l}:{hi}' for l in range(20, 40)]))
" "$HOT_LOW" "$HOT_HIGH")
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env -u CUDA_VISIBLE_DEVICES \
  CKPT="$CKPT" COUNTS="${COUNTS:-$HERE/expert-order}" GPUS="${GPUS:-0,1}" ${CONFIG:+CONFIG="$CONFIG"} \
  PARK=2-4,6-39 HOT="$HOT_HIGH" EXPORT_HOT="$EH" FLARE_DS41_EXL3_HOT_LAYERS="$HOT_LAYERS" \
  MAXC=8 PREFILL=4096 DRAFTS=5 MTP_ONLY=1 IMAGE_CONCURRENCY=1 SIDE_STREAMS=1 MOE_PREFILL=do \
  FLARE_DS41_EXL3_COLD_WIDTH=16 FLARE_DS41_EXL3_COLD_SIDE=16 FLARE_DS41_EXL3_SHARED_SIDE=1 \
  FLARE_DS41_EXL3_FUSED_PREP=1 FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS=24 \
  FLARE_DS41_EXL3_WOA_SINGLE=1 \
  FLARE_DS41_EXL3_PREFILL_GROUPS=2 FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS=4096 FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 \
  FLARE_DS41_EXL3_MERGE_TOPK=1 FLARE_DS41_EXL3_MERGE_EPILOGUE=1 FLARE_DS41_EXL3_MERGE_ROPEQ=0 \
  FLARE_DS41_EXL3_BULK_MIN_ROWS=0 \
  FLARE_EXPORT_LIBRARY_SCRATCH=1 FLARE_TUNE_SCRATCH_TOLERANCE=0.05 FLARE_TUNE_SCRATCH_MIN_SAVING_MIB=16 \
  tp3-tools/ds41-exl3-export-tp2.sh "$@"
