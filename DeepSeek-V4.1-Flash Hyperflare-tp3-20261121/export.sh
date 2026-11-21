#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh). Run it from the fork checkout with its .venv
# (uv sync, then re-apply tp3-patches/ to the installed FlashInfer; Hyperflare's torch/Triton fixes are applied in memory
# by ignition.runtime_patches). The defaults reproduce the release package 202611201530 except the steering sidecar,
# which you pass with STEERING= (download: README "Building the package"). Every switch is recorded in the package's
# tp3-config.json. EXPORT.md step 4 has the table of what each one sets.
#   CKPT     checkpoint view directory (required; the directory name becomes the model name, default view: ds41-1017-engram-fp8)
#   B12X_OVERLAY  b12x copy with both overlay patches (required while TWOSHOT=1 or GATHER_PULL=1, the defaults):
#            VENV=$PWD/.venv B12X_OVERLAY_PATCHES="<folder>/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch \
#              <folder>/tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch" tp3-tools/make-b12x-overlay.sh $PWD/.b12x-overlay-gpull
#   STEERING  path of the steering sidecar (.safetensors, layers 4-36) = --steering-sidecar; the package then takes a
#            per-request steering_scale in [0, 1] (0 = unsteered). Unset = no sidecar (nonzero scales get HTTP 400)
#   STEERING_PLAIN_ID  1 (default) = with STEERING, tune under the checkpoint's plain identity: the same tuning records
#            and plan selections as the package without the sidecar; 0 = upstream's sidecar-keyed identity
#   COUNTS   expert route counts that decide the hot experts (default: the blended code and prose order of the release,
#            expert-order/ next to this script in the release folder)
#   PARK     layers whose routed experts are split between VRAM and host RAM (default 20-39)
#   HOT      hot experts per parked layer kept in VRAM, out of 384 (default 355; fewer = more KV, slower decode)
#   MAXC     request slots the package captures graphs for (default 16)
#   PREFILL  prefill chunk size (default 4096)
#   DRAFTS   DSpark drafts (default 5); serve with the same DRAFTS (serve.sh)
#   PARKED_ROWS  BRRR row gate of the parked layers (default 0 = BRRR at every width)
#   TWOSHOT  1 (default) = race the BF16 two-shot pull all-reduce beside the upstream candidates (needs B12X_OVERLAY);
#            0 = upstream candidate set
#   GATHER_PULL  1 (default) = world-3 DCP column gathers pull instead of push (needs B12X_OVERLAY with the gather pull
#            patch; copies only); 0 = push, as before
#   ATTN     1 (default) = each attention decode's 16-head grid on a side stream beside the 8-head remainder grid
#   MLA_SPLITS  ROWS:SPLITS,... decode split count per prepared row count (default 18:10,37:5,47:4,62:3,94:2,96:3;
#            empty = the code's rule)
#   H24      ROWS:0|1,... row counts whose decode attention runs all 24 heads in one launch (default 9:0,96:1; empty = off)
#   MHC_PDL  1 (default) = programmatic dependent launch in b12x's mHC norm kernels
#   AR_PART_ROWS, ENGRAM_PART_ROWS  prefill all-reduce part caps for the decoder and the Engram projection (default 768 each)
#   INDEX_SIDE   1 (default) = indexer query chain on a side stream
#   CONTEXT_SIDE 1 (default) = DSpark context projection on a side stream beside verification
#   ENGRAM_SIDE  layer list for the Engram side stream (default empty = off)
#   WOA_UNEVEN   BRRR wo_a job split for <= 8 rows (default 1); WOA_SMALL default 0
#   EMBED_ROWS   1 (default) = vectorised host-row embedding gather
#   RECLAIM  1 (default) = the four VRAM reclaim switches (shared Engram token map, trimmed vocabulary gather, no
#            draft-vocabulary channel, shared vocabulary columns)
#   LIBRARY_SCRATCH  unset = code default (on): cuBLAS's own 64 MiB pool is kept out of the package; 0 = off
#   PAD_ONCE     1 (default) = one MXFP8 MMA-scale zero fill per forward instead of one per layer
#   VISION_ARENA_HOST  1 (default) = image feature arena (160 MiB/GPU) in device-visible host memory
#   MARKOV_HOST  1 (default) = DSpark Markov-head embedding (63 MiB/GPU) in device-visible host memory
#   MARKOV_STAGE 1 (default) = with MARKOV_HOST=1, copy each request's Markov row into VRAM once per draft step
#   MARKOV_TILE N = Markov score tiles cover N request rows per program (default unset = one program per row)
#   GLUE2    1 (default) = second decode-glue pass (fewer zero fills and copies around verification; same bytes)
#   SKIP_SUFFIX  prefill capacities that get an encoder-only twin (default 4096; empty = none): non-final chunks of long
#            prompts skip layers 21-39 and the head (serve with --scheduler-encoder-prefill false to turn it off)
#   IDX_SPLIT  prefill capacities whose index layers split the indexer rows across the GPUs (default 4096; empty = off)
#   IDX_SPLIT_MIN_CTX  only for prefill chunks from this many tokens on (default 32768; shorter prompts keep the
#            replicated graphs); IDX_SPLIT_VARIANTS = prefill, encoder_prefill or both (default both)
#   SHARE_OPERANDS  1 (default) = one computed copy per shared pointwise operand of opaque calls
#   ROTARY_ONCE  1 (default) = decode/verify/draft forwards compute each RoPE table once (bit-identical)
#   ROTARY_PREFILL  1 (default) = prefill graphs read one shared RoPE table per forward (bit-identical)
#   PIN_QPROJ    1 (default) = attention of <= 8 rows takes the pinned q projection plans
#   LOGPROB_ONE_EXCHANGE  1 (default) = the generation-logprob reduction exchanges once instead of twice (same values)
#   NO_GEN_LOGPROBS  1 = package without generation logprobs (requests that ask for them get HTTP 400); default 0
#   DRAFT_VOCAB, DRAFT_HEAD_PLANS  draft-head experiments (default unset; not used by the release)
#   BRRR_LAYERS  layers with the fused BRRR tail (default 0-39)
#   CONFIG   Flare config file for the export (--config; default: Flare's default config)
#   COMPILE_WORKERS  CPU compiler processes (--compile-workers; default automatic). The release export used 3 to stay
#            inside the test system's power-supply limit
#   FLARE_EXPORT_FORWARD_WORKERS=N, FLARE_EXPORT_FORWARD_WORKER_GIB=G, FLARE_EXPORT_FORWARD_WORKER_VRAM_MIB=M  (optional,
#            export speed only) forward compiler processes per host and their admission estimates; the release export
#            used 9 / 16 / 768. On an out-of-memory retry, unset them
#   FLARE_EXPORT_TUNE_ONLY=1  stop after kernel tuning (no package): a warm-up before a cold-cache export
#   FLARE_DS41_VISION_STAGE 1 (default) = vision tower in host RAM, weights staged through VRAM slots per block
#   GPUS     default 0,1,2
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with HOT=300,
# then export the target. The second export reuses the shared tuning cache.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
TWOSHOT=${TWOSHOT:-1}
GATHER_PULL=${GATHER_PULL:-1}
if [ -n "${B12X_OVERLAY:-}" ]; then
    if [ ! -d "$B12X_OVERLAY/b12x_tree" ] || [ ! -f "$B12X_OVERLAY/b12x/__init__.py" ]; then
        echo "B12X_OVERLAY=$B12X_OVERLAY needs b12x_tree and the b12x package shim" >&2
        exit 1
    fi
    export PYTHONPATH="$B12X_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
PCIE="${B12X_OVERLAY:-/nonexistent}/b12x_tree/comm/pcie"
if [ "$TWOSHOT" = 1 ]; then
    if [ -z "${B12X_OVERLAY:-}" ] ||
       ! grep -qs '^def twoshot_pull_enabled(' "$PCIE/_collective_tuning.py" ||
       ! grep -qs 'if twoshot_pull_enabled():' "$PCIE/pcie_allreduce.py"; then
        echo "TWOSHOT=1 needs B12X_OVERLAY with flashinfer-pcie-twoshot-pull-world3.patch" >&2
        exit 1
    fi
fi
if [ "$GATHER_PULL" = 1 ]; then
    if [ -z "${B12X_OVERLAY:-}" ] ||
       ! grep -qs '^def _gather_pull_enabled(' "$PCIE/_dcp_a2a_cute.py" ||
       ! grep -qs 'gather_pull = _gather_pull_enabled(world_size)' "$PCIE/_dcp_a2a_cute.py"; then
        echo "GATHER_PULL=1 needs B12X_OVERLAY with flashinfer-dcp-gather-pull-world3.patch" >&2
        exit 1
    fi
    export B12X_PCIE_DCP_GATHER_PULL=auto
else
    export B12X_PCIE_DCP_GATHER_PULL=0
fi
# Default expert order: expert-order/ next to this script in the release folder. In the fork checkout the release
# order is tp3-tools/release/expert-orders/order-mix-provisional-exl3S (tp3-tools/release/expert-order/ there keeps the
# 20261116 order, which other scripts use); the same file serves both places.
DEFAULT_COUNTS=$HERE/expert-order
[ -d "$HERE/expert-orders/order-mix-provisional-exl3S" ] && DEFAULT_COUNTS=$HERE/expert-orders/order-mix-provisional-exl3S
if [ "${MARKOV_STAGE-1}" = 1 ] && [ "${MARKOV_HOST-1}" != 1 ]; then
    echo "MARKOV_STAGE=1 needs MARKOV_HOST=1 (it stages rows of the parked table)" >&2; exit 1
fi
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=${PARK:-20-39} FLARE_DS41_HOT_EXPERTS=${HOT:-355} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$DEFAULT_COUNTS} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=${BRRR_LAYERS:-0-39} FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=${PARKED_ROWS:-0} \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_AR_ROWS=1024 FLARE_DS41_ENGRAM_CHUNK_ROWS=1024 FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1 \
       FLARE_DS41_AR_PART_ROWS=${AR_PART_ROWS:-768} FLARE_DS41_ENGRAM_PART_ROWS=${ENGRAM_PART_ROWS:-768} \
       FLARE_DS41_BRRR_WOA_UNEVEN=${WOA_UNEVEN:-1} FLARE_DS41_BRRR_WOA_SMALL=${WOA_SMALL:-0} \
       FLARE_DS41_VRAM_RECLAIM=${RECLAIM:-1} \
       FLARE_TUNE_SCRATCH_TOLERANCE=0.05 FLARE_TUNE_SCRATCH_MIN_SAVING_MIB=16 \
       B12X_PCIE_TWOSHOT_PULL=$TWOSHOT B12X_MLA_DECODE_REMAINDER_STREAM=${ATTN:-1} B12X_MHC_PDL=${MHC_PDL:-1}
export FLARE_DS41_INDEX_SIDE_STREAM=${INDEX_SIDE-1} \
       FLARE_DS41_CONTEXT_SIDE_STREAM=${CONTEXT_SIDE-1} \
       FLARE_DS41_EMBED_ROWS=${EMBED_ROWS-1} \
       FLARE_DS41_MMA_PAD_ONCE=${PAD_ONCE-1} \
       FLARE_DS41_VISION_ARENA_HOST=${VISION_ARENA_HOST-1} \
       FLARE_DS41_MARKOV_EMBED_HOST=${MARKOV_HOST-1} \
       FLARE_DS41_MARKOV_STAGE_ROWS=${MARKOV_STAGE-1} \
       FLARE_DS41_MLA_DECODE_SPLITS=${MLA_SPLITS-18:10,37:5,47:4,62:3,94:2,96:3} \
       FLARE_DS41_MLA_H24=${H24-9:0,96:1} \
       FLARE_DS41_GLUE_PASS2=${GLUE2-1} \
       FLARE_DS41_PREFILL_SKIP_SUFFIX=${SKIP_SUFFIX-4096} \
       FLARE_EXPORT_SHARE_COMPUTED_OPERANDS=${SHARE_OPERANDS-1} \
       FLARE_DS41_ROTARY_ONCE=${ROTARY_ONCE-1} \
       FLARE_DS41_ROTARY_ONCE_PREFILL=${ROTARY_PREFILL-1} \
       FLARE_DS41_PIN_SMALL_QPROJ=${PIN_QPROJ-1} \
       FLARE_DS41_LOGPROB_SINGLE_EXCHANGE=${LOGPROB_ONE_EXCHANGE-1}
# Omitted optional wrapper switches leave the underlying settings in place.
[ "${ENGRAM_SIDE+x}" ] && export FLARE_DS41_ENGRAM_SIDE_LAYERS=$ENGRAM_SIDE
[ "${LIBRARY_SCRATCH+x}" ] && export FLARE_EXPORT_LIBRARY_SCRATCH=$LIBRARY_SCRATCH
[ "${MARKOV_TILE+x}" ] && export FLARE_DS41_MARKOV_TILE_ROWS=$MARKOV_TILE
[ "${DRAFT_VOCAB+x}" ] && export FLARE_DS41_DRAFT_VOCAB=$DRAFT_VOCAB
[ "${DRAFT_HEAD_PLANS+x}" ] && export FLARE_DS41_DRAFT_HEAD_PLANS=$DRAFT_HEAD_PLANS
[ "${NO_GEN_LOGPROBS+x}" ] && export FLARE_DS41_NO_GENERATION_LOGPROBS=$NO_GEN_LOGPROBS
if [ "${RECLAIM:-}" = 0 ]; then
    export FLARE_DS41_SHARE_ENGRAM_TOKEN_MAP=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_TRIM_VOCAB_GATHER=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_SKIP_DRAFT_VOCAB_GATHER=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_SHARE_VOCAB_COLUMNS=$FLARE_DS41_VRAM_RECLAIM
fi
if [ "${STEERING_PLAIN_ID+x}" ] || [ -n "${STEERING:-}" ]; then
    export FLARE_DS41_STEERING_PLAIN_IDENTITY=${STEERING_PLAIN_ID-1}
fi

export FLARE_DS41_INDEX_ROW_SPLIT=${IDX_SPLIT-4096}
if [[ "$FLARE_DS41_INDEX_ROW_SPLIT" =~ ^[[:space:]]*(0)?[[:space:]]*$ ]]; then
    unset FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS
else
    export FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX=${IDX_SPLIT_MIN_CTX-32768}
    if [ -z "$FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX" ]; then
        unset FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS
    elif [ "${IDX_SPLIT_VARIANTS+x}" ]; then
        export FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS=$IDX_SPLIT_VARIANTS
    fi
fi
# The row split check uses a Python operator available only in debug-serve.
[ -n "${FLARE_DS41_INDEX_ROW_SPLIT_CHECK:-}" ] && { echo "FLARE_DS41_INDEX_ROW_SPLIT_CHECK is debug-serve only" >&2; exit 2; }
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency "${MAXC:-16}" \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --mtp-only --verbose \
  ${COMPILE_WORKERS:+--compile-workers "$COMPILE_WORKERS"} ${CONFIG:+--config "$CONFIG"} \
  ${STEERING:+--steering-sidecar "$STEERING"}
