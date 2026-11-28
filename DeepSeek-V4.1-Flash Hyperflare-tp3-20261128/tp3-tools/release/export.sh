#!/bin/bash
# Export the DS41 TP3 Wave 15.1 package from a checkpoint view (make_checkpoint_view.sh). Run with the flare-core
# .venv (uv sync --locked --extra prepare), patched hyperflare-recipes, and B12X_OVERLAY pointing to patched flashinfer
# (EXPORT.md steps 3-5; Hyperflare's torch/Triton fixes are applied in memory by ignition.runtime_patches). The defaults
# reproduce the Wave 15.1 release package gb202-tp3-202611281539 except the steering sidecar, which you pass with
# STEERING= (download: EXPORT.md step 4). The DS41 recipe settings are recorded in the package's tp3-config.json.
#   CKPT     checkpoint view directory (required; the directory name becomes the model name, default view: ds41-1017-engram-fp8)
#   B12X_OVERLAY  b12x overlay directory with b12x_tree symlinked to flashinfer/flashinfer/experimental/b12x and the
#            b12x/__init__.py shim (required while TWOSHOT=1 or GATHER_PULL=1, the defaults; see EXPORT.md step 3)
#   STEERING  path of the steering sidecar (.safetensors, layers 4-36) = --steering-sidecar; the package then takes a
#            per-request steering_scale in [0, 1] (0 = unsteered). Unset = no sidecar (nonzero scales get HTTP 400)
#   STEERING_PLAIN_ID  1 (default) = with STEERING, tune under the checkpoint's plain identity: the same tuning records
#            and plan selections as the package without the sidecar; 0 = upstream's sidecar-keyed identity
#   COUNTS   expert route counts that decide the hot experts (default: the blended code and prose order of the release,
#            expert-order/ next to this script in the release folder)
#   PARK     layers whose routed experts are split between VRAM and host RAM (default 20-39)
#   HOT      hot experts per parked layer kept in VRAM, out of 384 (default 375 for Wave 15.1 ~2.43M KV & peak C1/coding
#            speed on headless 96 GB GPUs; use 355 for the ~5.06M KV-King variant or extra desktop headroom)
#   GLOBAL_HOT_BUDGET  1 (default) = allocate len(PARK) * HOT resident experts globally across PARK layers by
#            normalized route frequency (1 <= hot_l <= 384, identical total VRAM, lower cold-route rate); 0 = uniform
#   MAXC     request slots the package captures graphs for (default 16)
#   PREFILL  prefill chunk size (default 7680, ladder 128,256,512,768,1536,3072,4608,6144,7680)
#   DRAFTS   DSpark drafts (default 5); serve with the same DRAFTS (serve.sh)
#   PARKED_ROWS  BRRR row gate of the parked layers (default 0 = BRRR at every width)
#   TWOSHOT  1 (default) = race the BF16 two-shot pull all-reduce beside the upstream candidates (needs B12X_OVERLAY);
#            0 = upstream candidate set
#   GATHER_PULL  1 (default) = world-3 DCP column gathers pull instead of push (needs B12X_OVERLAY with the gather pull
#            patch; copies only); 0 = push, as before
#   ATTN     1 (default) = each attention decode's 16-head grid on a side stream beside the 8-head remainder grid
#   MLA_SPLITS  ROWS:SPLITS,... decode split count per prepared row count (default 18:10,37:5,48:4,62:3,94:2,96:4;
#            empty = the code's rule)
#   H24      ROWS:0|1,... row counts whose decode attention runs all 24 heads in one launch (default 128:1; empty = off)
#   H24_PREFILL  1 (default) = 24-head one-launch DSV41 BF16 prefill MLA (replaces 16 + 8 partitions; bit-identical); 0 = off
#   PREFILL_DMA  1 (default) = copy-engine PCIe DMA all-reduce for TP3 prefill reductions (rows % 3 tail on SM collective),
#            shrinking the SM decode collective to decode capacity; 0 = shared SM collective
#   MHC_PDL  1 (default) = programmatic dependent launch in b12x's mHC norm kernels
#   AR_PART_ROWS, ENGRAM_PART_ROWS  prefill all-reduce part caps for the decoder and the Engram projection (default 0 with
#            PREFILL_DMA=1 for full-chunk DMA; 768 each with PREFILL_DMA=0)
#   INDEX_SIDE   1 (default) = indexer query chain on a side stream
#   CONTEXT_SIDE 1 (default) = DSpark context projection on a side stream beside verification
#   ENGRAM_SIDE  layer list for the Engram side stream (default 1,14; empty = off)
#   ENGRAM_SIDE_SPLIT  1 (default when ENGRAM_SIDE is non-empty) = split side-stream Engram across L-2 and L-1 attention
#   ENGRAM_LOCAL_LOOKUP  1 (default) = TP3 rank-local 8-head Engram lookup (no [tokens, 6144] fill or index_select)
#   ENGRAM_CHUNK_ROWS  Engram prefill projection chunk rows (default 768 = 6×128; 768×5=3840 is divisible by 3 and fits
#            the 4099-row DMA slab in one launch, cutting 4096-chunk Engram collectives from 24 to 14 with exact MMA alignment)
#   ENGRAM_FULL_PROJECTION  1 (default) = load projection_full (full-chunk Engram projection);
#            0 = omit Engram full-projection weights & 6/5 DMA slab headroom (~324 MiB/rank)
#   A16_QKV  0 (default on TP3) = TP3 split_k_norm (m<=8) + mla_reduce_norm (9..32), proven +7.7% at c1 & tighter
#            consistency; 1 = PR #9 fused A16 QKV projection for <= 28 rows (with mla_reduce_norm for 29..32)
#   FLAT_POOL  1 (default) = FLARE_FLAT_MEMORY_POOL=1 (1D flat Inductor pool, saves ~382 MiB/GPU graph capture pool);
#            0 = legacy per-dtype pool
#   ROW_SHARD_PREFILL  1 (default) = pass --row-shard-prefill (PCIe row-sharded encoder prefill; needs chunk divisible
#            by 2*world*128, e.g. PREFILL=3840, 4608, or 7680 on TP3); 0 = off
#   ROW_SHARD_ATTENTION  1 (default) = pass --row-shard-attention (64-head encoder attention on layers 0-19;
#            requires ROW_SHARD_PREFILL=1); 0 = off
#   ZERO_REP_ATTENTION  1 (default) = zero-replication encoder attention weight gather (keeps only 5 KiB of FP32
#            attn_sink resident and all-gathers wq_b/wo_a/wo_b on demand, saving ~2.26 GiB/rank); 0 = replicate MXFP8
#   MOE_RESERVE_SMS  SMs reserved on overlapped encoder MoE layers (0-19) for concurrent weight AllGather (default 4; 0 = off)
#   BRRR_DRAFT   1 (default) = DSpark draft layers (mtp.0..2, 128 experts, top-3) run through the fused BRRR kernel
#   BRRR_R1_M_THRESH  BRRR hot-expert R1 row threshold (default 17; empty = kernel default 48)
#   BRRR_PARKED_HOT_R1P  BRRR parked-layer hot R1P warps/expert (default 2; empty = kernel default 5)
#   BRRR_WOB_KS24_MAXM   BRRR wo_b ks=24 max row threshold (default unset = kernel default 32)
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
#   SKIP_SUFFIX  prefill capacities that get an encoder-only twin (default 3072,4608,6144,7680 at PREFILL=7680; empty = none).
#            Non-final chunks of long prompts skip layers 21-39 and the head.
#   IDX_SPLIT  prefill capacities whose index layers split the indexer rows across the GPUs (default empty when
#            ROW_SHARD_ATTENTION=1; otherwise PREFILL when captured)
#   IDX_SPLIT_MIN_CTX  only for prefill chunks from this many tokens on (default unset = always split captured IDX_SPLIT
#            capacities, reclaiming duplicate variant graphs now that GATHER_PULL=1 lowers break-even to ~3K);
#            IDX_SPLIT_VARIANTS = prefill, terminal_prefill or both (default both when IDX_SPLIT_MIN_CTX is set)
#   DRAFT_HEAD_QUANT  DSpark draft head quantization: nvfp4 (default), mxfp8, or bf16/none/empty (reuses BF16 target head)
#   SHARE_OPERANDS  1 (default) = one computed copy per shared pointwise operand of opaque calls
#   ROTARY_ONCE  1 (default) = decode/verify/draft forwards compute each RoPE table once (bit-identical)
#   ROTARY_PREFILL  1 (default) = prefill graphs read one shared RoPE table per forward (bit-identical)
#   PIN_QPROJ    1 (default) = attention of <= 8 rows takes the pinned q projection plans
#   LOGPROB_ONE_EXCHANGE  1 (default with generation logprobs) = use one exchange for their reduction
#   NO_GEN_LOGPROBS  1 = package without generation logprobs (requests that ask for them get HTTP 400); default 0
#   DRAFT_VOCAB, DRAFT_HEAD_PLANS  draft-head experiments (default unset; not used by the release)
#   COPY_VERIFY  compact wide copy-verify (draft_source=2) BxW shapes (default 1x16,2x16,4x16; empty = off)
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
# Ensure CUDA_HOME / nvcc >= 13.1 for SM120 (sm_120a) BRRR kernel compilation.
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/bin" ]; then
    case ":${PATH:-}:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) export PATH="$CUDA_HOME/bin${PATH:+:$PATH}" ;;
    esac
fi
NVCC_BIN=$(command -v nvcc || true)
if [ -z "$NVCC_BIN" ]; then
    echo "nvcc not found: install CUDA 13.1+ and export CUDA_HOME=/usr/local/cuda PATH=\$CUDA_HOME/bin:\$PATH" >&2
    exit 1
fi
NVCC_VER=$("$NVCC_BIN" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
if [ -n "$NVCC_VER" ] && [ "$(printf '%s\n%s\n' "$NVCC_VER" "13.1" | sort -V | head -1)" != "13.1" ]; then
    echo "nvcc $NVCC_VER at $NVCC_BIN is too old for SM120 (need CUDA >= 13.1; export CUDA_HOME=/usr/local/cuda PATH=\$CUDA_HOME/bin:\$PATH)" >&2
    exit 1
fi
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
NO_GEN_LOGPROBS=${NO_GEN_LOGPROBS-${FLARE_DS41_NO_GENERATION_LOGPROBS-0}}
DEFAULT_LOGPROB_EXCHANGE=1
if [ "$NO_GEN_LOGPROBS" = 1 ]; then
    DEFAULT_LOGPROB_EXCHANGE=0
fi
PREFILL_DMA=${PREFILL_DMA-1}
if [ "$PREFILL_DMA" = 1 ]; then
    DEFAULT_AR_ROWS=0
    DEFAULT_AR_PART_ROWS=0
    DEFAULT_ENGRAM_PART_ROWS=0
else
    DEFAULT_AR_ROWS=1024
    DEFAULT_AR_PART_ROWS=768
    DEFAULT_ENGRAM_PART_ROWS=768
fi
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=${PARK:-20-39} FLARE_DS41_HOT_EXPERTS=${HOT:-375} \
       FLARE_DS41_GLOBAL_HOT_BUDGET=${GLOBAL_HOT_BUDGET-1} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$DEFAULT_COUNTS} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=${BRRR_LAYERS:-0-39} FLARE_DS41_BRRR_DRAFT=${BRRR_DRAFT-1} FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=${PARKED_ROWS:-0} \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_PREFILL_DMA=$PREFILL_DMA \
       FLARE_DS41_AR_ROWS=${AR_ROWS-$DEFAULT_AR_ROWS} FLARE_DS41_ENGRAM_CHUNK_ROWS=${ENGRAM_CHUNK_ROWS-768} FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1 \
       FLARE_DS41_AR_PART_ROWS=${AR_PART_ROWS-$DEFAULT_AR_PART_ROWS} FLARE_DS41_ENGRAM_PART_ROWS=${ENGRAM_PART_ROWS-$DEFAULT_ENGRAM_PART_ROWS} \
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
       FLARE_DS41_MLA_DECODE_SPLITS=${MLA_SPLITS-18:10,37:5,48:4,62:3,94:2,96:4} \
       FLARE_DS41_MLA_H24=${H24-128:1} \
       FLARE_DS41_MLA_H24_PREFILL=${H24_PREFILL-1} \
       FLARE_DS41_MLA_H24_Q_ROPE=${H24_Q_ROPE-1} \
       FLARE_DS41_GLUE_PASS2=${GLUE2-1} \
       FLARE_EXPORT_SHARE_COMPUTED_OPERANDS=${SHARE_OPERANDS-1} \
       FLARE_DS41_ROTARY_ONCE=${ROTARY_ONCE-1} \
       FLARE_DS41_ROTARY_ONCE_PREFILL=${ROTARY_PREFILL-1} \
       FLARE_DS41_PIN_SMALL_QPROJ=${PIN_QPROJ-1} \
       FLARE_DS41_PIN_MID_QKV_NORM=${PIN_MID_QKV_NORM-1} \
       FLARE_DS41_INDEX_SIDE_PREFILL=${INDEX_SIDE_PREFILL-1} \
       FLARE_DS41_BRRR_PUSH=${BRRR_PUSH-1} \
       FLARE_DS41_BRRR_PUSH_V2=${BRRR_PUSH_V2-2} \
       FLARE_DS41_BRRR_ROPE_ANGLES=${BRRR_ROPE_ANGLES-1} \
       FLARE_DS41_BRRR_SKIP_PAD=${BRRR_SKIP_PAD-1} \
       FLARE_DS41_BRRR_COLD_BEFORE_SH1_ALL=${BRRR_COLD_BEFORE_SH1_ALL-0} \
       FLARE_DS41_BRRR_TWO_SHOT_MIN=${BRRR_TWO_SHOT_MIN-129} \
       FLARE_DS41_BRRR_SPIN_NS=${BRRR_SPIN_NS-32} \
       FLARE_DS41_ENGRAM_NO_READINESS_BARRIER=${ENGRAM_NO_READINESS_BARRIER-1} \
       FLARE_DS41_ENGRAM_FULL_PROJECTION=${ENGRAM_FULL_PROJECTION-1} \
       FLARE_DS41_A16_QKV=${A16_QKV-0} \
       FLARE_DS41_ZERO_REP_ATTENTION=${ZERO_REP_ATTENTION-1} \
       FLARE_FLAT_MEMORY_POOL=${FLAT_POOL-1} \
       FLARE_DS41_LOGPROB_SINGLE_EXCHANGE=${LOGPROB_ONE_EXCHANGE-$DEFAULT_LOGPROB_EXCHANGE} \
       FLARE_DS41_NO_GENERATION_LOGPROBS=$NO_GEN_LOGPROBS \
       FLARE_DS41_ENGRAM_LOCAL_LOOKUP=${ENGRAM_LOCAL_LOOKUP-1} \
       FLARE_DS41_ENGRAM_SIDE_LAYERS=${ENGRAM_SIDE-1,14}
if [ -n "${FLARE_DS41_ENGRAM_SIDE_LAYERS:-}" ]; then
    export FLARE_DS41_ENGRAM_SIDE_SPLIT=${ENGRAM_SIDE_SPLIT-1}
else
    export FLARE_DS41_ENGRAM_SIDE_SPLIT=0
fi
# Omitted optional wrapper switches leave the underlying settings in place.
[ "${LIBRARY_SCRATCH+x}" ] && export FLARE_EXPORT_LIBRARY_SCRATCH=$LIBRARY_SCRATCH
[ "${TUNING_MARGIN_MIB+x}" ] && export FLARE_EXPORT_TUNING_MARGIN_MIB=$TUNING_MARGIN_MIB
[ "${MARKOV_TILE+x}" ] && export FLARE_DS41_MARKOV_TILE_ROWS=$MARKOV_TILE
[ "${DRAFT_VOCAB+x}" ] && export FLARE_DS41_DRAFT_VOCAB=$DRAFT_VOCAB
[ "${DRAFT_HEAD_PLANS+x}" ] && export FLARE_DS41_DRAFT_HEAD_PLANS=$DRAFT_HEAD_PLANS
export FLARE_DS41_COPY_VERIFY=${COPY_VERIFY-1x16,2x16,4x16}
export FLARE_DS41_BRRR_R1_M_THRESH=${BRRR_R1_M_THRESH-17}
export FLARE_DS41_BRRR_PARKED_HOT_R1P=${BRRR_PARKED_HOT_R1P-2}
[ "${BRRR_WOB_KS24_MAXM+x}" ] && export FLARE_DS41_BRRR_WOB_KS24_MAXM=$BRRR_WOB_KS24_MAXM
export FLARE_DS41_MOE_RESERVE_SMS=${MOE_RESERVE_SMS-4}
if [ "${RECLAIM:-}" = 0 ]; then
    export FLARE_DS41_SHARE_ENGRAM_TOKEN_MAP=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_TRIM_VOCAB_GATHER=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_SKIP_DRAFT_VOCAB_GATHER=$FLARE_DS41_VRAM_RECLAIM \
           FLARE_DS41_SHARE_VOCAB_COLUMNS=$FLARE_DS41_VRAM_RECLAIM
fi
if [ "${STEERING_PLAIN_ID+x}" ] || [ -n "${STEERING:-}" ]; then
    export FLARE_DS41_STEERING_PLAIN_IDENTITY=${STEERING_PLAIN_ID-1}
fi

# The row split check uses a Python operator available only in debug-serve.
[ -n "${FLARE_DS41_INDEX_ROW_SPLIT_CHECK:-}" ] && { echo "FLARE_DS41_INDEX_ROW_SPLIT_CHECK is debug-serve only" >&2; exit 2; }
PREFILL=${PREFILL:-7680}
if [ "$PREFILL" = 7680 ] && [ -z "${FLARE_DS41_PREFILL_GRAPHS+x}" ]; then
    export FLARE_DS41_PREFILL_GRAPHS=128,256,512,768,1536,3072,4608,6144,7680
fi
DEFAULT_SKIP_SUFFIX=
DEFAULT_IDX_SPLIT=
if [ "$PREFILL" = 7680 ] && [ "${FLARE_DS41_PREFILL_GRAPHS:-}" = "128,256,512,768,1536,3072,4608,6144,7680" ]; then
    DEFAULT_SKIP_SUFFIX=3072,4608,6144,7680
    DEFAULT_IDX_SPLIT=7680
elif [ "$PREFILL" = 3840 ] && [ -z "${FLARE_DS41_PREFILL_GRAPHS:-}" ]; then
    DEFAULT_SKIP_SUFFIX=3072,3584,3840
    DEFAULT_IDX_SPLIT=3840
elif [ "$PREFILL" = 4096 ] && [ -z "${FLARE_DS41_PREFILL_GRAPHS:-}" ]; then
    DEFAULT_SKIP_SUFFIX=3072,3584,4096
    DEFAULT_IDX_SPLIT=4096
elif [ -z "${SKIP_SUFFIX+x}" ] || [ -z "${IDX_SPLIT+x}" ]; then
    if ! _prefill_defaults=$("${IGNITION_PYTHON:-.venv/bin/python}" - "$PREFILL" "${FLARE_DS41_PREFILL_GRAPHS:-}" <<'PY'
import sys

from ignition.components.capacity import prefill_capacities
from hyperflare_recipes.recipes.deepseek41.tp3_config import parse_capacities

try:
    chunk = int(sys.argv[1])
    selected = parse_capacities(sys.argv[2])
except ValueError:
    # Ignition reports invalid arguments and graph capacities during configuration.
    print("- -")
    sys.exit()
captured = set(prefill_capacities(chunk))
if selected is not None:
    if chunk % 768 == 0:
        captured.update(range(768, chunk + 1, 768))
    captured &= set(selected)
skip = ",".join(str(c) for c in (3072, 3584, 3840, 4096, 4608, 5376, 6144, 7680) if c in captured) or "-"
idx = str(chunk) if chunk in (3840, 4096, 4608, 5376, 6144, 7680) and chunk in captured else "-"
print(skip, idx)
PY
    ); then
        echo "failed to compute prefill defaults for PREFILL=$PREFILL using ${IGNITION_PYTHON:-.venv/bin/python}" >&2
        exit 1
    fi
    read -r DEFAULT_SKIP_SUFFIX DEFAULT_IDX_SPLIT <<< "$_prefill_defaults"
    [ "$DEFAULT_SKIP_SUFFIX" = "-" ] && DEFAULT_SKIP_SUFFIX=
    [ "$DEFAULT_IDX_SPLIT" = "-" ] && DEFAULT_IDX_SPLIT=
fi
ROW_SHARD_PREFILL=${ROW_SHARD_PREFILL-1}
ROW_SHARD_ATTENTION=${ROW_SHARD_ATTENTION-$ROW_SHARD_PREFILL}
if [ "$ROW_SHARD_ATTENTION" = 1 ] && [ "$ROW_SHARD_PREFILL" != 1 ]; then
    echo "ROW_SHARD_ATTENTION=1 requires ROW_SHARD_PREFILL=1" >&2; exit 2
fi
if [ "$ROW_SHARD_PREFILL" = 1 ] && [[ "$PREFILL" =~ ^[0-9]+$ ]] && (( PREFILL % 768 != 0 )); then
    echo "ROW_SHARD_PREFILL=1 on TP3 requires PREFILL to be a multiple of 768 (got $PREFILL; e.g. 3840, 4608, 6144, or 7680)" >&2
    exit 2
fi
if [ "$ROW_SHARD_ATTENTION" = 1 ]; then
    DEFAULT_IDX_SPLIT=
fi
export FLARE_DS41_PREFILL_SKIP_SUFFIX=${SKIP_SUFFIX-$DEFAULT_SKIP_SUFFIX}
export FLARE_DS41_INDEX_ROW_SPLIT=${IDX_SPLIT-$DEFAULT_IDX_SPLIT}
if [[ "$FLARE_DS41_INDEX_ROW_SPLIT" =~ ^[[:space:]]*(0)?[[:space:]]*$ ]]; then
    unset FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS
else
    export FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX=${IDX_SPLIT_MIN_CTX-}
    if [ -z "$FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX" ]; then
        unset FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS
    elif [ "${IDX_SPLIT_VARIANTS+x}" ]; then
        export FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS=$IDX_SPLIT_VARIANTS
    fi
fi
DRAFT_HEAD_QUANT=${DRAFT_HEAD_QUANT-nvfp4}
DRAFT_HEAD_FLAG=()
case "$DRAFT_HEAD_QUANT" in
    nvfp4|mxfp8) DRAFT_HEAD_FLAG=(--draft-head-quantization "$DRAFT_HEAD_QUANT") ;;
    ""|none|bf16) ;;
    *) echo "DRAFT_HEAD_QUANT must be nvfp4, mxfp8, bf16, none, or empty (got $DRAFT_HEAD_QUANT)" >&2; exit 2 ;;
esac
if [ -n "${DRAFT_VOCAB:-}" ] && [ "$DRAFT_HEAD_QUANT" != nvfp4 ]; then
    echo "DRAFT_VOCAB requires DRAFT_HEAD_QUANT=nvfp4" >&2; exit 2
fi
if [ -z "${IGNITION_PYTHON:-}" ] && [ -x .venv/bin/ignition ]; then
    IGNITION_CMD=(.venv/bin/ignition)
else
    IGNITION_CMD=("${IGNITION_PYTHON:-.venv/bin/python}" -m hyperflare_recipes.cli)
fi
ROW_SHARD_FLAG=()
[ "$ROW_SHARD_PREFILL" = 1 ] && ROW_SHARD_FLAG+=(--row-shard-prefill)
[ "$ROW_SHARD_ATTENTION" = 1 ] && ROW_SHARD_FLAG+=(--row-shard-attention)
exec "${IGNITION_CMD[@]}" export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency "${MAXC:-16}" \
  --prefill-chunk-size "$PREFILL" --mtp-draft-tokens "${DRAFTS:-5}" "${DRAFT_HEAD_FLAG[@]}" \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --mtp-only --verbose \
  ${COMPILE_WORKERS:+--compile-workers "$COMPILE_WORKERS"} ${CONFIG:+--config "$CONFIG"} \
  "${ROW_SHARD_FLAG[@]}" \
  ${STEERING:+--steering-sidecar "$STEERING"}
