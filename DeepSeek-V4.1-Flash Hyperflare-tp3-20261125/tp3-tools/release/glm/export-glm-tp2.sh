#!/bin/bash
# Export the GLM-5.3-Flash TP2 package (2 GPUs) from the nvidia/GLM-5.3-Flash-FP4-Local checkpoint directory.
# Run it from the fork checkout (EXPORT.md step 2: .venv synced, tp3-patches/ applied to its FlashInfer), GPUs idle.
# The settings of the hyperflare-glm-20261116 TP2 package: upstream's two-rank geometry with DCP2, a16, KDA side chain,
# token embedding in host memory, 8 slots, images with 2 image slots, --mtp-only, plus the export-time VRAM reclaim
# switches FLARE_GLM53_TRIAL_POOL=separate and FLARE_GLM53_ONESHOT_FIT=1 and the cuBLAS containment FLARE_BF16_BMM_CUBLAS=0.
#   CKPT       checkpoint directory (required). Its directory name becomes the model name (EXPORT.md step 3)
#   CONC       request slots the package captures graphs for (default 8 = this release)
#   GPUS       default 0,1
#   CUDA_HOME  CUDA 13.1 toolkit (default /usr/local/cuda-13.1)
#   B12X_OVERLAY  as in export-glm-tp3.sh (default <fork>/.b12x-overlay). At two GPUs the overlay compiles the same
#              push gather as upstream unless B12X_PCIE_DCP_GATHER_PULL=1 is exported (world-2 pull, not measured)
#   VISION     1 (default) = image input: vision tower blocks and tail parked in host memory and staged per image
#              (FLARE_GLM53_VISION_STAGE=1), IMAGE_CONCURRENCY image slots; 0 = text-only (--language-model-only)
#   IMAGE_CONCURRENCY  images encoded at once (default 2; 64 MiB per slot per GPU)
#   RECLAIM    1 (default) = FLARE_GLM53_TRIAL_POOL=separate (the eager "Measure graph memory" trial graphs get their own
#              pool, released before the serving capture) + FLARE_GLM53_ONESHOT_FIT=1 (the one-shot all-reduce sized
#              for the rows that use it, 512 at these settings, instead of 4096); 0 = upstream layout
#   COMPILE_WORKERS  CPU compiler processes (--compile-workers; default automatic). The release export used 3
#              (Hyperflare config [preparation] compile_workers) to stay inside the test system's power-supply limit
#   FLARE_EXPORT_FORWARD_WORKERS=N, FLARE_EXPORT_FORWARD_WORKER_GIB=G, FLARE_EXPORT_TUNE_ONLY=1  (fork, export speed;
#              unset = as before) forward compilers after tuning / tune-only warm-up, see tp3-tools/release/export.sh
#   CONFIG     a Hyperflare config.toml that selects another package store (paths.package_cache)
# Expect "Graph catalog: 56 graphs" (text-only: 50), the line "trial graph pool released: ... MiB returned before the
# serving capture", and at the end "Exported cache entry: .../gb202-tp2-<timestamp>".
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint directory (EXPORT.md step 3)}"
ROOT=$PWD
[ -x "$ROOT/.venv/bin/ignition" ] && [ -f "$ROOT/tp3-tools/make-b12x-overlay.sh" ] \
  || { echo "run this from the fork checkout (needs .venv/bin/ignition and tp3-tools/)" >&2; exit 2; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH="$CUDA_HOME/bin:$PATH" HF_HUB_OFFLINE=1
if [ -z "${B12X_OVERLAY+set}" ]; then
  B12X_OVERLAY=$ROOT/.b12x-overlay
  [ -d "$B12X_OVERLAY/b12x_tree" ] || bash "$ROOT/tp3-tools/make-b12x-overlay.sh" "$B12X_OVERLAY"
fi
export PYTHONPATH="${B12X_OVERLAY:+$B12X_OVERLAY:}$ROOT" TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext"
# Fork-only GLM export switches (ignition/recipes/glm53, unset = upstream choice). The TP3-only ones
# (EXPERT_COLUMNS, DRAFT_HEAD, DRAFT_EH_PROJ, DCP3, SLAB_SEGMENT, AR_ROWS) do not apply at two ranks.
export FLARE_GLM53_EXPERT_ACTIVATION=a16 FLARE_GLM53_KDA_SIDE_CHAIN=1 FLARE_GLM53_EMBED_HOST=1
unset FLARE_GLM53_DRAFT_HEAD FLARE_GLM53_DRAFT_EH_PROJ FLARE_GLM53_EXPAND_SCALES FLARE_GLM53_EXPERT_COLUMNS FLARE_GLM53_DCP3 \
      FLARE_GLM53_SLAB_SEGMENT FLARE_GLM53_AR_ROWS
case ${RECLAIM:-1} in
  1) export FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_ONESHOT_FIT=1 ;;
  0) unset FLARE_GLM53_TRIAL_POOL FLARE_GLM53_ONESHOT_FIT ;;
  *) echo "RECLAIM must be 1 or 0" >&2; exit 2 ;;
esac
# cuBLAS containment (fork, 2026-11-14): GLM's BF16 head BMMs race Triton tiles only (ignition/components/bf16_bmm.py).
export FLARE_BF16_BMM_CUBLAS=0
if [ "${VISION:-1}" = 1 ]; then
  export FLARE_GLM53_VISION_STAGE=1; MODALITY=(--image-concurrency "${IMAGE_CONCURRENCY:-2}")
else
  unset FLARE_GLM53_VISION_STAGE; MODALITY=(--language-model-only)
fi
CONFIG_FLAGS=(); [ -n "${CONFIG:-}" ] && CONFIG_FLAGS=(--config "$CONFIG")
exec .venv/bin/ignition export "$CKPT" ${CONFIG_FLAGS[@]+"${CONFIG_FLAGS[@]}"} --gpus "${GPUS:-0,1}" \
  --prefill-chunk-size 4096 --mtp-draft-tokens 5 --max-concurrency "${CONC:-8}" \
  "${MODALITY[@]}" --mtp-only --verbose ${COMPILE_WORKERS:+--compile-workers "$COMPILE_WORKERS"}
