#!/bin/bash
# Export the GLM-5.3-Flash TP2 package (2 GPUs) from the nvidia/GLM-5.3-Flash-FP4-Local checkpoint directory.
# Run it from the fork checkout (EXPORT.md step 2: .venv synced, tp3-patches/ applied to its FlashInfer), GPUs idle.
# Same settings as the fork's tp3-tools/glm53-export-tp2.sh with GPUS=0,1 MTP_ONLY=1, a16, KDA side chain and the
# token embedding in host memory, plus VISION=1 FLARE_GLM53_VISION_STAGE=1 IMAGE_CONCURRENCY=2, which is how the release
# package was exported. TP2 is upstream's own rank count
# (no padding) with upstream's decode context parallelism over the two ranks (DCP2, 512-token pages).
#   CKPT       checkpoint directory (required). Its directory name becomes the model name (EXPORT.md step 3)
#   CONC       request slots the package captures graphs for (default 8 = this release)
#   GPUS       default 0,1
#   CUDA_HOME  CUDA 13.1 toolkit (default /usr/local/cuda-13.1)
#   B12X_OVERLAY  as in export-glm-tp3.sh (default <fork>/.b12x-overlay). At two GPUs the overlay compiles the same
#              push gather as upstream unless B12X_PCIE_DCP_GATHER_PULL=1 is exported (world-2 pull, not measured)
#   VISION     1 (default) = image input: vision tower parked in host memory and staged per image
#              (FLARE_GLM53_VISION_STAGE=1), 2 image slots (IMAGE_CONCURRENCY); 0 = text-only (--language-model-only, more KV)
#   IMAGE_CONCURRENCY  images encoded at once (default 2; 64 MiB per slot per GPU)
#   CONFIG     a Hyperflare config.toml that selects another package store (paths.package_cache)
# Expect "Graph catalog: 56 graphs" (text-only: 50) and, at the end, "Exported cache entry: .../gb202-tp2-<timestamp>".
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
# (EXPERT_COLUMNS, DRAFT_HEAD, DRAFT_EH_PROJ, DCP3) do not apply at two ranks.
export FLARE_GLM53_EXPERT_ACTIVATION=a16 FLARE_GLM53_KDA_SIDE_CHAIN=1 FLARE_GLM53_EMBED_HOST=1
unset FLARE_GLM53_DRAFT_HEAD FLARE_GLM53_DRAFT_EH_PROJ FLARE_GLM53_EXPAND_SCALES FLARE_GLM53_EXPERT_COLUMNS FLARE_GLM53_DCP3
if [ "${VISION:-1}" = 1 ]; then
  export FLARE_GLM53_VISION_STAGE=1; MODALITY=(--image-concurrency "${IMAGE_CONCURRENCY:-2}")
else
  unset FLARE_GLM53_VISION_STAGE; MODALITY=(--language-model-only)
fi
CONFIG_FLAGS=(); [ -n "${CONFIG:-}" ] && CONFIG_FLAGS=(--config "$CONFIG")
exec .venv/bin/ignition export "$CKPT" ${CONFIG_FLAGS[@]+"${CONFIG_FLAGS[@]}"} --gpus "${GPUS:-0,1}" \
  --prefill-chunk-size 4096 --mtp-draft-tokens 5 --max-concurrency "${CONC:-8}" \
  "${MODALITY[@]}" --mtp-only --verbose
