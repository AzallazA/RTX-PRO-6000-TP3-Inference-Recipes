#!/bin/bash
# Export the GLM-5.3-Flash TP3 package (3 GPUs) from the nvidia/GLM-5.3-Flash-FP4-Local checkpoint directory.
# Run it from the fork checkout (EXPORT.md step 2: .venv synced, tp3-patches/ applied to its FlashInfer), GPUs idle.
# Same settings as the fork's tp3-tools/glm53-export-tp3.sh with MTP_ONLY=1 DCP=1 CONC=32, a16, KDA side chain and
# 704 expert columns, plus VISION=1 FLARE_GLM53_VISION_STAGE=1 IMAGE_CONCURRENCY=4, which is how the release package was exported.
#   CKPT       checkpoint directory (required). Its directory name becomes the model name (EXPORT.md step 3)
#   CONC       request slots the package captures graphs for (default 32 = this release; 8 = more KV, see README)
#   GPUS       default 0,1,2
#   CUDA_HOME  CUDA 13.1 toolkit (default /usr/local/cuda-13.1)
#   B12X_OVERLAY  b12x copy with the world-3 pull gather, ahead of the venv on PYTHONPATH (default <fork>/.b12x-overlay,
#              built on first use by the fork's tp3-tools/make-b12x-overlay.sh; the shared .venv stays unchanged).
#              B12X_OVERLAY= (empty) exports without it: upstream push gather, measured 7-8% slower at c1/c4/c8
#   VISION     1 (default) = image input: vision tower parked in host memory and staged per image
#              (FLARE_GLM53_VISION_STAGE=1), 4 image slots (IMAGE_CONCURRENCY); 0 = text-only (--language-model-only, more KV)
#   IMAGE_CONCURRENCY  images encoded at once (default 4; 64 MiB per slot per GPU)
#   CONFIG     a Hyperflare config.toml that selects another package store (paths.package_cache)
# Expect "Graph catalog: 74 graphs" (text-only: 68) and, at the end, "Exported cache entry: .../gb202-tp3-<timestamp>".
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
# Fork-only GLM export switches (ignition/recipes/glm53, unset = upstream choice):
export FLARE_GLM53_DCP3=1 FLARE_GLM53_EXPERT_ACTIVATION=a16 FLARE_GLM53_KDA_SIDE_CHAIN=1 FLARE_GLM53_EXPERT_COLUMNS=704
unset FLARE_GLM53_DRAFT_HEAD FLARE_GLM53_DRAFT_EH_PROJ FLARE_GLM53_EXPAND_SCALES FLARE_GLM53_EMBED_HOST
if [ "${VISION:-1}" = 1 ]; then
  export FLARE_GLM53_VISION_STAGE=1; MODALITY=(--image-concurrency "${IMAGE_CONCURRENCY:-4}")
else
  unset FLARE_GLM53_VISION_STAGE; MODALITY=(--language-model-only)
fi
CONFIG_FLAGS=(); [ -n "${CONFIG:-}" ] && CONFIG_FLAGS=(--config "$CONFIG")
exec .venv/bin/ignition export "$CKPT" ${CONFIG_FLAGS[@]+"${CONFIG_FLAGS[@]}"} --gpus "${GPUS:-0,1,2}" \
  --prefill-chunk-size 4096 --mtp-draft-tokens 5 --max-concurrency "${CONC:-32}" \
  "${MODALITY[@]}" --mtp-only --verbose
