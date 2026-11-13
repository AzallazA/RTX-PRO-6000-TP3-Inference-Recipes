#!/bin/bash
# Serve the GLM-5.3-Flash TP3 package on 3 GPUs (up to 32 concurrent requests, 1,048,576 tokens per request).
#   FLARE         path to this fork's flare binary (default: flare-cli/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the checkpoint directory name the package was exported from), name@version, or a
#                 package path (default glm53-fp4-local = the directory name in EXPORT.md). If you also exported the
#                 TP2 package under the same name, give the version: PACKAGE=glm53-fp4-local@gb202-tp3-<timestamp>
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 32G)
#   DRAFTS        MTP drafts (default 5 = the export's; the package is exported with --mtp-only and refuses 0)
#   ALPHA         adaptive-verification alpha (default 0.8, the c32 sweep winner; ALPHA= (empty) = Hyperflare's 1.0)
#   BUDGET        max verification rows per step (default unset = every captured width)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target: a new prompt's first
#                 token after ~0.5 s instead of 3-22 s behind 3-31 decoding streams, decode level; balanced = guard
#                 off, Hyperflare's default, for pure batch traffic; see README "Scheduler trade-off")
#   GPUS, PORT, HOST   default 0,1,2 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-glm53-fp4-local}
ALPHA=${ALPHA-0.8}
EXTRA=()
[ -n "${BUDGET:-}" ] && EXTRA+=(--mtp-max-verification-token-budget "$BUDGET")
[ -n "$ALPHA" ] && EXTRA+=(--mtp-adaptive-verification-alpha "$ALPHA")
exec "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" --host "${HOST:-127.0.0.1}" \
  --served-model-name GLM-5.3-Flash --prefix-cache-memory "${PREFIX_CACHE:-32G}" --mtp-draft-tokens "${DRAFTS:-5}" \
  --scheduler-policy "${SCHEDULER:-interactive}" ${EXTRA[@]+"${EXTRA[@]}"}
