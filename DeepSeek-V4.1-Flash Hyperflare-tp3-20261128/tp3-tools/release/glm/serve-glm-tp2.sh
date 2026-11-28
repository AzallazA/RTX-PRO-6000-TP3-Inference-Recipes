#!/bin/bash
# Serve the GLM-5.3-Flash TP2 package on 2 GPUs (up to 8 concurrent requests, 1,048,576 tokens per request).
#   FLARE         path to this fork's flare binary (default: flare-cli/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the checkpoint directory name the package was exported from), name@version, or a
#                 package path (default glm53-fp4-local = the directory name in EXPORT.md). If you also exported the
#                 TP3 package under the same name, give the version: PACKAGE=glm53-fp4-local@gb202-tp2-<timestamp>
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 32G)
#   DRAFTS        MTP drafts (default 5 = the export's; the package is exported with --mtp-only and refuses 0)
#   ALPHA         adaptive-verification alpha (default empty = Hyperflare's 1.0; at TP2 it beat 0.8, the TP3 setting,
#                 by 1-2% at c1/c4/c8: 253/504/710 vs 247/496/702 tok/s, 2 alternating rounds at 32K, 2026-11-12)
#   BUDGET        max verification rows per step (default unset = every captured width)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target; balanced = guard
#                 off, Hyperflare's default, for pure batch traffic; see README "Scheduler")
#   TTFT_ORDER    which overdue request the interactive guard serves first (default shortest = fewest prompt chunks
#                 left; oldest = the hyperflare-glm-20261112 behaviour; empty = do not pass the flag)
#   STACK_LIMIT   CUDA per-thread stack limit in bytes (FLARE_CUDA_STACK_LIMIT; default 256; empty = the driver
#                 default). The load refuses a value below the largest registered kernel stack
#   GPUS, PORT, HOST   default 0,1 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-glm53-fp4-local}
ALPHA=${ALPHA-}
SL=${STACK_LIMIT-256}; TO=${TTFT_ORDER-shortest}
ENV=() EXTRA=()
[ -n "${BUDGET:-}" ] && EXTRA+=(--mtp-max-verification-token-budget "$BUDGET")
[ -n "$ALPHA" ] && EXTRA+=(--mtp-adaptive-verification-alpha "$ALPHA")
[ -n "$TO" ] && EXTRA+=(--scheduler-ttft-order "$TO")
[ -n "$SL" ] && ENV+=(FLARE_CUDA_STACK_LIMIT=$SL)
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env ${ENV[@]+"${ENV[@]}"} "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name GLM-5.3-Flash --prefix-cache-memory "${PREFIX_CACHE:-32G}" \
  --mtp-draft-tokens "${DRAFTS:-5}" --scheduler-policy "${SCHEDULER:-interactive}" ${EXTRA[@]+"${EXTRA[@]}"}
