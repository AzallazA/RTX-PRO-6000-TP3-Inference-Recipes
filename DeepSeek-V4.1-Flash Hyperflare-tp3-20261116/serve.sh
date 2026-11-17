#!/bin/bash
# Serve the exported package on 3 GPUs with both FP8 Engram tables in host RAM (~189 GiB of tables; the server
# commits ~217 GiB of host RAM in total at PREFIX_CACHE=12G, so this needs a 256 GB machine, see README "Memory").
#   FLARE         path to this fork's flare binary   (default: flare-cli/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the checkpoint directory name the package was exported from) or a package path
#                 (default ds41-1017-engram-fp8 = the view name in EXPORT.md; Hyperflare picks its default version)
#   ENGRAM_RAM    1 (default) = both Engram tables in host RAM
#                 layer1 = layer 1's table in RAM, layer 14's read from disk through a row cache of ENGRAM_CACHE (default
#                 32GiB): ~60 GiB less committed host RAM at the same prefix cache (177.8 vs 237.3 GiB at 32G), decode
#                 level, 8-stream agent replay output -18% (README "Memory"); allows PREFIX_CACHE=32G
#                 0 = both tables from disk (not measured with the FP8 tables)
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 12G; 32G does not fit beside the FP8 tables in 256 GB)
#   BUDGET        max verification rows per step (default 32)
#   DRAFTS        DSpark drafts (default 5); must equal the DRAFTS the package was exported with
#   ALPHA         adaptive-verification alpha (default 0.8)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target; balanced = upstream)
#   TTFT_ORDER    which overdue request the interactive guard serves first (default shortest = fewest prompt chunks
#                 left; oldest = the 20261111 behaviour; empty = do not pass the flag)
#   TAIL_MERGE    --scheduler-prefill-tail-merge (default true)
#   PREFILL_MERGE short message pieces at a request's own checkpoint merge into the next chunk up to this many tokens
#                 (FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS; default 512; empty or 0 = off)
#   STACK_LIMIT   CUDA per-thread stack limit in bytes (FLARE_CUDA_STACK_LIMIT; default 256 = +247,552 KV tokens on
#                 the release package; empty = the recorded driver default 1,024)
#   FAST=0        turn off private Engram tables on 4 KiB pages and fast exit (FLARE_ENGRAM_PRIVATE_TABLES,
#                 FLARE_FAST_EXIT; on by default: faster start and stop)
#   GPUS, PORT, HOST   default 0,1,2 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-ds41-1017-engram-fp8}
ENV=() STORAGE=() EXTRA=()
case ${ENGRAM_RAM:-1} in
  1) ENV+=(FLARE_ENGRAM_RAM_TABLES=0) ;;
  layer1) ENV+=(FLARE_ENGRAM_RAM_LAYERS=0); STORAGE=(--embedding-table-cache-size "${ENGRAM_CACHE:-32GiB}") ;;
  0) STORAGE=(--embedding-table-storage disk) ;;
  *) echo "ENGRAM_RAM must be 1, layer1 or 0" >&2; exit 2 ;;
esac
[ "${FAST:-1}" = 1 ] && ENV+=(FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_FAST_EXIT=1)
SL=${STACK_LIMIT-256}; PM=${PREFILL_MERGE-512}; TO=${TTFT_ORDER-shortest}
[ -n "$SL" ] && ENV+=(FLARE_CUDA_STACK_LIMIT=$SL)
[ -n "$PM" ] && ENV+=(FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=$PM)
[ -n "$TO" ] && EXTRA+=(--scheduler-ttft-order "$TO")
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env "${ENV[@]}" "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory "${PREFIX_CACHE:-12G}" \
  --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification --mtp-max-verification-token-budget "${BUDGET:-32}" \
  --mtp-adaptive-verification-alpha "${ALPHA:-0.8}" --scheduler-policy "${SCHEDULER:-interactive}" \
  --scheduler-prefill-tail-merge "${TAIL_MERGE:-true}" "${EXTRA[@]}" "${STORAGE[@]}"
