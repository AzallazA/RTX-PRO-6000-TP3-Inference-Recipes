#!/bin/bash
# Serve the exported package on 3 GPUs with both FP8 Engram tables in host RAM (~189 GiB of tables; the server
# commits ~218 GiB of host RAM in total at PREFIX_CACHE=12G, so this needs a 256 GB machine, see README "Memory").
#   FLARE         path to this fork's flare binary   (default: flare-cli/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the checkpoint directory name the package was exported from) or a package path
#                 (default ds41-1017-engram-fp8 = the view name in EXPORT.md; Hyperflare picks its default version)
#   ENGRAM_RAM    1 (default) = both Engram tables in host RAM
#                 layer1 = layer 1's table in RAM, layer 14's read from disk through a row cache of ENGRAM_CACHE
#                 (default 32GiB); ~60 GiB less committed host RAM (README "Memory")
#                 0 = both tables from disk through a row cache of ENGRAM_CACHE (default 32GiB, shared by the tables)
#   READ_AHEAD    Legacy CPU pre-chunk Engram read-ahead (FLARE_ENGRAM_READ_AHEAD; default empty = use upstream #566+#70
#                 graph-driven advisory lookahead prefetch in disk mode; set READ_AHEAD=1 for the legacy CPU reader)
#   ENGRAM_PREFETCH  1 (default) = graph-driven advisory disk lookahead prefetch (#566+#70); 0 = pass --no-embedding-table-prefetch
#   STEERING_KEYS_FILE  file with steering keys (FLARE_STEERING_API_KEYS_FILE; one key per line or comma-separated,
#                 '#' comment lines, mode 600). Unset by default: any client may then set steering_scale.
#                 With keys, a nonzero steering_scale needs X-Steering-Key (README "Steering access")
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 12G; 32G does not fit beside the FP8 tables in 256 GB)
#   BUDGET        max verification rows per step (default 64)
#   DRAFTS        DSpark drafts (default 5); must equal the DRAFTS the package was exported with
#   ALPHA         adaptive-verification alpha scalar or comma-separated max_batch:alpha schedule (default 1.0)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target; balanced = upstream)
#   TTFT_ORDER    which overdue request the interactive guard serves first (default shortest = fewest prompt chunks
#                 left; oldest = the 20261111 behaviour; empty = do not pass the flag)
#   TAIL_MERGE    --scheduler-prefill-tail-merge (default true)
#   PREFILL_MERGE short message pieces at a request's own checkpoint merge into the next chunk up to this many tokens
#                 (FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS; default 512; empty or 0 = off)
#   HISTORY_DRAFTS  1 (default) = --history-drafts (prompt/history suffix-automaton copy drafts with wide copy-verify
#                 graphs); 0 = --no-history-drafts
#   STACK_LIMIT   CUDA per-thread stack limit in bytes (FLARE_CUDA_STACK_LIMIT; default 48; set 64 if an older nvcc
#                 produces >48 B stack; empty = use the limit recorded in the core, 1,024 for the release package)
#   FAST=0        turn off private Engram tables on 4 KiB pages and fast exit (FLARE_ENGRAM_PRIVATE_TABLES,
#                 FLARE_FAST_EXIT; on by default: faster start and stop)
#   GPUS, PORT, HOST   default 0,1,2 / 15015 / 127.0.0.1 (plain HTTP: see README "Over a network" before changing HOST)
#   Usage: serve.sh [--dry-run]. A dry run prints the memory plan for the selected settings.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo "usage: ${0##*/} [--dry-run]" >&2
  exit 2
fi
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-ds41-1017-engram-fp8}
# Start with the wrapper's settings even when a parent service exported Flare overrides.
unset FLARE_ENGRAM_RAM_TABLES FLARE_ENGRAM_RAM_LAYERS FLARE_ENGRAM_PRIVATE_TABLES FLARE_FAST_EXIT
unset FLARE_CUDA_STACK_LIMIT FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS FLARE_ENGRAM_READ_AHEAD FLARE_STEERING_API_KEYS_FILE
ENV=() STORAGE=() EXTRA=()
case "${ENGRAM_RAM:-1}" in
  1) ENV+=(FLARE_ENGRAM_RAM_TABLES=0) ;;
  layer1) ENV+=(FLARE_ENGRAM_RAM_LAYERS=0); STORAGE=(--embedding-table-storage disk --embedding-table-cache-size "${ENGRAM_CACHE:-32GiB}") ;;
  0) STORAGE=(--embedding-table-storage disk --embedding-table-cache-size "${ENGRAM_CACHE:-32GiB}") ;;
  *) echo "ENGRAM_RAM must be 1, layer1 or 0" >&2; exit 2 ;;
esac
[ "${FAST:-1}" = 1 ] && ENV+=(FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_FAST_EXIT=1)
SL=${STACK_LIMIT-48}; PM=${PREFILL_MERGE-512}; TO=${TTFT_ORDER-shortest}; RA=${READ_AHEAD-}
[ -n "$SL" ] && ENV+=("FLARE_CUDA_STACK_LIMIT=$SL")
[ -n "$PM" ] && ENV+=("FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=$PM")
[ -n "$RA" ] && ENV+=("FLARE_ENGRAM_READ_AHEAD=$RA")
[ -n "$TO" ] && EXTRA+=(--scheduler-ttft-order "$TO")
[ "${ENGRAM_PREFETCH:-1}" = 0 ] && STORAGE+=(--no-embedding-table-prefetch)
if [ "${HISTORY_DRAFTS:-1}" = 1 ]; then
  EXTRA+=(--history-drafts)
else
  EXTRA+=(--no-history-drafts)
fi
if [ -n "${STEERING_KEYS_FILE:-}" ]; then
  [ -r "$STEERING_KEYS_FILE" ] || { echo "STEERING_KEYS_FILE $STEERING_KEYS_FILE is not readable" >&2; exit 2; }
  ENV+=(FLARE_STEERING_API_KEYS_FILE="$STEERING_KEYS_FILE")
fi
exec /usr/bin/env "${ENV[@]}" "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory "${PREFIX_CACHE:-12G}" \
  --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification --mtp-max-verification-token-budget "${BUDGET:-64}" \
  --mtp-adaptive-verification-alpha "${ALPHA:-1.0}" --scheduler-policy "${SCHEDULER:-interactive}" \
  --scheduler-prefill-tail-merge "${TAIL_MERGE:-true}" \
  "${EXTRA[@]}" "${STORAGE[@]}" "$@"
