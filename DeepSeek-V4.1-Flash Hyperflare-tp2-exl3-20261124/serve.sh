#!/bin/bash
# Serve the DeepSeek-V4.1-Flash EXL3 TP2 2026-11-24 package on 2 GPUs (8 request slots), as measured in README.md.
# With both int4 Engram tables in host RAM the server commits ~151 GiB of host RAM; ENGRAM=one uses ~110 GiB (verified 0.969-0.979 consistency, 0 miss).
#   FLARE         path to the flare binary (default: flare-cli/target/release/flare)
#   PACKAGE       model name, name@version, or package path (default DeepSeek-V4.1-Flash-1017-EXL3-3bpw)
#   ENGRAM        ram (default) = both int4 Engram tables in host RAM
#                 one = layer-1 table in RAM, layer-14 table on disk (with ENGRAM_CACHE, default 4GiB)
#                 disk = both tables on disk through a row cache
#   ENGRAM_CACHE  row-cache size for ENGRAM=one/disk (default 4GiB for one)
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 8G)
#   BUDGET        max verification rows per step (default 32)
#   DRAFTS        DSpark drafts (default 5; must match export)
#   ALPHA         adaptive-verification alpha (default 0.8)
#   SCHEDULER     scheduler policy (default interactive)
#   TTFT_ORDER    overdue request order (default shortest)
#   TAIL_MERGE    --scheduler-prefill-tail-merge (default true)
#   ENCODER_PREFILL --scheduler-encoder-prefill (default true)
#   PREFILL_MERGE FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS (default 512; empty = off)
#   STACK_LIMIT   CUDA per-thread stack limit in bytes (default 144; max kernel stack 136 B)
#   STEERING_KEYS_FILE optional file with steering keys (FLARE_STEERING_API_KEYS_FILE) for sidecar packages
#   FAST=0        turn off private Engram tables and fast exit
#   GPUS, PORT, HOST   default 0,1 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-DeepSeek-V4.1-Flash-1017-EXL3-3bpw}
ENV=() EXTRA=()
case ${ENGRAM:-ram} in
  ram) EXTRA+=(--embedding-table-storage ram) ;;
  one) ENV+=(FLARE_ENGRAM_RAM_LAYERS=0)
       EXTRA+=(--embedding-table-storage disk --embedding-table-cache-size "${ENGRAM_CACHE:-4GiB}") ;;
  disk) EXTRA+=(--embedding-table-storage disk)
        [ -n "${ENGRAM_CACHE:-}" ] && EXTRA+=(--embedding-table-cache-size "$ENGRAM_CACHE") ;;
  *) echo "ENGRAM must be ram, one or disk" >&2; exit 2 ;;
esac
[ "${FAST:-1}" = 1 ] && ENV+=(FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_FAST_EXIT=1)
SL=${STACK_LIMIT-144}; PM=${PREFILL_MERGE-512}; TO=${TTFT_ORDER-shortest}
[ -n "$SL" ] && ENV+=(FLARE_CUDA_STACK_LIMIT=$SL)
[ -n "$PM" ] && ENV+=(FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=$PM)
[ -n "$TO" ] && EXTRA+=(--scheduler-ttft-order "$TO")
if [ -n "${STEERING_KEYS_FILE:-}" ]; then
  [ -r "$STEERING_KEYS_FILE" ] || { echo "STEERING_KEYS_FILE $STEERING_KEYS_FILE is not readable" >&2; exit 2; }
  ENV+=(FLARE_STEERING_API_KEYS_FILE="$STEERING_KEYS_FILE")
fi
exec /usr/bin/env -u CUDA_VISIBLE_DEVICES -u FLARE_ENGRAM_RAM_TABLES -u FLARE_ENGRAM_READ_AHEAD \
  ${ENV[@]+"${ENV[@]}"} "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1}" \
  --port "${PORT:-15015}" --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash \
  --prefix-cache-memory "${PREFIX_CACHE:-8G}" --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification \
  --mtp-max-verification-token-budget "${BUDGET:-32}" --mtp-adaptive-verification-alpha "${ALPHA:-0.8}" \
  --scheduler-policy "${SCHEDULER:-interactive}" --scheduler-prefill-tail-merge "${TAIL_MERGE:-true}" \
  --scheduler-encoder-prefill "${ENCODER_PREFILL:-true}" "${EXTRA[@]}" "$@"
