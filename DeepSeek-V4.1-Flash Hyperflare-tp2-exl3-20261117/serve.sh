#!/bin/bash
# Serve the DeepSeek-V4.1-Flash EXL3 TP2 package on 2 GPUs (8 request slots), exactly as measured in README.md.
# With the Engram tables in host RAM the server commits ~141 GiB of host RAM (README "Memory"); ENGRAM=one or disk for less.
#   FLARE         path to this fork's flare binary   (default: flare-cli/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the pack directory name the package was exported from), name@version, or a package
#                 path (default DeepSeek-V4.1-Flash-1017-EXL3-3bpw = the --local-dir name in EXPORT.md)
#   ENGRAM        ram (default) = both int4 Engram tables (51.5 GiB each) in host RAM
#                 one = the layer-14 table on disk, the other in RAM (= disk + FLARE_ENGRAM_RAM_LAYERS=0): ~41 GiB
#                 less RAM, no measurable speed cost
#                 disk = both tables read from the package on disk through a row cache: ~87 GiB less RAM, decode
#                 -1 to -3%, prefill -10 to -20%; needs a fast NVMe (README "Memory")
#   ENGRAM_CACHE  row-cache size for ENGRAM=disk (default empty = Hyperflare's default)
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 8G)
#   BUDGET        max verification rows per step (default 32)
#   DRAFTS        DSpark drafts (default 5); must equal the DRAFTS the package was exported with (--mtp-only package:
#                 it refuses 0)
#   ALPHA         adaptive-verification alpha (default 0.8)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target; balanced = upstream)
#   TTFT_ORDER    which overdue request the interactive guard serves first (default shortest = fewest prompt chunks
#                 left; oldest = Hyperflare's order; empty = do not pass the flag)
#   TAIL_MERGE    --scheduler-prefill-tail-merge (default true)
#   STACK_LIMIT   CUDA per-thread stack limit in bytes (FLARE_CUDA_STACK_LIMIT; default 128: the largest kernel this
#                 package registers needs 120 B and the load refuses a limit below it; empty = the driver default
#                 1,024 B, fewer KV tokens)
#   FAST=0        turn off private Engram tables on 4 KiB pages and fast exit (FLARE_ENGRAM_PRIVATE_TABLES,
#                 FLARE_FAST_EXIT; on by default: faster start and stop)
#   GPUS, PORT, HOST   default 0,1 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare-cli/target/release/flare}
PACKAGE=${PACKAGE:-DeepSeek-V4.1-Flash-1017-EXL3-3bpw}
ENV=() EXTRA=()
case ${ENGRAM:-ram} in
  ram) EXTRA+=(--embedding-table-storage ram) ;;
  disk|one) EXTRA+=(--embedding-table-storage disk)
        [ "$ENGRAM" = one ] && ENV+=(FLARE_ENGRAM_RAM_LAYERS=0)
        [ -n "${ENGRAM_CACHE:-}" ] && EXTRA+=(--embedding-table-cache-size "$ENGRAM_CACHE") ;;
  *) echo "ENGRAM must be ram, one or disk" >&2; exit 2 ;;
esac
[ "${FAST:-1}" = 1 ] && ENV+=(FLARE_ENGRAM_PRIVATE_TABLES=1 FLARE_FAST_EXIT=1)
SL=${STACK_LIMIT-128}; TO=${TTFT_ORDER-shortest}
[ -n "$SL" ] && ENV+=(FLARE_CUDA_STACK_LIMIT=$SL)
[ -n "$TO" ] && EXTRA+=(--scheduler-ttft-order "$TO")
# Do not also set FLARE_ENGRAM_RAM_TABLES: it overrides --embedding-table-storage per table.
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env -u CUDA_VISIBLE_DEVICES ${ENV[@]+"${ENV[@]}"} "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1}" \
  --port "${PORT:-15015}" --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash \
  --prefix-cache-memory "${PREFIX_CACHE:-8G}" --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification \
  --mtp-max-verification-token-budget "${BUDGET:-32}" --mtp-adaptive-verification-alpha "${ALPHA:-0.8}" \
  --scheduler-policy "${SCHEDULER:-interactive}" --scheduler-prefill-tail-merge "${TAIL_MERGE:-true}" "${EXTRA[@]}"
