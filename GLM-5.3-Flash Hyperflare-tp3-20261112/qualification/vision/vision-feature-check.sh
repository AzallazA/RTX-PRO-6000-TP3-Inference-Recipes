#!/bin/bash
# GLM-5.3-Flash vision tower OUTPUT FEATURES: old (upstream vision dataflow + staging) vs lean (in-place tower), one GPU.
#   OLD=<worktree> LEAN=<worktree> FLARE_GLM53_SNAPSHOT=<checkpoint dir> [PY=<venv python>] \
#     vision-feature-check.sh RESULT_DIR [extra vision_feature_dump.py args, e.g. --no-stage / --compiler none]
# OLD/LEAN are fork checkouts (each with its .venv and, if exported once, its .b12x-overlay); PY defaults to LEAN's venv.
# Runs dump(old), dump(lean), dump(lean again) as separate processes on GPU $GPU (default 0), then
#   cmp old vs lean        (expect rounding-level or better)
#   cmp lean vs lean-again (expect bit-identical: run-to-run stability)
# Writes RESULT_DIR/{old,lean,lean2}.pt + .log, cmp-*.json, SUMMARY.txt (the two VISION_FEATURES lines).
# Exit 0 only if every dump succeeded and both comparisons meet their expectation. ~4 min per dump expected
# (weights ~0.4 GiB, two graphs: 4096 and 32768 patches); each dump is capped at DUMP_TIMEOUT (default 900 s).
# Needs GPU $GPU otherwise idle (~3-4 GiB VRAM, 0.4 GiB pinned host).
set -uo pipefail
R=${1:?usage: vision-feature-check.sh RESULT_DIR [dump args]}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
OLD=${OLD:?set OLD to the worktree with the previous vision dataflow}
LEAN=${LEAN:?set LEAN to the worktree with the lean vision tower}
PY=${PY:-$LEAN/.venv/bin/python}
GPU=${GPU:-0}
DUMP_TIMEOUT=${DUMP_TIMEOUT:-900}
export FLARE_GLM53_SNAPSHOT=${FLARE_GLM53_SNAPSHOT:?set FLARE_GLM53_SNAPSHOT to the checkpoint directory}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}; export PATH=$CUDA_HOME/bin:$PATH HF_HUB_OFFLINE=1
KC=$HOME/.cache/flare/kernels
export TRITON_CACHE_DIR=$KC/triton B12X_COMPILE_CACHE_DIR=$KC/b12x TORCHINDUCTOR_CACHE_DIR=$KC/inductor
unset FLARE_GLM53_VISION_STAGE B12X_OVERLAY
mkdir -p "$R" || exit 2
START=$(date +%s)
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$R/SUMMARY.txt"; }
log "vision feature check: old=$OLD ($(git -C "$OLD" log -1 --format=%h)) lean=$LEAN ($(git -C "$LEAN" log -1 --format=%h)) GPU=$GPU args: $*"

dump(){ # dump NAME WORKTREE
  local name=$1 wt=$2 overlay=
  [ -d "$wt/.b12x-overlay/b12x_tree" ] && overlay="$wt/.b12x-overlay:"
  local t0=$(date +%s)
  ( cd /tmp && exec timeout "$DUMP_TIMEOUT" /usr/bin/env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$overlay$wt" \
      "$PY" "$HERE/vision_feature_dump.py" --out "$R/$name.pt" --label "$name" "${@:3}" ) > "$R/$name.log" 2>&1
  local rc=$?
  local last; last=$(grep -E 'saved .*problems' "$R/$name.log" | tail -1)
  [ $rc = 0 ] && [ -n "$last" ] || last="FAILED (rc $rc$([ $rc = 124 ] && echo ", timeout ${DUMP_TIMEOUT} s")): $(tail -1 "$R/$name.log")"
  log "dump $name $(( $(date +%s) - t0 )) s: $last"
  return $rc
}

FAIL=0
dump old "$OLD" "$@" || FAIL=1
dump lean "$LEAN" "$@" || FAIL=1
dump lean2 "$LEAN" "$@" || FAIL=1
cmp(){ # cmp A B LABEL EXPECT
  [ -f "$R/$1.pt" ] && [ -f "$R/$2.pt" ] || { log "VISION_FEATURES: $3 SKIPPED (missing dump)"; return 1; }
  local line
  line=$(cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" "$HERE/vision_feature_cmp.py" "$R/$1.pt" "$R/$2.pt" --label "$3" \
         --expect "$4" --json "$R/cmp-$3.json" 2> "$R/cmp-$3.err")
  local rc=$?
  log "${line:-VISION_FEATURES: $3 cmp failed rc=$rc: $(tail -1 "$R/cmp-$3.err")}"
  return $rc
}
cmp old lean old-vs-lean rounding || FAIL=1
cmp lean lean2 lean-vs-lean2 identical || FAIL=1
log "RESULT $([ $FAIL = 0 ] && echo PASS || echo FAIL) in $(( $(date +%s) - START )) s ($R)"
# last line = what the queue log shows: verdict + the old-vs-lean comparison
log "VISION_FEATURES: RESULT $([ $FAIL = 0 ] && echo PASS || echo FAIL) | $(grep -h 'VISION_FEATURES: old-vs-lean' "$R/SUMMARY.txt" | tail -1 | sed 's/.*VISION_FEATURES: //' | cut -c1-220)"
exit $FAIL
