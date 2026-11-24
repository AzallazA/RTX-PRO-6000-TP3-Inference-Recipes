#!/bin/bash
# Preflight for building and serving this release. Read-only: it changes nothing and starts nothing on the GPUs.
#   usage: ./check_env.sh <checkout dir>    a Flare fork checkout or the upstream clone for recipe source.patch
#   env:   CUDA_HOME     CUDA toolkit root (default /usr/local/cuda-13.1 if present, else the nvcc on PATH)
#          FLARE_CACHE   where packages and the kernel tuning cache go (default ~/.cache/flare)
#          MIN_FREE_GB   free space required on the drive holding FLARE_CACHE (default 1100: DeepSeek checkpoint ~510 GB,
#                        package ~505 GB (469 GiB), tuning cache 60-80 GB; lower it if the checkpoint lives on another drive)
#          GPUS          three distinct comma-separated GPU indices (default 0,1,2)
# Prints one PASS / WARN / FAIL line per check and exits 1 if anything FAILed. WARN lines are steps not done yet.
set -uo pipefail
FORK=${1:?usage: check_env.sh <checkout dir>}
HERE=$(dirname "$(readlink -f "$0")")
FLARE_CACHE=${FLARE_CACHE:-$HOME/.cache/flare}; MIN_FREE_GB=${MIN_FREE_GB:-1100}; GPUS=${GPUS:-0,1,2}
# Upstream Git revision targeted by the recipe's source.patch.
BASE=${BASE:-70069281cb3ace4365b86f4bfbbcb409da89434b}
fails=0
pass(){ echo "PASS  $*"; }; warn(){ echo "WARN  $*"; }; fail(){ echo "FAIL  $*"; fails=$((fails+1)); }
vge(){ [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # vge A B: version A >= B

# --- CUDA toolkit: 13.1+ with nvcc that targets SM120 (sm_120a). The export compiles the BRRR layer kernel with it.
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda-13.1/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda-13.1; fi
NVCC=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
[ -n "$NVCC" ] && [ -x "$NVCC" ] || NVCC=$(command -v nvcc || true)
if [ -z "$NVCC" ]; then
  fail "nvcc not found: install CUDA 13.1 and export CUDA_HOME=/usr/local/cuda-13.1 PATH=\$CUDA_HOME/bin:\$PATH"
else
  v=$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
  if vge "$v" 13.1; then pass "CUDA toolkit $v ($NVCC)"; else fail "CUDA toolkit $v at $NVCC; 13.1 or newer is required (12.x cannot target SM120)"; fi
  if "$NVCC" --list-gpu-code 2>/dev/null | grep -q '^sm_120'; then pass "nvcc accepts sm_120 (sm_120a)"; else fail "nvcc at $NVCC does not list sm_120"; fi
  if [ "${CUDA_HOME:-}" != "$(dirname "$(dirname "$NVCC")")" ]; then
    warn "CUDA_HOME is not set to this toolkit; before exporting: export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH=\$CUDA_HOME/bin:\$PATH"
  fi
fi

# --- locked memory: Flare pins host memory (cold experts, Engram tables, prefix cache)
if [ "$(ulimit -l)" = unlimited ]; then pass "memlock unlimited"
else fail "ulimit -l is $(ulimit -l); add '<user> - memlock unlimited' in /etc/security/limits.d/90-memlock.conf and log in again"; fi

# --- toolchain
if command -v rustc >/dev/null; then
  v=$(rustc --version | awk '{print $2}')
  if vge "$v" 1.88; then pass "rustc $v (fork requires >= 1.88)"; else fail "rustc $v; the fork requires 1.88 or newer (rustup update)"; fi
else fail "rustc not found (install rustup)"; fi
if command -v cargo >/dev/null; then pass "cargo $(cargo --version | awk '{print $2}')"; else fail "cargo not found"; fi
if command -v uv >/dev/null; then pass "uv $(uv --version | awk '{print $2}')"; else fail "uv not found (https://docs.astral.sh/uv/)"; fi
if command -v patch >/dev/null; then pass "patch present (applies tp3-patches/)"; else fail "patch not found (install the patch package)"; fi
command -v git >/dev/null || fail "git not found"

# --- fork checkout and source.patch
if ! checkout=$(git -C "$FORK" rev-parse --show-toplevel 2>/dev/null) ||
   ! head=$(git -C "$FORK" rev-parse --verify HEAD 2>/dev/null); then
  fail "$FORK is not a git checkout (EXPORT.md step 2)"
elif [ "$HERE" = "$checkout/tp3-tools/release" ]; then
  if ! git -C "$FORK" cat-file -e HEAD:tp3-tools/release/check_env.sh 2>/dev/null ||
     ! git -C "$FORK" cat-file -e HEAD:tp3-patches/flashinfer-tuning-world3.patch 2>/dev/null; then
    fail "checkout ${head:0:8} lacks the committed TP3 release tools or FlashInfer patch"
  elif git -C "$FORK" merge-base --is-ancestor "$BASE" "$head" 2>/dev/null; then
    pass "TP3 fork checkout ${head:0:8} contains upstream base ${BASE:0:8}"
  elif [ "$(git -C "$FORK" rev-parse --is-shallow-repository)" = true ] &&
       git -C "$FORK" tag --points-at "$head" | grep -E '^hyperflare-tp3-[0-9]{8}(\.[0-9]+)?$' >/dev/null; then
    pass "shallow TP3 release checkout ${head:0:8} has a release tag at HEAD"
  else
    fail "cannot verify TP3 fork checkout ${head:0:8}; use the tagged release or fetch history through ${BASE:0:8}"
  fi
else
  if [ "$head" = "$(git -C "$FORK" rev-parse --verify "$BASE^{commit}" 2>/dev/null)" ]; then
    pass "checkout at upstream Flare base ${BASE:0:8}"
  elif git -C "$FORK" merge-base --is-ancestor "$BASE" "$head" 2>/dev/null; then
    warn "checkout ${head:0:8} descends from ${BASE:0:8}; checking for the recipe patch"
  else
    fail "checkout ${head:0:8} requires upstream base $BASE (EXPORT.md step 2)"
  fi
  source_patch=$HERE/source.patch
  if [ ! -f "$source_patch" ] || [ ! -r "$source_patch" ]; then
    fail "recipe source patch missing or unreadable at $source_patch"
  elif git -C "$FORK" apply --check --reverse "$source_patch" 2>/dev/null; then
    pass "source.patch applied"
  elif git -C "$FORK" apply --check "$source_patch" 2>/dev/null; then
    warn "source.patch applies cleanly; run git -C '$FORK' apply '$source_patch'"
  else
    fail "source.patch neither applied nor applicable; check the base revision and local changes"
  fi
fi

# --- Python environment: uv-managed Python 3.12, FlashInfer patches, ignition.runtime_patches
VENV=$FORK/.venv
if [ -x "$VENV/bin/python" ]; then
  pv=$("$VENV/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  case $pv in 3.12.*) pass "fork .venv Python $pv" ;; *) fail ".venv Python is $pv; Flare pins 3.12 (uv sync --locked --extra prepare --extra test)" ;; esac
  SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  IMPL=$SP/flashinfer/experimental/b12x/moe/fused_moe/_impl.py
  if [ ! -f "$IMPL" ]; then
    fail "FlashInfer is not installed in .venv (uv sync --locked --extra prepare --extra test)"
  else
    # tiny_decode must honor swiglu_limit, or replies corrupt mid-way at TP3 (flare#193). FlashInfer aa6c4221 (pinned
    # since upstream #291) clamps inside the kernel (_tiny_decode_swiglu_supported); older pins needed the fork's
    # flashinfer-tiny-decode-swiglu.patch, which excluded tiny_decode whenever a limit was set (dropped 2026-11-14).
    if grep -q '_tiny_decode_swiglu_supported' "$IMPL"; then pass "tiny_decode honors swiglu_limit (FlashInfer applies the clamp in the kernel)"
    elif grep -q 'tiny_decode implements plain SiLU-GLU only' "$IMPL"; then pass "tiny_decode honors swiglu_limit (flashinfer-tiny-decode-swiglu.patch content present)"
    else fail "tiny_decode ignores swiglu_limit in $IMPL (replies corrupt at TP3): FlashInfer predates aa6c4221 and the old swiglu patch is not applied"; fi
    # The release folder carries its own tp3-patches/; run from the fork (tp3-tools/release/), check the fork's.
    PATCHES=$HERE/tp3-patches; [ -d "$PATCHES" ] || PATCHES=$FORK/tp3-patches
    for p in "$PATCHES"/*.patch; do
      b=$(basename "$p")
      if patch -R --dry-run -f -s -p1 -d "$SP" < "$p" >/dev/null 2>&1; then pass "$b applied"
      elif patch --dry-run -f -s -p1 -d "$SP" < "$p" >/dev/null 2>&1; then fail "$b not applied (applies cleanly): patch -p1 -d $SP < $p"
      else fail "$b neither applied nor applicable to the installed FlashInfer (wrong version? re-run uv sync --locked, then re-apply)"; fi
    done
    # The b12x two-shot overlay patch is not applied to the venv: export.sh needs a separate overlay built from it
    # (tp3-tools/make-b12x-overlay.sh, B12X_OVERLAY) while TWOSHOT=1 or GATHER_PULL=1, the export defaults.
    OV=$PATCHES/overlay/flashinfer-pcie-twoshot-pull-world3.patch
    if [ -f "$OV" ]; then pass "overlay patch present: $(basename "$OV") (build B12X_OVERLAY with tp3-tools/make-b12x-overlay.sh)"
    else warn "overlay patch $OV missing: export.sh then needs TWOSHOT=0 (upstream all-reduce candidates)"; fi
    # The DCP gather pull patch (from 20261121) goes into the same overlay; export.sh refuses GATHER_PULL=1 without it.
    GP=$PATCHES/overlay/flashinfer-dcp-gather-pull-world3.patch
    if [ -f "$GP" ]; then pass "overlay patch present: $(basename "$GP") (build the overlay with both patches)"
    else warn "overlay patch $GP missing: export.sh then needs GATHER_PULL=0"; fi
  fi
  # Upstream replaced `python -m ignition.patches` (on-disk edits of torch/Triton, re-run after every uv sync) with
  # ignition.runtime_patches: version-checked in-memory fixes that ignition export applies itself. Nothing to run;
  # check that the module is there and accepts the installed torch/Triton (it raises on a version mismatch).
  if (cd "$FORK" && "$VENV/bin/python" -c 'from ignition import runtime_patches as r; r.apply(); print(r.identity())') >/dev/null 2>&1; then
    pass "ignition.runtime_patches import and apply (torch Inductor and Triton fixes, in memory)"
  else fail "ignition.runtime_patches failed to import or apply: (cd $FORK && .venv/bin/python -c 'from ignition import runtime_patches as r; r.apply()') shows why; usually uv sync --locked --extra prepare --extra test"; fi
  if [ -x "$VENV/bin/ignition" ]; then pass "ignition CLI present"; else warn ".venv/bin/ignition missing (uv sync --locked --extra prepare --extra test)"; fi
  if [ -x "$VENV/bin/ninja" ]; then pass "ninja present (BRRR kernel compile)"; else warn ".venv/bin/ninja missing; the BRRR kernel compile needs it (uv sync --locked --extra prepare --extra test)"; fi
  tv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || echo missing)
  fv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))' 2>/dev/null || echo missing)
  cv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))' 2>/dev/null || echo missing)
  echo "      torch $tv, flashinfer-python $fv, nvidia-cutlass-dsl $cv (release: 2.14.1+cu132, 0.7.3, 4.8.1; see versions.txt). A cu132 torch wheel next to the"
  echo "      13.1 toolkit is expected: the wheel ships its own CUDA libraries, nvcc 13.1 only compiles the BRRR kernel."
else
  warn "no .venv in $FORK yet: uv sync --locked --extra prepare --extra test, then the FlashInfer patches (EXPORT.md step 2); re-run this check afterwards"
fi
# The flare binary moved to flare-cli/ upstream (flare/ is now the shared library crate).
if [ -x "$FORK/flare-cli/target/release/flare" ]; then pass "flare binary built: $FORK/flare-cli/target/release/flare"
else warn "flare binary not built yet: cargo build --locked --release --manifest-path $FORK/flare-cli/Cargo.toml"; fi

# --- disk
d=$FLARE_CACHE; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
free=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
if [ "$free" -ge "$MIN_FREE_GB" ]; then pass "$free GB free on the drive holding $FLARE_CACHE (need $MIN_FREE_GB)"
else fail "$free GB free on the drive holding $FLARE_CACHE; $MIN_FREE_GB GB needed (set MIN_FREE_GB if the checkpoints are on another drive)"; fi

# --- GPUs: 3 visible SM120 cards, idle. Kernel tuning captures the power limit and clocks in effect during the export.
IFS=, read -r gpu0 gpu1 gpu2 <<< "$GPUS"
if [[ ! $GPUS =~ ^[0-9]+,[0-9]+,[0-9]+$ || $gpu0 = "$gpu1" || $gpu0 = "$gpu2" || $gpu1 = "$gpu2" ]]; then
  fail "GPUS=$GPUS must specify exactly 3 distinct comma-separated GPU indices"
elif ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found (NVIDIA driver 615.71.09 or any driver supporting CUDA 13.1)"
elif ! gpu_rows=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.used,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader,nounits); then
  fail "nvidia-smi could not report visible GPUs"
else
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  n=0; bad=0
  want=("$gpu0" "$gpu1" "$gpu2")
  while IFS=, read -r idx name cc used plim smmax memmax; do
    idx=${idx// /}; cc=${cc// /}; used=${used// /}; plim=${plim// /}; smmax=${smmax// /}; memmax=${memmax// /}; name=${name# }
    for i in "${!want[@]}"; do
      [ "${want[i]}" = "$idx" ] || continue
      unset 'want[i]'
      n=$((n+1))
      [ "$cc" = "12.0" ] || { fail "GPU $idx: compute capability $cc, need 12.0 (SM120, RTX PRO 6000 Blackwell)"; bad=1; }
      [ "$used" -lt 1024 ] || { fail "GPU $idx: $used MiB in use; stop the server and other GPU processes before exporting"; bad=1; }
      echo "      GPU $idx: $name · power limit $plim W · max clocks SM $smmax MHz, memory $memmax MHz · $used MiB used"
      break
    done
  done <<< "$gpu_rows"
  if [ "$n" -ne 3 ]; then fail "$n of the 3 GPUs in GPUS=$GPUS are visible"
  elif [ "$bad" = 0 ]; then pass "3 SM120 GPUs ($GPUS) visible and idle, driver $drv"; fi
  echo "      Export with the power limit and clock offsets you will serve with: the tuning cache records kernels measured"
  echo "      at those settings. The release numbers were taken at 450 W, +75 MHz core, +6000 MHz memory offset; 350 W is 10-14% slower."
fi

echo
if [ "$fails" = 0 ]; then echo "All checks passed."; else echo "$fails check(s) FAILED."; exit 1; fi
