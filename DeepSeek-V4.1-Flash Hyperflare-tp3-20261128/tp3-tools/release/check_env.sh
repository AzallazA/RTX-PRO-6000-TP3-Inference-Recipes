#!/bin/bash
# Preflight for building and serving the DS41 TP3 Wave 15.1 (2026-11-28) release.
# Read-only: changes nothing and starts nothing on the GPUs.
#   usage: ./check_env.sh <workspace dir>   directory holding flare-core/, hyperflare-recipes/, flashinfer/, b12x-overlay/
#   env:   CUDA_HOME     CUDA toolkit root (default /usr/local/cuda if present, else nvcc on PATH)
#          FLARE_CACHE   where packages and the kernel tuning cache go (default ~/.cache/flare)
#          MIN_FREE_GB   free space required on the drive holding FLARE_CACHE (default 1100)
#          GPUS          three distinct comma-separated GPU indices (default 0,1,2)
# Prints one PASS / WARN / FAIL line per check and exits 1 if anything FAILed.
set -uo pipefail
WS=${1:?usage: check_env.sh <workspace dir>}
FLARE_CACHE=${FLARE_CACHE:-$HOME/.cache/flare}; MIN_FREE_GB=${MIN_FREE_GB:-1100}; GPUS=${GPUS:-0,1,2}
fails=0
pass(){ echo "PASS  $*"; }; warn(){ echo "WARN  $*"; }; fail(){ echo "FAIL  $*"; fails=$((fails+1)); }
vge(){ [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }

# --- CUDA toolkit: 13.1+ with nvcc that targets SM120 (sm_120a) for BRRR kernel compilation
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda; fi
NVCC=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
[ -n "$NVCC" ] && [ -x "$NVCC" ] || NVCC=$(command -v nvcc || true)
if [ -z "$NVCC" ]; then
  fail "nvcc not found: install CUDA 13.1+ and export CUDA_HOME=/usr/local/cuda PATH=\$CUDA_HOME/bin:\$PATH"
else
  v=$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
  if vge "$v" 13.1; then pass "CUDA toolkit $v ($NVCC)"; else fail "CUDA toolkit $v at $NVCC; 13.1+ is required (12.x cannot target SM120)"; fi
  if "$NVCC" --list-gpu-code 2>/dev/null | grep -q '^sm_120'; then pass "nvcc accepts sm_120 (sm_120a)"; else fail "nvcc at $NVCC does not list sm_120"; fi
  if [ "${CUDA_HOME:-}" != "$(dirname "$(dirname "$NVCC")")" ]; then
    warn "CUDA_HOME is not set to this toolkit; before exporting: export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH=\$CUDA_HOME/bin:\$PATH"
  fi
fi

# --- locked memory: Hyperflare pins host memory (cold experts, Engram tables, prefix cache)
if [ "$(ulimit -l)" = unlimited ]; then pass "memlock unlimited"
else fail "ulimit -l is $(ulimit -l); add '<user> - memlock unlimited' in /etc/security/limits.d/90-memlock.conf and log in again"; fi

# --- toolchain
if command -v rustc >/dev/null; then
  v=$(rustc --version | awk '{print $2}')
  if vge "$v" 1.88; then pass "rustc $v (requires >= 1.88 if building flare from source)"; else warn "rustc $v (< 1.88); needed only if building flare from source (prebuilt flare-linux-amd64.tgz needs no Rust)"; fi
else warn "rustc not found (optional if using prebuilt flare-linux-amd64.tgz)"; fi
if command -v uv >/dev/null; then pass "uv $(uv --version | awk '{print $2}')"; else fail "uv not found (https://docs.astral.sh/uv/)"; fi
command -v git >/dev/null || fail "git not found"

# --- 3-repo workspace layout (EXPORT.md step 3)
FLARE_CORE=$WS/flare-core
RECIPES=$WS/hyperflare-recipes
FLASHINFER=$WS/flashinfer
OVERLAY=$WS/b12x-overlay

if [ -d "$FLARE_CORE/.git" ] || [ -f "$FLARE_CORE/.git" ]; then
  pass "flare-core checkout ($(git -C "$FLARE_CORE" rev-parse --short HEAD 2>/dev/null || echo unknown))"
else fail "flare-core checkout missing at $FLARE_CORE (EXPORT.md step 3)"; fi

if [ -f "$RECIPES/hyperflare_recipes/recipes/deepseek41/tp3_config.py" ]; then
  pass "patched hyperflare-recipes present at $RECIPES"
else fail "hyperflare-recipes TP3 patch not applied at $RECIPES (EXPORT.md step 3)"; fi

if [ -d "$OVERLAY/b12x_tree/comm/pcie" ] && [ -f "$OVERLAY/b12x/__init__.py" ]; then
  if grep -qs '^def twoshot_pull_enabled(' "$OVERLAY/b12x_tree/comm/pcie/_collective_tuning.py" &&
     grep -qs '^def _gather_pull_enabled(' "$OVERLAY/b12x_tree/comm/pcie/_dcp_a2a_cute.py"; then
    pass "b12x-overlay present with TP3 two-shot pull & DCP gather pull ($OVERLAY)"
  else fail "b12x-overlay at $OVERLAY lacks TP3 flashinfer patches (use AzallazA/flashinfer@tp3-20261128 or apply b12x-tp3-20261128.patch)"; fi
else
  warn "b12x-overlay not built yet at $OVERLAY (EXPORT.md step 3)"
fi

# --- Python environment (.venv in flare-core)
VENV=$FLARE_CORE/.venv
if [ -x "$VENV/bin/python" ]; then
  pv=$("$VENV/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  case "$pv" in 3.12.*) pass "flare-core .venv Python $pv" ;; *) fail ".venv Python is $pv; Hyperflare pins 3.12 (uv sync --locked --extra prepare)" ;; esac
  if (cd "$FLARE_CORE" && "$VENV/bin/python" -c 'from ignition import runtime_patches as r; r.apply(); print(r.identity())') >/dev/null 2>&1; then
    pass "ignition.runtime_patches import and apply"
  else fail "ignition.runtime_patches failed to import or apply (run uv sync --locked --extra prepare in $FLARE_CORE)"; fi
  if [ -x "$VENV/bin/ninja" ]; then pass "ninja present (BRRR kernel compile)"; else warn ".venv/bin/ninja missing (uv sync --locked --extra prepare)"; fi
  tv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || echo missing)
  fv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))' 2>/dev/null || echo missing)
  cv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))' 2>/dev/null || echo missing)
  echo "      torch $tv, flashinfer-python $fv, nvidia-cutlass-dsl $cv (see versions.txt)"
else
  warn "no .venv in $FLARE_CORE yet (run: cd $FLARE_CORE && uv sync --locked --extra prepare)"
fi

# --- flare serving binary
if [ -x "$WS/flare-bin/flare" ]; then pass "prebuilt flare binary found: $WS/flare-bin/flare"
elif [ -x "$FLARE_CORE/flare-cli/target/release/flare" ]; then pass "source-built flare binary found: $FLARE_CORE/flare-cli/target/release/flare"
else warn "flare binary not found yet (download flare-linux-amd64.tgz or run cargo build --locked --release --manifest-path $FLARE_CORE/flare-cli/Cargo.toml)"; fi

# --- disk
d=$FLARE_CACHE; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
free=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
if [ "$free" -ge "$MIN_FREE_GB" ]; then pass "$free GB free on the drive holding $FLARE_CACHE (need $MIN_FREE_GB)"
else fail "$free GB free on the drive holding $FLARE_CACHE; $MIN_FREE_GB GB needed (set MIN_FREE_GB if checkpoints live elsewhere)"; fi

# --- GPUs: 3 visible SM120 cards, idle
IFS=, read -r gpu0 gpu1 gpu2 <<< "$GPUS"
if [[ ! "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+$ || "$gpu0" = "$gpu1" || "$gpu0" = "$gpu2" || "$gpu1" = "$gpu2" ]]; then
  fail "GPUS=$GPUS must specify exactly 3 distinct comma-separated GPU indices"
elif ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found (NVIDIA driver 615.71.09+ supporting CUDA 13.1+)"
elif ! gpu_rows=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.used,power.limit --format=csv,noheader,nounits); then
  fail "nvidia-smi could not report visible GPUs"
else
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  n=0; bad=0
  want=("$gpu0" "$gpu1" "$gpu2")
  while IFS=, read -r idx name cc used plim; do
    idx=${idx// /}; cc=${cc// /}; used=${used// /}; plim=${plim// /}; name=${name# }
    for i in "${!want[@]}"; do
      [ "${want[i]}" = "$idx" ] || continue
      unset 'want[i]'
      n=$((n+1))
      [ "$cc" = "12.0" ] || { fail "GPU $idx: compute capability $cc, need 12.0 (SM120, RTX PRO 6000 Blackwell)"; bad=1; }
      [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 1024 ] || { fail "GPU $idx: memory.used=$used MiB (stop server or reset GPU before exporting)"; bad=1; }
      echo "      GPU $idx: $name · power limit $plim W · $used MiB used"
      break
    done
  done <<< "$gpu_rows"
  if [ "$n" -ne 3 ]; then fail "$n of the 3 GPUs in GPUS=$GPUS are visible"
  elif [ "$bad" = 0 ]; then pass "3 SM120 GPUs ($GPUS) visible and idle, driver $drv"; fi
  echo "      Export with the power limit and clock settings you will serve with; see README.md for 450 W notes."
fi

echo
if [ "$fails" = 0 ]; then echo "All checks passed."; else echo "$fails check(s) FAILED."; exit 1; fi
