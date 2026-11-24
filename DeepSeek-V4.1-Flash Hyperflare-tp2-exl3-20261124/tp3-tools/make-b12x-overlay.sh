#!/bin/bash
# Build the b12x overlay GLM exports use (B12X_OVERLAY in glm53-export-tp3.sh): a copy of the venv's (already
# tp3-patches-patched) b12x plus tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch, so the shared venv (also used
# by the DS41 exports) stays unchanged. World-3 pull gather: +7-8 % GLM a16 c1/c4/c8 (2026-11-09, pkg 092215).
#   tp3-tools/make-b12x-overlay.sh [OUT]     default OUT: <worktree>/.b12x-overlay
#   VENV=<dir>  venv whose b12x is copied (default <worktree>/.venv)
#   B12X_OVERLAY_PATCHES="p1 p2"  patches to apply instead of the GLM pull-gather patch (DS41 two-shot overlay:
#     B12X_OVERLAY_PATCHES=$PWD/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch, see overlay/README.md)
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
OUT=${1:-$ROOT/.b12x-overlay}
while [[ "$OUT" == */ ]]; do OUT=${OUT%/}; done
[ -n "$OUT" ] || { echo "Refusing to replace /" >&2; exit 2; }
PARENT=$(dirname -- "$OUT")
mkdir -p -- "$PARENT"
exec {parent_fd}<"$PARENT"
flock -x "$parent_fd"

# Match the overlay layout, allowing the bytecode cache created by Python.
SHIM_ROOT='_root = Path(__file__).resolve().parent.parent / "b12x_tree"'
is_overlay() (
    candidate=$1
    shopt -s dotglob nullglob
    entries=("$candidate"/*)
    shim_entries=("$candidate/b12x"/*)
    if [ -d "$candidate/b12x/__pycache__" ] && [ ! -L "$candidate/b12x/__pycache__" ]; then
        shim_entries_count=2
    else
        shim_entries_count=1
    fi
    [ ! -L "$candidate" ] && [ -d "$candidate/b12x" ] && [ ! -L "$candidate/b12x" ] &&
        [ -d "$candidate/b12x_tree" ] && [ ! -L "$candidate/b12x_tree" ] &&
        [ "${#entries[@]}" -eq 2 ] && [ "${#shim_entries[@]}" -eq "$shim_entries_count" ] &&
        [ -f "$candidate/b12x/__init__.py" ] && [ ! -L "$candidate/b12x/__init__.py" ] &&
        grep -Fxq "$SHIM_ROOT" "$candidate/b12x/__init__.py"
)
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    is_overlay "$OUT" || { echo "Refusing to replace $OUT because it is not a b12x overlay" >&2; exit 2; }
fi

S=$("${VENV:-$ROOT/.venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
STAGE=$(mktemp -d -- "$PARENT/.b12x-overlay.XXXXXXXX")
cleanup() {
    if [ -f "$STAGE/exchanging" ] && ! is_overlay "$STAGE/overlay"; then
        echo "Inspect the b12x overlay staging directory at $STAGE" >&2
        return
    fi
    rm -rf -- "$STAGE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p -- "$STAGE/overlay/b12x"
cp -a -- "$S/flashinfer/experimental/b12x/." "$STAGE/overlay/b12x_tree"
find "$STAGE/overlay/b12x_tree" -name __pycache__ -prune -exec rm -rf {} +
sed "s#^_root = Path(_flashinfer.origin).parent / \"experimental\" / \"b12x\"#$SHIM_ROOT#" \
    "$S/b12x/__init__.py" > "$STAGE/overlay/b12x/__init__.py"
grep -Fxq "$SHIM_ROOT" "$STAGE/overlay/b12x/__init__.py"
if [ "${B12X_OVERLAY_PATCHES+x}" ]; then
    read -r -a PATCH_LIST <<< "${B12X_OVERLAY_PATCHES//$'\n'/ }"
else
    PATCH_LIST=("$ROOT/tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch")
fi
for P in "${PATCH_LIST[@]}"; do
    patch -d "$STAGE/overlay/b12x_tree" -p4 --batch --forward < "$P"
done

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    is_overlay "$OUT" || { echo "Refusing to replace $OUT because it is not a b12x overlay" >&2; exit 2; }
    # Swap sibling directories atomically so imports can keep using the destination.
    touch "$STAGE/exchanging"
    /usr/bin/python3 -I - "$STAGE/overlay" "$OUT" <<'PY'
import ctypes
import os
import sys

at_fdcwd = -100
rename_exchange = 2
renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
if renameat2(at_fdcwd, os.fsencode(sys.argv[1]), at_fdcwd, os.fsencode(sys.argv[2]), rename_exchange):
    sys.exit(f"Could not exchange b12x overlay directories: {os.strerror(ctypes.get_errno())}")
PY
    is_overlay "$STAGE/overlay" || { echo "The b12x overlay at $OUT changed during replacement" >&2; exit 2; }
else
    mv -nT -- "$STAGE/overlay" "$OUT"
    [ ! -d "$STAGE/overlay" ] || { echo "Refusing existing output $OUT" >&2; exit 2; }
fi
echo "overlay ready: $OUT"
