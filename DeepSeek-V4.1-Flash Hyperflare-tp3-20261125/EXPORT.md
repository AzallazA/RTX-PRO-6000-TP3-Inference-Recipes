# Exporting the DS41 TP3 2026-11-25 Package

This release uses the **Wave 7 upstream Hyperflare stack**: runtime fork **`AzallazA/hyperflare-inference@hyperflare-tp3-20261125`** (`ba6a4c11`) plus two clean patches in `tp3-patches/` applied to upstream `local-inference-lab/hyperflare-recipes @ 39c157a4` (`#23` + `#27`) and `local-inference-lab/flashinfer @ 0d48af53` (`#138`/`#139`/`#140`).

## 1. Requirements

- **GPUs:** 3 × RTX PRO 6000 Blackwell 96 GB (SM120), PCIe Gen5 x16 (tested at both **450 W PL** and stock **600 W TDP**; do **not** lock `-lmc` under a power limit).
- **Host RAM:** 256 GB minimum for `ENGRAM_RAM=1` (~218 GiB committed at `PREFIX_CACHE=12G`; use `ENGRAM_RAM=layer1` or `ENGRAM_RAM=0` for lower RAM).
- **Disk:** ~1.1 TB free (checkpoint ~510 GB, package ~470 GiB, tuning cache 60–80 GB).
- **Toolchain:** NVIDIA driver `615.71.09`+, CUDA toolkit `13.1`, Python `3.12`, Rust `1.93+`, `uv` `0.11+`, `ulimit -l unlimited` (see [versions.txt](versions.txt)).

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
ulimit -l   # must print "unlimited"
```

## 2. Clone & Build `flare`, Heliopause, Patched Recipes & `b12x` Overlay

```bash
RECIPE_DIR=/path/to/this-folder
mkdir -p ds41-tp3-20261125 && cd ds41-tp3-20261125

# 1. Runtime / Ignition fork (tag hyperflare-tp3-20261125)
git clone --branch hyperflare-tp3-20261125 --depth 1 https://github.com/AzallazA/hyperflare-inference.git flare-core

# 2. Upstream hyperflare-recipes @ 39c157a4 (#23 + #27) + TP3 Wave 7 patch
git clone https://github.com/local-inference-lab/hyperflare-recipes.git hyperflare-recipes
git -C hyperflare-recipes checkout 39c157a4
git -C hyperflare-recipes apply "$RECIPE_DIR/tp3-patches/hyperflare-recipes-tp3-20261125.patch"

# 3. Upstream FlashInfer @ 0d48af53 (#138/#139/#140) + TP3 Wave 7 b12x patch
git clone https://github.com/local-inference-lab/flashinfer.git flashinfer
git -C flashinfer checkout 0d48af53
git -C flashinfer apply "$RECIPE_DIR/tp3-patches/b12x-tp3-20261125.patch"

# Build the Rust serving binary (flare-cli/target/release/flare)
cargo build --locked --release --manifest-path flare-core/flare-cli/Cargo.toml

# Python environment & Heliopause extension (required for --row-shard-prefill export)
cd flare-core
uv sync --locked --extra prepare
cd ..
```

### Build the `b12x` Overlay

```bash
mkdir -p b12x-overlay/b12x
ln -s "$PWD/flashinfer/flashinfer/experimental/b12x" "$PWD/b12x-overlay/b12x_tree"
cat > b12x-overlay/b12x/__init__.py <<'PY'
import pathlib
_root = pathlib.Path(__file__).resolve().parent.parent / "b12x_tree"
__path__ = [str(_root)]
exec((_root / "__init__.py").read_text(), globals())
PY
```

## 3. Prepare the Checkpoint View & Optional WARMIND Sidecar

Download `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`, all 48 shards) and build the FP8-Engram checkpoint view:

```bash
./make_checkpoint_view.sh /path/to/DeepSeek-V4.1-Flash-1017 /path/to/views/ds41-1017-engram-fp8
```

Download the WARMIND v1.1 steering sidecar (MIT, 684 KB):
- Repo: [jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`
- File: `warmind-v1.1-orth4c-steering.safetensors` (`sha256: 68551f9b8deca315d0c23f31519f69e240d0d4bf0459676add70762635d86a2d`)

## 4. Export the Package (`PREFILL=3840`, `ROW_SHARD_PREFILL=1`)

```bash
export B12X_OVERLAY="$PWD/b12x-overlay"
export PYTHONPATH="$PWD/flare-core:$PWD/hyperflare-recipes"
export IGNITION_PYTHON="$PWD/flare-core/.venv/bin/python"

cd hyperflare-recipes
CKPT=/path/to/views/ds41-1017-engram-fp8 \
STEERING=/path/to/warmind-v1.1-orth4c-steering.safetensors \
STEERING_PLAIN_ID=1 \
GPUS=0,1,2 MAXC=16 PREFILL=3840 ROW_SHARD_PREFILL=1 DRAFTS=5 HOT=355 \
A16_QKV=0 FLAT_POOL=1 \
ENGRAM_SIDE=1,14 \
MLA_SPLITS="18:10,37:5,47:4,75:3,94:2,96:1" \
WOA_SMALL=0 \
BRRR_PUSH_V2=2 \
H24_Q_ROPE=1 \
COMPILE_WORKERS=3 \
FLARE_EXPORT_FORWARD_WORKERS=9 FLARE_EXPORT_FORWARD_WORKER_GIB=16 FLARE_EXPORT_FORWARD_WORKER_VRAM_MIB=768 \
bash tp3-tools/release/export.sh
```

*(Omit `STEERING=...` to export a plain unsteered package; `STEERING_PLAIN_ID=1` shares the tuning cache between plain and steered exports.)*

## 5. Serve

```bash
# Default: both FP8 Engram tables in host RAM (5,883,648 KV tokens, STACK_LIMIT=48, BUDGET=32)
FLARE="$PWD/flare-core/flare-cli/target/release/flare" ./serve.sh

# Hybrid / low-RAM: layer 1 in RAM, layer 14 on NVMe with 32 GiB cache
FLARE="$PWD/flare-core/flare-cli/target/release/flare" ENGRAM_RAM=layer1 ./serve.sh

# Full NVMe disk Engram (32 GiB cache)
FLARE="$PWD/flare-core/flare-cli/target/release/flare" ENGRAM_RAM=0 ./serve.sh

# Dry run (prints memory plan and KV capacity without starting the server)
FLARE="$PWD/flare-core/flare-cli/target/release/flare" ./serve.sh --dry-run
```
