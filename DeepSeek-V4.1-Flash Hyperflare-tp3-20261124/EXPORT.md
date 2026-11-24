# Exporting the DS41 TP3 2026-11-24 Package

This release uses the **post-split Hyperflare stack**: runtime fork **`AzallazA/hyperflare-inference@hyperflare-tp3-20261124`** plus two clean patches in `tp3-patches/` applied to upstream `local-inference-lab/hyperflare-recipes @ fee34e1` and `local-inference-lab/flashinfer @ 0aa01681`.

## 1. Requirements

- **GPUs:** 3 × RTX PRO 6000 Blackwell 96 GB (SM120), PCIe Gen5 x16, tested at **450 W PL** (`3165/17001 MHz` max clocks; do **not** lock `-lmc`).
- **Host RAM:** 256 GB minimum for `ENGRAM_RAM=1` (~218 GiB committed at `PREFIX_CACHE=12G`; use `ENGRAM_RAM=layer1` for lower RAM).
- **Disk:** ~1.1 TB free (checkpoint ~510 GB, package ~470 GiB, tuning cache 60–80 GB).
- **Toolchain:** NVIDIA driver `615.71.09`+, CUDA toolkit `13.1`, Python `3.12`, Rust `1.93+`, `uv` `0.11+`, `ulimit -l unlimited` (see [versions.txt](versions.txt)).

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
ulimit -l   # must print "unlimited"
```

## 2. Clone & Build `flare` + Patched Recipes & `b12x` Overlay

```bash
RECIPE_DIR=/path/to/this-folder
mkdir -p ds41-tp3-20261124 && cd ds41-tp3-20261124

# 1. Runtime / Ignition fork (tag hyperflare-tp3-20261124)
git clone --branch hyperflare-tp3-20261124 --depth 1 https://github.com/AzallazA/hyperflare-inference.git flare-core

# 2. Upstream hyperflare-recipes @ fee34e1 + tp3-patches/hyperflare-recipes-tp3-20261124.patch
git clone https://github.com/local-inference-lab/hyperflare-recipes.git hyperflare-recipes
git -C hyperflare-recipes checkout fee34e1
git -C hyperflare-recipes apply "$RECIPE_DIR/tp3-patches/hyperflare-recipes-tp3-20261124.patch"

# 3. Upstream FlashInfer @ 0aa01681 + tp3-patches/b12x-tp3-20261124.patch
git clone https://github.com/local-inference-lab/flashinfer.git flashinfer
git -C flashinfer checkout 0aa01681
patch -p1 -d flashinfer < "$RECIPE_DIR/tp3-patches/b12x-tp3-20261124.patch"

# Build the Rust serving binary (flare-cli/target/release/flare) and Python Heliopause extension
cargo build --locked --release --manifest-path flare-core/flare-cli/Cargo.toml

# Python environment (torch 2.14.1+cu132, triton 3.8.0, nvidia-cutlass-dsl 4.8.1)
cd flare-core
uv sync --locked --extra prepare
cd ..
```

### Build the `b12x` Overlay

```bash
mkdir -p b12x-overlay/b12x
ln -s "$PWD/flashinfer/flashinfer/experimental/b12x" "$PWD/b12x-overlay/b12x_tree"
cat > b12x-overlay/b12x/__init__.py <<'PY'
import pathlib, sys
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

## 4. Export the Package

```bash
export B12X_OVERLAY="$PWD/b12x-overlay"
export PYTHONPATH="$PWD/flare-core:$PWD/hyperflare-recipes"
export IGNITION_PYTHON="$PWD/flare-core/.venv/bin/python"

cd hyperflare-recipes
CKPT=/path/to/views/ds41-1017-engram-fp8 \
STEERING=/path/to/warmind-v1.1-orth4c-steering.safetensors \
STEERING_PLAIN_ID=1 \
GPUS=0,1,2 MAXC=16 PREFILL=4096 DRAFTS=5 HOT=355 \
ENGRAM_FULL_PROJECTION=0 A16_QKV=0 FLAT_POOL=1 ENGRAM_CHUNK_ROWS=768 \
COMPILE_WORKERS=3 \
FLARE_EXPORT_FORWARD_WORKERS=9 FLARE_EXPORT_FORWARD_WORKER_GIB=16 FLARE_EXPORT_FORWARD_WORKER_VRAM_MIB=768 \
bash tp3-tools/release/export.sh
```

*(Omit `STEERING=...` to export a plain unsteered package; `STEERING_PLAIN_ID=1` shares the tuning cache between plain and steered exports.)*

## 5. Serve

```bash
# Default: both FP8 Engram tables in host RAM (~218 GiB committed, 5,810,816 KV tokens, STACK_LIMIT=48)
FLARE="$PWD/flare-core/target/release/flare" ./serve.sh

# Low-RAM option: layer 1 in RAM, layer 14 on NVMe with 32 GiB cache
FLARE="$PWD/flare-core/target/release/flare" ENGRAM_RAM=layer1 ./serve.sh

# Dry run (prints memory plan and KV capacity without starting the server)
FLARE="$PWD/flare-core/target/release/flare" ./serve.sh --dry-run
```
