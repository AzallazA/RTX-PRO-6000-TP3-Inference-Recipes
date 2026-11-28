# Exporting & Serving the DS41 TP3 2026-11-28 (Wave 15.1) Package

This release uses the **Wave 15.1 Hyperflare stack**:
- **Runtime / Ignition:** [`AzallazA/hyperflare-inference@v0.4.1-tp3-20261128`](https://github.com/AzallazA/hyperflare-inference/releases/tag/v0.4.1-tp3-20261128) (`7eecfda6`, includes prebuilt **`flare-linux-amd64.tgz`** + `SHA256SUMS.txt`).
- **Kernels (`b12x`):** [`AzallazA/flashinfer@tp3-20261128`](https://github.com/AzallazA/flashinfer/releases/tag/tp3-20261128) (`9b87317d`) — or apply `tp3-patches/b12x-tp3-20261128.patch` to `local-inference-lab/flashinfer @ 0d48af53`.
- **Recipes / BRRR:** `local-inference-lab/hyperflare-recipes @ 39c157a4` + `tp3-patches/hyperflare-recipes-tp3-20261128.patch`.

## 1. Requirements

- **GPUs:** 3 × RTX PRO 6000 Blackwell 96 GB (SM120), PCIe Gen5 x16 (tested at **450 W PL** headless; do **not** lock `-lmc` under a power limit).
- **Host RAM:** 256 GB minimum for `ENGRAM_RAM=1` (~218 GiB committed at `PREFIX_CACHE=12G`; use `ENGRAM_RAM=layer1` or `ENGRAM_RAM=0` with `#566/#70` disk lookahead prefetch for lower RAM).
- **Disk:** ~1.1 TB free (checkpoint ~510 GB, package ~470 GiB, tuning cache 60–80 GB).
- **Toolchain:** NVIDIA driver `615.71.09`+, CUDA toolkit `13.3` (`/usr/local/cuda`), Python `3.12`, Rust `1.93+` (only if building `flare` from source; prebuilt `flare-linux-amd64.tgz` needs no Rust toolchain), `uv` `0.11+`, `ulimit -l unlimited` (see [versions.txt](versions.txt)).

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
ulimit -l   # must print "unlimited"
```

## 2. Get the `flare` Serving Binary (Prebuilt Tarball or Source Build)

### Option A — Prebuilt Binary Tarball (Fastest)

Download `flare-linux-amd64.tgz` and `SHA256SUMS.txt` from [AzallazA/hyperflare-inference@v0.4.1-tp3-20261128](https://github.com/AzallazA/hyperflare-inference/releases/tag/v0.4.1-tp3-20261128):

```bash
mkdir -p flare-bin && cd flare-bin
curl -LO https://github.com/AzallazA/hyperflare-inference/releases/download/v0.4.1-tp3-20261128/flare-linux-amd64.tgz
curl -LO https://github.com/AzallazA/hyperflare-inference/releases/download/v0.4.1-tp3-20261128/SHA256SUMS.txt
sha256sum -c --ignore-missing SHA256SUMS.txt
tar -xzf flare-linux-amd64.tgz
./flare --help
cd ..
```

### Option B — Build `flare` from Source

```bash
git clone --branch v0.4.1-tp3-20261128 --depth 1 https://github.com/AzallazA/hyperflare-inference.git flare-core
cargo build --locked --release --manifest-path flare-core/flare-cli/Cargo.toml
# Binary: flare-core/flare-cli/target/release/flare
```

## 3. Set Up Export Environment (Ignition, Heliopause, Recipes, and `b12x` Overlay)

Exporting a package requires Python + Ignition + the `flare_heliopause` extension (with SHA-256 `CUmodule` cubin deduplication) from `flare-core`, plus patched `hyperflare-recipes` and `flashinfer`:

```bash
RECIPE_DIR=/path/to/this-folder
mkdir -p ds41-tp3-20261128 && cd ds41-tp3-20261128

# 1. Runtime / Ignition + Heliopause source (skip if cloned in Option B)
git clone --branch v0.4.1-tp3-20261128 --depth 1 https://github.com/AzallazA/hyperflare-inference.git flare-core

# 2. Upstream hyperflare-recipes @ 39c157a4 + Wave 15.1 TP3 patch
git clone https://github.com/local-inference-lab/hyperflare-recipes.git hyperflare-recipes
git -C hyperflare-recipes checkout 39c157a4
git -C hyperflare-recipes apply "$RECIPE_DIR/tp3-patches/hyperflare-recipes-tp3-20261128.patch"

# 3. FlashInfer / b12x (either clone AzallazA/flashinfer@tp3-20261128 OR apply patch to upstream 0d48af53)
git clone --branch tp3-20261128 --depth 1 https://github.com/AzallazA/flashinfer.git flashinfer
# Equivalent patch workflow:
#   git clone https://github.com/local-inference-lab/flashinfer.git flashinfer
#   git -C flashinfer checkout 0d48af53
#   git -C flashinfer apply "$RECIPE_DIR/tp3-patches/b12x-tp3-20261128.patch"

# Python environment & Heliopause extension (required for export)
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

## 4. Prepare the Checkpoint View & Optional WARMIND Sidecar

Download `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`, all 48 shards) and build the FP8-Engram checkpoint view:

```bash
./make_checkpoint_view.sh /path/to/DeepSeek-V4.1-Flash-1017 /path/to/views/ds41-1017-engram-fp8
```

Download the WARMIND v1.1 steering sidecar (MIT, 684 KB):
- Repo: [jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`
- File: `warmind-v1.1-orth4c-steering.safetensors` (`sha256: 68551f9b8deca315d0c23f31519f69e240d0d4bf0459676add70762635d86a2d`)

## 5. Export the Package (`PREFILL=7680`, `GLOBAL_HOT_BUDGET=1`, `ZERO_REP_ATTENTION=1`)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export B12X_OVERLAY="$PWD/b12x-overlay"
export PYTHONPATH="$PWD/flare-core:$PWD/hyperflare-recipes"
export IGNITION_PYTHON="$PWD/flare-core/.venv/bin/python"

# Default Wave 15.1 Speed Build (HOT=375, ~2.43M KV tokens, 97.7% parked experts in VRAM):
CKPT=/path/to/views/ds41-1017-engram-fp8 \
STEERING=/path/to/warmind-v1.1-orth4c-steering.safetensors \
STEERING_PLAIN_ID=1 \
GPUS=0,1,2 MAXC=16 PREFILL=7680 \
ROW_SHARD_PREFILL=1 ROW_SHARD_ATTENTION=1 ZERO_REP_ATTENTION=1 \
MOE_RESERVE_SMS=4 GLOBAL_HOT_BUDGET=1 HOT=375 \
DRAFTS=5 COPY_VERIFY="1x16,2x16,4x16" \
A16_QKV=0 FLAT_POOL=1 ENGRAM_SIDE=1,14 \
MLA_SPLITS="18:10,37:5,48:4,62:3,94:2,96:4" \
WOA_SMALL=0 BRRR_PUSH_V2=2 H24_Q_ROPE=1 H24_PREFILL=1 \
BRRR_R1_M_THRESH=17 BRRR_PARKED_HOT_R1P=2 BRRR_SPIN_NS=32 \
COMPILE_WORKERS=3 \
FLARE_EXPORT_FORWARD_WORKERS=9 FLARE_EXPORT_FORWARD_WORKER_GIB=16 FLARE_EXPORT_FORWARD_WORKER_VRAM_MIB=768 \
bash "$RECIPE_DIR/export.sh"
```

> **KV-King / Desktop Variant (`5.06M` KV tokens):** Change only `HOT=375` → `HOT=355`. On a cold tuning cache, export once with `HOT=300` (or `FLARE_EXPORT_TUNE_ONLY=1`) if tuning runs tight on VRAM, then re-run at `HOT=375` — the second export reuses the tuning cache (~7 min total). Omit `STEERING=...` to export a plain unsteered package.

## 6. Serve

```bash
# Default: both FP8 Engram tables in host RAM (2,432,000 KV tokens @ HOT=375, --history-drafts, BUDGET=64, ALPHA=1.0)
FLARE=/path/to/flare ./serve.sh

# Hybrid / low-RAM: layer 1 in RAM, layer 14 on NVMe with #566+#70 graph-driven lookahead prefetch (32 GiB cache)
FLARE=/path/to/flare ENGRAM_RAM=layer1 ./serve.sh

# Full NVMe disk Engram with #566+#70 lookahead prefetch + #561 io_uring (32 GiB cache)
FLARE=/path/to/flare ENGRAM_RAM=0 ./serve.sh

# Disable disk lookahead prefetch (if benchmarking A/B)
FLARE=/path/to/flare ENGRAM_RAM=0 ENGRAM_PREFETCH=0 ./serve.sh

# Dry run (prints memory plan and KV capacity without starting the server)
FLARE=/path/to/flare ./serve.sh --dry-run
```
