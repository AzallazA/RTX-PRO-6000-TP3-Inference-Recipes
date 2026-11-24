# Exporting the DS41 EXL3 TP2 2026-11-24 Package

This release uses the **post-split Hyperflare stack**: runtime fork **`AzallazA/hyperflare-inference@hyperflare-tp2-exl3-20261124`** (identical runtime commit `5ffe4d54` to TP3) plus two clean patches in `tp3-patches/` applied to upstream `local-inference-lab/hyperflare-recipes @ fee34e1` and `local-inference-lab/flashinfer @ 0aa01681`.

## 1. Requirements

- **GPUs:** 2 × RTX PRO 6000 Blackwell 96 GB (SM120), PCIe Gen5 x16, tested at **450 W PL** (`3165/17001 MHz` max clocks; do **not** lock `-lmc`).
- **Host RAM:** ~151 GiB for `ENGRAM=ram` (both int4 Engram tables in RAM); ~110 GiB for `ENGRAM=one` (layer-14 table on NVMe with 4 GiB cache, verified `0.969–0.979` consistency / `0` miss).
- **Disk:** ~650 GB free (Diffbot EXL3 pack + exported package + tuning cache).
- **Toolchain:** NVIDIA driver `615.71.09`+, CUDA toolkit `13.1`, Python `3.12`, Rust `1.93+`, `uv` `0.11+`, `ulimit -l unlimited` (see [versions.txt](versions.txt)).

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
ulimit -l   # must print "unlimited"
```

## 2. Clone & Build `flare` + Patched Recipes & `b12x` Overlay

```bash
RECIPE_DIR=/path/to/this-folder
mkdir -p ds41-tp2-exl3-20261124 && cd ds41-tp2-exl3-20261124

# 1. Runtime / Ignition fork (tag hyperflare-tp2-exl3-20261124, same runtime as TP3)
git clone --branch hyperflare-tp2-exl3-20261124 --depth 1 https://github.com/AzallazA/hyperflare-inference.git flare-core

# 2. Upstream hyperflare-recipes @ fee34e1 + tp3-patches/hyperflare-recipes-tp2-exl3-20261124.patch
git clone https://github.com/local-inference-lab/hyperflare-recipes.git hyperflare-recipes
git -C hyperflare-recipes checkout fee34e1
git -C hyperflare-recipes apply "$RECIPE_DIR/tp3-patches/hyperflare-recipes-tp2-exl3-20261124.patch"

# 3. Upstream FlashInfer @ 0aa01681 + tp3-patches/b12x-tp2-exl3-20261124.patch
git clone https://github.com/local-inference-lab/flashinfer.git flashinfer
git -C flashinfer checkout 0aa01681
patch -p1 -d flashinfer < "$RECIPE_DIR/tp3-patches/b12x-tp2-exl3-20261124.patch"

# Build the Rust serving binary (flare-cli/target/release/flare)
cargo build --locked --release --manifest-path flare-core/flare-cli/Cargo.toml

# Python environment (torch 2.14.1+cu132, triton 3.8.0, nvidia-cutlass-dsl 4.8.1)
cd flare-core
uv sync --locked --extra prepare
cd ..

# Build the b12x overlay
mkdir -p b12x-overlay/b12x
ln -s "$PWD/flashinfer/flashinfer/experimental/b12x" "$PWD/b12x-overlay/b12x_tree"
cat > b12x-overlay/b12x/__init__.py <<'PY'
import pathlib, sys
_root = pathlib.Path(__file__).resolve().parent.parent / "b12x_tree"
__path__ = [str(_root)]
exec((_root / "__init__.py").read_text(), globals())
PY
```

## 3. Download Diffbot's EXL3 3-bit Pack

Download [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) to `/path/to/DeepSeek-V4.1-Flash-1017-EXL3-3bpw` (see `../DeepSeek-V4.1-Flash Hyperflare-tp2-exl3-20261117/EXPORT.md` for `hf download` details).

## 4. Export the Package (`344/316` hot experts, `MAXC=8`, `PREFILL=4096`, `DRAFTS=5`)

```bash
export B12X_OVERLAY="$PWD/b12x-overlay"
export PYTHONPATH="$PWD/flare-core:$PWD/hyperflare-recipes"
export IGNITION_PYTHON="$PWD/flare-core/.venv/bin/python"

HL=$(python3 -I -c "print(','.join([f'{l}:344' for l in [2,3,4]+list(range(6,20))] + [f'{l}:316' for l in range(20,40)]))")

CKPT=/path/to/DeepSeek-V4.1-Flash-1017-EXL3-3bpw \
COUNTS="$RECIPE_DIR/expert-order" \
GPUS=0,1 PREFILL=4096 DRAFTS=5 MAXC=8 COMPILE_WORKERS=3 \
ENGRAM_FULL_PROJECTION=0 ENGRAM_NO_READINESS_BARRIER=1 A16_QKV=0 \
ENGRAM_CHUNK_ROWS=768 GATHER_PULL=1 FLAT_POOL=1 \
B12X_PCIE_TWOSHOT_PULL=1 \
PARK=2-4,6-39 EXPORT_HOT=308 HOT=316 FLARE_DS41_EXL3_HOT_LAYERS="$HL" \
FLARE_EXPORT_LIBRARY_SCRATCH=1 MTP_ONLY=1 SIDE_STREAMS=1 \
FLARE_DS41_EXL3_COLD_SIDE=16 FLARE_DS41_EXL3_SHARED_SIDE=1 IMAGE_CONCURRENCY=1 \
FLARE_DS41_EXL3_FUSED_PREP=1 FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS=48 \
FLARE_TUNE_SCRATCH_TOLERANCE=0.05 FLARE_TUNE_SCRATCH_MIN_SAVING_MIB=16 \
FLARE_DS41_EXL3_WOA_SINGLE=1 FLARE_DS41_EXL3_PREFILL_GROUPS=2 \
FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS=4096 FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 \
FLARE_DS41_EXL3_MERGE_TOPK=1 FLARE_DS41_EXL3_MERGE_EPILOGUE=1 \
FLARE_DS41_EXL3_MERGE_ROPEQ=0 FLARE_DS41_EXL3_BULK_MIN_ROWS=1 \
FLARE_DS41_EXL3_COLD_WIDTH=16 MOE_PREFILL=do \
FLARE_DS41_EXL3_MASK_PADDING=1 FLARE_DS41_EXL3_POOL_SCRATCH=1 FLARE_FLAT_MEMORY_POOL=1 \
FLARE_DS41_PREFILL_SKIP_SUFFIX=3072,3584,4096 \
FLARE_DS41_MARKOV_EMBED_HOST=1 FLARE_DS41_MARKOV_STAGE_ROWS=1 \
FLARE_DS41_VISION_ARENA_HOST=1 FLARE_EXPORT_SHARE_COMPUTED_OPERANDS=1 \
FLARE_DS41_ROTARY_ONCE=1 FLARE_DS41_ROTARY_ONCE_PREFILL=1 \
FLARE_DS41_LOGPROB_SINGLE_EXCHANGE=1 FLARE_DS41_PIN_SMALL_QPROJ=1 \
FLARE_DS41_PIN_MID_QKV_NORM=1 FLARE_DS41_INDEX_SIDE_PREFILL=1 \
FLARE_DS41_MMA_PAD_ONCE=1 FLARE_DS41_GLUE_PASS2=1 \
FLARE_DS41_INDEX_ROW_SPLIT=4096 \
FLARE_DS41_ENGRAM_LOCAL_LOOKUP=1 \
FLARE_DS41_ENGRAM_SIDE_LAYERS=14 FLARE_DS41_ENGRAM_SIDE_SPLIT=1 \
FLARE_DS41_MLA_DECODE_SPLITS="9:10,18:5,23:4,31:3,47:2,94:1" \
FLARE_EXPORT_FORWARD_WORKERS=6 \
bash "$RECIPE_DIR/export.sh"
```

- **Optional WARMIND v1.1 Steering Sidecar:** Add `STEERING=/path/to/warmind-v1.1-orth4c-steering.safetensors FLARE_DS41_STEERING_PLAIN_IDENTITY=1` (fully compatible with `FLARE_DS41_EXL3_MERGE_EPILOGUE=1`).

## 5. Serve

```bash
# Default: both int4 Engram tables in host RAM (~151 GiB RSS, ~2.84M-2.87M KV, STACK_LIMIT=144)
FLARE="$PWD/flare-core/target/release/flare" ./serve.sh

# Low-RAM option: layer 1 in RAM, layer 14 from NVMe with 4 GiB row cache (~110 GiB RSS, consistency 0.969-0.979)
FLARE="$PWD/flare-core/target/release/flare" ENGRAM=one ENGRAM_CACHE=4GiB ./serve.sh
```
