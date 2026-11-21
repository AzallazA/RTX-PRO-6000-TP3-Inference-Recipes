#!/bin/bash
# FP8-Engram checkpoint view for DeepSeek-V4.1-Flash-1017: symlinks to EVERY file of DeepSeek's original snapshot
# (shards 1-48, config.json, model.safetensors.index.json, tokenizer, ...). Shards 47-48 hold DeepSeek's FP8 Engram tables
# (F8_E4M3 [rows, 256] + F8_E8M0 [rows, 8] scales, 189 GiB); DeepSeek's own config.json and index are used unchanged.
# No NVIDIA download is needed (the 20261111 folder's MXFP4 view took shards 47-48 from nvidia/DeepSeek-V4.1-Flash-1017-NVFP4).
# The output directory's name becomes the model name of the package (serve.sh default: ds41-1017-engram-fp8).
#   usage: make_checkpoint_view.sh <deepseek snapshot dir> <output dir (must not exist)>
#   expect: two "Engram table F8_E4M3 ... ok" lines and "96085 tensors in 48 shards, missing 0; payload 510286023000 vs index total_size 510286023000"
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <deepseek snapshot dir> <output dir>" >&2; exit 2; }
DS=$(readlink -f "$1"); OUT=$2
[ -e "$OUT" ] && { echo "refuse: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
for f in "$DS"/*; do ln -s "$(readlink -f "$f")" "$OUT/$(basename "$f")"; done
cd "$OUT"
/usr/bin/python3 -I - "$OUT" <<'PY'
import json, os, struct, sys
out = sys.argv[1]
idx = json.load(open(f"{out}/model.safetensors.index.json"))
seen, missing, total = {}, [], 0
for name, shard in idx["weight_map"].items():
    if shard not in seen:
        with open(f"{out}/{shard}", "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            seen[shard] = json.loads(h.read(n))
            end = max(v["data_offsets"][1] for k, v in seen[shard].items() if k != "__metadata__")
            if 8 + n + end != os.path.getsize(f"{out}/{shard}"):
                sys.exit(f"{shard}: size {os.path.getsize(f'{out}/{shard}')} != header end {8 + n + end}")
    if name not in seen[shard]:
        missing.append(name)
    else:
        o = seen[shard][name]["data_offsets"]; total += o[1] - o[0]
bad = []
for layer, shard in ((1, "model-00047-of-00048.safetensors"), (14, "model-00048-of-00048.safetensors")):
    w = seen[shard].get(f"layers.{layer}.engram.embed.weight", {}); s = seen[shard].get(f"layers.{layer}.engram.embed.scale", {})
    ok = w.get("dtype") == "F8_E4M3" and len(w.get("shape", [])) == 2 and w["shape"][1] == 256 \
         and s.get("dtype") == "F8_E8M0" and s.get("shape") == [w["shape"][0], 8]
    print(f"layer-{layer} Engram table {w.get('dtype')} {w.get('shape')} scales {s.get('dtype')} {s.get('shape')} {'ok' if ok else 'BAD'}")
    bad += [] if ok else [layer]
ts = idx.get("metadata", {}).get("total_size")
print(f"{len(idx['weight_map'])} tensors in {len(seen)} shards, missing {len(missing)}; payload {total} vs index total_size {ts}")
sys.exit(1 if missing or bad or ts != total else 0)
PY
echo "FP8 checkpoint view ready: $OUT"
