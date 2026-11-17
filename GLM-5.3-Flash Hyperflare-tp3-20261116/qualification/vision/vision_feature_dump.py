#!/usr/bin/env python3
"""Dump GLM-5.3-Flash vision tower OUTPUT FEATURES from one code version (real weights, real images, one GPU).

Run once per worktree (the worktree's `ignition` is imported from PYTHONPATH), then compare the files with
vision_feature_cmp.py. vision-feature-check.sh runs old / lean / lean-again and both comparisons.

  PYTHONPATH=<wt>/.b12x-overlay:<wt> CUDA_VISIBLE_DEVICES=0 <venv>/bin/python vision_feature_dump.py --out F.pt
  CUDA_VISIBLE_DEVICES= ... vision_feature_dump.py --check-cpu       CPU only: inputs, frontend parity, API wiring

Path exercised (as glm53/bootstrap.py builds it for serving; see the staging/vision CUDA tests it is taken from):
  * VisionWeights.load from the checkpoint (DirectWeightLoader, real b12x MXFP8/NVFP4 projections); with --stage
    (default, = the vision package's FLARE_GLM53_VISION_STAGE=1) the tower is parked in pinned host memory and
    staged through the two VRAM block slots + the gate-aliased tail slot on the arena's auxiliary stream.
  * One VisionScratch at max(PATCH_CAPACITIES)=32768, passed through scratch.overlay() as serving does, viewed
    with .rows(capacity) per graph; one VisionStaging shared by every capacity (built after the overlay).
  * Per needed capacity (smallest PATCH_CAPACITIES entry >= the image's patches; graphs built largest first):
    canonical HF module tree, VisionMetadata, PreparedVision, PreparedVisionGeneration (the served op: FP32 pixel
    upload buffer -> finiteness checks -> masked BF16 patches -> tower -> feature publication into the arena),
    compiled with ForwardCache.prepare_many (the "worker" compiler serving uses; --compiler none|trace for
    debugging), captured into a CUDA graph and replayed.
  * Features read from the published feature arena (VisionState.arena[slot, :patches/4]), BF16 as-is.
  Differences from a real export (identical for both versions): PreparationSession(autotune=False,
  compile_workers=0) and no operator workspaces (as the existing CUDA tests); no prefill scratch in the overlay.

Inputs (9): the first 4 images of the DS41 vision qualification `set` and `hard` dirs at their natural sizes, and
one 2800x2240 composite of set[30:34] (glm_images.grid(): exactly 100x80 = 8000 image tokens = 32000 patches, the
frontend's maximum). Preprocessing: transformers Glm5NextImageProcessor (defaults: <=8000 tokens, 28-px canvas,
bicubic, CLIP normalization) + transformers.vision_utils.get_vision_position_ids. --check-cpu verifies that this is
byte-identical to the Flare frontend (helios/src/frontend/images/glm.rs) on the frontend's own fixture
(tests/frontend/glm53_images.json: pixel and position SHA-256 per case) and that every input's grid equals the Rust
geometry() rule.

Each image is replayed --replays times (default 2) in-process; the file records whether those replays were
bit-identical and the op's error word. Pixel rows past the image are filled with NaN before each replay (the op must
mask them) and the feature arena with NaN (features must come from this replay).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
VISION = pathlib.Path(os.environ.get("VISION_SETS", HERE.parents[2] / "DeepSeek-V4.1-Flash Kraken-tp3-20261019/qualification/vision"))   # set/ and hard/ (the DeepSeek V4.1 vision qualification sets of this repo)
GLM_IMAGES = HERE.parents[1] / "tools" / "glm_images.py"   # this release folder's image check tool
SNAPSHOT = None                                    # --snapshot or FLARE_GLM53_SNAPSHOT: the checkpoint directory
FACTOR, MIN_TOKENS, MAX_TOKENS = 28, 16, 8000      # helios/src/frontend/images/glm.rs
COMPOSITE = (2800, 2240)
SLOT, GENERATION = 1, 7                            # request row / generation the features are published for


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path, help="torch.save output file (GPU run)")
    mode.add_argument("--check-cpu", action="store_true", help="CPU-only checks; no CUDA")
    p.add_argument("--snapshot", default=os.environ.get("FLARE_GLM53_SNAPSHOT", SNAPSHOT))
    p.add_argument("--images", default=str(VISION), help="dir with set/ and hard/ subdirs")
    p.add_argument("--per-dir", type=int, default=4, help="images taken from each of set/ and hard/")
    p.add_argument("--no-composite", action="store_true", help="skip the 2800x2240 near-max composite")
    p.add_argument("--stage", action=argparse.BooleanOptionalAction, default=True,
                   help="park + stage the tower like the vision package (default on)")
    p.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True,
                   help="staging copies on the arena's auxiliary stream (default on)")
    p.add_argument("--compiler", choices=("worker", "trace", "none"), default="worker")
    p.add_argument("--replays", type=int, default=2, help="graph replays per image (in-process stability)")
    p.add_argument("--label", default=None, help="label stored in the file (default: worktree name)")
    args = p.parse_args(argv)
    if args.replays < 1 or args.per_dir < 0:
        p.error("--replays must be >= 1 and --per-dir >= 0")
    if args.out is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("GPU run: assign one GPU explicitly (CUDA_VISIBLE_DEVICES=<n>)")
    return args


# Inputs ---------------------------------------------------------------------------------------------------------

def rust_geometry(width, height):
    """helios/src/frontend/images/glm.rs geometry(): (h, w) of the 28-aligned canvas."""
    align = lambda v: -(-v // FACTOR) * FACTOR  # noqa: E731
    h, w = align(height), align(width)
    min_pixels, max_pixels = MIN_TOKENS * FACTOR * FACTOR, MAX_TOKENS * FACTOR * FACTOR
    if h * w < min_pixels:
        scale = (min_pixels / (height * width)) ** .5
        h, w = align(max(1, int(-(-height * scale // 1)))), align(max(1, int(-(-width * scale // 1))))
    if h * w > max_pixels:
        low, high, (h, w) = 1, height, (FACTOR, FACTOR)
        while low <= high:
            content_h = (low + high) // 2
            content_w = max(1, int(width * content_h / height))
            ch, cw = align(content_h), align(content_w)
            if ch * cw <= max_pixels:
                h, w, low = ch, cw, content_h + 1
            else:
                high = content_h - 1
    return h, w


def composite(set_paths):
    """glm_images.grid(set[30:34], 2800x2240), loaded from glm_images.py itself (as lean_probe.py does)."""
    src = GLM_IMAGES.read_text().split('\nif mode == "latency":')[0]
    src = src.replace("mode, label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]", "")
    ns = {}
    exec(compile(src, "glm_images", "exec"), ns)
    import io

    from PIL import Image
    return Image.open(io.BytesIO(ns["grid"](set_paths[30:34], COMPOSITE))).convert("RGB")


def processor():
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    return Glm5NextImageProcessor()


def build_inputs(args):
    """[(name, source size, pixel_values FP32 [patches, 1176], position_ids int64 [patches, 2], grid)]."""
    from PIL import Image
    from transformers.vision_utils import get_vision_position_ids

    root = pathlib.Path(args.images)
    sets, hards = sorted((root / "set").glob("*.png")), sorted((root / "hard").glob("*.png"))
    if len(sets) < max(args.per_dir, 34 if not args.no_composite else 0) or len(hards) < args.per_dir:
        raise SystemExit(f"not enough images under {root}")
    images = [(f"{p.parent.name}/{p.name}", Image.open(p).convert("RGB"))
              for p in (*sets[:args.per_dir], *hards[:args.per_dir])]
    if not args.no_composite:
        images.append(("composite-2800x2240-set30-33", composite(sets)))
    proc, result = processor(), []
    for name, image in images:
        out = proc(images=[image], return_tensors="pt")
        grid = out.image_grid_thw
        pixels = out.pixel_values.float().contiguous()
        positions = get_vision_position_ids(grid, 2).to(dtype=__import__("torch").int64).contiguous()
        h, w = rust_geometry(*image.size)
        if grid[0].tolist() != [1, h // 14, w // 14] or pixels.shape != (h // 14 * (w // 14), 1176):
            raise SystemExit(f"{name}: processor grid {grid[0].tolist()} differs from the frontend rule {(h, w)}")
        if positions.shape != (pixels.shape[0], 2):
            raise SystemExit(f"{name}: position ids {tuple(positions.shape)} vs {pixels.shape[0]} patches")
        result.append(SimpleNamespace(name=name, size=image.size, pixels=pixels, positions=positions,
                                      grid=grid[0].tolist(), count=pixels.shape[0]))
    return result


def frontend_fixture_parity(worktree):
    """The Python preprocessing reproduces the Flare frontend's fixture hashes (pixels and positions)."""
    import torch
    from PIL import Image
    from transformers.vision_utils import get_vision_position_ids

    fixture = json.loads((pathlib.Path(worktree) / "tests/frontend/glm53_images.json").read_text())
    proc, rows = processor(), []
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    for case in fixture["cases"]:
        w, h, seed = case["width"], case["height"], case["seed"]
        data = bytes(v for y in range(h) for x in range(w)
                     for v in ((x * 7 + seed * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
        out = proc(images=[Image.frombytes("RGB", (w, h), data)], return_tensors="pt")
        pix = out.pixel_values.float().numpy().astype("<f4").tobytes()
        pos = get_vision_position_ids(out.image_grid_thw, 2).to(torch.int64).numpy().astype("<i8").tobytes()
        gh, gw = rust_geometry(w, h)
        rows.append((f"{w}x{h}", out.image_grid_thw[0].tolist() == case["grid"] == [1, gh // 14, gw // 14],
                     sha(pix) == case["pixels"], sha(pos) == case["positions"]))
    return rows


# GPU ------------------------------------------------------------------------------------------------------------

def capacity_for(count, capacities):
    for capacity in sorted(capacities):
        if count <= capacity:
            return capacity
    raise SystemExit(f"{count} patches exceed the largest vision capacity {max(capacities)}")


def checkpoint(snapshot):
    """tests/recipes/test_glm53_vision.py manifest(): the indexed shards holding the tower."""
    from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header

    snapshot = pathlib.Path(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({shard for name, shard in index.items() if name.startswith("model.visual.")})
    tensors, shards = {}, []
    for filename in files:
        record, entries = _read_header(snapshot / filename, filename)
        shards.append(record)
        tensors.update(entries)
    return IndexedCheckpoint(snapshot, tensors, tuple(shards))


def hf_module(snapshot):
    """tests/recipes/test_glm53_vision.py _module(); serving builds it from the package config (same values)."""
    import torch
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    config = Glm5NextVisionConfig.from_dict(
        json.loads((pathlib.Path(snapshot) / "config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Glm5NextVisionModel(config)
    return module.eval().requires_grad_(False)


class PinnedHost:
    """tests/recipes/test_glm53_vram_options.py _PinnedHost: pinned host pages viewed as a CUDA tensor (UVA)."""

    def __init__(self, device):
        self.device, self.regions = device, []

    def __call__(self, name, nbytes):
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        host = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
        self.regions.append((name, host))
        return _tensor_from_pointer(host.data_ptr(), shape=(nbytes,), dtype=torch.uint8, device=self.device,
                                    nbytes=host.nbytes)


def serving_state(device):
    """Upload buffers, request state and feature arena as tests/recipes/test_glm53_vision.py's generation test
    (one image slot, FEATURES=8192 features per request as serving)."""
    import torch
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.recipes.glm53.batch import BatchInputs
    from ignition.recipes.glm53.vision_state import FEATURES, VisionInputs

    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=2)
    batch = BatchInputs.allocate(3, 1, device, capacity=2)
    prefill = PrefillVisionInputs.allocate(3, device)
    state = VisionState(batch, prefill, requests, image_token=154854, features=FEATURES, hidden=4096, image_slots=1)
    completion = CompletionOutputs.allocate((2, 2, 32), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    return SimpleNamespace(inputs=inputs, requests=requests, state=state, completion=completion, frame=frame,
                           sequence=sequence)


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%H %D"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"? {error!r}"


def run_gpu(args, items):
    import tempfile

    import torch
    import b12x
    import ignition
    from b12x.preparation import PreparationSession
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.scratch import ScratchArena, overlay
    from ignition.recipes.glm53.vision import (PreparedVision, VisionMetadata, VisionScratch, VisionStaging,
                                               VisionWeights)
    from ignition.recipes.glm53.vision_state import PATCH_CAPACITIES, PreparedVisionGeneration

    worktree = pathlib.Path(ignition.__file__).resolve().parents[1]
    started = time.time()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    capacities = sorted({capacity_for(item.count, PATCH_CAPACITIES) for item in items}, reverse=True)
    log = lambda message: print(f"[{time.time() - started:7.1f}s] {message}", flush=True)  # noqa: E731
    log(f"ignition {ignition.__file__} ({git_head(worktree)}); b12x {b12x.__file__}; capacities {capacities}; "
        f"stage {args.stage} overlap {args.overlap} compiler {args.compiler}")

    if not args.snapshot:
        raise SystemExit("set --snapshot or FLARE_GLM53_SNAPSHOT to the checkpoint directory")
    manifest = checkpoint(args.snapshot)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(hf_module(args.snapshot), loader)
    host = None
    if args.stage:
        host = PinnedHost(device)
        weights = weights.park(host)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    log(f"weights loaded{' and parked' if args.stage else ''}; {torch.cuda.memory_allocated(device) >> 20} MiB")

    s = serving_state(device)
    # Serving: one scratch at the largest capacity, overlaid (with prefill scratch), viewed per capacity.
    scratch, = overlay((VisionScratch.allocate(max(PATCH_CAPACITIES), device),))
    arena = ScratchArena(device, overlap=args.overlap)
    ops, graphs, owners = {}, {}, []
    cache_root = tempfile.mkdtemp(prefix="glm-vision-featdump-")
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        staging = None
        if args.stage:
            staging = VisionStaging(weights, scratch, stream=arena.auxiliary_stream if args.overlap else None)
            log(f"staging slots {staging.allocated_bytes >> 20} MiB, tail aliased {staging.tail_aliased}")
        for capacity in capacities:
            name = f"vision{capacity}"
            metadata = VisionMetadata.allocate(capacity, 1, device)
            encoder = PreparedVision(hf_module(args.snapshot), scratch.rows(capacity), metadata, weights,
                                     session=session, name=name, staging=staging)
            ops[name] = PreparedVisionGeneration(encoder, s.inputs, s.state, s.completion, s.frame, s.sequence)
            owners.extend((metadata, encoder))
        session.freeze()
        log("prepared")
        forwards = dict(ops)
        if args.compiler == "worker":
            from ignition.preparation.aot import ForwardCache
            from ignition.preparation.execution import GraphSignature

            cache = ForwardCache(namespace={"vision": "glm53-featdump"}, device=device, root=cache_root)
            # As capture.py: the ops themselves (no partial), one signature per capacity.
            forwards = cache.prepare_many(dict(ops),
                                          {name: GraphSignature("vision", 1, int(name[6:])) for name in ops})
        elif args.compiler == "trace":
            from ignition.preparation.compilation import compile_forward
            forwards = {name: compile_forward(op) for name, op in ops.items()}
        log(f"compiled ({args.compiler})")
        stream = torch.cuda.Stream(device=device)
        for name, forward in forwards.items():
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graphs[name] = graph
        log(f"captured {list(graphs)}; {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
            f"{torch.cuda.memory_reserved(device) >> 20} MiB reserved")

        i, state, requests = s.inputs, s.state, s.requests
        features, info = {}, {}
        for item in items:
            capacity = capacity_for(item.count, PATCH_CAPACITIES)
            name, count = f"vision{capacity}", item.count
            results, errors = [], []
            for _ in range(args.replays):
                i.vision_pixel_values.fill_(float("nan"))
                i.vision_pixel_values[:count].copy_(item.pixels)
                i.vision_position_ids.zero_()
                i.vision_position_ids[:count].copy_(item.positions)
                i.vision_num_patches.fill_(count)
                i.vision_cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
                i.vision_slot.fill_(SLOT)
                i.vision_generation.fill_(GENERATION)
                i.vision_feature_offset.fill_(0)
                requests.generation[SLOT], requests.status[SLOT] = GENERATION, 1
                state.feature_slot.fill_(0)
                state.count.zero_()
                state.arena.fill_(float("nan"))
                torch.cuda.synchronize(device)
                graphs[name].replay()
                torch.cuda.synchronize(device)
                errors.append(int(ops[name].errors.item()))
                results.append(state.arena[0, :count // 4].clone())
            first = results[0]
            same = all(torch.equal(first.view(torch.int16), r.view(torch.int16)) for r in results[1:])
            finite = bool(torch.isfinite(first).all())
            features[item.name] = first.cpu()
            info[item.name] = dict(size=list(item.size), grid=item.grid, patches=count, features=count // 4,
                                   capacity=capacity, errors=errors, replays=args.replays,
                                   replays_identical=same, finite=finite,
                                   published=int(state.count[SLOT].item()))
            log(f"{item.name}: {item.size[0]}x{item.size[1]} grid {item.grid} patches {count} cap {capacity} "
                f"errors {errors} replays identical {same} finite {finite} max|x| {first.float().abs().max():.4g}")
    meta = dict(label=args.label or worktree.name, worktree=str(worktree), git=git_head(worktree),
                b12x=b12x.__file__, torch=torch.__version__, gpu=torch.cuda.get_device_name(device),
                snapshot=str(args.snapshot), stage=args.stage, overlap=args.overlap, compiler=args.compiler,
                capacities=capacities, seconds=round(time.time() - started, 1),
                inputs_sha256={item.name: hashlib.sha256(item.pixels.numpy().tobytes()).hexdigest()
                               for item in items})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=meta, info=info, features=features), args.out)
    bad = [n for n, v in info.items() if any(v["errors"]) or not v["finite"] or not v["replays_identical"]
           or v["published"] != v["features"]]
    log(f"saved {args.out}; {len(features)} images; problems: {bad or 'none'}")
    return 1 if bad else 0


# CPU check ------------------------------------------------------------------------------------------------------

def check_cpu(args):
    import torch
    import ignition

    if torch.cuda.is_available():
        raise SystemExit("--check-cpu must run with CUDA_VISIBLE_DEVICES= (no GPU)")
    worktree = pathlib.Path(ignition.__file__).resolve().parents[1]
    print(f"ignition {ignition.__file__} ({git_head(worktree)})")
    ok = True
    for case, grid, pixels, positions in frontend_fixture_parity(worktree):
        ok &= grid and pixels and positions
        print(f"  frontend fixture {case}: grid {grid} pixels {pixels} positions {positions}")
    # Every symbol and constructor signature the GPU run uses resolves in this worktree.
    import inspect

    from b12x.preparation import PreparationSession  # noqa: F401
    from b12x.sequence._shared.disk_table import _tensor_from_pointer  # noqa: F401
    from ignition.components.weights import DirectWeightLoader  # noqa: F401
    from ignition.preparation.aot import ForwardCache  # noqa: F401
    from ignition.preparation.compilation import compile_forward  # noqa: F401
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.scratch import ScratchArena, overlay  # noqa: F401
    from ignition.recipes.glm53.vision import (PreparedVision, VisionMetadata, VisionScratch, VisionStaging,
                                               VisionWeights)
    from ignition.recipes.glm53.vision_state import FEATURES, PATCH_CAPACITIES, PreparedVisionGeneration

    for function, names in ((PreparedVision.__init__, ("module", "scratch", "metadata", "weights", "session", "name",
                                                       "staging")),
                            (VisionStaging.__init__, ("weights", "scratch", "stream")),
                            (PreparedVisionGeneration.__init__, ("encoder", "inputs", "state", "completion",
                                                                 "frame", "sequence")),
                            (VisionWeights.park, ("allocate",)), (VisionScratch.allocate, ("patches", "device"))):
        missing = set(names) - set(inspect.signature(function).parameters)
        ok &= not missing
        print(f"  signature {function.__qualname__}: {'ok' if not missing else f'MISSING {sorted(missing)}'}")
    GraphSignature("vision", 1, 4096)
    # Serving geometry on the meta device: scratch at 32768, overlay-free views per capacity, serving state.
    scratch = VisionScratch.allocate(max(PATCH_CAPACITIES), "meta")
    scratch_mib = sum(t.untyped_storage().nbytes() for t in {id(getattr(scratch, f).untyped_storage()):
                      getattr(scratch, f) for f in scratch.__dataclass_fields__}.values()) >> 20
    s = serving_state("meta")
    ok &= s.state.arena.shape == (1, FEATURES, 4096)
    print(f"  VisionScratch(32768) {scratch_mib} MiB; feature arena {tuple(s.state.arena.shape)}")
    items = build_inputs(args)
    for item in items:
        capacity = capacity_for(item.count, PATCH_CAPACITIES)
        view = scratch.rows(capacity)
        VisionMetadata.allocate(capacity, 1, "meta")
        ok &= view.output.shape == (capacity // 4, 4096) and item.count // 4 <= FEATURES
        print(f"  input {item.name}: {item.size[0]}x{item.size[1]} -> grid {item.grid} patches {item.count} "
              f"tokens {item.count // 4} capacity {capacity}")
    print(f"CHECK_CPU {'PASS' if ok else 'FAIL'} ({worktree.name}, {len(items)} inputs)")
    return 0 if ok else 1


def main(argv=None):
    args = parse_args(argv)
    if args.check_cpu:
        return check_cpu(args)
    import torch
    torch.set_grad_enabled(False)
    return run_gpu(args, build_inputs(args))


if __name__ == "__main__":
    sys.exit(main())
