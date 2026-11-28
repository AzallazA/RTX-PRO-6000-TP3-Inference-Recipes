#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""exp/ds41-exl3: CPU-only plan check of a TP2 DS41 EXL3 export (no CUDA, no GPU tool, nothing written).

Runs, for the checkpoint and FLARE_DS41_* environment the export would use:
  1. recipe resolution (architecture, recipe, world-2 rank costs) and ModelConfig / TP3Config validation;
  2. inspect_indexed_checkpoint (the reader that rejected F16/I16 in seq220);
  3. the real DeepSeek weight loader (ModelWeights.load) for BOTH ranks on PyTorch's meta device: every tensor
     family (non-expert FP8/BF16, EXL3 routed banks incl. rank slicing, parked mixed banks + hot maps, drafter MCG
     banks, embedding/vision parking) is requested by name, sliced and shape-checked against its destination; no
     bytes are read. Steps that need CUDA kernels on real data are stubbed and listed in the report;
  4. Engram Tables.prepare on CPU (int4 detection, 144-byte rows, encoding name, plane spans);
  5. the TP2 graph planning of the DS41 preparation config (batch capacities, packed verification shapes, vocabulary
     channels) as recipe.prepare builds it.
Exit 0 = everything planned; any exception = nonzero with the traceback.
  CUDA_VISIBLE_DEVICES= .venv/bin/python tp3-tools/ds41_exl3_plan_check.py CKPT [--world 2]
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    sys.exit("plan check: run with CUDA_VISIBLE_DEVICES= (CPU only)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

STUBBED = []


def stub(name):
    if name not in STUBBED:
        STUBBED.append(name)


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class MetaWeight:
    """Shape-only stand-in for b12x's packed Weight (its packers require CUDA tensors)."""
    values: object
    scales: object
    recipe: object
    in_features: int


HOST_BACKED = {}   # meta storage -> (device bytes at its start, name): host_allocate (0) and mixed banks (device rows)


class MetaSession:
    """The checks of b12x DirectWeightSession.__call__ for checkpoint sources (shape, strides, extent within the
    tensor, element size), plus a count of the read records it would issue (one per contiguous run)."""
    payload_bytes = 0
    records = 0

    def __call__(self, destination, source):
        if tuple(destination.shape) != tuple(source.shape):
            raise ValueError(f"direct transfer shape mismatch: {tuple(source.shape)} -> {tuple(destination.shape)}")
        if any(st < 0 for st in (*source.stride(), *destination.stride())):
            raise ValueError("direct transfer with a negative stride")
        base = source.untyped_storage().nbytes()
        extent = (1 + sum((n - 1) * st for n, st in zip(source.shape, source.stride()))) * source.element_size()
        if source.numel() and source.storage_offset() * source.element_size() + extent > base:
            raise ValueError("source view exceeds the checkpoint tensor")
        elements, axis = 1, source.ndim - 1
        while axis >= 0:
            size = source.shape[axis]
            if size != 1 and (source.stride(axis) != elements or destination.stride(axis) != elements):
                break
            elements *= size
            axis -= 1
        outer = 1
        for n in (source.shape[:axis] if axis >= 0 else ()):
            outer *= n
        self.records += outer
        # b12x's batch executor (_batch.c / _cuda_range.h): the destination extent must lie in CUDA DEVICE memory of
        # the session device. Host-backed storage (host_allocate) or the host rows of a mixed bank are refused there.
        limit = HOST_BACKED.get(destination.untyped_storage()._cdata)
        if limit is not None:
            start = destination.storage_offset() * destination.element_size()
            d_extent = (1 + sum((n - 1) * st for n, st in zip(destination.shape, destination.stride()))) \
                * destination.element_size()
            if start + d_extent > limit[0]:
                raise ValueError(f"direct transfer into host-backed memory ({limit[1]}, bytes "
                                 f"{start}..{start + d_extent} beyond the {limit[0]} device bytes): b12x's batch "
                                 "executor accepts CUDA device allocations only")
        # b12x's session converts between floating dtypes (e.g. BF16 -> FP32); integer payloads must match exactly.
        if destination.dtype != source.dtype and not (destination.is_floating_point() and source.is_floating_point()):
            raise ValueError(f"direct transfer dtype mismatch: {source.dtype} -> {destination.dtype}")
        self.payload_bytes += source.numel() * source.element_size()
        return True

    def flush(self):
        pass

    def stats(self):
        return {}


def meta_sources(manifest):
    from ignition.preparation.checkpoint_source import tensor_dtype

    return {name: torch.empty(t.shape, dtype=tensor_dtype(t.dtype), device="meta")
            for name, t in manifest.tensors.items()}


def walk_weights(manifest, rank, world, tp3):
    """ModelWeights.load for one rank on the meta device (shapes only)."""
    from hyperflare_recipes.components import weights as cw
    from hyperflare_recipes.recipes.deepseek41 import weights as dw
    from hyperflare_recipes.components.parallel import TensorParallelShard

    sources = meta_sources(manifest)
    sources_session = []

    def enter(self):
        self.device = torch.device("meta")
        self._session = MetaSession()
        sources_session[:] = [self._session]
        self._sources = dict(sources)
        self._stack = contextlib.ExitStack()
        self._display = self._load_stats = None
        return self

    def exit_(self, *exc):
        self._session = None
        return False

    host = {}

    class MixedRegion:
        """core_storage.MixedRegion's size arithmetic (2 MiB granules; the boundary page stays on the device)."""
        granularity = 2 << 20

        def __init__(self, name, shape, dtype, device_rows):
            self.name = name
            self.device_view = torch.empty(shape, dtype=dtype, device="meta")
            row = self.device_view[0].numel() * self.device_view.element_size() if shape[0] else 0
            up = lambda n: -(-n // self.granularity) * self.granularity
            self.extent = up(shape[0] * row)
            self.device_bytes = min(up(device_rows * row), self.extent)
            self.row_bytes, self.mapped_rows = row, device_rows

    class Storage:
        def host_allocate(self, name, shape, dtype):
            host[name] = (tuple(shape), dtype)
            region = SimpleNamespace(device_view=torch.empty(shape, dtype=dtype, device="meta"),
                                     host_view=torch.empty(shape, dtype=dtype, device="meta"))
            for view in (region.device_view, region.host_view):
                HOST_BACKED[view.untyped_storage()._cdata] = (0, name, view)   # keep the view: no address reuse
            return region

        def mixed_allocate(self, name, shape, dtype, *, device_rows):
            region = MixedRegion(name, shape, dtype, device_rows)
            host[name] = (tuple(shape), dtype, device_rows, region)
            row = region.device_view[0].numel() * region.device_view.element_size() if shape[0] else 0
            HOST_BACKED[region.device_view.untyped_storage()._cdata] = (device_rows * row, name, region.device_view)
            return region

    storage = Storage()
    patches = [(cw.DirectWeightLoader, "__enter__", enter), (cw.DirectWeightLoader, "__exit__", exit_)]
    saved = [(o, n, o.__dict__.get(n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    try:
        with stubbed_kernels():
            weights = dw.ModelWeights.load(manifest, torch.device("meta"), shard=TensorParallelShard(rank, world),
                                           dspark=True, confidence=True, draft_head_quantization="nvfp4",
                                           markov_head_quantization="fp8", vocabulary_parallel_sampling=True,
                                           host_allocator=storage.host_allocate, tp3=tp3)
    finally:
        for o, n, v in saved:
            if v is None:
                delattr(o, n)
            else:
                setattr(o, n, v)
    print(f"  rank {rank}: {sources_session[0].records:,} direct read records, "
          f"{sources_session[0].payload_bytes / 2**30:.1f} GiB requested")
    return weights, host


@contextlib.contextmanager
def stubbed_kernels():
    """CUDA-only helpers that transform loaded data: replaced by shape-preserving stand-ins on meta tensors."""
    import hyperflare_recipes.components.weights as cw

    saved = {}

    def patch(module, name, replacement):
        if hasattr(module, name):
            saved[(module, name)] = getattr(module, name)
            setattr(module, name, replacement)

    def allocated(device=None):
        return 0

    from ignition.preparation import persistent
    patch(persistent, "allocated_tensor_bytes", allocated)
    import b12x.gemm.blockscaled as blockscaled
    import b12x._lib.intrinsics.reference as reference

    def pack_weight(values, scales=None, *, recipe=None, allocate=None, **options):
        stub(f"b12x pack_weight({recipe})")
        return MetaWeight(values, scales, recipe, values.shape[-1] if values is not None else 0)

    def swizzle_block_scale(scales, *, allocate=None):
        stub("b12x swizzle_block_scale")
        return scales

    import b12x.gemm._shared.wo_mxfp8 as wo_mxfp8

    def check_gpu_tensor(name, tensor, *args, **kwargs):
        stub("b12x CUDA-tensor checks (wo projection packing runs on meta tensors)")

    patch(wo_mxfp8, "_check_gpu_tensor", check_gpu_tensor)
    patch(blockscaled, "pack_weight", pack_weight)
    patch(reference, "swizzle_block_scale", swizzle_block_scale)
    import hyperflare_recipes.recipes.deepseek41.weights as dw
    patch(dw, "allocated_tensor_bytes", allocated)
    patch(dw, "release_unused", lambda: None)
    for name in ("nvfp4_draft_head", "mxfp8_draft_head", "fp8_markov_head"):
        def head(weight, _name=name):
            stub(f"{_name} (quantizes real values on CUDA)")
            return weight
        patch(dw, name, head)
    try:
        yield
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)


def summarize(weights, host, rank):
    from hyperflare_recipes.recipes.deepseek41.exl3.experts import Exl3Experts

    banks = [layer.experts.experts for layer in weights.layers]
    exl3 = [b for b in banks if isinstance(b, Exl3Experts)]
    parked = [i for i, b in enumerate(banks) if isinstance(b, Exl3Experts) and b.parked]
    draft = [] if weights.draft is None else [layer.experts.experts for layer in weights.draft.layers]
    print(f"  rank {rank}: {len(weights.layers)} layers, EXL3 banks {len(exl3)} (parked {parked or 'none'}), "
          f"drafter banks {len(draft)} codebook {sorted({d.codebook for d in draft})}, "
          f"local intermediate {exl3[0].local_intermediate if exl3 else '-'}, host regions {len(host)}")
    if len(exl3) != len(banks) or any(not isinstance(d, Exl3Experts) or d.codebook != "mcg" for d in draft):
        raise ValueError("not every routed/drafter expert bank is EXL3 with the expected codebook")
    check_wo_layouts(weights, rank)
    check_declared_layout(weights, host, rank)


def wo_tiled_copies(weights):
    """(attention layers, [(name, bytes)] of b12x tile-major WO copies kept) over target + drafter layers."""
    layers = [(f"layers.{i}", layer) for i, layer in enumerate(weights.layers)]
    if weights.draft is not None:
        layers += [(f"mtp.{i}", layer) for i, layer in enumerate(weights.draft.layers)]
    kept = []
    for name, layer in layers:
        output = layer.attention["output"]
        for matrix in ("wo_a", "wo_b"):
            tiled = getattr(output, matrix).values_tiled
            if tiled is not None:
                kept.append((f"{name}.{matrix}", tiled.numel() * tiled.element_size()))
    return len(layers), kept


def check_wo_layouts(weights, rank):
    """FLARE_DS41_EXL3_WOA_SINGLE: on = no tile-major WO copy is kept; off = b12x's copies as packed (TP2: WO-A of every
    attention layer). The WO plan never declares wo_a_tiled/wo_b_tiled, so a kept copy is never read."""
    from hyperflare_recipes.recipes.deepseek41.weights import woa_single

    count, kept = wo_tiled_copies(weights)
    nbytes = sum(n for _, n in kept)
    single = woa_single()
    print(f"  rank {rank}: WO tile-major copies kept {len(kept)} of {count} attention layers "
          f"({nbytes / 2**20:.0f} MiB){'; FLARE_DS41_EXL3_WOA_SINGLE=1' if single else ''}")
    if single and kept:
        raise ValueError(f"FLARE_DS41_EXL3_WOA_SINGLE=1 but tile-major WO copies remain: {[n for n, _ in kept][:4]}")


def check_declared_layout(weights, host, rank):
    """The package layout of the parked banks, proved on the meta regions: per bank and matrix the region mapped
    during the export (VRAM rows), the VRAM prefix the package declares (runtime_core.mixed_device_bytes, the name
    Helios reads) and the kernels' hot_rows must agree -- declared prefix == whole granules covering hot_rows rows,
    mapped <= declared <= extent, and every bank's matrices declare the same row count."""
    from ignition.preparation.runtime_core import mixed_device_bytes

    regions = {name: entry[3] for name, entry in host.items() if len(entry) == 4}
    parked = [(i, layer.experts.experts) for i, layer in enumerate(weights.layers) if layer.experts.experts.parked]
    mapped = declared = 0
    rows = set()
    for i, bank in parked:
        for matrix in ("w1", "w3", "w2"):
            name = f"layers.{i}.ffn.exl3.{matrix}"
            region = regions.get(name)
            if region is None:
                raise ValueError(f"parked layer {i}: no mixed region {name}")
            want_mapped = bank.hot_rows if bank.mapped_rows is None else bank.mapped_rows
            if region.mapped_rows != want_mapped:
                raise ValueError(f"{name}: mapped {region.mapped_rows} VRAM rows, the bank says {want_mapped}")
            up = -(-bank.hot_rows * region.row_bytes // region.granularity) * region.granularity
            got = mixed_device_bytes(region)
            if got != min(up, region.extent):
                raise ValueError(f"{name}: package declares {got} VRAM bytes, the kernels' hot_rows {bank.hot_rows} "
                                 f"need {min(up, region.extent)}")
            mapped += region.device_bytes
            declared += got
            rows.add((bank.hot_rows, bank.mapped_rows))
    from hyperflare_recipes.recipes.deepseek41.exl3.experts import cold_side_sms, hot_layers, side_budgets
    override = hot_layers()
    stray = sorted(set(override) - {i for i, _ in parked})
    if stray:
        raise ValueError(f"FLARE_DS41_EXL3_HOT_LAYERS names layers that are not parked: {stray}")
    for i, bank in parked:
        if i in override and bank.hot_rows != override[i]:
            raise ValueError(f"layer {i}: bank hot_rows {bank.hot_rows} != FLARE_DS41_EXL3_HOT_LAYERS {override[i]}")
    if override:
        print(f"  rank {rank}: per-layer hot counts: {len(override)} layers overridden, sum of hot rows over parked "
              f"banks {sum(b.hot_rows for _, b in parked):,} (uniform would be {len(parked)} x HOT)")
    k = cold_side_sms()
    if k:
        hot_sms, cold_sms = side_budgets(188, k)     # RTX PRO 6000 Blackwell: 188 SMs
        print(f"  cold side stream: hot flat kernel {hot_sms} SMs + cold group kernel {cold_sms} SMs (alternate locks)")
    if parked:
        print(f"  rank {rank}: {len(parked)} parked banks; kernel hot rows / export VRAM rows {sorted(rows)}; mixed "
              f"VRAM {mapped / 2**30:.2f} GiB mapped during the export, {declared / 2**30:.2f} GiB declared to the "
              f"package (+{(declared - mapped) / 2**30:.2f} GiB at serve)")


def check_engram(manifest, config, world):
    from hyperflare_recipes.recipes.deepseek41 import engram as recipe_engram
    from hyperflare_recipes.recipes.deepseek41.engram import Tables
    from hyperflare_recipes.components.parallel import TensorParallelShard

    saved = (recipe_engram.ReadinessTimings.allocate, recipe_engram.EngramInputs.allocate)
    recipe_engram.ReadinessTimings.allocate = staticmethod(lambda frame, count: None)
    seen = []
    recipe_engram.EngramInputs.allocate = staticmethod(
        lambda tokens, device, *, timings, interval=0, packed_fp4=False, scale_bytes=8:
        seen.append((packed_fp4, scale_bytes)) or SimpleNamespace(packed_fp4=packed_fp4))
    try:
        host = lambda name, shape, dtype: SimpleNamespace(host_view=torch.empty(shape, dtype=dtype, device="meta"))
        tables = Tables.prepare(manifest, config, tokens=4096, device="cpu", sequence=None, frame=None,
                                host_allocator=host, shard=TensorParallelShard(0, world), shared=True)
    finally:
        recipe_engram.ReadinessTimings.allocate, recipe_engram.EngramInputs.allocate = saved
    meta = tables.export_metadata()
    encodings = sorted({t["encoding"] for t in meta["tables"]})
    print(f"  Engram: {len(meta['tables'])} tables, int4 {tables.int4}, row bytes {tables.row_bytes}, "
          f"encodings {encodings}, staged scale bytes {sorted({s for _, s in seen})}")
    if not tables.int4 or tables.row_bytes != (128, 16) or encodings != ["INT4_FP16_BLOCK32"]:
        raise ValueError("Engram tables are not the expected int4 rows")


def xmoe_scratch_line(chunk, world, decode_rows):
    """exp/ds41-exl3-final-scratch: the shared xmoe scratch per GPU (core storage, persistent; the export's KV comes from
    the VRAM left after it, so a smaller scratch shows up as KV) for the FLARE_DS41_EXL3_PREFILL* switches as set,
    against the defaults; refuses the cold side stream outside the decode-once mode."""
    from hyperflare_recipes.recipes.deepseek41.exl3 import xmoe
    from hyperflare_recipes.recipes.deepseek41.exl3.experts import scratch_report

    r = scratch_report(chunk, xmoe.INTERMEDIATE // world, decode_rows)
    if os.environ.get("FLARE_DS41_EXL3_PREFILL_COLD_SIDE", "").strip() == "1" and r["mode"] != "do":
        raise ValueError("FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 needs FLARE_DS41_EXL3_PREFILL=do (MOE_PREFILL=do)")
    mib = lambda b: f"{b / 2**20:,.1f} MiB"   # noqa: E731
    return (f"xmoe scratch per GPU: {mib(r['bytes'])} (defaults {mib(r['default_bytes'])}, saved {mib(r['saved_bytes'])}; "
            f"prefill {r['mode']}, chunk {chunk}, groups {r['groups']}, decode-once pass {r['pass_routes']} routes, "
            f"cold side stream {'on' if r['cold_side'] else 'off'})")


RTX_PRO_6000 = 102_641_893_376   # NVML total bytes of one RTX PRO 6000 Blackwell (96 GB class)


def check_plan(export_argv):
    """The export CLI's planning (recipe, GPU selection, memory plan, speculation shapes) on two fake idle RTX PRO 6000
    GPUs, then recipe.prepare up to the point it hands the preparation config to the rank group (captured, not run)."""
    from ignition import planning
    from ignition.cli import _parser
    from ignition.cli import preparation as prep
    from ignition.cli.config import Configuration
    from hyperflare_recipes.recipes import recipes as all_recipes
    from hyperflare_recipes.recipes.deepseek41 import recipe as ds41

    args = _parser().parse_args(export_argv)
    configuration = Configuration.load(args.config)
    model = prep.resolve_model(configuration, args.model, recipes=all_recipes())
    gpus = tuple(planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell (plan check)", f"GPU-plan-{i}", RTX_PRO_6000,
                              RTX_PRO_6000 - (512 << 20), (12, 0)) for i in range(2))
    host = prep.Host(gpus, None)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency, args.max_concurrency)
    plan = prep.derive(configuration, model, host, gpus=args.gpus, concurrency=concurrency,
                       compile_workers=args.compile_workers, compile_headroom=args.compile_headroom,
                       tuning_gpus=args.tuning_gpus, autotune_effort=args.autotune_effort,
                       speculation=prep.speculation(args, model, configuration), prefill_chunk=args.prefill_chunk_size,
                       context_parallel=prep.context_parallel(args, model),
                       context_parallel_size=getattr(args, "dcp_size", None),
                       stream_overlap=prep.stream_overlap(args, model), mixed_prefill=args.mixed_prefill,
                       prompt_logprobs=args.prompt_logprobs, expert_scale_compression=args.expert_scale_compression,
                       image_concurrency=args.image_concurrency, mla_compute_mode=args.mla_compute_mode)
    print(f"  plan: world {plan.world_size}, concurrency {plan.concurrency}, prefill chunk {plan.prefill_chunk}, "
          f"context {plan.context}, minimum pages {plan.minimum_pages}, snapshots {plan.snapshots}, "
          f"speculation widths {plan.speculation.widths if plan.speculation else None}, "
          f"token capacity {plan.speculative_token_capacity}")
    captured = {}
    import ignition.preparation.rank_group as rank_group
    saved = rank_group.prepare_group
    rank_group.prepare_group = lambda snapshot, config, prepare, **kw: captured.setdefault("config", config)
    try:
        ds41.prepare(model.snapshot(configuration.directory("hub_cache")), plan, compile_forwards=True,
                     kernel_cache=None, progress=None)
    finally:
        rank_group.prepare_group = saved
    config = captured["config"]
    from hyperflare_recipes.recipes.deepseek41.bootstrap import vram_reclaim_report
    widths = config.speculative_widths
    print(f"  preparation config: devices {config.devices}, request capacity {config.request_capacity}, "
          f"batch capacities {config.batch_capacities}, speculative widths {widths}, draft tokens {config.draft_tokens}, "
          f"verification rows up to {config.request_capacity * max(widths)} (flat xmoe kernel <= 64: "
          f"{'yes' if config.request_capacity * max(widths) <= 64 else 'NO'})")
    print(f"  VRAM reclaim: {vram_reclaim_report(config, config.request_capacity * config.draft_tokens)}")
    print(f"  {xmoe_scratch_line(plan.prefill_chunk, plan.world_size, config.request_capacity * max(widths))}")
    if config.request_capacity * max(widths) > 64:
        raise ValueError("verification rows exceed the flat xmoe kernel's 64 rows (decode would take the group path)")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--world", type=int, default=2)
    argv = sys.argv[1:]
    export_argv = argv[argv.index("--") + 1:] if "--" in argv else []   # -- export MODEL ... (the export's own argv)
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    a.export_argv = export_argv
    t0 = time.time()
    from hyperflare_recipes.recipes import resolve
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
    from hyperflare_recipes.recipes.deepseek41.tp3_config import TP3Config
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    model = resolve(str(a.ckpt))
    costs = model.recipe.ranks.get(a.world)
    print(f"1 recipe {model.recipe.name if hasattr(model.recipe, 'name') else model.recipe} checkpoint "
          f"{model.checkpoint.alias}; world {a.world} rank costs {costs}")
    if costs is None:
        raise ValueError(f"recipe has no rank costs for world {a.world}")
    tp3 = TP3Config.from_environment()
    config = ModelConfig.load(a.ckpt, mini=tp3.mini)
    tp3.check_layers(config.layers)
    print(f"  ModelConfig: {config.layers} layers, exl3 {config.exl3}; TP3: host layers "
          f"{sorted(tp3.host_expert_layers) or 'none'} hot {tp3.hot_experts} mini {tp3.mini}")
    manifest = inspect_indexed_checkpoint(a.ckpt)
    print(f"2 checkpoint: {len(manifest.tensors)} tensors in {len(manifest.shards)} shards")
    print("3 weights (meta device, both ranks):")
    for rank in range(a.world):
        weights, host = walk_weights(manifest, rank, a.world, tp3)
        summarize(weights, host, rank)
    print("4 Engram tables:")
    check_engram(manifest, config, a.world)
    if a.export_argv:
        print("5 export planning and preparation config:")
        check_plan(a.export_argv)
    print(f"plan check PASS in {time.time() - t0:.1f} s" + (f"; stubbed: {', '.join(STUBBED)}" if STUBBED else ""))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        print("plan check FAILED")
        sys.exit(1)
