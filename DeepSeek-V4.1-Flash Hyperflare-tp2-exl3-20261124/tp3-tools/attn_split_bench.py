# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU microbench for DS41 TP3 decode lever 4 (attention 16 + 8 head grids): split counts per row count.

One GPU, nothing else running on it (checked through NVML before CUDA starts). Times b12x's prepared DSV41 BF16
compressed MLA decode exactly as a TP3 rank runs it (24 heads, SWA 128 + indexed 512 candidates, page table, attention
sink, provided output, remainder grid on a side stream = B12X_MLA_DECODE_REMAINDER_STREAM=1) inside a CUDA graph, for:

  * splits S in {10 (production), 5, 4, 3, 2, 1}, forced through ignition/components/mla_decode_splits.py (the
    FLARE_DS41_MLA_DECODE_SPLITS hook; B12X_MLA_SM120_NUM_SPLITS does not reach DSV41 decode);
  * 16 heads (one grid: the lower bound for a one-launch 24-head CTA) at S = 10 and 5, and S = 4 and 2 at >= 40 rows;
  * FP8 arithmetic with 8-head blocks (b12x's existing one-launch 3 x 8 path; numerics differ, timing reference only);
  * remainder stream off (serial grids) at S = 10, to price the stream itself;
  * --h24 / --h24-only: the one-launch 24-head kernel (ignition/components/mla_h24_decode.py, FLARE_DS41_MLA_H24) at
    every S of the sweep (decode lever 4b). --h24-only keeps just the 24-head 16 + 8 and H24 arms.

Cold gathers (review M1): every case binds SETS = 20 index sets (own SWA pages, page-table rows and candidates per set,
indexed pages drawn from a 1.2 GB pool built once) and the graph cycles through them, so consecutive calls never
re-read the same KV records; serving reads each layer's own KV.

Self-checks (review M2), exit 1 on failure:
  * every forced BF16 case: the hook applied, and the planned split count equals the forced one;
  * the serial arm's output equals the 10-split output bit for bit;
  * no forced case differs from 10 splits by more than 5e-2 (expected: about one BF16 step of a partial, <= ~1e-2);
  * every H24 case: the hook lowered the decode to the H24 program (applied, and the captured graph ran the
    prepared H24 program), and its output equals the 16 + 8 output at the same split count bit for bit.
Informational: 24-head 10-split times vs the 181718 trace (6.4 / 11.5 / 17 / 22.9 us at <=9 / 10-18 / 19-28 / 29-37
rows), flagged when off by more than 10 %.

Usage (GPUs exclusive: stop the server first; private compile caches are required):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 B12X_OVERLAY=... PYTHONPATH=<worktree> \\
      B12X_COMPILE_CACHE_DIR=<dir>/b12x TRITON_CACHE_DIR=<dir>/triton \\
      python tp3-tools/attn_split_bench.py --out <dir>/attn-splits.tsv [--quick]
  --quick: rows 5,13,23,32,40,80 and S 10,5,4,3,2 (about 6-8 min including compiles); full: about 12-15 min.
  --h24 adds 5 (quick) / 6 (full) H24 cases per row count (about +2 / +4 min); --h24-only --quick about 5 min.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

ROWS_FULL = (1, 5, 9, 10, 13, 18, 19, 23, 26, 28, 29, 32, 37, 40, 80)
ROWS_QUICK = (5, 13, 23, 32, 40, 80)
SPLITS_FULL = (10, 5, 4, 3, 2, 1)
SPLITS_QUICK = (10, 5, 4, 3, 2)
SWA_WIDTH, INDEXED_WIDTH, SWA_PAGE, INDEXED_PAGE = 128, 512, 128, 64
POOL_PAGES, TABLE_WIDTH = 65536, 2048  # 64-token indexed pages of 18 KB: 1.2 GB pool; 131K-token contexts per row
SETS = 20
SM_SCALE = 512 ** -0.5
TRACE_BANDS = ((9, 6.4), (18, 11.5), (28, 17.0), (37, 22.9))
MAX_DIFF = 5e-2


def preflight():
    """Refuse to touch a GPU that is not ours alone (before any CUDA call creates a context)."""
    for name in ("B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR"):
        if not os.environ.get(name):
            sys.exit(f"{name} must name a private cache directory (production caches stay pinned)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        sys.exit("set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to one GPU index")
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
    apps = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    pynvml.nvmlShutdown()
    if apps or used >= 1000:
        sys.exit(f"GPU {visible} is busy ({used} MiB used, {len(apps)} compute processes): stop the server first")


def build_pool(device, generator):
    import torch
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

    values = (torch.randn((INDEXED_PAGE, 512), device=device, generator=generator) * 0.2).bfloat16()
    page = pack_deepseek_v41_cache_reference(values, page_size=INDEXED_PAGE, cache_kind="indexed")
    pool = torch.empty((POOL_PAGES, page.shape[1]), dtype=torch.uint8, device=device)
    pool.copy_(page.expand(POOL_PAGES, -1))
    values = (torch.randn((SWA_PAGE, 512), device=device, generator=generator) * 0.2).bfloat16()
    swa_page = pack_deepseek_v41_cache_reference(values, page_size=SWA_PAGE, cache_kind="swa")
    return pool, swa_page


def build_sets(rows, heads, device, generator, swa_page):
    """SETS index sets over disjoint SWA pages and random indexed pages; one Q (same work per call)."""
    import torch

    pages_per_set = 2 * rows
    swa = swa_page.repeat(SETS * pages_per_set + 1, 1).contiguous()
    q = (torch.randn((rows, heads, 512), device=device, generator=generator) * 0.2).bfloat16()
    sets = []
    for index in range(SETS):
        first = 1 + index * pages_per_set
        base = (first + torch.arange(rows, device=device, dtype=torch.int32) * 2) * SWA_PAGE
        start = torch.randint(0, SWA_PAGE, (rows,), device=device, generator=generator, dtype=torch.int32)
        swa_indices = (base + start)[:, None] + torch.arange(SWA_WIDTH, device=device, dtype=torch.int32)[None]
        table = torch.randint(1, POOL_PAGES, (rows, TABLE_WIDTH), device=device, generator=generator,
                              dtype=torch.int32)
        logical = torch.stack([torch.randperm(TABLE_WIDTH * INDEXED_PAGE, device=device,
                                              generator=generator)[:INDEXED_WIDTH] for _ in range(rows)])
        sets.append(dict(q=q, swa_indices=swa_indices,
                         swa_lengths=torch.full((rows,), SWA_WIDTH, dtype=torch.int32, device=device),
                         indexed_indices=logical.to(torch.int32),
                         indexed_lengths=torch.full((rows,), INDEXED_WIDTH, dtype=torch.int32, device=device),
                         indexed_page_table=table))
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    return swa, sets, sink


def prepare(session, swa, pool, sets, sink, *, arithmetic, override_fp8_h8):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING
    from b12x.preparation import PreparedCall

    q = sets[0]["q"]
    rows, heads = q.shape[:2]
    output = torch.empty_like(q)
    caps = mla.Caps(device=q.device, num_q_heads=heads, max_q_rows=rows, max_width=SWA_WIDTH + INDEXED_WIDTH,
                    swa_width=SWA_WIDTH, indexed_width=INDEXED_WIDTH, max_page_table_width=TABLE_WIDTH,
                    swa_page_size=SWA_PAGE, indexed_page_size=INDEXED_PAGE, cache_format="deepseek_v41",
                    mode="decode", decode_row_capacity=rows, use_cuda_graph=True, compute_mode=arithmetic)
    invocation = mla.invocation_from_tensors(q=q, swa_k_cache=swa, indexed_k_cache=pool, attn_sink=sink, out=output)
    plan = mla.plan(caps, invocation=invocation)
    if override_fp8_h8:
        config = replace(TUNING.configure(plan.query, device=session.device.identity).default,
                         v41_compute_mode="fp8", v41_heads_per_block=8)
        plan = mla.plan(caps, invocation=invocation, override=config)
    run_args = dict(swa_k_cache=swa, indexed_k_cache=pool, attn_sink=sink, sm_scale=SM_SCALE)

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)
        trial = torch.empty_like(output)
        binding = state.bind_for_preparation(scratch=scratch, **sets[0])
        return PreparedCall(run=lambda: state.run(binding, **run_args, out=trial), output=trial,
                            owners=(scratch, binding))

    session.prepare((plan.request(name="attn-split-bench", prepare_call=prepare_call,
                                  benchmark_call=prepare_call),))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)  # calls are sequential: shared
    bindings = [mla.bind(plan, scratch=scratch, **index_set) for index_set in sets]

    def call(index):
        return mla.run(binding=bindings[index % SETS], swa_k_cache=swa, swa_page_size=SWA_PAGE,
                       indexed_k_cache=pool, indexed_page_size=INDEXED_PAGE, attn_sink=sink, sm_scale=SM_SCALE,
                       out=output)

    return call, output, (scratch, bindings)


def time_graph(call, *, calls, replays=60, warmup=10):
    import torch

    for index in range(SETS):
        call(index)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for index in range(calls):
                call(index)
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls)
    samples.sort()
    return samples[len(samples) // 2], samples[len(samples) // 10]


def run_case(device, pool, swa_page, rows, heads, splits, variant, cache_dir):
    """variant: bf16 (production), fp8h8 (b12x one-launch 3 x 8 heads, FP8 arithmetic), serial (stream off),
    h24 (one-launch 24-head kernel, remainder stream on as in production)."""
    import torch
    from b12x.preparation import PreparationSession
    from ignition.components import mla_decode_splits, mla_h24_decode, mla_remainder_stream

    mla_h24_decode.uninstall()  # the remainder-stream hook must sit under the H24 hook, never above it
    os.environ[mla_remainder_stream.ENV] = "0" if variant == "serial" else "1"
    mla_remainder_stream.install()
    hook = mla_decode_splits.install(((rows, splits),)) if splits is not None else None
    fused = mla_h24_decode.install(((rows, True),)) if variant == "h24" else None
    generator = torch.Generator(device=device).manual_seed(4118 + rows)
    swa, sets, sink = build_sets(rows, heads, device, generator, swa_page)
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=0, cache_dir=cache_dir) as session:
            call, output, owners = prepare(session, swa, pool, sets, sink,
                                           arithmetic="auto" if variant == "fp8h8" else "bf16",
                                           override_fp8_h8=variant == "fp8h8")
            median, p10 = time_graph(call, calls=100 if rows <= 18 else 40)
            call(0)
            torch.cuda.synchronize()
            result = output.float().clone()
    finally:
        mla_decode_splits.uninstall()
        mla_h24_decode.uninstall()
    applied = list(hook.applied) if hook else []
    info = dict(rows=rows, heads=heads, forced_splits=splits, variant=variant, us=round(median, 3),
                us_p10=round(p10, 3), applied=applied)
    if fused is not None:
        info.update(h24_applied=list(fused.applied), h24_replays=fused.replays)
    return info, result


def cases(rows_list, splits_list, h24="off"):
    """h24: off | on (add H24 arms) | only (24-head 16 + 8 and H24 arms only)."""
    for rows in rows_list:
        for splits in splits_list:
            yield rows, 24, splits, "bf16"
        if h24 != "off":
            for splits in splits_list:  # after the 16 + 8 arm at the same split count (its bit-identity reference)
                yield rows, 24, splits, "h24"
        if h24 == "only":
            continue
        for splits in (10, 5) + ((4, 2) if rows >= 40 else ()):
            yield rows, 16, splits, "bf16"
        yield rows, 24, None, "fp8h8"
        yield rows, 24, 10, "serial"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--h24", action="store_true", help="add one-launch 24-head (H24) arms at every split count")
    parser.add_argument("--h24-only", action="store_true", help="only the 24-head 16 + 8 and H24 arms")
    parser.add_argument("--cache-dir", default=None, help="b12x preparation cache (default: <out dir>/prep-cache)")
    args = parser.parse_args()
    preflight()
    import torch

    if torch.cuda.device_count() != 1:
        sys.exit("run with exactly one visible GPU (CUDA_VISIBLE_DEVICES=N)")
    device = torch.device("cuda", 0)
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "prep-cache")
    os.makedirs(cache_dir, exist_ok=True)
    rows_list, splits_list = (ROWS_QUICK, SPLITS_QUICK) if args.quick else (ROWS_FULL, SPLITS_FULL)
    pool, swa_page = build_pool(device, torch.Generator(device=device).manual_seed(1118))
    h24_mode = "only" if args.h24_only else ("on" if args.h24 else "off")
    reference, same_split, failures, times = {}, {}, [], {}
    started = time.time()
    with open(args.out, "w") as handle:
        handle.write("rows\theads\tvariant\tforced_splits\tplanned_splits\tus_median\tus_p10\t"
                     "max_abs_diff_vs_S10\tapplied\n")
        for rows, heads, splits, variant in cases(rows_list, splits_list, h24_mode):
            info, out = run_case(device, pool, swa_page, rows, heads, splits, variant, cache_dir)
            key = (rows, heads)
            if variant == "bf16" and splits == 10:
                reference[key] = out
            if variant == "bf16" and heads == 24:
                same_split[(rows, splits)] = out
            diff = (out - reference[key]).abs().max().item() if key in reference else float("nan")
            planned = info["applied"][-1][2] if info["applied"] else None
            if variant == "h24":
                twin = same_split.get((rows, splits))
                same = twin is not None and torch.equal(out.view(torch.int32), twin.view(torch.int32))
                info["bit_identical_vs_16p8"] = same
                if not same:
                    gap = float("nan") if twin is None else (out - twin).abs().max().item()
                    failures.append(f"rows {rows} H24 S{splits}: differs from 16 + 8 at S{splits} (max {gap:.3e})")
                if info["h24_applied"] != [(rows, planned)] * len(info["h24_applied"]) or not info["h24_applied"] \
                        or info["h24_replays"] < 1:
                    failures.append(f"rows {rows} H24 S{splits}: hook applied {info['h24_applied']}, "
                                    f"prepared runs {info['h24_replays']}")
            if variant != "fp8h8":
                if not info["applied"] or planned != splits:
                    failures.append(f"rows {rows} heads {heads} {variant} S{splits}: planned {planned}, "
                                    f"applied {info['applied']}")
                if variant == "serial" and diff != 0.0:
                    failures.append(f"rows {rows}: serial arm differs from 10 splits by {diff:.3e}")
                if not diff <= MAX_DIFF:
                    failures.append(f"rows {rows} heads {heads} S{splits}: max diff {diff:.3e} > {MAX_DIFF}")
            times[(rows, heads, variant, splits)] = info["us"]
            handle.write(f"{rows}\t{heads}\t{variant}\t{splits}\t{planned}\t{info['us']}\t{info['us_p10']}\t"
                         f"{diff:.3e}\t{json.dumps(info['applied'])}\n")
            handle.flush()
            print(json.dumps(dict(info, diff=diff, elapsed_s=round(time.time() - started))), flush=True)
        for rows in rows_list:
            expected = next((us for bound, us in TRACE_BANDS if rows <= bound), None)
            ours = times.get((rows, 24, "bf16", 10))
            best = min((us, s) for (r, h, v, s), us in times.items() if r == rows and h == 24 and v == "bf16")
            trace = ("n/a" if expected is None else
                     f"trace {expected} us {'ok' if abs(ours - expected) <= 0.1 * expected else 'OFF>10%'}")
            line = (f"SUMMARY rows={rows} S10={ours} us best=S{best[1]} {best[0]} us ({best[0] - ours:+.2f}); {trace}; "
                    f"16-head S10={times.get((rows, 16, 'bf16', 10))} S5={times.get((rows, 16, 'bf16', 5))}; "
                    f"fp8h8={times.get((rows, 24, 'fp8h8', None))} serial={times.get((rows, 24, 'serial', 10))}")
            handle.write("# " + line + "\n")
            print(line, flush=True)
            fused = {s: us for (r, h, v, s), us in times.items() if r == rows and v == "h24"}
            if fused:
                h_us, h_s = min((us, s) for s, us in fused.items())
                same = " ".join(f"S{s}:{times[(rows, 24, 'bf16', s)]}->{us}" for s, us in sorted(fused.items(),
                                key=lambda item: -item[0]) if (rows, 24, "bf16", s) in times)
                line = (f"H24 rows={rows} best=S{h_s} {h_us} us vs 16+8 best S{best[1]} {best[0]} us "
                        f"({h_us - best[0]:+.2f}) vs S10 {ours} us ({h_us - ours:+.2f}); same-S {same}")
                handle.write("# " + line + "\n")
                print(line, flush=True)
    if failures:
        print("SELF-CHECK FAILED:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)
    print("SELF-CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
