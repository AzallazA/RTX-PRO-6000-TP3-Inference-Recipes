"""CPU-only import/API check for the b12x bump image (no GPU: run with CUDA_VISIBLE_DEVICES= and no --gpus).

  docker run --rm --name b12xbump-check-$$ -e CUDA_VISIBLE_DEVICES= -v "$PWD/patch-b12x-bump:/chk:ro" \
      --entrypoint /opt/venv/bin/python ds41-kraken-tp3:20261019.13-b12x-test /chk/check_imports.py

Prints one line per check (ok / FAIL / info) and exits 1 if any check failed. Also runs on the .11 image as a
control (expect the 'new API' checks to fail there and everything else to pass).
"""
import importlib
import inspect
import os
import sys
import traceback

FAILED = []


def check(name, fn):
    try:
        r = fn()
        print(f"ok    {name}{': ' + str(r) if r not in (None, True) else ''}", flush=True)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
        if os.environ.get("CHECK_TRACE"):
            traceback.print_exc()


def imp(mod):
    return lambda: importlib.import_module(mod) and None


def b12x_origin():
    import b12x  # noqa: F401
    import b12x.moe.fused_moe._impl as impl

    meta = open("/opt/ds41/b12x-bump.txt").read().split() if os.path.exists("/opt/ds41/b12x-bump.txt") else ["(not a bump image)"]
    return f"{os.path.dirname(impl.__file__)} {' '.join(meta)}"


def indexer_api():
    from b12x.attention import dsa_indexer

    missing = [n for n in ("plan", "Caps", "bind", "run", "scratch_specs", "quantize_q_mxfp4", "quantize_write_index_k_mxfp4") if not hasattr(dsa_indexer, n)]
    gone = [n for n in ("score", "select") if hasattr(dsa_indexer, n)]
    if missing:
        raise AssertionError(f"missing {missing}")
    if gone:
        raise AssertionError(f"staged API still present {gone} (old b12x)")
    from b12x.attention.dsa_indexer._mxfp4_preparation import MXFP4PreparedState

    need = [n for n in ("bind", "run", "quantize_query", "write_index_keys", "layout") if not hasattr(MXFP4PreparedState, n) and n != "layout"]
    if need:
        raise AssertionError(f"MXFP4PreparedState lacks {need}")
    caps = inspect.signature(dsa_indexer.Caps)
    for kw in ("device", "num_q_heads", "max_q_rows", "max_page_table_width", "topk", "mode", "cache_format", "page_size", "max_candidates", "candidate_topk_blocks"):
        if kw not in caps.parameters:
            raise AssertionError(f"Caps lacks {kw}")
    b = inspect.signature(dsa_indexer.bind)
    for kw in ("scratch", "q_mxfp4", "q_scales", "query_weights", "index_k_cache", "page_table", "cache_lengths", "active_width", "score_width", "output_indices", "candidate_indices", "candidate_lengths", "candidate_output", "candidate_output_lengths"):
        if kw not in b.parameters and not any(p.kind is p.VAR_KEYWORD for p in b.parameters.values()):
            raise AssertionError(f"bind lacks {kw}")
    return "fused run(); staged score/select gone"


def tiny_guard():
    from b12x.moe.fused_moe import _impl

    f = _impl._w4a8_mx_micro_supported
    params = inspect.signature(f).parameters
    if "swiglu_limit" not in params:
        raise AssertionError("no swiglu_limit parameter: tiny_decode guard/clamp NOT applied")
    common = dict(num_tokens=1, k=4096, n=1024, num_topk=6, weight_E=256, activation="silu", source_format="fp4_e8m0_k32")
    clamped = f(**common, swiglu_limit=10.0)
    plain = f(**common)
    tiny_src = inspect.getsource(importlib.import_module("b12x.moe._shared.kernels.tiny_decode"))
    mode = "clamp(PR83)" if "swiglu_limit" in tiny_src else "guard"
    if mode == "guard" and clamped:
        raise AssertionError("guard image routes swiglu_limit=10 to tiny_decode")
    return f"{mode}: tiny selected with limit 10 -> {clamped}, without limit -> {plain}"


def engram_patch():
    from b12x.sequence.engram import _impl

    src = inspect.getsource(_impl)
    if "PATCH(ds41-nvfp4)" not in src:
        raise AssertionError("packed-FP4 Engram patch missing")
    from b12x.sequence.engram import _preparation

    psrc = inspect.getsource(_preparation)
    return "packed_fp4 present; _preparation uses " + ("CompileJob.DEVICE_ORDINAL (upstream kept)" if "CompileJob.DEVICE_ORDINAL" in psrc else "device.ordinal (old)")


def brrr_hook(mode):
    def run():
        import brrr.patches as p

        os.environ["BRRR_RADIX_TOPK"] = mode
        m = importlib.import_module("b12x.attention.dsa_indexer.mxfp4")
        if mode == "0":
            return "BRRR indexer patch skipped (BRRR_RADIX_TOPK=0)"
        p.patch_indexer(m)
        return f"BRRR indexer patch applied (mode {mode})"

    return run


def brrr_structs():
    from b12x.moe.fused_moe._impl import _PreparedW4A8Weights
    import dataclasses

    fields = {f.name for f in dataclasses.fields(_PreparedW4A8Weights)} if dataclasses.is_dataclass(_PreparedW4A8Weights) else set(vars(_PreparedW4A8Weights))
    need = {"w13_rp", "w2_rp", "w13_sfb", "w2_sfb", "n64_repack", "intermediate_size", "w13_layout", "source_format"}
    miss = sorted(need - fields)
    if miss and dataclasses.is_dataclass(_PreparedW4A8Weights):
        raise AssertionError(f"_PreparedW4A8Weights lacks {miss}")
    from b12x._lib.quant import csf_blocks

    return f"_PreparedW4A8Weights fields ok; CsfBlockPayload={hasattr(csf_blocks, 'CsfBlockPayload')}"


def native_loader():
    # The io_uring/bounce loader is a C extension b12x builds at first use (gcc + CUDA headers, no GPU needed).
    from b12x.loader import _native

    path = _native._build()
    return str(path)


check("import b12x (origin, bump metadata)", b12x_origin)
for mod in (
    "b12x.preparation",
    "b12x.moe.fused_moe",
    "b12x.attention.dsa_indexer",
    "b12x.attention.dsa_indexer.mxfp4",
    "b12x.attention.compressed_sparse_mla",
    "b12x.attention.sparse_mla",
    "b12x.attention.paged_decode",
    "b12x.gemm.blockscaled",
    "b12x.gemm.mxfp8_linear",
    "b12x.gemm.bf16_vocab_projection",
    "b12x.gemm.mla_query_projection",
    "b12x.gemm.wo_projection",
    "b12x.norm.mhc",
    "b12x.norm.hyperconnection",
    "b12x.sequence.engram",
    "b12x.sequence.mtp_feedback",
    "b12x.comm.pcie",
    "b12x.comm.pcie._oneshot_preparation",
    "b12x.comm.pcie._twoshot_preparation",
):
    check(f"import {mod}", imp(mod))
check("dsa_indexer fused API", indexer_api)
check("tiny_decode SwiGLU fix", tiny_guard)
check("Engram packed-FP4 patch", engram_patch)
check("BRRR-consumed b12x structures", brrr_structs)
check("BRRR indexer hook, BRRR_RADIX_TOPK=0", brrr_hook("0"))
if os.environ.get("CHECK_BRRR_BLOCKS"):
    check("BRRR indexer hook, BRRR_RADIX_TOPK=blocks (expected FAIL on the bump)", brrr_hook("blocks"))
for mod in (
    "vllm.utils.b12x",
    "vllm.model_executor.layers.fused_moe.b12x",
    "vllm.model_executor.layers.fused_moe.expert_residency",
    "vllm.model_executor.offloader.uva_p4",
    "vllm.distributed.device_communicators.b12x_pcie_all_reduce",
    "vllm.models.deepseek_v4_1.attention",
    "vllm.models.deepseek_v4_1.b12x_layers",
    "vllm.models.deepseek_v4_1.common.engram",
    "vllm.models.deepseek_v4_1.nvidia.model",
    "vllm.models.deepseek_v4_1.nvidia.dspark",
    "vllm.v1.worker.b12x_startup",
    "vllm.model_executor.warmup.b12x_prepare",
    "brrr.decoder",
):
    check(f"import {mod}", imp(mod))
check("b12x native loader C build (CPU)", native_loader)
print(f"\n{'ALL OK' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
