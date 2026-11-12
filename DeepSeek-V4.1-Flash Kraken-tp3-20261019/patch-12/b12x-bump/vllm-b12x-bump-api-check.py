"""CPU check (no GPU, no vLLM install): vLLM's B12xBlockscaledLinear.run against b12x blockscaled.mm.

usage: CUDA_VISIBLE_DEVICES= python -I check_blockscaled_view.py B12X_PARENT CPUDEPS VLLM_FILE [expect=pass|attrerror]
  B12X_PARENT  directory holding a `b12x` package (symlink to flashinfer/experimental/b12x) [+ _flashinfer_filelock.py];
               "-" = the installed b12x (inside the image)
  CPUDEPS      site dir with nvidia-cutlass-dsl (pip --target); "-" = already importable (inside the image)
  VLLM_FILE    vllm/model_executor/kernels/linear/b12x_blockscaled.py to test (loaded with stub vllm modules)
It builds the composite plan exactly like ensure_plan() (NVFP4, plan_regimes with exact-M counts + A16 cutoff),
stubs require_prepared (no kernels) and b12x.gemm.blockscaled._ops.linear (records what the custom op would get),
then calls run() for every planned row count and one in-between count.
"""
import importlib.util, site, sys, types
from dataclasses import dataclass

parent, cpudeps, vfile = sys.argv[1:4]
expect = sys.argv[4] if len(sys.argv) > 4 else "pass"
if cpudeps != "-":
    site.addsitedir(cpudeps)
if parent != "-":
    sys.path.insert(0, parent)
import os  # noqa: E402
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
import torch  # noqa: E402
import b12x.gemm.blockscaled as bs  # noqa: E402
from b12x.gemm.blockscaled import _ops  # noqa: E402
import b12x.preparation as prep  # noqa: E402

# ---- stub vllm modules the wrapper imports
class PreparationResourceUnavailableError(RuntimeError):
    pass

@dataclass
class Workload:
    max_tokens: int
    fixed_token_counts: tuple
    output_dtype: torch.dtype = torch.bfloat16

mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.envs", "vllm.logger", "vllm.utils", "vllm.utils.b12x")}
mods["vllm.envs"].VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES = 64 << 20
class _Log:
    def __getattr__(self, name):
        return lambda *a, **k: None
mods["vllm.logger"].init_logger = lambda name: _Log()
u = mods["vllm.utils.b12x"]
u.B12xPreparationUnit = object
u.B12xWorkload = Workload
u.PreparationResourceUnavailableError = PreparationResourceUnavailableError
A16 = 4
u.get_b12x_a16_max_tokens = lambda: A16
u.get_b12x_blockscaled = lambda: bs.api if hasattr(bs, "api") else bs
mods["vllm"].envs = mods["vllm.envs"]
sys.modules.update(mods)
spec = importlib.util.spec_from_file_location("b12x_blockscaled_under_test", vfile)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

# ---- CPU NVFP4 weight with LM-head-like geometry (K=4096, one TP3 vocab shard)
K, N = 4096, 43008
packed = bs.NVFP4LinearWeight(
    values=torch.zeros((N, K // 2), dtype=torch.uint8),
    scale_mma=torch.zeros((N * K // 16,), dtype=torch.uint8),
    global_scale=torch.ones((1,), dtype=torch.float32),
    global_scale_kind="multiplier",
    in_features=K,
    out_features=N,
)
packed.padded_in_features  # property exists
holder = wrapper.B12xBlockscaledLinear(packed, recipe="nvfp4", activation_mode="auto", layer_name="lm_head")
plan = holder.ensure_plan(Workload(max_tokens=64, fixed_token_counts=(1, 2, 3, 6, 8, 12)))
print("plan", type(plan).__name__, "token_counts", plan.token_counts, "has query:", hasattr(plan, "query"))

class _State:
    required_workspace = 0
prep.require_prepared = lambda *a, **k: _State()
seen = []
def fake_linear(source, values, scales, global_scale, *, plan, global_scale_kind, **kw):
    seen.append((plan.handle, source.shape[0], global_scale_kind))
    return torch.zeros((source.shape[0], N), dtype=torch.bfloat16)
_ops.linear = fake_linear

rows_list = (*plan.token_counts, 5)
try:
    for rows in rows_list:
        out = holder.run(torch.zeros((rows, K), dtype=torch.bfloat16), None)
        assert out.shape == (rows, N), out.shape
except AttributeError as exc:
    print("AttributeError:", exc)
    sys.exit(0 if expect == "attrerror" else 1)
assert all(h == plan.handle for h, _, _ in seen), seen
assert [r for _, r, _ in seen] == list(rows_list)
assert holder._mm_plan_source is plan if hasattr(holder, "_mm_plan_source") else True
view = getattr(holder, "_mm_plan", plan)
print("mm saw handle", seen[0][0], "== composite handle", plan.handle, "| view", type(view).__name__,
      "query.num_tokens", getattr(getattr(view, "query", None), "num_tokens", None),
      "source_activation", getattr(getattr(view, "query", None), "source_activation", "n/a"))
# wrong-width source must still be rejected by blockscaled.mm
try:
    holder.run(torch.zeros((2, 2 * K), dtype=torch.bfloat16), None)
    print("ERROR: 2K-wide source accepted"); sys.exit(1)
except ValueError as exc:
    print("2K-wide source rejected:", exc)
print("RESULT pass")
sys.exit(0 if expect == "pass" else 1)
