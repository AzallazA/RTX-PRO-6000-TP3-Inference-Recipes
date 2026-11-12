"""CPU check for vllm-ds41-lazy-engram-boot-ram.patch (no torch, no GPU).

Extracts the patched functions from a patched site-packages tree with ast and runs
them against fakes. Usage: python3 -I this.py <patched-root containing vllm/>
"""
import ast, os, sys, tempfile, types

root = sys.argv[1]
nv = os.path.join(root, "vllm/models/deepseek_v4_1/nvidia")


def grab(path, names, cls=None):
    tree = ast.parse(open(path).read())
    body = tree.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in out} == set(names), (path, names)
    for n in out:
        n.decorator_list = []
    return ast.unparse(ast.Module(body=out, type_ignores=[]))


class FakeTensor:
    def __init__(self, v): self.v = v
    def item(self): return self.v[0]


class FakeTorch:
    int32 = "int32"
    calls = []
    class cuda:
        @staticmethod
        def synchronize(): FakeTorch.calls.append("sync")
    class distributed:
        class ReduceOp: MIN = "min"
        others = []  # other ranks' flags
        @staticmethod
        def all_reduce(t, op, group):
            assert op == "min" and group == "cpu"
            t.v = [min([t.v[0]] + FakeTorch.distributed.others)]
    @staticmethod
    def tensor(v, dtype): return FakeTensor(list(v))


class Log:
    def __init__(self): self.msgs = []
    def info(self, m, *a): self.msgs.append(m % a)
    def warning(self, m, *a): self.msgs.append(m % a)


class Emb:
    def __init__(self, pending=True, nbytes=16 << 30):
        self.lazy_pending, self.mapped_host_nbytes, self.filled = pending, nbytes, 0
    def fill_lazy(self): self.filled += 1; self.lazy_pending = False


class Engram:
    def __init__(self, emb): self.embed_tokens = emb


class Layer:
    def __init__(self, engram=None): self.engram = engram


class Group:
    def __init__(self, rank, world): self.rank_in_group, self.world_size, self.barriers, self.cpu_group = rank, world, 0, "cpu"
    def barrier(self): self.barriers += 1


def build(flags, avail_gib, group):
    log = Log()
    promoted = []
    vd = types.ModuleType("vllm.distributed"); vd.get_tp_group = lambda: group
    b12 = types.ModuleType("vllm.model_executor.layers.fused_moe.b12x"); b12.promote_pending_inplace = lambda: promoted.append(1)
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.distributed": vd,
                        "vllm.model_executor.layers.fused_moe.b12x": b12})
    import itertools, time
    g = dict(os=os, torch=FakeTorch, logger=log, time=time, islice=itertools.islice,
             _host_mem_available=lambda: int(avail_gib * (1 << 30)), **flags)
    exec(grab(os.path.join(nv, "model.py"), ["self_arm_lazy_engram", "_fill_lazy_engram"], "DeepseekV4Model"), g)
    M = type("M", (), {"self_arm_lazy_engram": g["self_arm_lazy_engram"], "_fill_lazy_engram": g["_fill_lazy_engram"]})
    return M, log, promoted


def model(M, embs):
    m = M(); m.layers = [Layer()] + [Layer(Engram(e)) for e in embs]; m.start_layer, m.end_layer = 0, len(m.layers)
    return m


ok = 0
with tempfile.TemporaryDirectory() as d:
    arm = os.path.join(d, "arm"); os.environ["B12X_DEFERRED_BANK_ARM_FILE"] = arm
    # 1. self-arm off: nothing happens
    M, log, _ = build(dict(_LAZY_SELF_ARM=False, _LAZY_PARALLEL=False), 200, Group(0, 3))
    model(M, [Emb()]).self_arm_lazy_engram(); assert not os.path.exists(arm); ok += 1
    # 2. self-arm on, no lazy tables pending: no arm file (non-lazy configs unchanged)
    M, log, _ = build(dict(_LAZY_SELF_ARM=True, _LAZY_PARALLEL=False), 200, Group(0, 3))
    model(M, [Emb(pending=False)]).self_arm_lazy_engram(); assert not os.path.exists(arm); ok += 1
    # 3. self-arm on, pending: arm file created, idempotent
    m = model(M, [Emb()]); m.self_arm_lazy_engram(); assert os.path.exists(arm); m.self_arm_lazy_engram(); ok += 1
    # 4. fill not armed -> returns without filling
    os.remove(arm); e = Emb(); m = model(M, [e]); m._fill_lazy_engram([Engram(e)]); assert e.filled == 0; ok += 1
    # 5. sequential fill (rank 1 of 3): promotes, fills once, 3 barriers
    open(arm, "w").close(); grp = Group(1, 3); M, log, promoted = build(dict(_LAZY_SELF_ARM=True, _LAZY_PARALLEL=False), 100, grp)
    e = Emb(); m = model(M, [e]); m._fill_lazy_engram([Engram(e)])
    assert promoted and e.filled == 1 and grp.barriers == 3 and m._lazy_engram_pending is False; ok += 1
    # 6. sequential, too little RAM -> raises
    M, log, _ = build(dict(_LAZY_SELF_ARM=True, _LAZY_PARALLEL=False), 20, Group(0, 3)); e = Emb()
    try: model(M, [e])._fill_lazy_engram([Engram(e)]); raise SystemExit("expected RuntimeError")
    except RuntimeError: ok += 1
    # 7. parallel fits on every rank: one fill, one barrier
    FakeTorch.distributed.others = [1, 1]; grp = Group(2, 3)
    M, log, _ = build(dict(_LAZY_SELF_ARM=True, _LAZY_PARALLEL=True), 3 * 32 + 8 + 1, grp)
    e1, e2 = Emb(), Emb(); m = model(M, [e1, e2]); m._fill_lazy_engram([Engram(e1), Engram(e2)])
    assert e1.filled == e2.filled == 1 and grp.barriers == 1 and "parallel" in log.msgs[-1]; ok += 1
    # 8. parallel requested but another rank lacks room -> falls back to sequential (3 barriers)
    FakeTorch.distributed.others = [0, 1]; grp = Group(0, 3)
    M, log, _ = build(dict(_LAZY_SELF_ARM=True, _LAZY_PARALLEL=True), 200, grp)
    e = Emb(); m = model(M, [e]); m._fill_lazy_engram([Engram(e)])
    assert e.filled == 1 and grp.barriers == 3 and any("one rank at a time" in x for x in log.msgs); ok += 1

# _is_real_batch
g = {"InputBatch": object, "RequestState": object, "os": os}
exec(grab(os.path.join(nv, "model_state.py"), ["_is_real_batch"]), g)
B = types.SimpleNamespace; real = g["_is_real_batch"]
known = B(req_id_to_index={"chatcmpl-1": 0, "chatcmpl-2": 1})
assert real(B(req_ids=["chatcmpl-1", "chatcmpl-2"]), known)
assert not real(B(req_ids=["req_0_ab12"]), known)
assert not real(B(req_ids=[]), known)
assert not real(B(req_ids=["chatcmpl-1", "req_1_x"]), known)
ok += 1
print(f"LAZY-ENGRAM PATCH CHECK PASS ({ok}/9)")
