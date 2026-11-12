"""CPU unit test for vision_stage.VisionWeightStager (no GPU needed).

Runs a miniature tower (patch embed + 6 blocks) and aligner, two images per call, and checks:
  1. outputs equal the unstaged reference for: plain CPU path, simulated streams with overlap on and off, and the
     fallback path (borrowed scratch too small -> a dedicated buffer is allocated);
  2. every linear's ``weight`` is the host Parameter again after each call;
  3. stream protocol: with fake torch.cuda streams/events, every pair of operations touching overlapping bytes
     (slot copies, staged weight reads, scratch writes) where one writes is ordered by a happens-before path that
     agrees with issue order (no copy may overwrite a slot or scratch buffer that queued work still reads).
Usage (inside the .11 image or any env with torch):  python test_vision_stage.py <dir containing vision_stage.py>
"""

import contextlib
import importlib.util
import sys

import torch
from torch import nn

spec = importlib.util.spec_from_file_location("vision_stage", sys.argv[1] + "/vision_stage.py")
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

torch.manual_seed(0)
H, QKV, I2, I, PATCH = 8, 24, 44, 22, 12
A_IN, A_OUT, R = 72, 40, 3

# ----------------------------------------------------------------------------- fake streams (happens-before graph)
OPS = []  # (node, kind, lo, hi, label)


class Node:
    __slots__ = ("deps", "idx")

    def __init__(self, deps):
        self.deps = [d for d in deps if d is not None]
        self.idx = len(NODES)
        NODES.append(self)


NODES = []


class FakeStream:
    def __init__(self, name="side", device=None):
        self.name, self.last, self.pending, self.device = name, None, [], device

    def add(self):
        node = Node([self.last, *self.pending])
        self.last, self.pending = node, []
        return node

    def wait_event(self, event):
        if event.node is not None:
            self.pending.append(event.node)

    def wait_stream(self, other):
        if other is not self and other.last is not None:
            self.pending.append(other.last)


class FakeEvent:
    def __init__(self, *a, **k):
        self.node = None

    def record(self, stream=None):
        self.node = (stream or CURRENT[0]).add()


MAIN = FakeStream("main")
CURRENT = [MAIN]


@contextlib.contextmanager
def fake_stream_ctx(stream):
    prev = CURRENT[0]
    CURRENT[0] = stream
    try:
        yield
    finally:
        CURRENT[0] = prev


def span(t):
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def op(kind, tensor, label):
    lo, hi = span(tensor)
    OPS.append((CURRENT[0].add(), kind, lo, hi, label))


def reach_sets():
    reach = []
    for node in NODES:
        bits = 0
        for dep in node.deps:
            bits |= reach[dep.idx] | (1 << dep.idx)
        reach.append(bits)
    return reach


def check_races():
    reach = reach_sets()
    bad = 0
    for i, (n1, k1, lo1, hi1, l1) in enumerate(OPS):
        for n2, k2, lo2, hi2, l2 in OPS[i + 1 :]:
            if "w" not in (k1, k2) or hi1 <= lo2 or hi2 <= lo1:
                continue
            if not (reach[n2.idx] >> n1.idx) & 1:  # later-issued op must depend on the earlier one
                bad += 1
                if bad <= 5:
                    print(f"  RACE: {l1}({k1}) vs {l2}({k2}) not ordered")
    return bad


# ----------------------------------------------------------------------------- miniature model
class Lin(nn.Module):
    def __init__(self, i, o, scratch):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) * 0.3, requires_grad=False)
        self.scratch = scratch

    def forward(self, x):
        op("r", self.weight, f"read {self.tag}")
        out = x @ self.weight.t()
        dst = self.scratch()
        op("w", dst, f"scratch {self.tag}")
        return out


class Workspace:
    def __init__(self, sizes):
        self.bufs = [torch.empty(n, dtype=torch.float32) for n in sizes]  # same dtype as the weights below


class Block(nn.Module):
    def __init__(self, ws):
        super().__init__()
        s = lambda: ws.bufs[0]
        self.wqkv, self.wo, self.w1, self.w2 = Lin(H, QKV, s), Lin(H, H, s), Lin(H, I2, s), Lin(I, H, s)

    def forward(self, x):
        y = x + self.wo(torch.tanh(self.wqkv(x)[:, :H]))
        return y + self.w2(torch.nn.functional.silu(self.w1(y)[:, :I]))


class Tower(nn.Module):
    def __init__(self, ws, n):
        super().__init__()
        self.patch = Lin(PATCH, H, lambda: ws.bufs[0])
        self.blocks = nn.ModuleList(Block(ws) for _ in range(n))

    def forward(self, p):
        x = self.patch(p)
        for b in self.blocks:
            x = b(x)
        return x


class Aligner(nn.Module):
    def __init__(self, ws):
        super().__init__()
        s = lambda: ws.bufs[0]  # like merged/hidden/activated: the buffer that also holds the tower slots
        self.w1, self.w2 = Lin(A_IN, A_OUT, s), Lin(A_OUT, A_OUT, s)

    def forward(self, x):
        merged = x.reshape(-1, A_IN)
        return self.w2(torch.nn.functional.gelu(self.w1(merged)))


def build(tower_ws_sizes, aligner_ws_sizes, n=6):
    torch.manual_seed(1)
    tws, aws = Workspace(tower_ws_sizes), Workspace(aligner_ws_sizes)
    tower, aligner = Tower(tws, n), Aligner(aws)
    for name, m in [*tower.named_modules(), *aligner.named_modules()]:
        if isinstance(m, Lin):
            m.tag = name or "lin"
    return tower, aligner, tws, aws


def run(tower, aligner, images):
    outs = []
    for p in images:
        outs.append(aligner(tower(p)))
    return torch.cat(outs)


def park(mods):
    # stand-in for the vLLM UVA offload / MappedHostAllocation: a fresh Parameter in other memory
    for m in mods:
        m.weight = nn.Parameter(m.weight.detach().clone(), requires_grad=False)


def make_stager(tower, aligner, tws, aws, overlap, kernel_copy=False):
    groups = [[b.wqkv, b.wo, b.w1, b.w2] for b in tower.blocks]
    groups[0].insert(0, tower.patch)
    return vs.VisionWeightStager(
        tower, list(tower.blocks), groups, aligner, [aligner.w1, aligner.w2],
        lambda: tws.bufs, lambda: aws.bufs, overlap=overlap, kernel_copy=kernel_copy, log=print,
    )


def case(name, simulate, overlap, small=False, kernel_copy=False):
    global OPS, NODES
    OPS, NODES = [], []
    MAIN.last, MAIN.pending = None, []
    block = QKV * H + H * H + I2 * H + H * I  # elements (float32)
    if small:  # borrowed scratch too small for the slots -> fallback buffer
        tsz, asz = [A_OUT * A_IN + 64, 64], [64, 64]
    else:  # as in the real model: one big buffer per weight-sized need, plus spare room (alignment padding)
        tsz = [A_OUT * A_IN + 128, A_OUT * A_OUT + 128]  # aligner weights land in tower scratch
        asz = [2 * block + PATCH * H + 512, 64]  # both tower slots in the aligner's large scratch buffer
    tower, aligner, tws, aws = build(tsz, asz)
    images = [torch.randn(18, PATCH) for _ in range(2)]  # 18 patches -> 2 merged rows of 9 x 8
    with torch.no_grad():
        ref = run(tower, aligner, images)
        del OPS[:]
        del NODES[:]
        MAIN.last, MAIN.pending = None, []
        linears = [m for m in [*tower.modules(), *aligner.modules()] if isinstance(m, Lin)]
        park(linears)
        host_params = {id(m): m.weight for m in linears}
        st = make_stager(tower, aligner, tws, aws, overlap, kernel_copy)
        vs.VisionWeightStager._simulate_streams = simulate
        out1 = run(tower, aligner, images)
        out2 = run(tower, aligner, images)
    ok = torch.equal(out1, ref) and torch.equal(out2, ref)
    restored = all(m._parameters["weight"] is host_params[id(m)] for m in linears)
    races = check_races() if simulate else 0
    staged_reads = sum(1 for o in OPS if o[1] == "r" and any(
        o[2] >= span(h)[0] and o[2] < span(h)[1] for h in (*tws.bufs, *aws.bufs, *st._owned)))
    print(f"{name}: outputs {'equal' if ok else 'DIFFER'}, host params restored {restored}, races {races}, "
          f"staged reads {staged_reads}/{sum(1 for o in OPS if o[1] == 'r')}, borrowed {st.borrowed_bytes} B, "
          f"allocated {st.allocated_bytes} B")
    return ok and restored and races == 0 and (not simulate or staged_reads == sum(1 for o in OPS if o[1] == "r"))


_real_copy = vs.VisionWeightStager._copy


def _traced_copy(self, linears, stream):
    # every slot copy is a write of the staged view on the stream it is issued to
    with fake_stream_ctx(stream or CURRENT[0]):
        for linear in linears:
            op("w", self._staged[id(linear)], f"copy {linear.tag}")
    return _real_copy(self, linears, stream)


vs.VisionWeightStager._copy = _traced_copy
torch.cuda.current_stream = lambda *a, **k: CURRENT[0]
torch.cuda.Stream = FakeStream
torch.cuda.Event = FakeEvent
torch.cuda.stream = fake_stream_ctx
results = [
    case("plain CPU path", simulate=False, overlap=True),
    case("simulated streams, overlap", simulate=True, overlap=True),
    case("simulated streams, no overlap", simulate=True, overlap=False),
    case("simulated streams, fallback buffer", simulate=True, overlap=True, small=True),
    case("simulated streams, kernel copy", simulate=True, overlap=True, kernel_copy=True),
]
print("VISION STAGE TEST", "PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
