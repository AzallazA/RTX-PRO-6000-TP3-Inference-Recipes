# SPDX-License-Identifier: Apache-2.0
"""TP3 (DS41_VISION_STAGE): stage host-resident vision weights through VRAM slots borrowed from idle workspaces.

The image keeps the vision tower and aligner in host memory already (ds41-serve passes --cpu-offload-params experts
vision aligner: vLLM's UVA offloader turns every vision/aligner parameter into a CUDA view of pinned host pages, 0.90
GiB/GPU; DS41_VISION_HOST parks whatever is still in VRAM the same way). Read in place, every GEMV row block re-reads
its weight tiles over PCIe (Flare measured ~3 s per image). This stager copies each group's weights into VRAM just
before use instead (one PCIe pass per image, 0.90 GiB) without any new VRAM:

* vision blocks: two slots (double buffer) carved from the aligner's workspace, which is idle while the tower runs;
  group 0 = patch embedding + block 0, group i = block i. While block i computes, block i+1's weights copy on a side
  stream (DS41_VISION_STAGE_OVERLAP=1, default) into the other slot.
* aligner: its two projections are copied into the tower's MLP/QKV scratch, which is idle once the tower returned.

All weights stay host-resident; the staged copies are views of existing buffers, so the freed VRAM still goes to expert
promotion. If a slot does not fit in the borrowed buffers, a dedicated buffer is allocated once and logged.
The encoder runs eagerly (never inside a CUDA graph), so swapping each linear's ``weight`` Parameter around the call is
safe: every kernel captures its weight pointer at launch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

_ALIGN = 256


def _flat_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError("staging donor must be contiguous")
    return tensor.detach().reshape(-1).view(torch.uint8)


class _Arena:
    """First-fit byte allocator over borrowed buffers (views only; nothing is allocated)."""

    def __init__(self, donors: Sequence[torch.Tensor]):
        self.buffers = [[_flat_bytes(t), 0] for t in donors]

    def take(self, nbytes: int) -> torch.Tensor | None:
        for entry in sorted(self.buffers, key=lambda e: e[0].numel() - e[1]):
            buf, used = entry
            pad = (-(buf.data_ptr() + used)) % _ALIGN
            if used + pad + nbytes <= buf.numel():
                entry[1] = used + pad + nbytes
                return buf[used + pad : used + pad + nbytes]
        return None


def _typed(raw: torch.Tensor, like: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(raw.view(like.dtype).view(like.shape), requires_grad=False)


class VisionWeightStager:
    """Hooks a tower (blocks) and an aligner so their parked ``weight`` Parameters are staged per group."""

    # Unit tests set this to drive the stream/event path with CPU tensors and fake torch.cuda streams.
    _simulate_streams = False

    def __init__(
        self,
        tower: nn.Module,
        blocks: Sequence[nn.Module],
        tower_groups: Sequence[Sequence[nn.Module]],
        aligner: nn.Module,
        aligner_group: Sequence[nn.Module],
        tower_donors: Callable[[], Sequence[torch.Tensor] | None],
        aligner_donors: Callable[[], Sequence[torch.Tensor] | None],
        *,
        overlap: bool = True,
        kernel_copy: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        if len(tower_groups) != len(blocks):
            raise ValueError("one tower group per block")
        self.tower, self.blocks, self.aligner = tower, list(blocks), aligner
        self.tower_groups = [list(g) for g in tower_groups]
        self.aligner_group = list(aligner_group)
        # Donors for the tower phase are the aligner's buffers and vice versa.
        self._tower_slot_donors, self._aligner_slot_donors = aligner_donors, tower_donors
        self.overlap = overlap
        self.log = log or (lambda msg: None)
        self.kernel_copy = kernel_copy
        # The host-resident Parameter of every staged linear (a CUDA alias of host pages on the GPU): copy source, and
        # what ``weight`` is restored to after each call.
        self._host = {id(m): m._parameters["weight"]
                      for m in (*(m for g in self.tower_groups for m in g), *self.aligner_group)}
        self._staged: dict[int, nn.Parameter] = {}
        self._ready_tower = self._ready_aligner = False
        self._owned: list[torch.Tensor] = []
        self._active = False
        self._stream = None
        self._events = None
        self.borrowed_bytes = self.allocated_bytes = 0
        self.copied_bytes_per_image = sum(
            m.weight.numel() * m.weight.element_size()
            for m in (*(m for g in self.tower_groups for m in g), *self.aligner_group)
        )
        self._handles = [
            tower.register_forward_pre_hook(self._tower_pre),
            tower.register_forward_hook(self._tower_post),
            aligner.register_forward_pre_hook(self._aligner_pre),
            aligner.register_forward_hook(self._aligner_post),
        ]
        for index, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(self._block_pre(index)))
            self._handles.append(block.register_forward_hook(self._block_post(index)))

    # ---------------------------------------------------------------- layout
    def _place(self, arena: _Arena, linears: Sequence[nn.Module], device) -> list[torch.Tensor]:
        views: list[torch.Tensor | None] = [None] * len(linears)
        order = sorted(range(len(linears)), key=lambda j: -linears[j].weight.numel() * linears[j].weight.element_size())
        missing = []
        for j in order:
            nbytes = linears[j].weight.numel() * linears[j].weight.element_size()
            view = arena.take(nbytes)
            if view is None:
                missing.append(j)
            else:
                views[j] = view
                self.borrowed_bytes += nbytes
        if missing:
            need = sum(linears[j].weight.numel() * linears[j].weight.element_size() + _ALIGN for j in missing)
            own = torch.empty(need, dtype=torch.uint8, device=device)
            self._owned.append(own)
            self.allocated_bytes += need
            extra = _Arena([own])
            for j in missing:
                views[j] = extra.take(linears[j].weight.numel() * linears[j].weight.element_size())
        return views  # type: ignore[return-value]

    def _setup_tower(self) -> bool:
        donors = self._tower_slot_donors()
        if donors is None:
            return False
        arena = _Arena(donors)
        device = self.tower_groups[0][0].weight.device
        # Slot s serves groups s, s+2, ...; it is laid out for its largest group, and every other group in the slot
        # must have the same per-role shapes (true for identical ViT blocks; group 0 adds the patch embedding first).
        for slot in (0, 1):
            members = [i for i in range(slot, len(self.tower_groups), 2)]
            if not members:
                continue
            template = max(members, key=lambda i: len(self.tower_groups[i]))
            views = self._place(arena, self.tower_groups[template], device)
            for i in members:
                group = self.tower_groups[i]
                offset = len(self.tower_groups[template]) - len(group)  # extra leading roles (patch embedding)
                for j, linear in enumerate(group):
                    view = views[j + offset] if i != template else views[j]
                    if view.numel() != linear.weight.numel() * linear.weight.element_size():
                        raise ValueError("vision staging: block groups differ in shape")
                    self._staged[id(linear)] = _typed(view, linear.weight)
        self._ready_tower = True
        return True

    def _setup_aligner(self) -> bool:
        donors = self._aligner_slot_donors()
        if donors is None:
            return False
        device = self.aligner_group[0].weight.device
        views = self._place(_Arena(donors), self.aligner_group, device)
        for linear, view in zip(self.aligner_group, views):
            self._staged[id(linear)] = _typed(view, linear.weight)
        self._ready_aligner = True
        self.log(
            "DS41_VISION_STAGE: %d MiB of staging slots borrowed from vision/aligner scratch, %d MiB allocated, "
            "%d MiB copied per image, overlap=%s"
            % (self.borrowed_bytes >> 20, self.allocated_bytes >> 20, self.copied_bytes_per_image >> 20, self.overlap)
        )
        return True

    # ---------------------------------------------------------------- streams
    def _streams(self):
        main = torch.cuda.current_stream()
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=main.device) if self.overlap else None
            self._events = {
                "copy": [torch.cuda.Event(), torch.cuda.Event()],
                "done": [torch.cuda.Event(), torch.cuda.Event()],
                "aligner": torch.cuda.Event(),
            }
        return main, (self._stream or main)

    def _copy(self, linears: Sequence[nn.Module], stream) -> None:
        # Source and destination are both CUDA tensors: torch issues one cudaMemcpyAsync (device-to-device; the source
        # pointer is mapped pinned host memory, so the copy engine pulls it over PCIe). kernel_copy (DS41_VISION_STAGE_COPY
        # =kernel) uses an elementwise kernel instead, for a driver that mishandles memcpy from mapped host memory.
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx:
            for linear in linears:
                staged, host = self._staged[id(linear)], self._host[id(linear)]
                if self.kernel_copy:
                    torch.mul(host.data, 1, out=staged.data)
                else:
                    staged.data.copy_(host.data, non_blocking=True)

    def _bind(self, linears: Sequence[nn.Module]) -> None:
        for linear in linears:
            linear._parameters["weight"] = self._staged[id(linear)]

    def _unbind(self, linears: Sequence[nn.Module]) -> None:
        for linear in linears:
            linear._parameters["weight"] = self._host[id(linear)]

    def _usable(self, device: torch.device) -> bool:
        return device.type != "cuda" or not torch.cuda.is_current_stream_capturing()

    def _streamed(self, device: torch.device) -> bool:
        return device.type == "cuda" or self._simulate_streams

    # ---------------------------------------------------------------- hooks
    def _tower_pre(self, module, args):
        device = self.tower_groups[0][0].weight.device
        self._active = False
        if not self._usable(device):
            return
        if not self._ready_tower and not self._setup_tower():
            return
        if not self._streamed(device):  # CPU unit tests: synchronous copies
            self._copy(self.tower_groups[0], None)
            self._bind(self.tower_groups[0])
            self._active = True
            return
        main, copy = self._streams()
        ev = self._events
        if copy is not main:
            copy.wait_stream(main)  # the aligner (previous image) may still read its scratch = our slots
        for slot in (0, 1):
            if slot < len(self.tower_groups):
                self._copy(self.tower_groups[slot], copy)
                ev["copy"][slot].record(copy)
        main.wait_event(ev["copy"][0])
        self._bind(self.tower_groups[0])
        self._active = True

    def _block_pre(self, index):
        def hook(module, args):
            if not self._active:
                return
            group = self.tower_groups[index]
            if self._streamed(group[0].weight.device):
                torch.cuda.current_stream().wait_event(self._events["copy"][index % 2])
            elif index > 0:
                self._copy(group, None)
            self._bind(group)

        return hook

    def _block_post(self, index):
        def hook(module, args, output):
            if not self._active:
                return
            group = self.tower_groups[index]
            self._unbind(group)
            nxt = index + 2
            if not self._streamed(group[0].weight.device):
                return
            main, copy = self._streams()
            ev = self._events
            ev["done"][index % 2].record(main)
            if nxt < len(self.tower_groups):
                if copy is not main:
                    copy.wait_event(ev["done"][index % 2])
                self._copy(self.tower_groups[nxt], copy)
                ev["copy"][index % 2].record(copy)

        return hook

    def _tower_post(self, module, args, output):
        if self._active:
            for group in self.tower_groups:
                self._unbind(group)
        self._active = False

    def _aligner_pre(self, module, args):
        device = self.aligner_group[0].weight.device
        if not self._usable(device):
            return
        if not self._ready_aligner and not self._setup_aligner():
            return
        if not self._streamed(device):
            self._copy(self.aligner_group, None)
            self._bind(self.aligner_group)
            return
        main, copy = self._streams()
        if copy is not main:
            copy.wait_stream(main)  # the tower's scratch (our slots) is free once the tower's kernels ran
        self._copy(self.aligner_group, copy)
        self._events["aligner"].record(copy)
        main.wait_event(self._events["aligner"])
        self._bind(self.aligner_group)

    def _aligner_post(self, module, args, output):
        self._unbind(self.aligner_group)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
