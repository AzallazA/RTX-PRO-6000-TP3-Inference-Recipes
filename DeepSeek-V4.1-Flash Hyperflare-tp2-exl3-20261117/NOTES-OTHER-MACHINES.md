# Running this build on another machine

The numbers in [README.md](README.md) come from one machine ([Test system](README.md#test-system)). This build
depends on the machine more than our TP3 build does, because the routed experts that are not resident in VRAM
(40-68 per split layer) are read from host RAM over PCIe on every step that routes to them. Check these before
comparing numbers.

## GPU power limit and clocks

- All numbers are at a **450 W power limit with +75 MHz core / +6000 MHz memory offsets** (memory clock 16,365 MHz).
  On our TP3 build, 450 W with the offsets gave about +10% decode over a lower power setting. Stock settings will be
  slower; not measured on this build.
- Export with the power limit and clocks you will serve with. The export tunes kernels by timing them on your cards.
- Persistence mode on (`nvidia-smi -pm 1`) and ECC off were our settings.

## PCIe

- **PCIe Gen5 x16 per GPU, directly to the CPU.** Our cards measured ~37 GB/s host-to-GPU each. A Gen4 link, x8
  lanes, or a PCIe switch shared by both cards cuts that bandwidth, and decode slows with it.
- Check the negotiated link under load: `nvidia-smi -q | grep -A3 "Link Width"` and `"Max Link Gen"` / `"Current"`
  (the link can drop to Gen1 at idle; check while serving).

## Peer-to-peer

- The two GPUs talk to each other directly (tensor-parallel all-reduce). We ran with **peer-to-peer enabled**; without
  P2P this build is untested.
- Check with `nvidia-smi topo -p2p r` (both directions should be `OK`). On workstation boards P2P may depend on the
  driver and the platform's IOMMU / ACS settings.

## Host RAM

- With the Engram tables in RAM (the default) the server plans ~140.5 GiB (151 GiB RSS measured): tables 103 GiB, host-resident experts
  ~25 GiB, prefix cache ~8 GiB, mapped buffers ~4 GiB. **~168 GiB of total RAM is the floor, 256 GB comfortable**, plus
  whatever else the machine runs. Less RAM: see the next section.
- The export needs ~80 GiB of free host RAM.
- RAM bandwidth matters a little too: host-resident experts are read from it. We ran 8 channels of DDR5-6000.
- `memlock unlimited` is required (Hyperflare pins host memory; [EXPORT.md, step 1](EXPORT.md#1-requirements)).

## Engram tables on disk

`ENGRAM=one ./serve.sh` (one table on disk) and `ENGRAM=disk ./serve.sh` (both; `--embedding-table-storage disk`)
read the int4 Engram rows from the package through a row cache (`ENGRAM_CACHE`, 4 GiB in our runs, ~56-57% hit rate
on the random-prompt benchmark). Disk reads bypass the OS page cache (O_DIRECT), so RAM use is predictable. Measured
on our machine, same package, KV 2,140,800 tokens in each mode:

| | Both in RAM (default) | One on disk (`ENGRAM=one`) | Both on disk (`ENGRAM=disk`) |
| --- | --- | --- | --- |
| Server RAM (measured RSS) | 151 GiB | 110 GiB | 64 GiB |
| Flare's planned host RAM | 140.5 GiB | 98.7 GiB | 52.9 GiB |
| Decode 32K, 1 / 4 / 8 streams (tok/s) | 247 / 541 / 733 | 239 / 541 / 745 | 239 / 528 / 729 |
| Decode, 1 stream, fixed 2K / 46K prompt (tok/s) | 262 / 290 | 299 / 298 | 288 / 277 |
| Prefill 32K (tok/s) | 11.1-11.6K | 11.1-11.2K | 8.9-10.4K |
| Four 46K streams (aggregate tok/s) | 562-592 | 575-588 | 551-563 |

- One table on disk: ~41 GiB less RAM, no measurable speed cost (within run-to-run noise).
- Both on disk: ~87 GiB less RAM, decode -1 to -3%, prefill -10 to -20% (cold rows come from NVMe).
- Each table is 51.5 GiB; you need a fast NVMe (ours: PCIe Gen5 x4, ~12 GB/s). A slower drive will cost more
  prefill than we measured. Not measured on a slower drive.
- Total machine RAM, leaving ~15 GiB for the OS: both in RAM ~168 GiB minimum (256 GB comfortable); one on disk
  ~125 GiB minimum (160-192 GB comfortable); both on disk ~80 GiB minimum (96-128 GB comfortable).
- Do not also set `FLARE_ENGRAM_RAM_TABLES`: it overrides the storage flag per table.

## CPU and NUMA

- Ours: one socket, 24 cores, **one NUMA node**. The CPU itself does little at decode time. The host-RAM experts are
  read by the GPUs over PCIe and computed on the GPUs, not on the CPU.
- On a multi-node CPU (dual socket, or NPS2/NPS4 settings), the host RAM the GPUs read should sit on the node their
  PCIe root ports belong to. Both GPUs on one node, and the server bound to it, is the simple case:
  `numactl --cpunodebind=<n> --membind=<n> ./serve.sh`. Untested here.

## Drive

- Start-up reads the whole package (~314 GiB). ~1 min from a warm page cache on our
  Gen5 NVMe; a slow drive mostly costs start time, not speed (except with the Engram tables on disk).

## Package

- The package is built for **2 × RTX PRO 6000 Blackwell 96 GB (SM120)** and tuned on the exporting machine's cards
  and driver. Export it yourself on your machine ([EXPORT.md](EXPORT.md)); do not copy a package between machines
  with different GPUs or drivers.
- Fewer resident experts give more KV at lower decode speed, and the speed cost grows with a slower PCIe link:
  [KV-TRADEOFF.md](KV-TRADEOFF.md).
