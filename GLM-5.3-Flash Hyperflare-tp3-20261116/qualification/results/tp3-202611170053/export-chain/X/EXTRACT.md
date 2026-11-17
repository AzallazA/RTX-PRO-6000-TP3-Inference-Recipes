# GLM TP3 export extract

## Phases ([ OK ] lines: plan, timings)

    [    OK    ] Resolve checkpoint.  glm-5.3-flash · <ckpt-dir>/glm53-…  0.3s
    [    OK    ] Select GPUs.  GPUs 0-2 · concurrency 32 · images 4 · up to 5…  0.1s library scratch primed on cuda:0: 64 MiB at 0x744f32000000
    [    OK    ] Connect tensor-parallel ranks.  4.1s
    [    OK    ] Inspect checkpoint.  2.5s
    [    OK    ] Load weights.  52.6s
    [    OK    ] Load vision weights.  0.2s
    [    OK    ] Allocate model state and collectives.  prefill slab 528.26…  642.2s
    [    OK    ] Construct graph catalog.  vision staging slots 22 MiB…  74/74  7.1s
    [    OK    ] Prepare and tune kernels.  117.6s
    [    OK    ] Warm serving graphs.  74/74  11.6s
    [    OK    ] Measure graph memory.  captured snapshot1 in 0.00s  74/74  10.1s
    [    OK    ] Compile model forwards.  3 compiler workers; per ra…  25/25  765.0s
    [    OK    ] Allocate KV cache.  trial graph pool released: 3574 of 3574 …  0.3s
    [    OK    ] Load and warm compiled forwards.  529 MiB portable …  74/74  169.5s
    [    OK    ] Release serialized compiler state.  0 MiB; guards and…  74/74  0.0s
    [    OK    ] Warm serving graphs.  74/74  0.2s
    [    OK    ] Prepare serving graphs.  captured snapshot1 in 0.00s  74/74  10.7s
    [    OK    ] Wait for tensor-parallel ranks.  2/2  0.1s
    [    OK    ] Clear disposable storage.  1467106898 bytes  0.0s
    [    OK    ] Inspect CUDA graph functions.  5582 functions, 5554 modules  0.4s
    [    OK    ] Serialize CUDA graphs.  74 graphs, 97915004 bytes  0.4s
    [    OK    ] Capture compiled modules and globals.  299491072 bytes, 605308 global bytes downloaded  0.4s
    [    OK    ] Scan and copy reachable storage.  64665111834 bytes scanned, 64273726746 bytes downloaded, 0 bytes stored; 44 allocations omitted  2.1s
    [    OK    ] Export frontend assets.  0.0s
    [    OK    ] Materialize weight payloads.  111.5s
    [    OK    ] Wait for rank exports.  2/2  115.2s
    [    OK    ] Release preparation resources.  9.1s
    [    OK    ] Assemble core payloads.  3 ranks  43.5s
    [    OK    ] Validate core publication.  1 deployments  0.0s
    [    OK    ] Verify frontend payloads.  20228464 bytes  0.0s
    [    OK    ] Compress and publish rank 0 executables.  396800768 plaintext bytes  0.5s
    [    OK    ] Verify and publish rank 0 data.  62821526464 source bytes verified  3.4s
    [    OK    ] Compress and publish rank 1 executables.  396803704 plaintext bytes  0.5s
    [    OK    ] Verify and publish rank 1 data.  62821489496 source bytes verified  3.4s
    [    OK    ] Compress and publish rank 2 executables.  396803015 plaintext bytes  0.5s
    [    OK    ] Verify and publish rank 2 data.  60566573224 source bytes verified  3.3s
    [    OK    ] Verify and publish shared data.  0 source bytes verified  0.0s
    [    OK    ] Sign and publish core metadata.  27421454 metadata bytes  0.1s
    [    OK    ] Commit core.  <store>/models/glm53-fp4-local/gb202-tp3-202611170053  0.0s
    [    OK    ] Clear export staging.  0.5s
    [    OK    ] Export native core.  <store>/models/glm53-fp4…  313.8s

## Vision staging

    vision blocks and tail parked in device-visible host memory and staged per image (FLARE_GLM53_VISION_STAGE=1)
    vision staging slots 22 MiB (parked 373 MiB, tail copied per image 108 MiB): free=29507 MiB, allocated=4590 MiB, reserved=4672 MiB, rss=3727 MiB
    capturing vision32768: 32650 MiB free, 1242 MiB allocated, 1508 MiB reserved, 63183 MiB outside PyTorch
    capturing vision16384: 30196 MiB free, 1242 MiB allocated, 3958 MiB reserved, 63187 MiB outside PyTorch
    capturing vision4096: 30188 MiB free, 1242 MiB allocated, 3960 MiB reserved, 63193 MiB outside PyTorch
    capturing vision1024: 30184 MiB free, 1242 MiB allocated, 3960 MiB reserved, 63197 MiB outside PyTorch
    capturing vision256: 30178 MiB free, 1242 MiB allocated, 3960 MiB reserved, 63203 MiB outside PyTorch
    capturing vision64: 30164 MiB free, 1242 MiB allocated, 3966 MiB reserved, 63211 MiB outside PyTorch
    capturing vision32768: 31084 MiB free, 1242 MiB allocated, 2550 MiB reserved, 63707 MiB outside PyTorch
    capturing vision16384: 31020 MiB free, 1242 MiB allocated, 2614 MiB reserved, 63707 MiB outside PyTorch
    capturing vision4096: 31020 MiB free, 1242 MiB allocated, 2614 MiB reserved, 63707 MiB outside PyTorch
    capturing vision1024: 31020 MiB free, 1242 MiB allocated, 2614 MiB reserved, 63707 MiB outside PyTorch
    capturing vision256: 31020 MiB free, 1242 MiB allocated, 2614 MiB reserved, 63707 MiB outside PyTorch
    capturing vision64: 31020 MiB free, 1242 MiB allocated, 2614 MiB reserved, 63707 MiB outside PyTorch

## Per-graph pool requests (largest text graph vs every vision graph; full list pools.tsv)

    Graph pool requests (MiB, 74 compiled forwards, largest first; max 1008.4 in prefill4096)
    text top 5: prefill4096 1008.4, speculative32x4 980.8, prefill3584 948.4, prefill3072 888.3, prefill2560 842.5
    vision: vision32768 976.0, vision16384 488.0, vision4096 194.5, vision1024 136.0, vision256 120.1, vision64 110.2
    PASS largest vision pool vision32768 976.0 MiB vs largest text pool prefill4096 1008.4 MiB (margin 32.4 MiB; 6 vision graphs, 68 text graphs)

## MoE kernel choice

    

## Tuning (sharded vs reconciled, b3bf45f9)

    last b12x ready lines:
    b12x ready comm.pcie: 1357/1357 ready, candidates 0/12 prepared, 0 measured, 830 cached, 0 restored, 0 rejected, 0 compilations, 1:18

## Memory inventory per rank (inventory.tsv)

    PASS ranks 0/1/2 MiB: used 66,321/66,319/66,315 (r2-max -6); unattributed 1,333/1,331/1,333 (r2-max +0); retained:Allocator slack 19/19/18 (r2-max -1); core 62,448/62,448/62,442 (r2-max -6); retained 62,449/62,449/62,443 (r2-max -6)
    kv_b: no kv_b owner among the listed top groups
    tail example: retained: 22.06 MiB resources[28][*]<VisionStaging>.blocks<WeightStaging>.slots[*]['block'] — ignition/components/vision_staging.py:80
    Rank 0 / CUDA 0: 66,321.31 MiB used, 31,019.56 MiB free
    Core fixed mappings: 62,448.00 MiB; kernel globals: 591.12 KiB
    Retained storage: 62,448.58 MiB
    Graph capture pool: 1,110.00 MiB
    Allocator slack: 19.33 MiB
    Preparation-only storage: 2,540.13 MiB
    Allocator slack: 247.34 MiB
    Mapped host: 373.25 MiB; imported peers: 1,648.00 MiB (excluded from local VRAM)
    Device usage minus tracked physical storage: 1,332.61 MiB
    Rank 1 / CUDA 1: 66,319.31 MiB used, 31,021.56 MiB free
    Core fixed mappings: 62,448.00 MiB; kernel globals: 591.12 KiB
    Retained storage: 62,448.58 MiB
    Graph capture pool: 1,110.00 MiB
    Allocator slack: 19.37 MiB
    Preparation-only storage: 2,540.13 MiB
    Allocator slack: 247.34 MiB
    Mapped host: 373.25 MiB; imported peers: 1,648.00 MiB (excluded from local VRAM)
    Device usage minus tracked physical storage: 1,330.61 MiB
    Rank 2 / CUDA 2: 66,315.31 MiB used, 31,025.56 MiB free
    Core fixed mappings: 62,442.00 MiB; kernel globals: 591.12 KiB
    Retained storage: 62,442.58 MiB
    Graph capture pool: 1,110.00 MiB
    Allocator slack: 18.18 MiB
    Preparation-only storage: 2,540.13 MiB
    Allocator slack: 247.34 MiB
    Mapped host: 373.25 MiB; imported peers: 1,648.00 MiB (excluded from local VRAM)
    Device usage minus tracked physical storage: 1,332.61 MiB

