# GLM TP2 export extract

## Phases ([ OK ] lines)

    [ OK ] Resolve checkpoint. glm-5.3-flash · <ckpt-dir>/glm53-… 0.3s
    [ OK ] Select GPUs. GPUs 0-1 · concurrency 8 · images 2 · up to 5 … 0.1s library scratch primed on cuda:0: 64 MiB at 0x78204e000000
    [ OK ] Connect tensor-parallel ranks. 3.7s
    [ OK ] Inspect checkpoint. 2.4s
    [ OK ] Load weights. 55.7s
    [ OK ] Load vision weights. 0.3s
    [ OK ] Allocate model state and collectives. one-shot all-reduce s… 0.1s
    [ OK ] Construct graph catalog. vision staging slots 22 MiB… 56/56 4.9s
    [ OK ] Prepare and tune kernels. 270.4s
    [ OK ] Warm serving graphs. 56/56 8.9s
    [ OK ] Measure graph memory. captured snapshot1 in 0.00s 56/56 8.0s
    [ OK ] Compile model forwards. 3 compiler workers; per ra… 28/28 311.1s
    [ OK ] Allocate KV cache. trial graph pool released: 2990 of 2990 … 0.2s
    [ OK ] Load and warm compiled forwards. 413 MiB portable … 56/56 120.3s
    [ OK ] Release serialized compiler state. 0 MiB; guards and… 56/56 0.0s
    [ OK ] Warm serving graphs. 56/56 0.0s
    [ OK ] Prepare serving graphs. captured snapshot1 in 0.00s 56/56 8.4s
    [ OK ] Wait for tensor-parallel ranks. 1/1 0.1s
    [ OK ] Clear disposable storage. 2579950922 bytes 0.0s
    [ OK ] Inspect CUDA graph functions. 4309 functions, 4281 modules 0.3s
    [ OK ] Serialize CUDA graphs. 56 graphs, 81053876 bytes 0.2s
    [ OK ] Capture compiled modules and globals. 275954531 bytes, 572143 global bytes downloaded 0.4s
    [ OK ] Scan and copy reachable storage. 90314074274 bytes scanned, 88653912226 bytes downloaded, 0 bytes stored; 35 allocations omitted 1.8s
    [ OK ] Export frontend assets. 0.0s
    [ OK ] Materialize weight payloads. 147.9s
    [ OK ] Wait for rank exports. 1/1 0.1s
    [ OK ] Release preparation resources. 4.8s
    [ OK ] Assemble core payloads. 2 ranks 21.8s
    [ OK ] Validate core publication. 1 deployments 0.0s
    [ OK ] Verify frontend payloads. 20228464 bytes 0.0s
    [ OK ] Compress and publish rank 0 executables. 356436264 plaintext bytes 0.5s
    [ OK ] Verify and publish rank 0 data. 89803469267 source bytes verified 4.9s
    [ OK ] Compress and publish rank 1 executables. 356436600 plaintext bytes 0.5s
    [ OK ] Verify and publish rank 1 data. 89803428923 source bytes verified 4.9s
    [ OK ] Verify and publish shared data. 0 source bytes verified 0.0s
    [ OK ] Sign and publish core metadata. 23739150 metadata bytes 0.1s
    [ OK ] Commit core. <store>/models/glm53-fp4-local/gb202-tp2-202611170201 0.0s
    [ OK ] Clear export staging. 0.3s
    [ OK ] Export native core. <store>/models/glm53-fp4… 204.7s

## Embedding / vision staging

    token embedding in device-visible host memory (FLARE_GLM53_EMBED_HOST=1)
    vision blocks and tail parked in device-visible host memory and staged per image (FLARE_GLM53_VISION_STAGE=1)
    vision staging slots 22 MiB (parked 373 MiB, tail copied per image 108 MiB): free=6877 MiB, allocated=4418 MiB, reserved=4546 MiB, rss=4870 MiB

## Per-graph pool requests (pools.tsv)

    Graph pool requests (MiB, 56 compiled forwards, largest first; max 1072.1 in prefill4096)
    text top 5: prefill4096 1072.1, prefill3584 1020.1, prefill3072 968.1, prefill2560 916.1, prefill2048 832.3
    vision: vision32768 976.0, vision16384 488.0, vision4096 194.5, vision1024 136.0, vision256 119.4, vision64 110.4
    PASS largest vision pool vision32768 976.0 MiB vs largest text pool prefill4096 1072.1 MiB (margin 96.1 MiB; 6 vision graphs, 50 text graphs)

## Tuning

    last progress line: 846/1176 ready, 69 measured, 603 cached, 0 restored, 0 compilations at 1:08
    [ OK ] Prepare and tune kernels. 270.4s
    selected kernels: {'cached': 21384, 'tuned': 2616, 'fixed': 11986}
    [ OK ] Compile model forwards. 3 compiler workers; per ra… 28/28 311.1s
    b12x selecting attention.dsa_indexer.glm_pool: 595/1176 ready, candidates 0/1 prepared, 69 measured, 596 cached, 0 restored, 0 rejected, 0 compilations, 1:04
    b12x selecting norm.mhc: 729/1176 ready, candidates 0/1 prepared, 69 measured, 600 cached, 0 restored, 0 rejected, 0 compilations, 1:06
    b12x selecting gemm.blockscaled.fixed: 846/1176 ready, candidates 0/1 prepared, 69 measured, 603 cached, 0 restored, 0 rejected, 0 compilations, 1:08

## Memory inventory (inventory.tsv)

    ranks 0/1 MiB: used 89,929/89,929; unattributed 1,243/1,243; retained:Allocator slack 17/17; core 85,696/85,696; retained 85,697/85,697; mapped_host 1,583/1,583

