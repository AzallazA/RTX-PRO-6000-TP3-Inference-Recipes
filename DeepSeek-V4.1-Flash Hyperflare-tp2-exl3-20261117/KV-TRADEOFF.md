# KV cache vs speed: choosing the resident expert counts

The release package keeps **344 experts per layer resident in VRAM on layers 2-19 and 316 on layers 20-39** (of 384;
layers 0, 1 and 5 are fully resident), which leaves **2,140,800 KV tokens**. You can move that point in either direction
with one re-export ([EXPORT.md](EXPORT.md)); nothing else in the recipe changes.

- **More KV, slower decode:** fewer resident experts.
- **Less KV, faster decode:** more resident experts.

## More KV (fewer resident experts)

The resident expert counts are the main KV lever. Every expert that is not resident in VRAM lives in host RAM and is
read over PCIe whenever a token routes to it. Lowering `HOT_LOW` / `HOT_HIGH` (they set
`FLARE_DS41_EXL3_HOT_LAYERS`) frees VRAM for the KV cache. In exchange, more routes go to host-RAM experts, so
**decode gets slower**. How much slower depends on your PCIe link and workload; we have not measured it on this build.

**Rule of thumb:** one fewer resident expert on all 37 split layers (`HOT_LOW` and `HOT_HIGH` both −1) frees about
**234 MiB per GPU ≈ +265K KV tokens** (our planning model; the KV cache costs 890 bytes per token per GPU). One layer alone is about 1/37
of that (~7K tokens). Each step of the rule also moves 37 experts into host RAM: about +0.45 GiB of host RAM
(estimate).

| `HOT_LOW` / `HOT_HIGH` | KV tokens | Note |
| --- | --- | --- |
| **344 / 316** (release) | **2,140,800** | measured |
| 342 / 314 | ≈ 2.7M | estimate |
| 340 / 312 | ≈ 3.2M | estimate |
| 336 / 308 | ≈ 4.3M | estimate |
| 332 / 304 | ≈ 5.3M | estimate; `HOT_HIGH` below 308 not tested |
| 320 / 292 | ≈ 8.5M | estimate, not tested; ~11 GiB more host RAM |

Estimates come from the rule and the measured release point; check the `Allocate cache` line of your own export. The
two counts do not have to move together. `export.sh --plan-only` prints the declared VRAM
per GPU for a setting before any GPU work: 1 GiB less declared ≈ +1.2M tokens. `export.sh` lowers `EXPORT_HOT` to the
smaller of the two counts when you go below 308.

```bash
CKPT=~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw HOT_LOW=332 HOT_HIGH=304 /path/to/this-folder/export.sh
```

**Other, smaller levers:**

- **Image slots** (`IMAGE_CONCURRENCY`, export): already at the minimum, 1. Hyperflare's default (4 here) would cost
  120 MiB more per GPU.
- **CUDA stack limit** (`STACK_LIMIT` in `serve.sh`): already at 128 B, just above the largest registered kernel
  (120 B). The driver default (1,024 B, `STACK_LIMIT=`) would cost roughly 290K KV tokens (estimate).
- **Verification cap** (`BUDGET` in `serve.sh`): a lower cap frees a little VRAM (on the TP3 build, 24 instead of 32
  gave +36K tokens); not measured here.
- **Prefix cache** (`PREFIX_CACHE` in `serve.sh`): host RAM, not VRAM. It does not change the KV token count, but a
  larger one keeps more reusable prompt KV for returning sessions, which then re-prefill less.

## Less KV, more speed (more resident experts)

The same rule works the other way: one more resident expert on all 37 split layers costs about 234 MiB per GPU, about
265K KV tokens, and moves 37 fewer experts to host RAM. Fewer routes then go over PCIe, so decode gets a little faster.
The gain shrinks with every step: the experts added last are the least used ones (the order puts the most used experts
first), and with the release order only about 0.5 % of decode routes still go to host RAM. We have not measured speeds
for settings above the release point.

| `HOT_LOW` / `HOT_HIGH` | KV tokens | Note |
| --- | --- | --- |
| **344 / 316** (release) | **2,140,800** | measured |
| 345 / 317 | ≈ 1.9M | estimate |
| 346 / 318 | ≈ 1.6M | estimate |
| 348 / 320 | ≈ 1.1M | estimate; about the KV of the pack's own two-GPU vLLM recipe (1.07M) |

Keep enough KV for your longest sessions: one full 1M-token session needs 1M tokens of KV plus room for anything else
running. The export prints the KV it allocated (`Allocate cache`); if it is lower than you need, step back down.

## Measuring your own traffic first

The resident experts are chosen by route counts. If your traffic is not code and prose like ours, measuring your own
routes ([EXPERT-ORDER.md](EXPERT-ORDER.md)) usually helps more than adding experts: a better order sends fewer routes to
host RAM at the same KV.
