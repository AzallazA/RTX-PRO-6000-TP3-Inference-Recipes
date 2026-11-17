# Building an expert order from your own traffic

On two GPUs about 15 % of the experts live in host RAM, and every route to one of them is read over PCIe. Which experts
stay resident is decided by an **expert order**: per-layer route counts, most used first. The release order
(`expert-order/` in this folder) was measured on our traffic, a blend of 0.6 code (agentic coding) and 0.4 prose. If your
traffic looks different (another language, another domain, long documents instead of code), measuring your own routes
and exporting with them usually sends fewer routes to host RAM, at the same KV. That often helps more than adding
resident experts ([KV-TRADEOFF.md](KV-TRADEOFF.md)).

What it costs: one debug server run (about 30-40 min to prepare, then as long as you drive traffic) and one export.
Nothing here changes the model's outputs; the order only decides which experts sit in VRAM.

## How it works

1. A **debug server** (`ignition debug-serve`, the in-process Python server) runs the model with a route counter on.
   It counts, per layer and expert, how often a token routes there, separately for prefill rows and decode rows.
2. You send **your traffic** and take **snapshots** of the counters between phases.
3. `tp3-tools/ds41_route_order.py build` turns the snapshot differences into an order. We recommend `--groups step`:
   count decode rows only. Decode is where host-RAM reads cost speed; prefill rows would otherwise dominate the counts.
4. You **export** with that order.

The debug server is slower than the exported package (it is not built for speed) and has a limited KV cache by
default; `FLARE_DS41_DEBUG_KV_PAGES` raises it.

## Steps

Run from the fork checkout (`<fork>`, built as in [EXPORT.md](EXPORT.md)), with the pack at `<pack>` and this folder at
`<recipes>`. The GPUs must be free.

**1. Build the debug-serve support library** (once):

```bash
cd <fork>/heliopause && maturin build --release --features qualification -o <wheel-dir>
python -m zipfile -e <wheel-dir>/*.whl <site>
```

**2. Start the debug server with the route counter on:**

```bash
cd <fork>
DEBUG_SERVE=1 HELIOPAUSE_PATH=<site> \
FLARE_DS41_ROUTE_COUNTS=$PWD/counts FLARE_DS41_DEBUG_KV_PAGES=4096 \
CKPT=<pack> <recipes>/export.sh
```

It prepares the model (30-40 min) and then serves on `127.0.0.1:15015`, like the package. `FLARE_DS41_DEBUG_KV_PAGES=4096`
gives it 524,288 KV tokens (about 445 MiB per GPU); without it the debug server holds only a few thousand tokens and
cannot run real prompts. These switches are for the debug server only; the export refuses them.

**3. Snapshot, send traffic, snapshot.** Take a snapshot as soon as the server answers:

```bash
python3 -I tp3-tools/ds41_route_order.py snap --counts counts start
```

Then send your first traffic class (for us: code). Use real requests if you can replay them; otherwise generate some
from a text file of typical prompts:

```bash
python3 -I tp3-tools/ds41_route_order.py drive --corpus my-code.txt --seconds 450
python3 -I tp3-tools/ds41_route_order.py snap --counts counts code-a
python3 -I tp3-tools/ds41_route_order.py drive --corpus my-code.txt --seconds 450
python3 -I tp3-tools/ds41_route_order.py snap --counts counts code-b
```

The two halves (`-a`, `-b`) let the builder check that the order is stable (it trains on one half and checks on the
other). If you have a second class (for us: prose), repeat with snapshots `prose-a` and `prose-b`. With one class, stop
after `code-b`.

How much traffic: aim for at least about 200,000 decode routes per layer per class (the builder reports it). In our run
that took about 15 minutes of code traffic at 4 streams; prose with long prompts needs more time because prefill rows
are not counted with `--groups step`.

**4. Stop the debug server** (Ctrl-C).

**5. Build the order:**

```bash
python3 -I tp3-tools/ds41_route_order.py build --counts counts --out my-order --groups step \
  --release <recipes>/expert-order
```

- The default blend is 0.6 of the first class and 0.4 of the second. With one class use `--code-share 1.0`.
- `--release` points at this release's order (`<recipes>/expert-order`). The builder compares your order with it and
  falls back to it for any layer with too few routes; `my-order/ROUTES.md` reports the routes per layer, how much
  the two halves agree, and how many routes would go to host RAM with each order at the release resident counts.

**6. Export with your order:**

```bash
COUNTS=$PWD/my-order CKPT=<pack> <recipes>/export.sh
```

Everything else (resident counts, switches) stays as in the release. You can combine it with other resident counts
(`HOT_LOW` / `HOT_HIGH`, see [KV-TRADEOFF.md](KV-TRADEOFF.md)).

## Tips

- **Match your real mix.** If your traffic is half chat and half code, measure both and set `--code-share` to your mix.
  An order measured on one kind of traffic only can be clearly worse on the other (we saw this with a code-only order
  on prose).
- **Use enough traffic.** Small samples make a noisy order. Check the half-to-half agreement in `ROUTES.md`.
- **Decode rows matter most.** Keep `--groups step` unless your workload is mostly long prompts with short answers.
- **Speculative decoding:** the debug server picks its own verification widths, which may differ from the package. That
  changes how many rows each step has, not which experts are used, so the order still applies.
- Keep a copy of the release `expert-order/` to fall back to.
