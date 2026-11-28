#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Build a DeepSeek V4.1 expert order from route counts measured on YOUR traffic (stdlib only; run with python3 -I).

The expert order decides which routed experts of a split (parked) MoE layer stay resident in VRAM: the export keeps
the HOT experts with the largest counts of each layer resident, the rest are served from host memory. The release
order (tp3-tools/release/expert-order) was measured on a coding + prose mix; an order measured on your own traffic
keeps more of your routes on the resident path.

Workflow (details: --help of each subcommand):
  1. Start `ignition debug-serve` with the export's settings and the route counter on, e.g.
       DEBUG_SERVE=1 FLARE_DS41_ROUTE_COUNTS=$PWD/counts FLARE_DS41_DEBUG_KV_PAGES=4096 HELIOPAUSE_PATH=<site dir> \\
         CKPT=<pack> <the release export.sh environment> tp3-tools/ds41-exl3-export-tp2.sh
     The counter adds every live row's top-k router expert IDs per layer (prefill graphs and decode/verify graphs
     separately) into a table in every rank process; it is debug-serve only (an export refuses the switch).
  2. Snapshots between traffic phases, taken while the server is idle (cumulative tables, one per rank process):
       ds41_route_order.py snap --counts counts start
       ... class-1 traffic, first half ...   ds41_route_order.py snap --counts counts code-a
       ... class-1 traffic, second half ...  ds41_route_order.py snap --counts counts code-b
       ... class-2 traffic, first half ...   ds41_route_order.py snap --counts counts prose-a
       ... class-2 traffic, second half ...  ds41_route_order.py snap --counts counts prose-b
     "code" and "prose" are just the two traffic classes of the blend (step 4); with one class only, take start,
     code-a and code-b and build with --code-share 1.0. `drive` sends synthetic traffic from a text file if you
     have no live traffic to replay.
  3. Stop the debug-serve (Ctrl-C).
  4. ds41_route_order.py build --counts counts --out my-order [--release tp3-tools/release/expert-order]
     writes expert-routes-rank0-layer{0..39}.json (the export's format, router-ID space) and a report.
  5. Export with COUNTS=my-order (tp3-tools/ds41-exl3-export-tp2.sh / the release export.sh read COUNTS).

Blend (the method of the release order): per layer, CODE_SHARE x normalized class-1 distribution + (1 - CODE_SHARE)
x normalized class-2 distribution; default 0.6 / 0.4. Normalizing per class keeps the larger class from dominating
by volume alone. --groups selects the rows counted: all (prefill + decode/verify, default) or step (decode and
verification rows only: the rows whose expert reads decide decode speed; the EXL3 TP2 release used step) or prefill.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

LAYERS = 40
TAGS = ("start", "code-a", "code-b", "prose-a", "prose-b")


def parse_layers(spec):
    out = set()
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def rank_dirs(counts):
    return sorted(p for p in Path(counts).iterdir() if p.is_dir() and p.name.startswith("dev"))


# ---- snap -------------------------------------------------------------------------------------------------------
def snap(counts, tag, timeout=120):
    """Ask every rank process for a cumulative snapshot <tag>; True when all answered."""
    if not rank_dirs(counts):
        raise SystemExit(f"no rank directories (dev*) in {counts}: is the debug-serve up with FLARE_DS41_ROUTE_COUNTS={counts}?")
    Path(counts, f"request-{tag}").touch()
    t0 = time.time()
    while time.time() - t0 < timeout:
        procs = rank_dirs(counts)
        done = [p for p in procs if (p / f"done-{tag}").exists()]
        if len(done) == len(procs) and time.time() - t0 > 1:
            print(f"snapshot {tag}: {len(done)} rank processes")
            return True
        time.sleep(0.5)
    print(f"snapshot {tag}: TIMEOUT ({len(done)} of {len(procs)} answered)")
    return False


# ---- drive (optional synthetic traffic) -------------------------------------------------------------------------
def drive(a):
    """Requests built from slices of a text file at a fixed concurrency for a fixed time."""
    text = Path(a.corpus).read_text(errors="replace")
    tasks = [line.strip() for line in Path(a.tasks).read_text().splitlines() if line.strip()] if a.tasks else [""]
    deadline, lock, stats, seq = time.time() + a.seconds, threading.Lock(), {"requests": 0, "errors": 0}, [a.seed]
    url = f"http://{a.host}:{a.port}/v1/chat/completions"

    def worker():
        while time.time() < deadline:
            with lock:
                seed = seq[0]
                seq[0] += 1
            rng = random.Random(seed)
            chars = int(rng.choice(a.sizes) * 3.6)  # ~3.6 characters per token
            off = rng.randrange(max(1, len(text) - chars))
            prompt = (text[off:off + chars] + "\n\n" + rng.choice(tasks)).strip()
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
                    "temperature": 0.6, "top_p": 0.95,
                    "chat_template_kwargs": {"enable_thinking": a.thinking_every > 0 and seed % a.thinking_every == 0}}
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                json.load(urllib.request.urlopen(request, timeout=1800))
                key = "requests"
            except Exception as error:  # keep going; failures are counted
                print(f"request failed: {error!r}"[:200], file=sys.stderr)
                key = "errors"
                time.sleep(2)
            with lock:
                stats[key] += 1
    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"drive done: {stats}")
    return 0 if stats["requests"] else 1


# ---- build ------------------------------------------------------------------------------------------------------
def load_snaps(counts, code_only):
    procs = rank_dirs(counts)
    if not procs:
        raise SystemExit("ROUTES FAIL no rank snapshot directories (dev*) in " + str(counts))
    snaps, unmasked = {}, set()
    for tag in TAGS:
        tables = []
        for p in procs:
            f = p / f"snap-{tag}.json"
            if not f.is_file():
                if code_only and tag.startswith("prose"):
                    break
                raise SystemExit(f"ROUTES FAIL {f} missing")
            d = json.loads(f.read_text())
            unmasked.update(d.get("unmasked", []))
            tables.append(d["groups"])
        else:
            if any(t != tables[0] for t in tables[1:]):
                raise SystemExit(f"ROUTES FAIL rank processes disagree on snapshot {tag} (routing is replicated)")
            snaps[tag] = tables[0]
            continue
        snaps[tag] = snaps["code-b"]  # one traffic class: empty class-2 phases
    return snaps, len(procs), sorted(unmasked)


def diff(a, b, group):
    """b - a for one group ('prefill', 'step', 'all') -> [layer][expert]."""
    def g(s, name):
        return s[name] if name != "all" else [[x + y for x, y in zip(p, q)] for p, q in zip(s["prefill"], s["step"])]
    return [[y - x for x, y in zip(ra, rb)] for ra, rb in zip(g(a, group), g(b, group))]


def norm(row):
    t = sum(row)
    return [x / t for x in row] if t > 0 else [0.0] * len(row)


def blend(code, prose, share):
    return [[share * c + (1 - share) * p for c, p in zip(norm(cr), norm(pr))] for cr, pr in zip(code, prose)]


def hot_set(row, hot):
    return set(sorted(range(len(row)), key=lambda e: (-row[e], e))[:hot])  # ties by lower ID, as weights.py


def cold(dist, hot):
    return sum(x for e, x in enumerate(dist) if e not in hot)


def build(a):
    code_only = a.code_share >= 1.0
    snaps, nproc, unmasked = load_snaps(a.counts, code_only)
    park = parse_layers(a.park)
    phases = {}
    for group in ("all", "step", "prefill"):
        phases[group] = {
            "code": diff(snaps["start"], snaps["code-b"], group), "prose": diff(snaps["code-b"], snaps["prose-b"], group),
            "code-a": diff(snaps["start"], snaps["code-a"], group), "code-b": diff(snaps["code-a"], snaps["code-b"], group),
            "prose-a": diff(snaps["code-b"], snaps["prose-a"], group), "prose-b": diff(snaps["prose-a"], snaps["prose-b"], group)}
    P = phases[a.groups]
    totals = {k: sum(map(sum, v)) for k, v in P.items()}
    lines, problems = [], []
    if unmasked:
        lines.append(f"WARNING rows counted without a live mask (shape mismatch) on layers {unmasked}")
    for L in park:
        for ph in ("code",) if code_only else ("code", "prose"):
            n = sum(P[ph][L])
            if n < a.min_routes:
                problems.append(f"layer {L} {ph}: {n} routes < {a.min_routes}")
    if problems:
        print("\n".join(problems[:10]))
        raise SystemExit(f"ROUTES FAIL too few routes ({len(problems)} layer/phase cells), e.g. {problems[0]}; "
                         "send more traffic or lower --min-routes")
    full = blend(P["code"], P["prose"], a.code_share)
    rel = None
    if a.release:
        rel = {L: json.loads((Path(a.release) / f"expert-routes-rank0-layer{L}.json").read_text())["counts"]
               for L in range(LAYERS)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = (f"debug-serve route counts, blend {a.code_share:g} class 1 / {1 - a.code_share:g} class 2 of normalized "
           f"distributions, groups {a.groups}; router-ID space")
    for L in range(LAYERS):
        counts = [int(round(x * 1e9)) for x in full[L]]
        if sum(counts) == 0:
            if rel is None:
                raise SystemExit(f"ROUTES FAIL layer {L} has no routes and no --release order to fall back on")
            counts = rel[L]
            lines.append(f"layer {L}: no routes measured, --release counts kept")
        (out / f"expert-routes-rank0-layer{L}.json").write_text(json.dumps(
            {"rank": 0, "layer": L, "total_routes": sum(counts), "counts": counts, "source": src,
             "raw_routes": {ph: sum(P[ph][L]) for ph in ("code", "prose")}}))
    summary = (f"ROUTES OK order {out}: {nproc} rank processes agree; routes class 1 {totals['code']:,} class 2 "
               f"{totals['prose']:,} ({a.groups})")
    rows = []
    if rel is not None:
        H, agg = a.hot, {"ov": [], "in_rel": 0.0, "in_new": 0.0, "ho_rel": 0.0, "ho_new": 0.0}
        fa = blend(P["code-a"], P["prose-a"], a.code_share)
        fb = blend(P["code-b"], P["prose-b"], a.code_share)
        for L in park:
            new, old = hot_set(full[L], H), hot_set(rel[L], H)
            ov = len(new & old) / H
            in_rel, in_new = cold(full[L], old), cold(full[L], new)
            ho_rel = (cold(fb[L], old) + cold(fa[L], old)) / 2
            ho_new = (cold(fb[L], hot_set(fa[L], H)) + cold(fa[L], hot_set(fb[L], H))) / 2
            rows.append(f"| {L} | {ov * 100:.1f} | {len(new - old)} | {in_rel * 100:.2f} | {in_new * 100:.2f} | "
                        f"{ho_rel * 100:.2f} | {ho_new * 100:.2f} |")
            agg["ov"].append(ov)
            for k, v in (("in_rel", in_rel), ("in_new", in_new), ("ho_rel", ho_rel), ("ho_new", ho_new)):
                agg[k] += v / len(park)
        summary += (f"; hot {H} on {len(park)} split layers: overlap with --release mean "
                    f"{sum(agg['ov']) / len(park) * 100:.1f}% min {min(agg['ov']) * 100:.1f}%; cold-route share "
                    f"--release -> new: in-sample {agg['in_rel'] * 100:.2f}% -> {agg['in_new'] * 100:.2f}%, held-out "
                    f"{agg['ho_rel'] * 100:.2f}% -> {agg['ho_new'] * 100:.2f}%")
    report = Path(a.report) if a.report else out / "ROUTES.md"
    report.write_text("\n".join([
        "# Route counts -> expert order", "", summary, "", *lines, "",
        f"Per-phase routes ({a.groups}): " + ", ".join(f"{k} {v:,}" for k, v in totals.items()), "",
        *(["| layer | overlap % | new experts | cold release % (in) | cold new % (in) | cold release % (held-out) | "
           "cold new % (held-out) |", "|---|---|---|---|---|---|---|", *rows, ""] if rows else [])]))
    print(summary)


def selftest():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = random.Random(1)
        rel = d / "release"
        rel.mkdir()
        base = [[rng.paretovariate(1.2) for _ in range(384)] for _ in range(LAYERS)]
        for L in range(LAYERS):
            (rel / f"expert-routes-rank0-layer{L}.json").write_text(json.dumps({"counts": [int(x * 1000) for x in base[L]]}))
        cum = {"prefill": [[0] * 384 for _ in range(LAYERS)], "step": [[0] * 384 for _ in range(LAYERS)]}
        for proc in ("dev0-pid1", "dev1-pid2"):
            (d / "c" / proc).mkdir(parents=True)
        for tag in TAGS:
            if tag != "start":
                for g in cum:
                    for L in range(LAYERS):
                        w = [b * (1.3 if (e % 7 == 0 and tag.startswith("code")) else 1.0) for e, b in enumerate(base[L])]
                        for e in rng.choices(range(384), weights=w, k=6000):
                            cum[g][L][e] += 1
            for proc in ("dev0-pid1", "dev1-pid2"):
                (d / "c" / proc / f"snap-{tag}.json").write_text(json.dumps({"groups": cum, "unmasked": []}))
        for share, groups in ((0.6, "all"), (0.6, "step"), (1.0, "step")):
            a = argparse.Namespace(counts=d / "c", release=rel, out=d / f"o{share}{groups}", report=None, hot=320,
                                   park="2-4,6-39", code_share=share, groups=groups, min_routes=1000)
            build(a)
            files = sorted(a.out.glob("expert-routes-rank0-layer*.json"))
            assert len(files) == LAYERS and len(json.loads(files[0].read_text())["counts"]) == 384
        for proc in ("dev0-pid1", "dev1-pid2"):  # one traffic class: no prose snapshots at all
            for tag in ("prose-a", "prose-b"):
                (d / "c" / proc / f"snap-{tag}.json").unlink()
        build(argparse.Namespace(counts=d / "c", release=None, out=d / "o1", report=None, hot=320, park="2-4,6-39",
                                 code_share=1.0, groups="step", min_routes=1000))
        print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("snap", help="take a cumulative snapshot <tag> in every rank process (server idle)",
                       description=f"Writes request-<tag> into the counts directory and waits until every rank process "
                                   f"answered (dev*/snap-<tag>.json + done-<tag>). build uses the tags {', '.join(TAGS)}.")
    s.add_argument("--counts", required=True, help="the FLARE_DS41_ROUTE_COUNTS directory of the running debug-serve")
    s.add_argument("tag", choices=TAGS)
    s.add_argument("--timeout", type=float, default=120)
    b = sub.add_parser("build", help="snapshots -> expert order (export format) + report",
                       description="Phases are snapshot differences: class 1 = code-b - start (halves code-a, code-b), "
                                   "class 2 = prose-b - code-b (halves prose-a, prose-b). Every rank process's tables "
                                   "must be identical (routing is replicated over tensor-parallel ranks). Output: "
                                   "OUT/expert-routes-rank0-layer{0..39}.json ({'counts': [per expert, router-ID space], "
                                   "...}), pass OUT as COUNTS to the export; report OUT/ROUTES.md. With --release, the "
                                   "report compares the hot sets at --hot on the split layers (--park): overlap and the "
                                   "share of routes that would go to host-resident (cold) experts, in-sample and held-out "
                                   "(set from one half, evaluated on the other).")
    b.add_argument("--counts", required=True, help="the counts directory (dev*/snap-<tag>.json from snap)")
    b.add_argument("--out", required=True, help="output directory for the 40 layer files")
    b.add_argument("--release", help="an existing order to compare with (e.g. tp3-tools/release/expert-order); also "
                                     "the fallback for a layer without routes")
    b.add_argument("--report", help="report path (default OUT/ROUTES.md)")
    b.add_argument("--code-share", type=float, default=0.6,
                   help="weight of class 1 (code) in the blend; class 2 (prose) gets 1 - share (default 0.6; 1.0 = one "
                        "traffic class, prose snapshots optional)")
    b.add_argument("--groups", default="all", choices=("all", "step", "prefill"),
                   help="rows counted: all = prefill + decode/verify (default), step = decode and verification rows "
                        "only (used for the EXL3 TP2 release order), prefill = prefill rows only")
    b.add_argument("--hot", type=int, default=320, help="hot experts per split layer for the comparison (default 320)")
    b.add_argument("--park", default="2-4,6-39", help="split (parked) layers for the checks (default 2-4,6-39)")
    b.add_argument("--min-routes", type=int, default=200000,
                   help="minimum routes per split layer and class (default 200000); fewer = FAIL")
    d = sub.add_parser("drive", help="optional synthetic traffic from a text file (one class per call)",
                       description="Sends chat requests built from random slices of CORPUS (+ one line of TASKS, if "
                                   "given) at CONCURRENCY streams for SECONDS. Size the prompts to the debug-serve's KV "
                                   "capacity (FLARE_DS41_DEBUG_KV_PAGES x 128 tokens): concurrency x (size + max_tokens) "
                                   "must fit.")
    d.add_argument("--corpus", required=True)
    d.add_argument("--tasks", help="file with one instruction per line appended to each prompt")
    d.add_argument("--seconds", type=float, default=450)
    d.add_argument("--concurrency", type=int, default=4)
    d.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[1000, 2000, 4000, 8000],
                   help="prompt sizes in tokens, comma list (default 1000,2000,4000,8000)")
    d.add_argument("--max-tokens", type=int, default=1536)
    d.add_argument("--thinking-every", type=int, default=3, help="every n-th request with thinking on (0 = never)")
    d.add_argument("--seed", type=int, default=5000)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=15015)
    d.add_argument("--model", default="DeepSeek-V4.1-Flash")
    sub.add_parser("selftest", help="synthetic snapshots through build (temporary directory only)")
    a = ap.parse_args()
    if a.mode == "snap":
        return 0 if snap(a.counts, a.tag, a.timeout) else 1
    if a.mode == "drive":
        return drive(a)
    if a.mode == "build":
        build(a)
        return 0
    selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
