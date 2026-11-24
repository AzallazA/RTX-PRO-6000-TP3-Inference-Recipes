"""Snapshot/diff Helios speculative metrics: per (batch,width) rounds, GPU ms/round, tokens/round,
GPU busy share, and per-depth draft acceptance (flare_speculative_drafts_total).
usage: spec_metrics.py snap OUT.json | spec_metrics.py diff A.json B.json"""
import json, re, sys, time, urllib.request
NAMES = ("rounds", "gpu_nanoseconds", "emitted_tokens", "accepted_drafts", "offered_drafts")
_BW_RE = re.compile(r'flare_speculative_(\w+)_total\{batch="(\d+)",width="(\d+)"\} (\S+)')
_DEPTH_RE = re.compile(r'flare_speculative_drafts_total\{([^}]+)\} (\S+)')
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_metrics(text):
    out = {"t": time.time(), "m": {}, "d": {}}
    for line in text.splitlines():
        m = _BW_RE.match(line)
        if m and m[1] in NAMES:
            out["m"].setdefault(f"{m[2]},{m[3]}", {})[m[1]] = float(m[4])
            continue
        dm = _DEPTH_RE.match(line)
        if dm:
            labels = dict(_LABEL_RE.findall(dm[1]))
            depth, source, outcome = labels.get("depth"), labels.get("source"), labels.get("outcome")
            if depth and source and outcome in ("offered", "accepted"):
                out["d"].setdefault(source, {}).setdefault(depth, {})[outcome] = float(dm[2])
    return out


def snap():
    text = urllib.request.urlopen("http://127.0.0.1:15015/metrics", timeout=10).read().decode()
    return parse_metrics(text)


def format_diff(a, b):
    lines = []
    wall = b["t"] - a["t"]
    gpu_total = 0.0
    lines.append(f"wall {wall:.1f} s")
    lines.append(f"{'batch,width':>11} {'rounds':>8} {'GPU ms/round':>12} {'tok/round':>9} {'accept':>6}")
    for k in sorted(b["m"], key=lambda s: tuple(map(int, s.split(",")))):
        d = {n: b["m"][k].get(n, 0) - a["m"].get(k, {}).get(n, 0) for n in NAMES}
        if d["rounds"] <= 0:
            continue
        gpu_total += d["gpu_nanoseconds"]
        acc = d["accepted_drafts"] / d["offered_drafts"] if d["offered_drafts"] else 0
        lines.append(f"{k:>11} {d['rounds']:8.0f} {d['gpu_nanoseconds']/d['rounds']/1e6:12.3f} "
                     f"{d['emitted_tokens']/d['rounds']:9.2f} {acc:6.2f}")
    busy_pct = (gpu_total / 1e9 / wall * 100) if wall > 0 else 0.0
    lines.append(f"GPU busy (speculative) {gpu_total/1e9:.1f} s of {wall:.1f} s wall = {busy_pct:.0f}%")

    a_depths, b_depths = a.get("d", {}), b.get("d", {})
    for source in sorted(b_depths):
        rows = []
        offered_d1 = 0.0
        for depth in sorted(b_depths[source], key=int):
            off = b_depths[source][depth].get("offered", 0.0) - a_depths.get(source, {}).get(depth, {}).get("offered", 0.0)
            acc = b_depths[source][depth].get("accepted", 0.0) - a_depths.get(source, {}).get(depth, {}).get("accepted", 0.0)
            if off <= 0 and acc <= 0:
                continue
            if int(depth) == 1:
                offered_d1 = off
            rows.append((int(depth), off, acc))
        if not rows:
            continue
        lines.append(f"\nper-depth drafts (source={source}):")
        lines.append(f"{'depth':>5} {'offered':>10} {'accepted':>10} {'cond_acc':>9} {'surv_d1':>9}")
        for depth, off, acc in rows:
            cond = acc / off if off > 0 else 0.0
            surv = acc / offered_d1 if offered_d1 > 0 else 0.0
            lines.append(f"{depth:5d} {off:10.0f} {acc:10.0f} {cond:9.3f} {surv:9.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "snap":
        with open(sys.argv[2], "w") as fh:
            json.dump(snap(), fh)
    else:
        with open(sys.argv[2]) as fa, open(sys.argv[3]) as fb:
            a, b = json.load(fa), json.load(fb)
        print(format_diff(a, b))
