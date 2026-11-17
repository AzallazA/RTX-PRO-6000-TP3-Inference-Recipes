#!/usr/bin/env python3
"""Compare two vision_feature_dump.py files image by image; prints ONE line starting VISION_FEATURES:.

  vision_feature_cmp.py A.pt B.pt [--label NAME] [--expect identical|rounding] [--json OUT.json]

Per image: max abs diff, max rel diff (max abs diff / max |A|), fraction of elements bit-identical (BF16 bit
patterns), cosine similarity (float64). Verdict per image and overall (the worst image):
  bit-identical   every element has the same bits
  rounding-level  max rel <= --max-rel (1e-2) and cosine >= --min-cos (0.9999), all finite
  LARGER          anything else (incl. NaN/Inf, shape mismatch, missing image)
Exit status: 0 if the overall verdict is at least --expect (default rounding) and neither run reported a problem
(op error word, non-finite features, in-process replays not bit-identical, short publication); 1 otherwise;
2 on unreadable input.
"""

from __future__ import annotations

import argparse
import json
import sys

ORDER = ("bit-identical", "rounding-level", "LARGER")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--label", default="")
    p.add_argument("--expect", choices=("identical", "rounding"), default="rounding")
    p.add_argument("--max-rel", type=float, default=1e-2)
    p.add_argument("--min-cos", type=float, default=0.9999)
    p.add_argument("--json", default=None, help="write per-image numbers here")
    return p.parse_args(argv)


def compare(a, b, max_rel_limit=1e-2, min_cos=0.9999):
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(verdict="LARGER", note=f"shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}")
    if a.dtype == torch.bfloat16:
        bits = a.view(torch.int16) == b.view(torch.int16)
    else:
        bits = a == b
    x, y = a.double(), b.double()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    diff = (x - y).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.
    scale = float(x.abs().max()) if x.numel() else 0.
    max_rel = max_abs / scale if scale > 0 else (0. if max_abs == 0 else float("inf"))
    denominator = float(x.norm() * y.norm())
    cosine = float((x * y).sum()) / denominator if denominator > 0 else (1. if max_abs == 0 else 0.)
    if bool(bits.all()):
        verdict = "bit-identical"
    elif finite and max_rel <= max_rel_limit and cosine >= min_cos:
        verdict = "rounding-level"
    else:
        verdict = "LARGER"
    return dict(verdict=verdict, max_abs=max_abs, max_rel=max_rel, bit_identical=float(bits.double().mean()),
                cosine=cosine, finite=finite, max_abs_a=scale)


def run_problems(data, tag):
    problems = []
    for name, info in data.get("info", {}).items():
        if any(info.get("errors", [])):
            problems.append(f"{tag}:{name}:errors{info['errors']}")
        if not info.get("finite", True):
            problems.append(f"{tag}:{name}:non-finite")
        if not info.get("replays_identical", True):
            problems.append(f"{tag}:{name}:replays-differ")
        if info.get("published", info.get("features")) != info.get("features"):
            problems.append(f"{tag}:{name}:published{info.get('published')}!={info.get('features')}")
    return problems


def short(name):
    return name.split("/")[-1].removesuffix(".png")


def main(argv=None):
    args = parse_args(argv)
    import torch

    try:
        a, b = (torch.load(path, map_location="cpu", weights_only=False) for path in (args.a, args.b))
        fa, fb = a["features"], b["features"]
    except Exception as error:  # noqa: BLE001
        print(f"VISION_FEATURES: {args.label} ERROR cannot read inputs: {error!r}")
        return 2
    rows, worst = {}, 0
    for name in [*fa, *(n for n in fb if n not in fa)]:
        if name not in fa or name not in fb:
            rows[name] = dict(verdict="LARGER", note="missing in " + ("A" if name not in fa else "B"))
        else:
            rows[name] = compare(fa[name], fb[name], args.max_rel, args.min_cos)
        worst = max(worst, ORDER.index(rows[name]["verdict"]))
    problems = run_problems(a, "A") + run_problems(b, "B")
    overall = ORDER[worst] if rows else "LARGER"
    allowed = 0 if args.expect == "identical" else 1
    ok = rows and worst <= allowed and not problems
    parts = []
    for name, r in rows.items():
        if "max_abs" in r:
            parts.append(f"{short(name)} abs {r['max_abs']:.3g} rel {r['max_rel']:.3g} "
                         f"same {r['bit_identical']:.4f} cos {r['cosine']:.7f} {r['verdict']}")
        else:
            parts.append(f"{short(name)} {r['verdict']} ({r['note']})")
    label = f"{args.label} " if args.label else ""
    sides = f"{a.get('meta', {}).get('label', args.a)} vs {b.get('meta', {}).get('label', args.b)}"
    print(f"VISION_FEATURES: {label}{sides}: overall {overall} (expect {args.expect}: {'PASS' if ok else 'FAIL'}"
          f"{'; run problems ' + ','.join(problems) if problems else ''}) | " + " | ".join(parts))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(dict(label=args.label, a=args.a, b=args.b, meta_a=a.get("meta"), meta_b=b.get("meta"),
                           overall=overall, expect=args.expect, ok=bool(ok), problems=problems, images=rows),
                      handle, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
