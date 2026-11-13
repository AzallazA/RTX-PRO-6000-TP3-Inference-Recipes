#!/usr/bin/env python3
"""GLM vision lean-tower GPU equivalence probe (system python3 + Pillow), against a live server on PORT (15015).

  lean_probe.py run LABEL OUT.json     12 image prompts (6 set, 4 hard, 2 near-max 8000-token grids), first-token
                                       top-5 logprobs, max_tokens 1, serial; the 12 are sent twice (same-server check)
  lean_probe.py cmp A.json B.json      compares two runs prompt by prompt: exact top-5 match, top-1 match, max |delta|

The first token's logprobs come from the prefill over the image features. If the lean in-place tower is
bit-identical to the old tower on GPU, the old and lean packages give identical values (the GLM first token was
24/24 identical across restarts on the text-only TP3 package). Prints one LEAN_PROBE line.
"""

import base64, json, os, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "15015"))
V = pathlib.Path(os.environ.get("VISION_SETS", HERE.parents[2] / "DeepSeek-V4.1-Flash Kraken-tp3-20261019/qualification/vision"))
GLM_IMAGES = HERE.parents[1] / "tools" / "glm_images.py"   # this release folder's image check tool
SET, HARD = sorted((V / "set").glob("*.png")), sorted((V / "hard").glob("*.png"))
QUESTION = "What is shown in this image? Answer briefly."


def helpers():
    """grid() from glm_images.py without running its mode dispatch."""
    src = GLM_IMAGES.read_text().split('\nif mode == "latency":')[0]
    src = src.replace("mode, label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]", "")
    ns = {}
    exec(compile(src, "glm_images", "exec"), ns)
    return ns


def ask(img):
    content = [dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(img).decode())),
               dict(type="text", text=QUESTION)]
    body = dict(model="GLM-5.3-Flash", max_tokens=1, temperature=0, logprobs=True, top_logprobs=5,
                reasoning_effort="low", messages=[dict(role="user", content=content)])
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        top = (lp[0].get("top_logprobs") or []) if lp else []
        return dict(ok=True, top=[[t.get("token"), t.get("logprob")] for t in top],
                    prompt_tokens=r["usage"]["prompt_tokens"])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, top=[], error=repr(e)[:300])


def run(label, out):
    grid = helpers()["grid"]
    images = [p.read_bytes() for p in SET[:6]] + [p.read_bytes() for p in HARD[:4]] + \
             [grid(SET[30:34], (2800, 2240)), grid(HARD[10:14], (2800, 2240))]
    rows = []
    for rep in range(2):
        for i, img in enumerate(images):
            rows.append(dict(ask(img), index=i, rep=rep))
    json.dump(dict(label=label, n=len(images), rows=rows), open(out, "w"), indent=1)
    n = len(images)
    stable = sum(1 for i in range(n) if rows[i]["top"] and rows[i]["top"] == rows[i + n]["top"])
    print(f"LEAN_PROBE {label}: ok {sum(r['ok'] for r in rows)}/{len(rows)}, with logprobs "
          f"{sum(1 for r in rows if r['top'])}/{len(rows)}, same-server repeat identical {stable}/{n}")


def cmp(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    n = a["n"]
    exact = top1 = both = 0
    deltas = []
    for i in range(n):
        x, y = a["rows"][i]["top"], b["rows"][i]["top"]
        if not (x and y):
            continue
        both += 1
        exact += x == y
        top1 += x[0][0] == y[0][0]
        dy = {t: lp for t, lp in y}
        deltas += [abs(lp - dy[t]) for t, lp in x if t in dy]
    print(f"LEAN_PROBE cmp {a['label']} vs {b['label']}: compared {both}/{n}, exact {exact}, top-1 same {top1}, "
          f"max |dlogprob| {max(deltas):.4g}" if deltas else
          f"LEAN_PROBE cmp {a['label']} vs {b['label']}: no logprobs to compare ({both}/{n})")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3])
