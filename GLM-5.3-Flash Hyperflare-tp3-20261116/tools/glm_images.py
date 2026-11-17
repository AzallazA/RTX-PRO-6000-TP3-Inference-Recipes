#!/usr/bin/env python3
"""GLM-5.3-Flash image checks against a live Flare server (system python3 + Pillow).

  glm_images.py latency LABEL OUT.json   cold/warm image latency (run FIRST after startup: the first request is cold)
  glm_images.py smoke   LABEL OUT.json   real images incl. one near the current max size (8000 image tokens), one
                                         larger than the max (frontend downscale), 2 and 4 images in one request
  glm_images.py eval    LABEL OUT.json   DS41 vision sets (set + hard, 60 + 60 chart/diagram questions, exact last line)

Sampling within the GLM recipe ranges (Flare clamps others silently): temperature 1.0, top_p 0.95; thinking is always
on, reasoning_effort low. Each mode prints one grep-able line (IMAGE_LATENCY / IMAGE_SMOKE / IMAGE_EVAL).
Images: the DeepSeek V4.1 vision qualification sets (VISION_SETS, default: the Kraken-tp3-20261019 folder of this repo).
"""

import base64, concurrent.futures, io, json, os, pathlib, re, statistics, sys, time, urllib.request

from PIL import Image

mode, label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
PORT = int(os.environ.get("PORT", "15015"))
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash")
V = pathlib.Path(os.environ.get("VISION_SETS", "../DeepSeek-V4.1-Flash Kraken-tp3-20261019/qualification/vision"))   # set/ and hard/ with answers.json
SET, HARD = sorted((V / "set").glob("*.png")), sorted((V / "hard").glob("*.png"))
FACTOR, MAX_TOKENS_IMAGE = 28, 8000   # helios/src/frontend/images/glm.rs: 28-pixel canvas, <= 8000 tokens per image


def png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def load(path):
    return Image.open(path).convert("RGB")


def grid(paths, size):
    """Distinct composite of four images at an exact canvas size (multiples of 28: no resize by the frontend)."""
    w, h = size[0] // 2, size[1] // 2
    canvas = Image.new("RGB", size, "white")
    for i, p in enumerate(paths[:4]):
        canvas.paste(load(p).resize((w, h)), ((i % 2) * w, (i // 2) * h))
    return png(canvas)


def tokens(size):
    return (size[0] // FACTOR) * (size[1] // FACTOR)


def ask(images, text, max_tokens=8192, timeout=1800):
    content = [dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(b).decode()))
               for b in images] + [dict(type="text", text=text)]
    body = dict(model=MODEL, max_tokens=max_tokens, temperature=1.0, top_p=0.95, seed=7, reasoning_effort="low",
                messages=[dict(role="user", content=content)])
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        msg = r["choices"][0]["message"]
        return dict(ok=True, secs=round(time.time() - t, 3), content=msg.get("content") or "",
                    finish=r["choices"][0].get("finish_reason"), prompt_tokens=r["usage"]["prompt_tokens"],
                    completion_tokens=r["usage"]["completion_tokens"])
    except Exception as e:  # noqa: BLE001
        detail = getattr(e, "read", lambda: b"")()
        return dict(ok=False, secs=round(time.time() - t, 3), content=f"ERROR {e!r} {detail[:300]!r}"[:500],
                    finish="error", prompt_tokens=0, completion_tokens=0)


def norm(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[*`\"'.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace(" - ", "-").strip()


def correct(out, answer):
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return norm(last) == norm(answer) or norm(out) == norm(answer) or norm(last).endswith(" " + norm(answer))


def answers(directory):
    return {item["file"]: item for item in json.load(open(directory / "answers.json"))}


if mode == "latency":
    q = "Describe this image in one word."
    res = {"label": label}
    first = ask([SET[0].read_bytes()], q, max_tokens=1)
    res["cold_small"] = first["secs"]
    res["warm_small"] = ask([SET[1].read_bytes()], q, max_tokens=1)["secs"]
    seq = [ask([p.read_bytes()], q, max_tokens=1) for p in SET[5:13]]
    res["seq8"] = [s["secs"] for s in seq]
    near = (2800, 2240)   # 100 x 80 = 8000 tokens = 32000 patches (largest capacity 32768)
    mid = (1792, 1792)    # 4096 tokens
    big = [ask([grid(SET[13 + 4 * i:17 + 4 * i], near)], q, max_tokens=1) for i in range(2)]
    res["large_first"], res["large_second"] = big[0]["secs"], big[1]["secs"]
    res["large_prompt_tokens"] = big[0]["prompt_tokens"]
    res["mid"] = ask([grid(SET[21:25], mid)], q, max_tokens=1)["secs"]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        par = list(ex.map(lambda p: ask([p.read_bytes()], q, max_tokens=1), HARD[:4]))
    res["par4_wall"] = round(time.time() - t0, 3)
    res["par4"] = [s["secs"] for s in par]
    res["text"] = statistics.median(ask([], q, max_tokens=1)["secs"] for _ in range(3))
    res["ok"] = all(x["ok"] for x in (first, *seq, *big, *par))
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"IMAGE_LATENCY {label} cold {res['cold_small']:.2f} s warm {res['warm_small']:.2f} s "
          f"seq8_med {statistics.median(res['seq8']):.2f} s large8000 first {res['large_first']:.2f} s "
          f"second {res['large_second']:.2f} s ({res['large_prompt_tokens']} prompt tok) mid4096 {res['mid']:.2f} s "
          f"par4_wall {res['par4_wall']:.2f} s text {res['text']:.2f} s ok {res['ok']}")

elif mode == "smoke":
    A = answers(V / "set")
    files = sorted(A)
    cases = []
    item = A[files[0]]
    cases.append(("small", [(V / "set" / item["file"]).read_bytes()], item["question"], item["answer"], 50))
    item = A[files[1]]
    src = load(V / "set" / item["file"])
    scale = min(2800 / src.width, 2240 / src.height)
    near = src.resize((int(src.width * scale) // FACTOR * FACTOR, int(src.height * scale) // FACTOR * FACTOR))
    cases.append((f"near-max {near.size[0]}x{near.size[1]} ({tokens(near.size)} tok)", [png(near)], item["question"],
                  item["answer"], int(0.9 * tokens(near.size))))
    item = A[files[2]]
    src = load(V / "set" / item["file"])
    over = src.resize((src.width * 8, src.height * 8))
    cases.append((f"over-max {over.size[0]}x{over.size[1]} (downscaled)", [png(over)], item["question"], item["answer"],
                  int(0.8 * MAX_TOKENS_IMAGE) if tokens(over.size) > MAX_TOKENS_IMAGE else 50))
    two = [(V / "set" / files[i]).read_bytes() for i in (3, 4)]
    cases.append(("two images", two, "How many images are attached? Answer with just the number.", "2", 100))
    four = [(V / "set" / files[i]).read_bytes() for i in (5, 6, 7, 8)]
    cases.append(("four images", four, "How many images are attached? Answer with just the number.", "4", 200))
    rows, passed = [], 0
    for name, images, question, answer, min_prompt in cases:
        r = ask(images, question)
        good = r["ok"] and r["finish"] == "stop" and r["prompt_tokens"] >= min_prompt and bool(r["content"].strip())
        right = good and correct(r["content"], answer)
        passed += good
        rows.append(dict(case=name, served=good, correct=right, answer=answer, **r))
    json.dump(rows, open(out_path, "w"), indent=1)
    print(f"IMAGE_SMOKE {label} served {passed}/{len(cases)} correct {sum(r['correct'] for r in rows)}/{len(cases)} | "
          + " | ".join(f"{r['case']}: {'ok' if r['served'] else 'FAIL'}{'' if r['correct'] else ' (answer differs)'} "
                       f"{r['prompt_tokens']}p {r['secs']:.1f}s" for r in rows))

elif mode == "eval":
    res = []
    for name in ("set", "hard"):
        items = json.load(open(V / name / "answers.json"))
        def one(item, name=name):
            r = ask([(V / name / item["file"]).read_bytes()], item["question"], max_tokens=16000)
            return dict(set=name, id=item["id"], kind=item["kind"], answer=item["answer"], ok=r["ok"] and correct(r["content"], item["answer"]),
                        served=r["ok"], output=r["content"][-200:], finish=r["finish"], tokens=r["completion_tokens"], secs=r["secs"])
        with concurrent.futures.ThreadPoolExecutor(int(os.environ.get("EVAL_CONC", "8"))) as ex:
            res += list(ex.map(one, items))
    json.dump(res, open(out_path, "w"), indent=1)
    parts = []
    for name in ("set", "hard"):
        rr = [r for r in res if r["set"] == name]
        parts.append(f"{name} {sum(r['ok'] for r in rr)}/{len(rr)} (errors {sum(not r['served'] for r in rr)}, "
                     f"truncated {sum(r['finish'] == 'length' for r in rr)})")
    print(f"IMAGE_EVAL {label} T=1.0 effort low: " + " | ".join(parts)
          + " | DS41 Flare ref (T=0+1, set): results-vision-staged-real-set.json")
else:
    sys.exit(f"unknown mode {mode}")
