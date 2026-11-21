#!/usr/bin/env python3
"""Live check of the steering access gate (stdlib only).

  steering_access_check.py --key-file FILE --out FILE.json [--host 127.0.0.1] [--port 15015]
      for a server started with steering keys (--steering-api-keys-file FILE, or FLARE_STEERING_API_KEYS_FILE=FILE)
  steering_access_check.py --sanity --set PROMPTS.json --out FILE.json [--key-file FILE] [--host ...] [--port ...]
      does steering move anything: the set's "trigger" prompts in order (at most 8), each with its own nonce line,
      scale 0 then 1, temperature 0, max_tokens 24, until one pair differs in text (format: see refusal_set.py)
  steering_access_check.py --self-test

Last line 'ACCESS PASS|FAIL: ...' or 'STEER SANITY PASS|FAIL: ...'. The key is read from FILE (first non-comment entry)
and never printed or written: the JSON carries only statuses, token counts and cached-token counts (--sanity also
stores the generated texts in --out). A wrong key is generated so that it differs from the right one.

Requests (chat completions unless noted; temperature 0, seed 7, enable_thinking false, max_tokens 16). P = one ~3.4K-token
prompt with a per-run nonce (the later steps reuse its KV through the prefix cache):
  warm      P, steering_scale 0, no key                 -> 200 (fills the prefix cache)
  nokey     P, s=1, no key                              -> 403 permission_error (OpenAI shape), message names X-Steering-Key
  wrongx    P, s=1, X-Steering-Key: wrong               -> 403
  wrongb    P, s=1, Authorization: Bearer wrong         -> 403
  kwargs    P, chat_template_kwargs.steering_scale 0.5, no key -> 403
  compl     /v1/completions, s=1, no key                -> 403 (OpenAI shape)
  msgs      /v1/messages, s=1, no key                   -> 403 (Anthropic shape: type error, error.type permission_error)
  stream    P, s=1, stream, no key                      -> 403 (JSON error, no SSE)
  rightx    P, s=1, X-Steering-Key: right               -> 200, >= 1 token, cached_tokens 0 (a nonzero scale changes the
                                                           prefix identity: proof the scale reached the request)
  rightb    P, s=1, Authorization: Bearer right         -> 200, cached_tokens > 0 (reuses rightx's steered prefix)
  streamk   P, s=1, stream, X-Steering-Key: right       -> 200 SSE with >= 1 content/reasoning delta and [DONE]
  zero      P, s=0, no key                              -> 200, cached_tokens > 0 (reuses warm's unsteered prefix)
  omitted   P, no steering_scale, no key                -> 200, cached_tokens > 0
INFO: rightx text vs warm text (moved or not). Every 403 body is also checked to not contain the right or the wrong key.
If the server also has a serving API key, this check does not apply as written (Authorization then carries that key).
"""
import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request

MODEL = "DeepSeek-V4.1-Flash"


def read_key(path):
    for line in open(path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(",")[0].strip()
    raise SystemExit("ACCESS FAIL: key file holds no key")


def prompt(nonce):
    return (f"Reference id {nonce}.\n"
            + "".join(f"Clause {i}: the tenant shall keep unit {i % 41} in good repair for {i * 13 % 97} days.\n" for i in range(170))
            + "Summarize the clauses in one sentence.")


HOST = "127.0.0.1"


def request(port, path, body, headers=None, timeout=900):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    if path.startswith("/v1/messages"):
        h["anthropic-version"] = "2023-06-01"
    h.update(headers or {})
    req = urllib.request.Request(f"http://{HOST}:{port}{path}", data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("content-type", ""), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, "", f"{type(e).__name__}: {e}"


def chat_body(p, scale=None, stream=False, kwargs_scale=None):
    b = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 16, "temperature": 0, "seed": 7,
         "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
    if scale is not None:
        b["steering_scale"] = scale
    if kwargs_scale is not None:
        b["chat_template_kwargs"]["steering_scale"] = kwargs_scale
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def parse_chat(status, text):
    out = {"status": status, "tokens": 0, "cached": None, "text": ""}
    try:
        j = json.loads(text)
    except ValueError:
        return out, None
    if status == 200:
        m = (j.get("choices") or [{}])[0].get("message") or {}
        out["text"] = (m.get("content") or "") + (m.get("reasoning_content") or m.get("reasoning") or "")
        u = j.get("usage") or {}
        out["tokens"] = u.get("completion_tokens") or 0
        out["cached"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return out, j


def parse_sse(status, text):
    deltas, done = 0, False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            done = True
            continue
        try:
            j = json.loads(d)
        except ValueError:
            continue
        for c in j.get("choices") or []:
            dl = c.get("delta") or {}
            if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                deltas += 1
    return {"status": status, "deltas": deltas, "done": done}


def forbidden(status, text, anthropic, secrets_):
    """'' when the answer is the gate's 403 in the route's shape, else the reason."""
    if status != 403:
        return f"HTTP {status} (want 403)"
    if any(s and s in text for s in secrets_):
        return "403 body contains a key"
    try:
        j = json.loads(text)
    except ValueError:
        return "403 body is not JSON"
    e = j.get("error") or {}
    if e.get("type") != "permission_error":
        return f"error.type {e.get('type')!r}"
    if anthropic and j.get("type") != "error":
        return "Anthropic shape lacks type 'error'"
    if not anthropic and e.get("code") != "permission_error":
        return f"error.code {e.get('code')!r}"
    if "X-Steering-Key" not in (e.get("message") or ""):
        return "message does not name X-Steering-Key"
    return ""


def evaluate(rows, keys):
    """rows: name -> result dict. Returns (ok, line)."""
    bad, notes = [], []
    w = rows.get("warm", {})
    if w.get("status") != 200 or not w.get("tokens"):
        bad.append(f"warm HTTP {w.get('status')} tokens {w.get('tokens')}")
    for name in ("nokey", "wrongx", "wrongb", "kwargs", "compl", "msgs", "stream"):
        r = rows.get(name, {})
        if r.get("why"):
            bad.append(f"{name}: {r['why']}")
    rx, rb, sk, z, o = (rows.get(n, {}) for n in ("rightx", "rightb", "streamk", "zero", "omitted"))
    if rx.get("status") != 200 or not rx.get("tokens"):
        bad.append(f"rightx HTTP {rx.get('status')} tokens {rx.get('tokens')}")
    elif rx.get("cached") != 0:
        bad.append(f"rightx cached_tokens {rx.get('cached')} (want 0: steered prefix must not reuse the unsteered one)")
    if rb.get("status") != 200 or not rb.get("tokens"):
        bad.append(f"rightb HTTP {rb.get('status')} tokens {rb.get('tokens')}")
    elif not (rb.get("cached") or 0) > 0:
        bad.append(f"rightb cached_tokens {rb.get('cached')} (want > 0: same steered prefix as rightx)")
    if sk.get("status") != 200 or not sk.get("deltas") or not sk.get("done"):
        bad.append(f"streamk HTTP {sk.get('status')} deltas {sk.get('deltas')} done {sk.get('done')}")
    for name, r in (("zero", z), ("omitted", o)):
        if r.get("status") != 200 or not r.get("tokens"):
            bad.append(f"{name} HTTP {r.get('status')} tokens {r.get('tokens')}")
        elif not (r.get("cached") or 0) > 0:
            bad.append(f"{name} cached_tokens {r.get('cached')} (want > 0: unsteered prefix reuse)")
    for name, r in rows.items():
        if any(k and k in json.dumps(r) for k in keys):
            bad.append(f"{name}: result row contains a key")
    moved = rows.get("rightx", {}).get("text") != rows.get("warm", {}).get("text")
    notes.append(f"INFO steered text {'moved' if moved else 'same'} vs s=0")
    summary = (f"403 x{sum(1 for n in ('nokey', 'wrongx', 'wrongb', 'kwargs', 'compl', 'msgs', 'stream') if not rows.get(n, {}).get('why'))}/7 "
               f"(no key, wrong X-Steering-Key, wrong Bearer, chat_template_kwargs, completions, messages (Anthropic), stream); "
               f"right key X-Steering-Key {rx.get('status')} cached {rx.get('cached')}, Bearer {rb.get('status')} cached {rb.get('cached')}, "
               f"stream {sk.get('status')} deltas {sk.get('deltas')}; s=0 no key {z.get('status')} cached {z.get('cached')}, "
               f"omitted {o.get('status')} cached {o.get('cached')}; {'; '.join(notes)}")
    if bad:
        return False, f"ACCESS FAIL: {'; '.join(bad)} | {summary}"
    return True, f"ACCESS PASS: {summary}"


def run(a):
    key = read_key(a.key_file)
    wrong = "wrong-" + secrets.token_hex(12)
    while wrong == key:
        wrong = "wrong-" + secrets.token_hex(12)
    p = prompt(time.time_ns())
    rows = {}
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, 0.0))
    rows["warm"], _ = parse_chat(st, tx)
    for name, path, body, headers, anth in (
        ("nokey", "/v1/chat/completions", chat_body(p, 1.0), {}, False),
        ("wrongx", "/v1/chat/completions", chat_body(p, 1.0), {"X-Steering-Key": wrong}, False),
        ("wrongb", "/v1/chat/completions", chat_body(p, 1.0), {"Authorization": f"Bearer {wrong}"}, False),
        ("kwargs", "/v1/chat/completions", chat_body(p, kwargs_scale=0.5), {}, False),
        ("compl", "/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 16, "temperature": 0, "steering_scale": 1.0}, {}, False),
        ("msgs", "/v1/messages", {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": p}], "steering_scale": 1.0}, {}, True),
        ("stream", "/v1/chat/completions", chat_body(p, 1.0, stream=True), {}, False),
    ):
        st, ct, tx = request(a.port, path, body, headers)
        rows[name] = {"status": st, "content_type": ct, "why": forbidden(st, tx, anth, (key, wrong))}
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, 1.0), {"X-Steering-Key": key})
    rows["rightx"], _ = parse_chat(st, tx)
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, 1.0), {"Authorization": f"Bearer {key}"})
    rows["rightb"], _ = parse_chat(st, tx)
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, 1.0, stream=True), {"X-Steering-Key": key})
    rows["streamk"] = parse_sse(st, tx)
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, 0.0))
    rows["zero"], _ = parse_chat(st, tx)
    st, ct, tx = request(a.port, "/v1/chat/completions", chat_body(p, None))
    rows["omitted"], _ = parse_chat(st, tx)
    ok, line = evaluate(rows, (key, wrong))
    for r in rows.values():   # texts are not needed in the record
        r.pop("text", None)
    json.dump({"rows": rows, "verdict": line}, open(a.out, "w"), indent=1)
    print(line)
    return 0 if ok else 1


def self_test():
    k = "sk-steer-selftest"
    good = {"warm": {"status": 200, "tokens": 16, "cached": 0, "text": "a"},
            **{n: {"status": 403, "why": ""} for n in ("nokey", "wrongx", "wrongb", "kwargs", "compl", "msgs", "stream")},
            "rightx": {"status": 200, "tokens": 16, "cached": 0, "text": "b"},
            "rightb": {"status": 200, "tokens": 16, "cached": 3392, "text": "b"},
            "streamk": {"status": 200, "deltas": 9, "done": True},
            "zero": {"status": 200, "tokens": 16, "cached": 3392, "text": "a"},
            "omitted": {"status": 200, "tokens": 16, "cached": 3392, "text": "a"}}
    checks = []
    ok, line = evaluate(good, (k, "wrong-x")); checks.append(("good", ok and "moved" in line))
    b = json.loads(json.dumps(good)); b["nokey"]["why"] = "HTTP 200 (want 403)"
    checks.append(("nokey honored", not evaluate(b, (k,))[0]))
    b = json.loads(json.dumps(good)); b["rightx"]["cached"] = 3392
    checks.append(("silently ignored", not evaluate(b, (k,))[0]))
    b = json.loads(json.dumps(good)); b["rightb"]["status"] = 403
    checks.append(("bearer refused", not evaluate(b, (k,))[0]))
    b = json.loads(json.dumps(good)); b["zero"]["status"] = 403
    checks.append(("zero refused", not evaluate(b, (k,))[0]))
    b = json.loads(json.dumps(good)); b["warm"]["note"] = k
    checks.append(("key in row", not evaluate(b, (k,))[0]))
    checks.append(("403 shape openai", forbidden(403, json.dumps({"error": {"type": "permission_error", "code": "permission_error", "message": "send X-Steering-Key"}}), False, (k,)) == ""))
    checks.append(("403 shape anthropic", forbidden(403, json.dumps({"type": "error", "error": {"type": "permission_error", "message": "X-Steering-Key"}}), True, (k,)) == ""))
    checks.append(("403 echoes key", forbidden(403, json.dumps({"error": {"type": "permission_error", "code": "permission_error", "message": f"X-Steering-Key {k}"}}), False, (k,)) != ""))
    checks.append(("400 is not 403", forbidden(400, "{}", False, (k,)) != ""))
    s = parse_sse(200, 'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n')
    checks.append(("sse", s["deltas"] == 1 and s["done"]))
    z, o = {"status": 200, "tokens": 5, "text": "I can't"}, {"status": 200, "tokens": 5, "text": "Sure"}
    checks.append(("sanity moved", sanity_verdict([(8, z, z), (0, z, o)])[0]))
    checks.append(("sanity flat", not sanity_verdict([(8, z, z), (0, o, o)])[0]))
    checks.append(("sanity errors", not sanity_verdict([(8, {"status": 500}, o)])[0]))
    bad = [n for n, c in checks if not c]
    print(f"steering_access_check self-test {'OK' if not bad else 'FAILED: ' + ', '.join(bad)} ({len(checks)} checks)")
    return 1 if bad else 0


def sanity_verdict(pairs):
    """pairs: list of (index, zero_row, one_row). PASS when a pair answered 200 twice and the texts differ."""
    errs = [i for i, z, o in pairs if z.get("status") != 200 or o.get("status") != 200 or not z.get("tokens") or not o.get("tokens")]
    moved = [i for i, z, o in pairs if i not in errs and z.get("text") != o.get("text")]
    if moved:
        return True, f"STEER SANITY PASS: trigger #{moved[0]} moved (s=0 vs s=1 text differs; {len(pairs)} tried, errors {len(errs)})"
    return False, f"STEER SANITY FAIL: no trigger moved in {len(pairs)} tried (errors {len(errs)})"


def run_sanity(a):
    trig = json.load(open(a.set, encoding="utf-8"))["trigger"]
    order = list(range(min(8, len(trig))))
    hdr = {"X-Steering-Key": read_key(a.key_file)} if a.key_file else {}
    pairs = []
    for i in order:
        rows = []
        for scale in (0.0, 1.0):
            p = f"(sanity {i} {scale} {time.time_ns() % 10**9})\n" + trig[i]
            body = chat_body(p, scale)
            body["max_tokens"] = 24
            st, ct, tx = request(a.port, "/v1/chat/completions", body, hdr if scale else None)
            rows.append(parse_chat(st, tx)[0])
        pairs.append((i, rows[0], rows[1]))
        if pairs[-1][1].get("text") != pairs[-1][2].get("text") and rows[0].get("status") == 200 and rows[1].get("status") == 200:
            break
    ok, line = sanity_verdict(pairs)
    json.dump({"pairs": [{"index": i, "zero": z, "one": o} for i, z, o in pairs], "verdict": line}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(line)
    return 0 if ok else 1


def main():
    global HOST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=15015)
    ap.add_argument("--key-file")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--set")
    a = ap.parse_args()
    HOST = a.host
    if a.self_test:
        return self_test()
    if a.sanity:
        if not (a.port and a.set and a.out):
            ap.error("--sanity needs --set and --out")
        return run_sanity(a)
    if not (a.port and a.key_file and a.out):
        ap.error("--key-file and --out are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
