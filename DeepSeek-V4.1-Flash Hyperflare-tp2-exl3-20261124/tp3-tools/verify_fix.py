"""flare#193 fix check: count TRUE decode corruptions (decode junk where teacher-forced prefill is confident).

The one-token reference reply can be empty: Flare returns no logprob record for a sampled terminating end-of-sequence
token (helios/src/server/runtime.rs, `terminating_eos`), so its top_logprobs list is empty when the single sampled token
is EOS. The top-5 distribution does not depend on the sample, so such a request is retried (RETRIES extra attempts); a
candidate still empty after that is skipped (counted neither as TRUE nor as natural) and counted. The counts are
appended at the end of the summary line, which otherwise keeps its format.
"""
import json, os, sys, urllib.request

URL = os.environ.get("VERIFY_FIX_URL", "http://localhost:15015")
RETRIES = 4
PROMPTS = [
    ("# Test suite for is_palindrome: 60 checks, one per line.\n"
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n\n"
     "print(is_palindrome(\"racecar\"))  # True\nprint(is_palindrome(\"hello\"))  # False\n"
     "print(is_palindrome(\"A man, a plan, a canal: Panama\"))  # True\n"),
    ("// debounce tests: 40 cases\nfunction debounce(fn, ms) {\n  let t;\n  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };\n}\n"
     "const a = debounce(log, 100);\nconst b = debounce(log, 200);\n"),
]


def post(body):
    req = urllib.request.Request(URL + "/v1/completions", json.dumps({"model": "DeepSeek-V4.1-Flash", **body}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
true_bad, natural, tokens, retried, skipped = [], 0, 0, 0, 0
for run in range(runs):
    P = PROMPTS[run % len(PROMPTS)]
    lp = post({"prompt": P, "max_tokens": 300, "top_p": 0.95, "logprobs": 5})["choices"][0]["logprobs"]
    toks, tlp, tops = lp["tokens"], lp["token_logprobs"], lp["top_logprobs"]
    tokens += len(toks)
    for i in range(2, len(toks) - 1):
        if max(tops[i].values()) < -1.6 and tlp[i - 1] > -0.02 and tlp[i - 2] > -0.02:
            top = None
            for attempt in range(1 + RETRIES):
                ref = post({"prompt": P + "".join(toks[:i]), "max_tokens": 1, "top_p": 0.95, "logprobs": 5})
                top = ((ref["choices"][0].get("logprobs") or {}).get("top_logprobs") or [None])[0]
                if top:
                    break
                if attempt < RETRIES:
                    retried += 1
            if not top:
                skipped += 1
                continue
            rtop = max(top.items(), key=lambda kv: kv[1])
            if rtop[1] > -0.3:
                true_bad.append((run, i, toks[i - 1], toks[i], round(max(tops[i].values()), 2), rtop))
            else:
                natural += 1
print(f"{runs} runs, {tokens} tokens: TRUE corruptions {len(true_bad)}, natural-uncertainty flags {natural}, "
      f"empty one-token replies retried {retried}, candidates skipped {skipped}")
for b in true_bad[:10]:
    print("  ", b)
