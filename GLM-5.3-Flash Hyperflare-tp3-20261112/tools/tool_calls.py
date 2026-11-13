#!/usr/bin/env python3
"""Tool-call qualification (flare#193 lesson): N tool-requiring prompts at concurrency C.

Per request: no HTTP/server error; the model calls a tool (finish_reason tool_calls); every call names a
declared tool, its arguments parse as a JSON object with non-empty unique keys, all required parameters
present with the right JSON types and enum values. A second turn returns the tool result and must end
with a non-empty assistant answer.
GLM: Flare parses the model's <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> output
into OpenAI tool_calls (arguments typed by the declared schema), so the checks below are the DS41 ones; added:
any GLM markup left in content or a tool name/argument (a parser miss) fails the request, and the reasoning
length is recorded. The second turn sends the assistant message back as returned (reasoning_content included,
which the GLM template re-renders as <think>...</think> since clear_thinking defaults to false).
usage: tool_calls.py [N=60] [C=4]   (MODEL, FLARE_URL, TOOLS_EFFORT, QUAL_TEMPERATURE, QUAL_TOP_P from the environment)
"""
import json
import os, random, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_common as g

URL = g.URL + "/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
         "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Find files matching a glob under a directory.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}},
         "required": ["pattern", "path"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
         "required": ["command", "timeout_seconds"]}}},
]
SPEC = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
CITIES = ["Lisbon", "Osaka", "Nairobi", "Calgary", "Hanoi", "Reykjavik", "Lima", "Tbilisi"]
TASKS = [
    lambda r: (f"What's the weather in {r.choice(CITIES)} right now, in {r.choice(['celsius', 'fahrenheit'])}? Use the tool.", "get_weather"),
    lambda r: (f"Find all {r.choice(['*.py', '*.rs', '*.md', '*.json'])} files under {r.choice(['/srv/app', '~/projects', '/etc'])}, at most {r.randint(5, 50)} results.", "search_files"),
    lambda r: (f"Run `{r.choice(['df -h', 'uname -a', 'ls -la /tmp', 'git status'])}` with a {r.randint(5, 120)} second timeout and tell me what it says.", "run_command"),
]
FAKE = {"get_weather": {"temperature": 18, "conditions": "light rain"},
        "search_files": {"matches": ["a/one.py", "b/two.py"], "truncated": False},
        "run_command": {"exit_code": 0, "stdout": "ok\n"}}


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def check_call(call):
    fn = call.get("function", {})
    name = fn.get("name")
    if name not in SPEC:
        return f"unknown tool {name!r}"
    try:
        args = json.loads(fn.get("arguments") or "")
    except json.JSONDecodeError as e:
        return f"arguments not JSON: {e}"
    if not isinstance(args, dict):
        return "arguments not an object"
    if any(not k for k in args):
        return "empty argument name"
    spec = SPEC[name]
    for key in spec["required"]:
        if key not in args:
            return f"missing required {key}"
    for key, value in args.items():
        prop = spec["properties"].get(key)
        if prop is None:
            return f"undeclared argument {key}"
        if not isinstance(value, TYPES[prop["type"]]) or (prop["type"] == "integer" and isinstance(value, bool)):
            return f"{key} has type {type(value).__name__}"
        if "enum" in prop and value not in prop["enum"]:
            return f"{key}={value!r} not in enum"
    return None


def one(i):
    rng = random.Random(1000 + i)
    prompt, expected = rng.choice(TASKS)(rng)
    msgs = [{"role": "user", "content": prompt}]
    base = dict(model=g.MODEL, tools=TOOLS, max_tokens=int(os.environ.get("TOOLS_MAX_TOKENS", "8192")),
                reasoning_effort=g.TOOLS_EFFORT, temperature=g.TEMPERATURE, top_p=g.TOP_P, seed=i)
    d, err = post({**base, "messages": msgs})
    if err:
        return i, "FAIL", err
    choice = d["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        return i, "FAIL", f"no tool call (finish={choice.get('finish_reason')}): {(choice['message'].get('content') or '')[:120]!r}"
    names = [c.get("function", {}).get("name") for c in calls]
    leak = g.leaked_markup(choice["message"].get("content")) + [t for c in calls for t in g.leaked_markup(
        (c.get("function", {}).get("name") or "") + (c.get("function", {}).get("arguments") or ""))]
    if leak:
        return i, "FAIL", f"GLM markup in parsed output: {leak}"
    if len(set(c.get("id") for c in calls)) != len(calls):
        return i, "FAIL", "duplicate tool call ids"
    for c in calls:
        problem = check_call(c)
        if problem:
            return i, "FAIL", problem
    msgs.append(choice["message"])
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(FAKE[c["function"]["name"]])})
    d2, err = post({**base, "messages": msgs})
    if err:
        return i, "FAIL", "turn 2: " + err
    answer = (d2["choices"][0]["message"].get("content") or "").strip()
    if not answer and not d2["choices"][0]["message"].get("tool_calls"):
        return i, "FAIL", f"turn 2: empty answer (finish={d2['choices'][0].get('finish_reason')})"
    if g.leaked_markup(answer):
        return i, "FAIL", f"turn 2: GLM markup in content: {g.leaked_markup(answer)}"
    thought = len(choice["message"].get("reasoning_content") or "")
    return i, "OK", f"{','.join(names)}{'' if expected in names else ' (unexpected tool)'} reasoning {thought} chars"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t0 = time.time()
    with ThreadPoolExecutor(c) as pool:
        results = sorted(pool.map(one, range(n)))
    ok = sum(1 for _, s, _ in results if s == "OK")
    odd = [r for r in results if r[1] == "OK" and "unexpected" in r[2]]
    for r in results:
        if r[1] != "OK":
            print("  FAIL", r[0], r[2])
    print(f"tool calls: {ok}/{n} OK ({len(odd)} chose a different tool), concurrency {c}, {time.time() - t0:.0f} s")
    json.dump([dict(index=i, status=s, detail=d) for i, s, d in results],
              open(os.path.join(os.environ.get("QUAL_OUT", os.path.dirname(os.path.abspath(__file__))), "tool_calls.json"), "w"), indent=2)
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
