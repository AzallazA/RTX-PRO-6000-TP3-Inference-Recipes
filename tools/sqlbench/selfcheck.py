#!/usr/bin/env python3
"""Run the 25 reference SQL statements through the local DuckDB load and grade them with the
site's criterion against the expected values extracted from the bundle (questions.json).
If data/answers.json (full expected rows, fetched optionally) is present, also report whether
every row matches (informational; the site only grades the first row)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb  # noqa: E402
import fetch_site  # noqa: E402
import replay  # noqa: E402

ds = replay.load_questions()
ans = None
if os.path.exists(fetch_site.ANSWERS_FILE):
    with open(fetch_site.ANSWERS_FILE) as f:
        ans = {q["id"]: q for q in json.load(f)["questions"]}
r = replay.Runner()
schema = r.get_schema()
print(f"duckdb {duckdb.__version__}; tables loaded: {', '.join(f'{t}({len(c)} cols)' for t, c in schema.items())}")
print("schema prompt as sent to the model:\n" + replay.format_schema(schema))
ok = 0
for q in ds["questions"]:
    t0 = time.perf_counter()
    try:
        res = r.run_query(q["sql"])
    except Exception as e:
        print(f"Q{q['id']:>2} EXEC ERROR: {e}")
        continue
    chk = replay.check_result(q, res)
    p = replay.passed(chk)
    full = "n/a"
    if ans and q["id"] in ans:  # full-row comparison (stringified, numeric tolerance 1e-6)
        exp_rows = ans[q["id"]]["rows"]
        match = len(exp_rows) == res["numRows"]
        for er, ar in zip(exp_rows, res["rows"]):
            for c, ev in zip(ans[q["id"]]["columns"], er):
                av = ar.get(c)
                en, an_ = replay.js_number(ev), replay.js_number(av)
                if not ((en == en and an_ == an_ and abs(en - an_) <= 1e-6) or str(ev) == str(av)):
                    match = False
        full = "yes" if match else "NO"
    ok += p
    print(f"Q{q['id']:>2} {q['difficulty']:<7} rows={res['numRows']:>3} (exp {q['row_count']:>3}) cols={len(res['columns'])} "
          f"grade={'PASS' if p else 'FAIL'} all_rows_match={full} {time.perf_counter() - t0:.2f}s"
          + ("" if p else f"  {chk}"))
n = len(ds["questions"])
print(f"\n{ok}/{n} reference queries pass the site's criterion against the expected values from the bundle")
sys.exit(0 if ok == n else 1)
