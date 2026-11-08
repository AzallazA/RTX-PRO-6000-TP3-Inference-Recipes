#!/usr/bin/env bash
# End-to-end self-check without a model (no GPU needed):
#   1. fetch questions + data from the benchmark site (if missing)
#   2. run the 25 reference SQL statements through DuckDB and grade them (expect 25/25)
#   3. run the full tool-calling loop against the stdlib mock server with fault injection
#      (expect 22 pass / 2 fail / 1 error, 1 truncated, 1 junk, 2 sql errors, 1 no-tool-call)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
"$PY" -c 'import duckdb' 2>/dev/null || { echo "duckdb not importable by $PY. Run: python3 -m venv .venv && .venv/bin/pip install duckdb" >&2; exit 2; }

echo "== 1/3 fetch (questions.json + data/)"; "$PY" fetch_site.py
echo "== 2/3 reference-SQL self-check"; "$PY" selfcheck.py | tail -n 1

echo "== 3/3 mock loop"
PORT="${MOCK_PORT:-$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')}"
"$PY" tests/mock_server.py "$PORT" & MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
for _ in $(seq 50); do "$PY" -c "import socket;socket.create_connection(('127.0.0.1',$PORT),1).close()" 2>/dev/null && break; sleep 0.1; done
OUT="$(mktemp -t sqlbench_mock_XXXXXX.json)"
"$PY" replay.py --base-url "http://127.0.0.1:$PORT/v1" --model mock-model --concurrency 4 --quiet --out "$OUT" | tail -n 3
"$PY" - "$OUT" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
exp = {"total": 25, "passed": 22, "failed": 2, "errored": 1, "truncated": 1, "junk": 1, "sql_errors": 2,
       "no_tool_call_errors": 1, "timeouts": 0}
bad = {k: (s.get(k), v) for k, v in exp.items() if s.get(k) != v}
print("mock loop:", "OK" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
PYEOF
rm -f "$OUT"
echo "self-check passed"
