#!/usr/bin/env python3
"""Mock OpenAI-compatible server for testing replay.py without a model (stdlib only).
Answers every question with its reference SQL via run_sql_query, then results_ok.
Fault injection by question id:
  13 -> truncated reply (finish_reason=length, no tool call) on every attempt -> error after 3 attempts
  14 -> reference SQL with a DeepSeek-style junk marker appended (DuckDB rejects it; the mock then
        calls results_ok) -> fail, junk flag set
  15 -> one SQL error (unknown table) then the correct query -> pass with sql_errors=1
  16 -> wrong SQL -> fail with a row-0 diff
Expected summary: 22/25 pass, 2 fail, 1 error.
Usage: python3 tests/mock_server.py <port>"""
import http.server
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import replay  # noqa: E402

QS = {q["question"]: q for q in replay.load_questions()["questions"]}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0.1 and body["max_tokens"] == 2048 and len(body["tools"]) == 2
        msgs = body["messages"]
        q = QS[msgs[1]["content"]]
        n_tool = sum(1 for m in msgs if m["role"] == "tool")
        fin, tc, content = "tool_calls", None, None
        if q["id"] == 13:
            fin, content = "length", "Let me think about this very carefully and at great length"
        elif n_tool == 0 or (q["id"] == 15 and n_tool == 1):
            sql = q["sql"]
            if q["id"] == 14:
                sql = sql + " <｜end▁of▁sentence｜>"
            if q["id"] == 15 and n_tool == 0:
                sql = "SELECT * FROM NoSuchTable"
            if q["id"] == 16:
                sql = "SELECT 'FY1999' AS \"Fiscal Year\""
            tc = {"id": f"call_{q['id']}_{n_tool}", "type": "function",
                  "function": {"name": "run_sql_query", "arguments": json.dumps({"sql": sql})}}
        else:
            tc = {"id": f"call_ok_{q['id']}", "type": "function", "function": {"name": "results_ok", "arguments": "{}"}}
        resp = {"id": "x", "object": "chat.completion", "model": "mock-model",
                "choices": [{"index": 0, "finish_reason": fin,
                             "message": {"role": "assistant", "content": content, "tool_calls": [tc] if tc else None}}],
                "usage": {"prompt_tokens": 1000 + 10 * len(msgs), "completion_tokens": 50}}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"mock server on http://127.0.0.1:{port}/v1", file=sys.stderr, flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
