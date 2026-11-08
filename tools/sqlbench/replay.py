#!/usr/bin/env python3
"""
Local, scriptable replay of https://sql-benchmark.nicklothian.com/ ("Run Your Own Benchmark").

Faithful port of the site's in-browser runner (Astro island BenchmarkRunner.<hash>.js). Comments
of the form ":NNN" are line numbers in a prettier-formatted copy of that bundle (hash CELwKJm9,
2026-11) and are kept only as pointers for anyone verifying the port against the site's code.

Questions and data are NOT shipped with this harness: fetch_site.py downloads them from the site
on first use (questions.json + data/). Dependencies: python3 >= 3.10 stdlib + duckdb.
"""
import argparse
import concurrent.futures
import datetime as _dt
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_site  # noqa: E402  (same directory)

TABLES_DIR = fetch_site.TABLES_DIR
QUESTIONS_FILE = fetch_site.QUESTIONS_FILE
TABLES = fetch_site.TABLES

MAX_TOOL_CALLS = 20          # BenchmarkRunner.pretty.js:323  const X = 20
MAX_RETRIES = 2              # :760 maxRetries default (consecutive SQL errors)
MAX_NO_TOOLCALL_RETRIES = 2  # :1515 maxNoToolCallRetries -> 3 attempts
DEFAULT_TIMEOUT_S = 120      # :2572 (parseInt(f,10)||120)*1e3

# ---------------------------------------------------------------- tools (:223-:245)
TOOL_RUN_SQL = {
    "name": "run_sql_query",
    "description": "Execute a SQL query against the DuckDB database. Call this when you have a SQL query ready to run.",
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The DuckDB SQL query to execute. ONLY ever pass SQL to this",
            }
        },
        "required": ["sql"],
    },
}
TOOL_RESULTS_OK = {
    "name": "results_ok",
    "description": "Confirm that the query results correctly answer the user question. Call this when the results look correct.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def openai_tools(defs):  # :1456 Rt
    return [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                              "parameters": d["parameters"]}} for d in defs]


# ---------------------------------------------------------------- prompts (:246-:318)
def format_schema(schema):  # :246 we(e)
    items = list(schema.items())
    if not items:
        return "No tables exist yet."
    out = ["Available tables:\n"]
    for table, cols in items:
        colstr = ", ".join(f"{c['col']} ({c['type']})" for c in cols)
        out.append(f'Table "{table}":')
        out.append(f"  Columns: {colstr}\n")
    return "\n".join(out)


def system_prompt_tool(schema):  # :261 at(e)
    return f"""You are a SQL query generator for DuckDB.
Generate SQL queries that answer the user's questions using DuckDB SQL syntax.
Make sure you quote all field, column and table names.
Think through the problem step by step before writing the query.

Here is the database schema:

{format_schema(schema)}

You have these tools available: run_sql_query(sql) - Execute a SQL query against the DuckDB database.

Workflow:
- Call run_sql_query with your SQL query. Important: only ever pass SQL to run_sql_query.
- If the query returns an error, call run_sql_query with a corrected query.
- When results look correct, call results_ok."""


def system_prompt_grammar(schema):  # :278 lt(e)  (CLI grammar mode; NOT used by the browser)
    return f"""You are a SQL query generator for DuckDB.
Generate SQL queries that answer the user's questions using DuckDB SQL syntax.
Make sure you quote all field, column and table names.

Here is the database schema:

{format_schema(schema)}

Instructions:
- Output a SQL SELECT query that answers the question.
- Do not output anything other than the SQL query"""


def result_message(res, mode="tool"):  # :291 Ae(e, t)
    first = res["rows"][0] if res["rows"] else None
    s = ", ".join(f"{c}={first[c] if first[c] is not None else 'NULL'}" for c in res["columns"]) if first else "(empty)"
    if mode == "grammar":
        body = "If they do, output OK\nIf they do not, output a corrected SQL query.\nDo not output anything other than the SQL query or OK."
    else:
        body = "If they do, call results_ok.\nIf they do not, call run_sql_query with a corrected query."
    return (f"Query executed successfully. Verify these results match the request.\n\n{body}\n\n"
            f"Returned {res['numRows']} row(s).\nColumns: {', '.join(res['columns'])}\nFirst row: {s}")


# ---------------------------------------------------------------- JS-compat helpers
def js_str(v):
    """Mimic String(v) of a duckdb-wasm cell value (:1430-:1443 runQuery stringifies every cell)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if v == int(v) and abs(v) < 1e21:
            return str(int(v))
        return repr(v)
    return str(v)


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|Infinity)$")


def js_number(v):
    """Mimic Number(v) in JS (Number(null)=0, Number('')=0, non-numeric -> NaN)."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return 0.0
    if _NUM_RE.match(s):
        try:
            return float(s.replace("Infinity", "inf"))
        except ValueError:
            return math.nan
    return math.nan


# ---------------------------------------------------------------- grading (:33-:97 Je / He)
def check_result(q, res):
    if res is None:
        return {"rowCountMatch": False, "columnCountMatch": False, "columnNamesMatch": False,
                "firstRowMatch": False, "actualRowCount": None, "actualColumnCount": None,
                "missingColumns": list(q["columns"]), "extraColumns": [], "firstRowDiffs": []}

    def norm(c):
        return re.sub(r"[^a-z0-9]", "", c.lower())

    actual_by_norm = {norm(c): c for c in res["columns"]}
    expected_by_norm = {norm(c): c for c in q["columns"]}
    missing = [c for c in q["columns"] if norm(c) not in actual_by_norm]
    extra = [c for c in res["columns"] if norm(c) not in expected_by_norm]
    row_count_match = res["numRows"] == q["row_count"]
    col_count_match = len(res["columns"]) == len(q["columns"])
    names_match = not missing and not extra
    diffs = []
    if res["rows"]:
        row0 = res["rows"][0]
        for col in q["columns"]:
            if col not in q["first_row"]:
                continue
            expected = q["first_row"][col]
            acol = actual_by_norm.get(norm(col))
            if acol is None:
                continue
            actual = row0[acol]
            e_num, a_num = js_number(expected), js_number(actual)
            num_match = False
            if not math.isnan(e_num) and not math.isnan(a_num):
                es = js_str(expected) if not isinstance(expected, str) else expected
                dot = es.find(".")
                decimals = 0 if dot == -1 else len(es) - dot - 1
                tol = 5 * 10 ** (-(decimals + 1))
                eps = sys.float_info.epsilon * max(abs(a_num), abs(e_num))
                num_match = abs(a_num - e_num) <= tol + eps
            str_match = (js_str(actual) if not isinstance(actual, str) else actual) == \
                        (js_str(expected) if not isinstance(expected, str) else expected)
            if not num_match and not str_match:
                diffs.append({"column": col, "expected": expected, "actual": actual})
    return {"rowCountMatch": row_count_match, "columnCountMatch": col_count_match,
            "columnNamesMatch": names_match,
            "firstRowMatch": len(diffs) == 0 and len(res["rows"]) > 0,
            "actualRowCount": res["numRows"], "actualColumnCount": len(res["columns"]),
            "missingColumns": missing, "extraColumns": extra, "firstRowDiffs": diffs}


def passed(check):  # :91 He
    return bool(check) and check["rowCountMatch"] and check["columnCountMatch"] \
        and check["columnNamesMatch"] and check["firstRowMatch"]


# ---------------------------------------------------------------- extra flags (not on the site)
JUNK_MARKERS = ["<｜", "｜>", "<ds_safety>", "</ds_safety>", "##END##", "<|im_end|>", "<|im_start|>",
                "<|end", "<|begin", "<|tool", "<|User|>", "<|Assistant|>"]
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u0080-\u009f�￰-￿]")


def is_junk(text):
    if not text:
        return False
    if _CTRL_RE.search(text):
        return True
    return any(m in text for m in JUNK_MARKERS)


# ---------------------------------------------------------------- DuckDB runner (:1394-:1455 Se)
class Runner:
    def __init__(self, tables_dir=TABLES_DIR):
        if tables_dir == TABLES_DIR and any(not os.path.exists(os.path.join(tables_dir, f"{t}.csv")) for t in TABLES):
            ensure_assets()
        self.db = duckdb.connect(database=":memory:")
        for t in TABLES:  # :793 yt / SqlRunner Gl: CREATE VIEW t AS SELECT * FROM read_csv_auto('t.csv')
            path = os.path.join(tables_dir, f"{t}.csv")
            if not os.path.exists(path):
                sys.exit(f"missing {path} (run: python3 fetch_site.py)")
            self.db.execute(f"CREATE VIEW {t} AS SELECT * FROM read_csv_auto('{path}')")
        self._local = threading.local()

    def _conn(self):
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self.db.cursor()
            self._local.conn = c
        return c

    def get_schema(self):  # :1405
        rows = self._conn().execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position").fetchall()
        schema = {}
        for t, c, ty in rows:
            schema.setdefault(t, []).append({"col": c, "type": ty})
        return schema

    def run_query(self, sql):  # :1418 runQuery -> all cells stringified
        t0 = time.perf_counter()
        cur = self._conn()
        rel = cur.execute(sql)
        cols = [d[0] for d in rel.description] if rel.description else []
        raw = rel.fetchall()
        elapsed = f"{time.perf_counter() - t0:.2f}"
        rows = [{c: js_str(v) for c, v in zip(cols, r)} for r in raw]
        return {"columns": cols, "rows": rows, "numRows": len(rows), "elapsed": elapsed}


# ---------------------------------------------------------------- OpenAI-compatible client (:1466-:1564)
def to_openai_messages(system_prompt, messages):  # :1466 qe
    out = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            if "toolCall" in m:
                tc = m["toolCall"]
                out.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], separators=(",", ":"))}}]})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "tool_result":
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": m["content"]})
    return out


class Client:
    def __init__(self, endpoint, model, api_key, temperature, max_tokens, extra_body, seed, mode="tool"):
        self.endpoint, self.model, self.api_key = endpoint, model, api_key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body, self.seed, self.mode = extra_body or {}, seed, mode

    def build_body(self, system_prompt, messages, with_tools=True):  # :1517-:1523
        body = {"model": self.model, "messages": to_openai_messages(system_prompt, messages)}
        if with_tools:
            body["tools"] = openai_tools([TOOL_RUN_SQL, TOOL_RESULTS_OK])
        body["max_tokens"] = self.max_tokens
        body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.extra_body)
        return body

    def post(self, body, timeout):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=max(0.001, timeout)) as resp:
                txt = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"API error ({e.code}): {txt}")
        latency = time.perf_counter() - t0
        if status != 200:
            raise RuntimeError(f"API error ({status}): {txt}")
        return json.loads(txt), latency


# ---------------------------------------------------------------- per-question agent loop (:338-:557 dt)
class Aborted(Exception):
    pass


def run_question_tool(q, schema, client, runner, timeout_s, log):
    deadline = time.monotonic() + timeout_s
    messages = [{"role": "user", "content": q["question"]}]
    sysprompt = system_prompt_tool(schema)
    generated_sql, query_result = None, None
    attempts = consec_err = calls = in_tok = out_tok = 0
    model_name = None
    calls_log = []

    def base(err):
        return {"generatedSql": generated_sql, "queryResult": query_result, "error": err,
                "attempts": attempts, "inputTokens": in_tok, "outputTokens": out_tok,
                "modelName": model_name, "calls": calls_log}

    try:
        while calls < MAX_TOOL_CALLS:
            if time.monotonic() >= deadline:
                return base(f"Timeout after {timeout_s}s")
            # ---- vt(): up to 3 HTTP attempts until a tool call appears (:1525-:1563)
            tool_call = None
            for attempt in range(MAX_NO_TOOLCALL_RETRIES + 1):
                if calls >= MAX_TOOL_CALLS:  # :327 onClientCallAttempt -> D()
                    raise RuntimeError(f"Exceeded maximum tool calls ({MAX_TOOL_CALLS})")
                calls += 1
                log(f"[Q{q['id']}] Calling LLM (attempt {calls})")
                body = client.build_body(sysprompt, messages)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return base(f"Timeout after {timeout_s}s")
                try:
                    resp, latency = client.post(body, remaining)
                except (TimeoutError, OSError) as e:
                    if time.monotonic() >= deadline or "timed out" in str(e).lower():
                        return base(f"Timeout after {timeout_s}s")
                    return base(str(e))
                usage = resp.get("usage") or {}
                in_tok += usage.get("prompt_tokens") or 0
                out_tok += usage.get("completion_tokens") or 0
                model_name = resp.get("model_alias") or resp.get("model") or model_name
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                tcs = msg.get("tool_calls") or []
                rec = {"latency_s": round(latency, 3), "finish_reason": choice.get("finish_reason"),
                       "usage": usage, "content": msg.get("content"),
                       "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                       "tool_calls": tcs, "n_messages_sent": len(body["messages"])}
                calls_log.append(rec)
                if tcs:
                    tc = tcs[0]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}
                    tool_call = {"toolCallId": tc.get("id") or f"call_{int(time.time()*1000)}",
                                 "functionName": tc["function"]["name"], "arguments": args}
                    break
                if attempt >= MAX_NO_TOOLCALL_RETRIES:
                    raise RuntimeError(f"No tool call in response after {MAX_NO_TOOLCALL_RETRIES + 1} attempts")
            # ---- dispatch (:446-:514)
            if tool_call["functionName"] == "results_ok":
                return base(None)
            if tool_call["functionName"] == "run_sql_query":
                sql = tool_call["arguments"].get("sql", "")
                if not isinstance(sql, str):
                    sql = json.dumps(sql)
                generated_sql = sql
                attempts += 1
                messages.append({"role": "assistant", "toolCall": {"id": tool_call["toolCallId"],
                                                                    "name": "run_sql_query", "arguments": tool_call["arguments"]}})
                log(f"[Q{q['id']}] Executing SQL (attempt {attempts})")
                try:
                    res = runner.run_query(sql)
                    query_result, consec_err = res, 0
                    calls_log[-1]["sql_result"] = {"numRows": res["numRows"], "columns": res["columns"]}
                    messages.append({"role": "tool_result", "toolCallId": tool_call["toolCallId"],
                                     "content": result_message(res)})
                except Exception as e:
                    err = str(e)
                    calls_log[-1]["sql_error"] = err
                    if consec_err >= MAX_RETRIES:
                        query_result = None
                        return base(f"Query failed after {MAX_RETRIES} retries: {err}")
                    consec_err += 1
                    messages.append({"role": "tool_result", "toolCallId": tool_call["toolCallId"],
                                     "content": f"Error executing query. Fix this error and call run_sql_query again. Error: {err}"})
            else:
                messages.append({"role": "assistant", "toolCall": {"id": tool_call["toolCallId"],
                                                                    "name": tool_call["functionName"], "arguments": tool_call["arguments"]}})
                messages.append({"role": "tool_result", "toolCallId": tool_call["toolCallId"],
                                 "content": f'Unknown tool "{tool_call["functionName"]}". Available tools: run_sql_query, results_ok'})
        return base(f"Exceeded maximum tool calls ({MAX_TOOL_CALLS})")
    except Exception as e:
        if time.monotonic() >= deadline:
            return base(f"Timeout after {timeout_s}s")
        return base(str(e))


def strip_think(text):  # :320 it(e)
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()


def extract_sql_text(text):
    """Only for --mode prompt (no GBNF grammar available): strip <think>, code fences."""
    t = strip_think(text)
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", t, re.I)
    if m:
        t = m.group(1).strip()
    return t


def run_question_prompt(q, schema, client, runner, timeout_s, log):
    """Port of pt() (:558-:722), the CLI --grammar mode, minus the grammar constraint."""
    deadline = time.monotonic() + timeout_s
    sysprompt = system_prompt_grammar(schema)
    messages = [{"role": "user", "content": q["question"]}]
    generated_sql, query_result = None, None
    attempts = consec_err = calls = in_tok = out_tok = 0
    model_name = None
    calls_log = []

    def base(err):
        return {"generatedSql": generated_sql, "queryResult": query_result, "error": err,
                "attempts": attempts, "inputTokens": in_tok, "outputTokens": out_tok,
                "modelName": model_name, "calls": calls_log}

    try:
        while calls < MAX_TOOL_CALLS:
            if time.monotonic() >= deadline:
                return base(f"Timeout after {timeout_s}s")
            calls += 1
            log(f"[Q{q['id']}] Calling LLM prompt mode (attempt {attempts + 1})")
            body = client.build_body(sysprompt, messages, with_tools=False)
            try:
                resp, latency = client.post(body, deadline - time.monotonic())
            except (TimeoutError, OSError) as e:
                return base(f"Timeout after {timeout_s}s" if time.monotonic() >= deadline else str(e))
            usage = resp.get("usage") or {}
            in_tok += usage.get("prompt_tokens") or 0
            out_tok += usage.get("completion_tokens") or 0
            model_name = resp.get("model_alias") or resp.get("model") or model_name
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = extract_sql_text(msg.get("content") or "")
            calls_log.append({"latency_s": round(latency, 3), "finish_reason": choice.get("finish_reason"),
                              "usage": usage, "content": msg.get("content"),
                              "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                              "tool_calls": msg.get("tool_calls") or [], "n_messages_sent": len(body["messages"])})
            messages.append({"role": "assistant", "content": text})
            if text == "OK":
                return base(None)
            generated_sql = text
            attempts += 1
            try:
                res = runner.run_query(text)
                query_result, consec_err = res, 0
                calls_log[-1]["sql_result"] = {"numRows": res["numRows"], "columns": res["columns"]}
                messages.append({"role": "user", "content": result_message(res, "grammar")})
            except Exception as e:
                err = str(e)
                calls_log[-1]["sql_error"] = err
                if consec_err >= MAX_RETRIES:
                    query_result = None
                    return base(f"Query failed after {MAX_RETRIES} retries: {err}")
                consec_err += 1
                messages.append({"role": "user", "content": f"Error executing query: {err}\n\nOutput corrected SQL only."})
        return base(f"Exceeded maximum tool calls ({MAX_TOOL_CALLS})")
    except Exception as e:
        return base(f"Timeout after {timeout_s}s" if time.monotonic() >= deadline else str(e))


# ---------------------------------------------------------------- orchestration (:743-:792 gt)
def ensure_assets():
    """Download questions.json and data/ from the site if they are missing (see fetch_site.py)."""
    try:
        fetch_site.ensure()
    except fetch_site.FetchError as e:
        sys.exit(f"fetch_site: {e}")


def load_questions(path=QUESTIONS_FILE):
    if path == QUESTIONS_FILE and not os.path.exists(path):
        ensure_assets()
    with open(path) as f:
        return json.load(f)


def reference_from_sql(runner, q):
    res = runner.run_query(q["sql"])
    return {**q, "row_count": res["numRows"], "columns": res["columns"],
            "first_row": dict(res["rows"][0]) if res["rows"] else {}}


def _tc_text(tc):
    """Human-readable text of a tool call (arguments JSON-decoded so \\uXXXX escapes do not hide markers)."""
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return f"{fn.get('name', '')} {json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args}"


def grade(q, out):
    check = None if out["error"] else check_result(q, out["queryResult"])
    status = "error" if out["error"] else ("pass" if passed(check) else "fail")
    calls = out["calls"]
    return {
        "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
        "status": status, "error": out["error"], "check": check,
        "generatedSql": out["generatedSql"],
        "queryResult": out["queryResult"],
        "attempts": out["attempts"], "llm_calls": len(calls),
        "inputTokens": out["inputTokens"], "outputTokens": out["outputTokens"],
        "latency_s": round(sum(c["latency_s"] for c in calls), 3),
        "finish_reasons": [c["finish_reason"] for c in calls],
        "truncated": any(c["finish_reason"] == "length" for c in calls),
        "junk": is_junk(out["generatedSql"]),
        "junk_in_reply": any(is_junk(c.get("content") or "") or is_junk(c.get("reasoning") or "") or
                             any(is_junk(_tc_text(tc)) for tc in c.get("tool_calls") or []) for c in calls),
        "sql_errors": sum(1 for c in calls if "sql_error" in c),
        "modelName": out["modelName"],
        "calls": calls,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible base URL (…/v1) or full …/chat/completions URL (default: vLLM's http://localhost:8000/v1)")
    ap.add_argument("--model", required=True, help="model id sent in the request body (as listed by GET /v1/models)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--extra-body", default=None,
                    help='JSON merged into the request body, e.g. \'{"chat_template_kwargs":{"thinking":false}}\'')
    ap.add_argument("--seed", type=int, default=None, help="adds `seed` to the body (site does not send one)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-question wall clock (site default 120 s)")
    ap.add_argument("--questions", default=None, help="comma-separated question ids (default all 25)")
    ap.add_argument("--difficulty", default=None, help="comma-separated: trivial,easy,medium,hard")
    ap.add_argument("--mode", choices=["tool", "prompt"], default="tool",
                    help="tool = browser behaviour (default); prompt = CLI grammar-mode prompts without grammar")
    ap.add_argument("--ref-from-sql", action="store_true",
                    help="grade against reference SQL run locally instead of the shipped first_row/row_count")
    ap.add_argument("--out", default=None, help="output JSON path (default runs/<ts>_<model>.json next to this script)")
    ap.add_argument("--fetch", action="store_true", help="(re)download questions/data from the site first, then run")
    ap.add_argument("--dry-run", action="store_true", help="print the exact first request body and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    extra = json.loads(args.extra_body) if args.extra_body else {}

    if args.fetch:
        try:
            fetch_site.ensure(force=True)
        except fetch_site.FetchError as e:
            sys.exit(f"fetch_site: {e}")
    dataset = load_questions()
    questions = dataset["questions"]
    if args.questions:
        ids = [int(x) for x in args.questions.split(",")]
        byid = {q["id"]: q for q in questions}
        questions = [byid[i] for i in ids if i in byid]
    if args.difficulty:
        ds = set(args.difficulty.split(","))
        questions = [q for q in questions if q["difficulty"] in ds]

    runner = Runner()
    schema = runner.get_schema()
    if args.ref_from_sql:
        questions = [reference_from_sql(runner, q) for q in questions]

    client = Client(endpoint, args.model, args.api_key, args.temperature, args.max_tokens, extra, args.seed, args.mode)

    if args.dry_run:
        q = questions[0]
        sp = system_prompt_tool(schema) if args.mode == "tool" else system_prompt_grammar(schema)
        body = client.build_body(sp, [{"role": "user", "content": q["question"]}], with_tools=(args.mode == "tool"))
        print(f"POST {endpoint}")
        print("Content-Type: application/json" + ("\nAuthorization: Bearer ***" if args.api_key else ""))
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return

    lock = threading.Lock()

    def log(msg):
        if not args.quiet:
            with lock:
                print(msg, file=sys.stderr, flush=True)

    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    fn = run_question_tool if args.mode == "tool" else run_question_prompt
    jobs = [(r, q) for r in range(args.rounds) for q in questions]
    records = [None] * len(jobs)

    def work(idx):
        r, q = jobs[idx]
        t0 = time.time()
        out = fn(q, schema, client, runner, args.timeout, log)
        rec = grade(q, out)
        rec["round"] = r
        rec["durationMs"] = int((time.time() - t0) * 1000)
        log(f"[Q{q['id']}] round {r} -> {rec['status'].upper()}"
            + (f" ({rec['error'][:80]})" if rec['error'] else "")
            + (" TRUNCATED" if rec["truncated"] else "") + (" JUNK" if rec["junk"] else ""))
        return idx, rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        for idx, rec in ex.map(work, range(len(jobs))):
            records[idx] = rec
    completed = _dt.datetime.now(_dt.timezone.utc).isoformat()

    summary = {
        "total": len(records),
        "passed": sum(r["status"] == "pass" for r in records),
        "failed": sum(r["status"] == "fail" for r in records),
        "errored": sum(r["status"] == "error" for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "junk": sum(r["junk"] for r in records),
        "junk_in_reply": sum(r["junk_in_reply"] for r in records),
        "sql_errors": sum(r["sql_errors"] for r in records),
        "wrong_results": sum(r["status"] == "fail" for r in records),
        "no_tool_call_errors": sum(1 for r in records if r["error"] and r["error"].startswith("No tool call")),
        "timeouts": sum(1 for r in records if r["error"] and r["error"].startswith("Timeout")),
        "llm_calls": sum(r["llm_calls"] for r in records),
        "totalInputTokens": sum(r["inputTokens"] for r in records),
        "totalOutputTokens": sum(r["outputTokens"] for r in records),
        "total_latency_s": round(sum(r["latency_s"] for r in records), 1),
    }
    report = {
        "meta": {"site": "https://sql-benchmark.nicklothian.com/", "datasetId": dataset["id"],
                 "datasetName": dataset["name"], "startedAt": started, "completedAt": completed,
                 "endpoint": endpoint, "model": args.model, "modelName_reported": next(
                     (r["modelName"] for r in records if r["modelName"]), None),
                 "mode": args.mode, "temperature": args.temperature, "max_tokens": args.max_tokens,
                 "extra_body": extra, "seed": args.seed, "concurrency": args.concurrency,
                 "rounds": args.rounds, "timeout_s": args.timeout, "ref_from_sql": args.ref_from_sql,
                 "duckdb_version": duckdb.__version__},
        "schema_prompt": format_schema(schema),
        "summary": summary,
        "results": records,
    }
    out = args.out
    if not out:
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
        out = os.path.join(HERE, "runs", f"{ts}_{safe}{'_' + args.mode if args.mode != 'tool' else ''}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    # ---- stdout table
    print(f"\n{dataset['name']} -- {args.model} @ {endpoint}  mode={args.mode} T={args.temperature} max_tokens={args.max_tokens}"
          f"{' extra=' + json.dumps(extra) if extra else ''}")
    print(f"{'Q':>3} {'rnd':>3} {'diff':<8} {'status':<6} {'calls':>5} {'lat_s':>7} {'out_tok':>7} {'trunc':>5} {'junk':>4} {'sqlerr':>6}  note")
    for r in records:
        note = r["error"] or ""
        if r["status"] == "fail" and r["check"]:
            c = r["check"]
            bits = []
            if not c["rowCountMatch"]:
                bits.append(f"rows {c['actualRowCount']}!={next(q['row_count'] for q in questions if q['id']==r['id'])}")
            if not c["columnNamesMatch"]:
                bits.append(f"cols missing={c['missingColumns']} extra={c['extraColumns']}")
            if not c["firstRowMatch"] and c["firstRowDiffs"]:
                bits.append("row0 " + "; ".join(f"{d['column']}: {d['actual']}!={d['expected']}" for d in c["firstRowDiffs"][:3]))
            note = " | ".join(bits)[:140]
        print(f"{r['id']:>3} {r['round']:>3} {r['difficulty']:<8} {r['status']:<6} {r['llm_calls']:>5} {r['latency_s']:>7.1f} "
              f"{r['outputTokens']:>7} {'Y' if r['truncated'] else '-':>5} {'Y' if r['junk'] else '-':>4} {r['sql_errors']:>6}  {note[:140]}")
    s = summary
    print(f"\nPASS {s['passed']}/{s['total']}  fail(wrong result) {s['failed']}  error {s['errored']}"
          f"  | truncated {s['truncated']}  junk-in-sql {s['junk']} (in any reply {s['junk_in_reply']})"
          f"  sql-errors {s['sql_errors']}  no-tool-call {s['no_tool_call_errors']}  timeouts {s['timeouts']}")
    print(f"LLM calls {s['llm_calls']}  tokens in/out {s['totalInputTokens']}/{s['totalOutputTokens']}  "
          f"sum latency {s['total_latency_s']} s")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
