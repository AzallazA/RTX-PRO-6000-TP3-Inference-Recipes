# sqlbench — local replay of the LLM SQL Benchmark

A scriptable, GPU-free reproduction of the **"Run Your Own Benchmark"** button on
Nick Lothian's LLM SQL Benchmark site, <https://sql-benchmark.nicklothian.com/>
(CLI source: <https://github.com/nlothian/llm-sql-benchmark>). Point it at any
OpenAI-compatible `/v1/chat/completions` endpoint (vLLM, Flare, llama.cpp, OpenRouter, ...)
and it runs the same 25 AdventureWorks questions, with the same prompts, tool definitions,
agent loop and grader as the browser, then writes a JSON report with extra diagnostics.

**Credit.** The benchmark — questions, reference SQL, expected results, the AdventureWorks
CSV extract and the in-browser runner this harness ports — is the work of Nick Lothian and
belongs to the site. **Nothing from the site is redistributed here**: `fetch_site.py` downloads
the current bundle and data at run time into gitignored local files (`questions.json`, `data/`).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install duckdb      # Python 3.10+, duckdb is the only dependency
.venv/bin/python fetch_site.py                              # optional: replay.py does this automatically
.venv/bin/python replay.py --base-url http://localhost:8000/v1 --model <model-id>
```

`fetch_site.py` discovers the hashed `BenchmarkRunner.<hash>.js` bundle from the site's
`index.html`, extracts the 25 questions (`question`, reference `sql`, expected `row_count`,
`columns`, `first_row`), writes `questions.json`, and downloads the 7 CSV tables
(`data/tables/*.csv.gz`, decompressed next to them, ~17 MB) plus the optional
`data/answers.json` (full expected rows, used only by `selfcheck.py`). If the bundle format
changes and extraction fails, it exits with the bundle URL and the list of extraction
assumptions; `--bundle-url` lets you point it at a different bundle. `replay.py --fetch`
forces a refresh.

## What the benchmark does (as ported)

* **Data.** Seven CSVs derived from Microsoft AdventureWorks DW: `Customer`, `Date`, `Product`,
  `Reseller`, `Sales`, `Sales_Order`, `Sales_Territory`. Each is exposed to DuckDB as
  `CREATE VIEW <t> AS SELECT * FROM read_csv_auto('<t>.csv')` — no explicit types, no database file.
* **Schema prompt.** `information_schema.columns` for schema `main`, rendered as
  `Table "X":\n  Columns: col (TYPE), ...` and inlined into the system prompt (no sample rows, no DDL).
* **Mode.** Tool calling only (the browser never uses the CLI's grammar mode). Two tools are
  offered in OpenAI "function" format: `run_sql_query(sql: string)` and `results_ok()`.
  System prompt and user message (= the question text) are reproduced verbatim; `--dry-run`
  prints the exact first request body.
* **Request body.** `{model, messages, tools, max_tokens: 2048, temperature: 0.1}` — nothing
  else (no `stream`, `tool_choice`, `response_format`, `seed`, ...). The assistant turn is
  replayed with only its **first** tool call, arguments re-serialised from the parsed JSON, and any
  assistant text/reasoning dropped.
* **Agent loop per question.** The SQL is taken from `tool_calls[0].function.arguments.sql`
  with no text parsing. If a reply contains no tool call, the same request is re-sent, up to
  **3 attempts**, then the question is an *error* (`No tool call in response after 3 attempts`).
  `run_sql_query` executes in DuckDB; on success the model sees
  `Query executed successfully... Returned N row(s). Columns: ... First row: ...` (first row only),
  on failure `Error executing query. Fix this error and call run_sql_query again. Error: ...`;
  three *consecutive* SQL failures end the question as an error. `results_ok` stops the loop and the
  last successful result is graded. Limits: 20 LLM calls per question and a **120 s wall clock per
  question** (`Timeout after 120s`). The browser runs questions one at a time.
* **Grading** — pass requires all four:
  1. row count equals the expected `row_count`;
  2. same number of columns;
  3. same column names (compared after lower-casing and stripping `[^a-z0-9]`, so
     `total_revenue` == `Total Revenue`), no extras;
  4. **first row only**, per expected column: numeric match if the actual value rounds to the
     expected value's printed precision (`|a-e| <= 5·10^-(decimals+1)` plus machine epsilon),
     otherwise exact string equality. Result order matters; no sorting; rows beyond the first are
     not checked.

  Status is `error` (loop/LLM/SQL exhaustion/timeout), `fail` (ran, wrong) or `pass`.

## What the harness adds

Defaults reproduce the browser exactly; everything below is opt-in or purely diagnostic.

* Per answer: `finish_reasons` of every LLM call, `truncated` (any call with
  `finish_reason == "length"`), `junk` (generated SQL contains control characters, U+FFFD or
  tokenizer markers such as `<｜...｜>`, `<|im_end|>`, `##END##`), `junk_in_reply` (same test over
  content, reasoning and tool-call arguments), `sql_errors` (failed executions), `llm_calls`,
  latency, token usage, and the full call log (content, reasoning, tool calls, SQL result shape or
  error).
* Summary: pass/fail/error plus `truncated`, `junk`, `sql_errors`, `wrong_results`,
  `no_tool_call_errors`, `timeouts`, tokens and summed latency.
* `--rounds N` repeats the 25 questions N times (variance), `--concurrency N` runs N questions in
  parallel (use 1 for a browser-faithful latency picture), `--questions 1,9`, `--difficulty hard`.
* `--extra-body '{"chat_template_kwargs": {...}}'` merges arbitrary keys into the request body
  (e.g. to switch a model's thinking mode); `--seed N` adds `seed`; `--temperature`,
  `--max-tokens`, `--timeout` override the site defaults (0.1 / 2048 / 120 s).
* `--ref-from-sql` grades against the reference SQL executed locally instead of the shipped
  expected values; `--mode prompt` uses the CLI grammar-mode prompts without a grammar (not what
  the website does; for experiments only).

## Running against a server

vLLM (default base URL is `http://localhost:8000/v1`; `--model` is the id served under `/v1/models`):

```bash
.venv/bin/python replay.py --base-url http://localhost:8000/v1 --model <served-model-name>
# e.g. disable thinking through the chat template, 3 rounds, 4 questions in flight:
.venv/bin/python replay.py --base-url http://localhost:8000/v1 --model <served-model-name> \
    --extra-body '{"chat_template_kwargs":{"thinking":false}}' --rounds 3 --concurrency 4
```

Flare / any other OpenAI-compatible server — same command with its base URL (or a full
`.../chat/completions` URL) and an API key if required:

```bash
OPENAI_API_KEY=... .venv/bin/python replay.py --base-url http://<host>:<port>/v1 --model <model-id>
```

`--api-key` overrides `OPENAI_API_KEY`. Progress goes to stderr; a per-question table and the
summary line go to stdout; the JSON report is written to `runs/<timestamp>_<model>.json`
(or `--out`).

## Output JSON

```
meta          site, datasetId/Name, startedAt/completedAt, endpoint, model, modelName_reported,
              mode, temperature, max_tokens, extra_body, seed, concurrency, rounds, timeout_s,
              ref_from_sql, duckdb_version
schema_prompt the exact schema block sent in the system prompt
summary       total, passed, failed, errored, truncated, junk, junk_in_reply, sql_errors,
              wrong_results, no_tool_call_errors, timeouts, llm_calls, totalInput/OutputTokens,
              total_latency_s
results[]     id, difficulty, question, round, status (pass|fail|error), error, check
              {rowCountMatch, columnCountMatch, columnNamesMatch, firstRowMatch, actualRowCount,
              actualColumnCount, missingColumns, extraColumns, firstRowDiffs[]}, generatedSql,
              queryResult {columns, rows (all rows, cells stringified as the browser does), numRows,
              elapsed}, attempts, llm_calls, inputTokens, outputTokens, latency_s, finish_reasons,
              truncated, junk, junk_in_reply, sql_errors, modelName, durationMs,
              calls[] {latency_s, finish_reason, usage, content, reasoning, tool_calls,
              n_messages_sent, sql_result | sql_error}
```

## Self-check (no model, no GPU)

```bash
make check          # or: ./selfcheck.sh
```

1. `fetch_site.py` — downloads/refreshes `questions.json` and `data/`.
2. `selfcheck.py` — runs the 25 reference SQL statements through the local DuckDB load and grades
   them with the site's criterion against the extracted expected values: must print **25/25**.
   If `data/answers.json` is present it also reports whether *every* row matches (informational).
3. `tests/mock_server.py` — a stdlib OpenAI-compatible mock that answers with the reference SQL
   and injects faults (Q13 truncated replies → error after 3 attempts; Q14 junk marker in the SQL →
   fail + `junk`; Q15 one SQL error then the fix → pass with `sql_errors=1`; Q16 wrong SQL → fail
   with a row-0 diff). `replay.py` is run against it and the summary must be
   22 pass / 2 fail / 1 error, 1 truncated, 1 junk, 2 sql_errors, 1 no-tool-call, 0 timeouts.

Set `PYTHON=/path/to/python` to use an interpreter other than `.venv/bin/python`.

## Known differences from the browser

* **DuckDB engine.** The site runs duckdb-wasm (1.32.x at the time of writing) in the browser;
  this harness runs the Python `duckdb` package (2.x). Same SQL dialect, same CSV bytes, same
  `read_csv_auto` sniffing on these files (the self-check shows identical column types and 25/25).
  Floating-point summation order can differ in later rows (e.g. a cent in a running total), which
  is outside the first-row grading criterion but visible in `selfcheck.py`'s all-rows column.
* **Endpoint handling.** The browser POSTs to the endpoint string exactly as typed; `replay.py`
  appends `/chat/completions` to a base URL that does not already end with it.
* **Timeouts.** The browser aborts the in-flight HTTP request at the 120 s mark; the harness passes
  the remaining budget as the socket timeout and reports `Timeout after 120s` the same way.
* **Additions** (`--rounds`, `--concurrency`, `--extra-body`, `--seed`, `--ref-from-sql`,
  `--mode prompt`, the truncated/junk/sql_errors flags) are not on the site; with defaults the
  request bodies, loop and grading are identical.
* **Published leaderboard** numbers on the site were produced by the author's CLI runner (same
  prompts and grader) over OpenRouter, not necessarily by the browser button.

## Files

| path | what |
|---|---|
| `replay.py` | the harness (python3 + duckdb). Auto-fetches questions/data when missing |
| `fetch_site.py` | bundle discovery, question extraction, data download (`--force`, `--bundle-url`, `--no-data`) |
| `selfcheck.py` | reference SQL for all 25 questions through the local DuckDB load, graded with the site criterion |
| `selfcheck.sh`, `Makefile` | `make venv`, `make fetch`, `make check`, `make clean` |
| `tests/mock_server.py` | stdlib mock OpenAI server with fault injection |
| `requirements.txt` | `duckdb` |
| `questions.json`, `data/` | **fetched at run time, gitignored** — property of the benchmark site |
| `runs/` | output reports (gitignored) |

Comments of the form `:NNN` in `replay.py` are line numbers in a prettier-formatted copy of the
site's `BenchmarkRunner` bundle (hash `CELwKJm9`, 2026-11) for anyone verifying the port against
the site's own code; that copy is not shipped.
