#!/usr/bin/env python3
"""
Fetch the benchmark definition and data from https://sql-benchmark.nicklothian.com/ at run time.

Nothing from the site is redistributed with this harness. This script:
  1. downloads index.html and discovers the hashed BenchmarkRunner bundle
     (``component-url="/_astro/BenchmarkRunner.<hash>.js"``);
  2. downloads that bundle and extracts the embedded dataset object
     (``{id:"adventureworks", name:..., questions:<array>, ...}``) -- the 25 questions with their
     reference ``sql`` and the grader's expected ``row_count`` / ``columns`` / ``first_row``;
  3. writes them to ``questions.json`` next to this script;
  4. downloads the 7 AdventureWorks CSV tables (``data/tables/<Table>.csv.gz``, decompressed to
     ``.csv``) and ``data/answers.json.gz`` (full expected rows, optional, used by selfcheck.py)
     into ``data/`` (gitignored).

The data and questions belong to the benchmark site / its author (Nick Lothian).

Usage:  python3 fetch_site.py [--force] [--bundle-url URL] [--no-data]
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.request

SITE = "https://sql-benchmark.nicklothian.com/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TABLES_DIR = os.path.join(DATA_DIR, "tables")
QUESTIONS_FILE = os.path.join(HERE, "questions.json")
ANSWERS_FILE = os.path.join(DATA_DIR, "answers.json")
TABLES = ["Customer", "Date", "Product", "Reseller", "Sales", "Sales_Order", "Sales_Territory"]
N_QUESTIONS = 25

# Pinned fallback: the bundle name observed when this harness was written (2026-11).
# Used only if index.html no longer reveals a BenchmarkRunner bundle.
PINNED_BUNDLE = "/_astro/BenchmarkRunner.CELwKJm9.js"

ASSUMPTIONS = """Extraction assumptions (fetch_site.py):
  * index.html contains an Astro island  component-url="/_astro/BenchmarkRunner.<hash>.js"
  * that bundle contains a minified JS object literal  {id:"adventureworks",name:"...",questions:<VAR>,...}
    and the array  <VAR>=[{id:1,question:"...",difficulty:"...",sql:"...",row_count:N,columns:[...],first_row:{...}},...]
    (plain JS literal syntax: unquoted keys, "..."/'...'/`...` strings, numbers such as 160243023e-2)
  * exactly 25 questions, each with id, question, difficulty, sql, row_count, columns, first_row
  * tables are served as  <site>/data/tables/<Table>.csv.gz  (gzip; the site fetches "<path>.gz")
If the site changed, update fetch_site.py (or pass --bundle-url) and re-run."""


class FetchError(Exception):
    pass


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "sqlbench-replay/1.0 (+fetch_site.py)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} for {url}") from None
    except (urllib.error.URLError, OSError) as e:
        raise FetchError(f"cannot fetch {url}: {e}") from None


def maybe_gunzip(b):
    return gzip.decompress(b) if len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B else b


# ---------------------------------------------------------------- minimal JS literal parser
class _JS:
    """Parses a JS value literal (object/array/string/number/true/false/null/!0/!1/void 0)
    as emitted by minifiers. Good enough for data literals; no expressions."""
    _ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0",
            "\\": "\\", "'": "'", '"': '"', "`": "`", "/": "/"}

    def __init__(self, s, i=0):
        self.s, self.i = s, i

    def err(self, what):
        return FetchError(f"JS literal parse error at offset {self.i}: {what}: {self.s[self.i:self.i + 60]!r}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def value(self):
        self.ws()
        c = self.s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c in "\"'`":
            return self.string()
        if c == "-" or c == "." or c.isdigit():
            return self.number()
        for lit, v in (("true", True), ("false", False), ("null", None), ("!0", True), ("!1", False), ("void 0", None)):
            if self.s.startswith(lit, self.i):
                self.i += len(lit)
                return v
        raise self.err("unexpected token")

    def string(self):
        q = self.s[self.i]
        self.i += 1
        out = []
        while True:
            if self.i >= len(self.s):
                raise self.err("unterminated string")
            c = self.s[self.i]
            if c == q:
                self.i += 1
                return "".join(out)
            if c == "\\":
                e = self.s[self.i + 1]
                if e == "u":
                    if self.s[self.i + 2] == "{":
                        end = self.s.index("}", self.i)
                        out.append(chr(int(self.s[self.i + 3:end], 16)))
                        self.i = end + 1
                    else:
                        out.append(chr(int(self.s[self.i + 2:self.i + 6], 16)))
                        self.i += 6
                    continue
                if e == "x":
                    out.append(chr(int(self.s[self.i + 2:self.i + 4], 16)))
                    self.i += 4
                    continue
                if e == "\n":  # line continuation
                    self.i += 2
                    continue
                out.append(self._ESC.get(e, e))
                self.i += 2
                continue
            if q == "`" and c == "$" and self.s[self.i + 1] == "{":
                raise self.err("template literal with substitution")
            out.append(c)
            self.i += 1

    _NUM = re.compile(r"-?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

    def number(self):
        m = self._NUM.match(self.s, self.i)
        if not m:
            raise self.err("bad number")
        self.i = m.end()
        t = m.group(0)
        if t[1:3].lower() == "x" or t[:2].lower() == "0x":
            return int(t, 16)
        v = float(t)
        return int(v) if v == int(v) and "." not in t and "e" not in t.lower() and abs(v) < 2 ** 53 else v

    def key(self):
        self.ws()
        c = self.s[self.i]
        if c in "\"'":
            return self.string()
        m = re.compile(r"[A-Za-z_$][\w$]*").match(self.s, self.i)
        if not m:
            raise self.err("bad object key")
        self.i = m.end()
        return m.group(0)

    def obj(self):
        self.i += 1
        out = {}
        self.ws()
        if self.s[self.i] == "}":
            self.i += 1
            return out
        while True:
            k = self.key()
            self.ws()
            if self.s[self.i] != ":":
                raise self.err("expected ':'")
            self.i += 1
            out[k] = self.value()
            self.ws()
            c = self.s[self.i]
            self.i += 1
            if c == "}":
                return out
            if c != ",":
                raise self.err("expected ',' or '}'")
            self.ws()
            if self.s[self.i] == "}":  # trailing comma
                self.i += 1
                return out

    def arr(self):
        self.i += 1
        out = []
        self.ws()
        if self.s[self.i] == "]":
            self.i += 1
            return out
        while True:
            out.append(self.value())
            self.ws()
            c = self.s[self.i]
            self.i += 1
            if c == "]":
                return out
            if c != ",":
                raise self.err("expected ',' or ']'")
            self.ws()
            if self.s[self.i] == "]":
                self.i += 1
                return out


def parse_js_literal(s, i):
    return _JS(s, i).value()


# ---------------------------------------------------------------- discovery + extraction
def discover_bundle(site=SITE):
    """Return (bundle_url, how) from index.html; falls back to the pinned name."""
    html = http_get(site).decode("utf-8", "replace")
    m = re.search(r'component-url="(/_astro/BenchmarkRunner\.[^"]+\.js)"', html)
    if not m:
        m = re.search(r'["\'(](/_astro/BenchmarkRunner[^"\')\s]*\.js)', html)
    if m:
        return site.rstrip("/") + m.group(1), "index.html"
    return site.rstrip("/") + PINNED_BUNDLE, "pinned fallback (index.html did not reference a BenchmarkRunner bundle)"


def _find_array_var(js, name):
    """Locate 'name=[' as a variable initialiser (preceded by a non-identifier char)."""
    for m in re.finditer(r"(?<![\w$])" + re.escape(name) + r"=\[", js):
        return m.end() - 1
    return None


def extract_dataset(js):
    """Return the dataset dict {id, name, questions:[...]} from the bundle source."""
    # 1. the dataset object: {id:"adventureworks",name:"...",questions:VAR,...}
    m = re.search(r'\{id:"adventureworks",([^{}]*)\}', js)
    name, qvar = "AdventureWorks Benchmark", None
    if m:
        body = m.group(1)
        mn = re.search(r'name:"([^"]*)"', body)
        if mn:
            name = mn.group(1)
        mq = re.search(r"questions:([A-Za-z_$][\w$]*)", body)
        if mq:
            qvar = mq.group(1)
    start = _find_array_var(js, qvar) if qvar else None
    if start is None:
        # 2. fallback: the array literal itself, recognisable by its first element
        m2 = re.search(r"=\[\{id:1,question:", js)
        if not m2:
            raise FetchError("could not locate the questions array in the bundle")
        start = m2.start() + 1
    questions = parse_js_literal(js, start)
    if not isinstance(questions, list):
        raise FetchError("questions literal is not an array")
    required = {"id", "question", "difficulty", "sql", "row_count", "columns", "first_row"}
    for q in questions:
        if not isinstance(q, dict) or not required <= set(q):
            raise FetchError(f"question entry missing keys {sorted(required - set(q or {}))}: {str(q)[:120]}")
    if len(questions) != N_QUESTIONS:
        raise FetchError(f"expected {N_QUESTIONS} questions, found {len(questions)}")
    tables = TABLES
    mt = re.search(r"tableFiles:([A-Za-z_$][\w$]*)", m.group(1)) if m else None
    if mt:
        ts = _find_array_var(js, mt.group(1))
        if ts is not None:
            files = parse_js_literal(js, ts)
            names = [re.sub(r"^tables/|\.csv$", "", f) for f in files if isinstance(f, str)]
            if names:
                tables = names
    return {"id": "adventureworks", "name": name, "questions": questions}, tables


def fetch_questions(bundle_url=None, site=SITE, verbose=True):
    how = "command line"
    if not bundle_url:
        bundle_url, how = discover_bundle(site)
    if verbose:
        print(f"[fetch_site] bundle: {bundle_url}  ({how})", file=sys.stderr)
    js = http_get(bundle_url).decode("utf-8", "replace")
    try:
        ds, tables = extract_dataset(js)
    except FetchError as e:
        raise FetchError(f"extraction failed: {e}\n  bundle URL: {bundle_url}\n{ASSUMPTIONS}") from None
    ds["source"] = {"site": site, "bundle": bundle_url}
    return ds, tables


def fetch_tables(tables, site=SITE, force=False, verbose=True):
    os.makedirs(TABLES_DIR, exist_ok=True)
    for t in tables:
        csv_path = os.path.join(TABLES_DIR, f"{t}.csv")
        if os.path.exists(csv_path) and not force:
            continue
        url = f"{site.rstrip('/')}/data/tables/{t}.csv.gz"
        if verbose:
            print(f"[fetch_site] {url}", file=sys.stderr)
        raw = http_get(url, timeout=300)
        with open(os.path.join(TABLES_DIR, f"{t}.csv.gz"), "wb") as f:
            f.write(raw)
        data = maybe_gunzip(raw)
        if not data or data[:1] == b"<":
            raise FetchError(f"{url} did not return CSV data (got {data[:40]!r})\n{ASSUMPTIONS}")
        with open(csv_path + ".part", "wb") as f:
            f.write(data)
        os.replace(csv_path + ".part", csv_path)


def fetch_answers(site=SITE, force=False, verbose=True):
    """Optional: full expected rows (Question Explorer data) for selfcheck.py."""
    if os.path.exists(ANSWERS_FILE) and not force:
        return True
    url = f"{site.rstrip('/')}/data/answers.json.gz"
    try:
        raw = http_get(url)
        data = maybe_gunzip(raw)
        json.loads(data)
    except (FetchError, ValueError) as e:
        if verbose:
            print(f"[fetch_site] optional answers.json not available: {e}", file=sys.stderr)
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ANSWERS_FILE, "wb") as f:
        f.write(data)
    return True


def ensure(force=False, bundle_url=None, with_data=True, verbose=True):
    """Make sure questions.json and data/tables/*.csv exist; fetch what is missing."""
    need_q = force or not os.path.exists(QUESTIONS_FILE)
    tables = TABLES
    if need_q:
        ds, tables = fetch_questions(bundle_url, verbose=verbose)
        with open(QUESTIONS_FILE, "w") as f:
            json.dump(ds, f, indent=1, ensure_ascii=False)
        if verbose:
            print(f"[fetch_site] wrote {os.path.relpath(QUESTIONS_FILE, os.getcwd())} ({len(ds['questions'])} questions)", file=sys.stderr)
    else:
        with open(QUESTIONS_FILE) as f:
            ds = json.load(f)
    if with_data:
        missing = [t for t in tables if not os.path.exists(os.path.join(TABLES_DIR, f"{t}.csv"))]
        if force or missing:
            fetch_tables(tables, force=force, verbose=verbose)
        fetch_answers(force=force, verbose=verbose)
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    ap.add_argument("--bundle-url", default=None, help="skip discovery and use this BenchmarkRunner bundle URL")
    ap.add_argument("--no-data", action="store_true", help="only write questions.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        ds = ensure(force=a.force, bundle_url=a.bundle_url, with_data=not a.no_data, verbose=not a.quiet)
    except FetchError as e:
        sys.exit(f"fetch_site.py: {e}")
    qs = ds["questions"]
    by_diff = ", ".join(f"{d}={sum(q['difficulty'] == d for q in qs)}" for d in ("trivial", "easy", "medium", "hard"))
    present = ", ".join(t for t in TABLES if os.path.exists(os.path.join(TABLES_DIR, f"{t}.csv")))
    print(f"{len(qs)} questions ({by_diff}); tables in {os.path.relpath(TABLES_DIR, os.getcwd())}: {present}")


if __name__ == "__main__":
    main()
