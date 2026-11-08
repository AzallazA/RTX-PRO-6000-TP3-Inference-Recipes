PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: venv fetch check clean

venv:            ## create .venv and install the only dependency (duckdb)
	python3 -m venv .venv && .venv/bin/pip install -q duckdb

fetch:           ## download questions + data from the benchmark site
	$(PY) fetch_site.py

check:           ## fetch + reference-SQL self-check (25/25) + mock tool-calling loop
	PYTHON=$(PY) ./selfcheck.sh

clean:           ## remove fetched site data and run outputs (keeps .venv)
	rm -rf data questions.json runs
