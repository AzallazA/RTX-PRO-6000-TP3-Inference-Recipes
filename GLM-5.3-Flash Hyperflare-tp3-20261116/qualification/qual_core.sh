# qual_core.sh: gates, the "core" profile, GPQA Diamond / tool-eval-bench stages and stage reuse for qual_all.sh.
# Sourced by qual_all.sh after it sets O (output dir), S (stages), MODEL, PORT, QC, LLM_BENCH_DIR and note().
# Identical copies live in each qualification kit (DS41 release folder, GLM glm-qual); keep them in sync.
#
# QUAL_PROFILE   full (default: the kit's original stage list) | core (see qual_all.sh header)
# GATES          every stage writes "time<TAB>name<TAB>PASS|WARN|FAIL|INFO<TAB>detail" to $O/GATES.tsv; SUMMARY.md at the end
# REUSE_FROM     an earlier run dir of this kit: a stage whose gates all PASSED there is not re-run (gate copied and
#                marked REUSED) when $O/fingerprint.txt (package, server binary checksum, serve arguments) is identical
# Baselines      GPQA_BASE / TOOLEVAL_BASE = a results JSON or "auto" = newest $BASELINE_DIR/$BASE_TARGET-*/{gpqa,tooleval}.json
#                (BASELINE_DIR default $HOME/bench-results/baselines). The GPQA run reuses the baseline's max_tokens
#                (GPQA_MAX_TOKENS otherwise). Gates: GPQA FAIL on a significant paired drop (McNemar p < 0.05) or
#                truncated answers up by > TRUNC_UP_MAX (3), WARN on a non-significant drop > GPQA_DROP_PP (3) points;
#                tool-eval FAIL below baseline - TOOLEVAL_DROP (3). Without a baseline both are INFO.
# MINFREE_MIB    stress min free VRAM margin (480): below it the stress gate is WARN when every request passed
QUAL_PROFILE=${QUAL_PROFILE:-full}
BASELINE_DIR=${BASELINE_DIR:-$HOME/bench-results/baselines}; GPQA_BASE=${GPQA_BASE:-auto}; TOOLEVAL_BASE=${TOOLEVAL_BASE:-auto}
GPQA_MAX_TOKENS=${GPQA_MAX_TOKENS:-}; GPQA_DROP_PP=${GPQA_DROP_PP:-3}; TRUNC_UP_MAX=${TRUNC_UP_MAX:-3}; TOOLEVAL_DROP=${TOOLEVAL_DROP:-3}
MINFREE_MIB=${MINFREE_MIB:-480}; REUSE_FROM=${REUSE_FROM:-}; QC_REUSED=""
QCPY=${QCPY:-python3}

gate(){ local n=$1 lv=$2; shift 2; printf '%s\t%s\t%s\t%s\n' "$(date -u +%T)" "$n" "$lv" "$*" >> "$O/GATES.tsv"; note "GATE $n $lv: $*"; }

qc_fingerprint(){  # package + server binary checksum + serve arguments of the running Flare server
  local pid; pid=$(ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1; exit}')
  if [ -z "$pid" ]; then echo "no flare run process" > "$O/fingerprint.txt"; return; fi
  { echo "binary $(md5sum < "/proc/$pid/exe" 2>/dev/null | cut -c1-32)"
    echo "args $(tr '\0' ' ' < "/proc/$pid/cmdline" | sed -E 's/^[^ ]+ //')"
    echo "qc ${QC:-8} profile-independent"; } > "$O/fingerprint.txt"; }

qc_reuse_ok(){ [ -n "$REUSE_FROM" ] && [ -s "$REUSE_FROM/fingerprint.txt" ] && cmp -s "$REUSE_FROM/fingerprint.txt" "$O/fingerprint.txt"; }
qc_reused(){  # qc_reused STAGE -> 0 (gate copied) when REUSE_FROM passed gate STAGE on the same fingerprint
  [ -n "$REUSE_FROM" ] || return 1
  qc_reuse_ok || return 1
  local row; row=$(awk -F'\t' -v n="$1" '$2==n {r=$0} END {print r}' "$REUSE_FROM/GATES.tsv" 2>/dev/null)
  [ -n "$row" ] && [ "$(cut -f3 <<< "$row")" = PASS ] || return 1
  gate "$1" PASS "REUSED from $(basename "$REUSE_FROM"): $(cut -f4 <<< "$row")"; QC_REUSED="$QC_REUSED $1"; return 0; }

qc_newest_base(){  # qc_newest_base FILE
  local d; for d in $(ls -td "$BASELINE_DIR/${BASE_TARGET:?}"-*/ 2>/dev/null); do
    [ -s "$d$1" ] && $QCPY -I -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get('accuracy') or (d.get('tool_eval') or {}).get('summary')) else 1)" "$d$1" 2>/dev/null \
      && { echo "$d$1"; return 0; }; done; return 1; }

qc_rqline(){ grep -h "$1" "$O/RESULTS.md" 2>/dev/null | tail -1 | cut -d'—' -f2- | sed 's/^ *//'; }
qc_gate_profile(){  # qc_gate_profile NAME (lavd | hotel-lights) from the kit's DONE line
  local v; v=$(qc_rqline "DONE $1-c${QC:-8}")
  if [[ "$v" =~ exact\ ([0-9]+)\ /\ wrong\ ([0-9]+) ]]; then local e=${BASH_REMATCH[1]}
    if [ "$e" -ge 30 ]; then gate "$1" PASS "$v"; elif [ "$e" -ge 29 ]; then gate "$1" WARN "$v"; else gate "$1" FAIL "$v"; fi
  else gate "$1" FAIL "no DONE line"; fi; }
qc_gate_retrieval(){ local v; v=$(qc_rqline "DONE long retrieval")
  case "$v" in *"128000 pass"*"900000 pass"*) gate retrieval PASS "$v";; *) gate retrieval FAIL "${v:-no result}";; esac; }
qc_gate_longgen(){  # qc_gate_longgen WANT
  $QCPY -I - "$O/long-generation.json" "$1" <<'PY' | while IFS=$'\t' read -r lv msg; do gate longgen "$lv" "$msg"; done
import json, sys
try: rows = json.load(open(sys.argv[1]))
except Exception as e: print(f"FAIL\tno result ({e!r})"[:200]); sys.exit()
want = int(sys.argv[2]); bad = [r["seed"] for r in rows if r["fail"]]
print(f"{'PASS' if len(rows) == want and not bad else 'FAIL'}\t{len(rows)}/{want} runs, fails {bad}, completions {[r['completion'] for r in rows]}, worst 8-gram {max((r['repeat_8gram'] for r in rows), default=None)}")
PY
}
qc_gate_stress(){  # qc_gate_stress PRESET
  $QCPY -I - "$O/stress_$1.json" "$MINFREE_MIB" <<'PY' | while IFS=$'\t' read -r lv msg; do gate stress "$lv" "$msg"; done
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print(f"FAIL\tno result ({e!r}; timeout = hang)"[:200]); sys.exit()
m = int(sys.argv[2]); line = (f"{d['verdict']} {d['long_ok']}/{d['long_total']} long, ping {d['ping_ok']}/{d['ping_total']}, min free {d['min_free_mib']} MiB, "
                              f"bad lines {len(d['bad_log_lines'])}, wall {d['wall_min']:.1f} min")
if not str(d["verdict"]).startswith("PASS"): print(f"FAIL\t{line}")
elif d["min_free_mib"] < m: print(f"WARN\t{line} (every request passed; min free below the {m} MiB margin)")
else: print(f"PASS\t{line}")
PY
}

qc_gpqa(){  # GPQA Diamond, temperature 0, profile concurrency QC, paired against the baseline
  local gb=$GPQA_BASE mt=$GPQA_MAX_TOKENS args=() rc=0 bmt
  [ "$gb" = auto ] && gb=$(qc_newest_base gpqa.json || true)
  if [ -n "$gb" ] && [ -s "$gb" ]; then args+=(--compare-baseline "$gb")
    bmt=$($QCPY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['metadata'].get('max_tokens') or '')" "$gb" 2>/dev/null); [ -n "$bmt" ] && mt=$bmt; fi
  [ -n "$mt" ] && args+=(--max-tokens "$mt")
  note "gpqa: GPQA Diamond, model $MODEL, concurrency ${QC:-8}, max_tokens ${mt:-bench default}, baseline ${gb:-none}"
  (cd "$LLM_BENCH_DIR" && timeout 14400 .venv/bin/python llm_decode_bench.py --port "$PORT" --model "$MODEL" --test-profile gpqa-diamond \
     --profile-concurrency "${QC:-8}" "${args[@]}" --display-mode plain --output "$O/gpqa.json" > "$O/gpqa.log" 2>&1) || rc=$?
  $QCPY -I - "$O/gpqa.json" "${gb:-}" "$GPQA_DROP_PP" "$TRUNC_UP_MAX" "$rc" <<'PY' | while IFS=$'\t' read -r lv msg; do gate gpqa "$lv" "$msg"; done
import json, math, sys
p, gb, drop, tmax, rc = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
try: d = json.load(open(p)); a = d["accuracy"]
except Exception as e: print(f"FAIL\tno result (exit {rc}, {e!r})"[:200]); sys.exit()
head = (f"{a['correct']}/{a['scored']} = {100 * a['accuracy']:.1f}% (Wilson95 {100 * a['wilson95_low']:.1f}-{100 * a['wilson95_high']:.1f}), "
        f"truncated {a.get('truncated_no_answer')}, hit max_tokens {a.get('hit_max_tokens')}, errors {a.get('errors')}")
c = d.get("comparison")
if not c and gb:  # pair by item id here when the bench wrote no comparison (exact two-sided McNemar)
    try:
        def runs(x): return {r["item_id"]: r for r in x.get("runs", []) if r.get("item_id") and r.get("ok")}
        B, C = runs(json.load(open(gb))), runs(d); ids = sorted(set(B) & set(C))
        b = sum(1 for i in ids if B[i].get("correct") and not C[i].get("correct")); cc = sum(1 for i in ids if C[i].get("correct") and not B[i].get("correct"))
        n = b + cc; pv = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, cc) + 1)) / 2 ** n) if n else 1.0
        tr = lambda R: sum(1 for i in ids if R[i].get("hit_max_tokens") and not R[i].get("correct") and not R[i].get("parsed_answer"))
        c = dict(paired_items=len(ids), delta_pp=100 * (cc - b) / len(ids) if ids else 0, mcnemar_exact_p=pv, significant_at_0_05=pv < 0.05,
                 truncated_no_answer=dict(baseline=tr(B), candidate=tr(C)), profile_match=True, dataset_match=True, own=True)
    except Exception as e: print(f"WARN\t{head}; baseline unreadable ({e!r})"[:250]); sys.exit()
if not c: print(f"INFO\t{head}; no baseline: this run can serve as one"); sys.exit()
if not (c.get("profile_match") and c.get("dataset_match")): print(f"WARN\t{head}; baseline profile/dataset differ: not compared"); sys.exit()
tu = c["truncated_no_answer"]["candidate"] - c["truncated_no_answer"]["baseline"]
lv = "FAIL" if (c["delta_pp"] < 0 and c["significant_at_0_05"]) or tu > tmax else ("WARN" if c["delta_pp"] < -drop else "PASS")
print(f"{lv}\t{head}; vs baseline {c['delta_pp']:+.1f} pp on {c['paired_items']} paired, McNemar p {c['mcnemar_exact_p']:.3f}, "
      f"truncated {c['truncated_no_answer']['baseline']} -> {c['truncated_no_answer']['candidate']}{' (own pairing)' if c.get('own') else ''}")
PY
}
qc_tooleval(){  # tool-eval-bench with the community leaderboard settings, against the baseline score
  local tb=$TOOLEVAL_BASE rc=0
  [ "$tb" = auto ] && tb=$(qc_newest_base tooleval.json || true)
  (cd "$LLM_BENCH_DIR" && timeout 5400 .venv/bin/python llm_decode_bench.py --port "$PORT" --model "$MODEL" --tool-eval \
     --display-mode plain --output "$O/tooleval.json" > "$O/tooleval.log" 2>&1) || rc=$?
  $QCPY -I - "$O/tooleval.json" "${tb:-}" "$TOOLEVAL_DROP" "$rc" <<'PY' | while IFS=$'\t' read -r lv msg; do gate tooleval "$lv" "$msg"; done
import json, sys
p, tb, drop, rc = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
def summ(x):
    t = json.load(open(x)).get("tool_eval") or {}
    return t.get("summary") or {}, t
try: s, t = summ(p)
except Exception as e: print(f"FAIL\tno result (exit {rc}, {e!r})"[:200]); sys.exit()
if s.get("final_score") is None: print(f"FAIL\texit {rc}, {str(t.get('error') or t.get('status') or 'no score')[:160]}"); sys.exit()
head = f"score {s['final_score']}/100 {s.get('rating') or ''}"
try: b, _ = summ(tb) if tb else ({}, {})
except Exception: b = {}
if b.get("final_score") is None: print(f"INFO\t{head}; no baseline: this run can serve as one"); sys.exit()
print(f"{'PASS' if s['final_score'] >= b['final_score'] - drop else 'FAIL'}\t{head}; baseline {b['final_score']} (min baseline - {drop:g})")
PY
}

qc_summary(){
  touch "$O/GATES.tsv"
  local np nw nf; np=$(awk -F'\t' '$3=="PASS"' "$O/GATES.tsv" | wc -l); nw=$(awk -F'\t' '$3=="WARN"' "$O/GATES.tsv" | wc -l); nf=$(awk -F'\t' '$3=="FAIL"' "$O/GATES.tsv" | wc -l)
  { echo "# qualification $(basename "$O")"; echo; echo "profile: $QUAL_PROFILE  "; echo "stages: $S  "
    echo "reused: ${QC_REUSED:-none}${REUSE_FROM:+ (from $REUSE_FROM)}  "; echo "gates: $np PASS, $nw WARN, $nf FAIL"; echo
    echo "| time | gate | result | detail |"; echo "|---|---|---|---|"
    awk -F'\t' '{gsub(/\|/, "/", $4); printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4}' "$O/GATES.tsv"; } > "$O/SUMMARY.md"
  note "gates: $np PASS, $nw WARN, $nf FAIL (profile $QUAL_PROFILE${QC_REUSED:+; reused:$QC_REUSED}; SUMMARY.md)"; }
