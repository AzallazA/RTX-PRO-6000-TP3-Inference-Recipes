#!/bin/bash
# Full qualification of the running Flare build (2x build: H342 + BRRR tail + MXFP4 Engram in RAM).
# Same protocol as the .11 vLLM release where possible, at concurrency 8 instead of 4 (about 1 h instead of 2.5 h); Flare deltas noted inline.
# Outputs go to $QUAL_OUT (default run-<date>/ next to this script), never into the shipped results/ and RESULTS.md.
# Needs: FLARE_LOG=<server log> (stress stage), CKPT=<checkpoint view> (long generation detokenizes with its tokenizer),
# llm-inference-bench and LiveBench checkouts: LLM_BENCH_DIR (default ~/llm-inference-bench), LIVEBENCH_DIR (default ~/LiveBench).
# Stages (arguments, or the profile's list): eval tools lavd hotel retrieval longgen eos livebench stress gpqa tooleval.
# QUAL_PROFILE=full (default): eval tools lavd hotel retrieval longgen eos livebench stress (about 1 h at c8).
# QUAL_PROFILE=core (about 75 min): eval tools hotel retrieval longgen gpqa tooleval stress (LIVEBENCH=1 adds livebench):
#   no lavd and no respect-EOS bench, long generation 2 seeds (LONGGEN_SEEDS), LiveBench replaced by GPQA Diamond
#   (llm-inference-bench gpqa-diamond profile, temperature 0, paired with a stored baseline) and tool-eval-bench.
#   Time: eval ~2 min, tools ~1, hotel-lights ~11, retrieval ~2, longgen x2 ~5, GPQA ~20-30, tool-eval ~3-5, stress ~18.
# Every stage writes a gate line to GATES.tsv (PASS/WARN/FAIL/INFO) and SUMMARY.md at the end. Gates, baselines,
# REUSE_FROM (skip stages already passed on the same package, binary and serve arguments): see qual_core.sh.
cd "$(dirname "$(readlink -f "$0")")"; Q=$PWD
LLM_BENCH_DIR=${LLM_BENCH_DIR:-$HOME/llm-inference-bench}; LIVEBENCH_DIR=${LIVEBENCH_DIR:-$HOME/LiveBench}
export LLM_BENCH=${LLM_BENCH:-$LLM_BENCH_DIR/llm_decode_bench.py}   # run_qualification.py (longgen) uses this path
export QUAL_OUT=${QUAL_OUT:-$Q/run-$(date +%Y%m%d-%H%M)}; mkdir -p "$QUAL_OUT"; O=$QUAL_OUT
LABEL=${LABEL:-flare-v3-2x}   # result labels; LiveBench caches answers per display name, so give each build its own LABEL
export MODEL=${MODEL:-DeepSeek-V4.1-Flash} PORT=${PORT:-15015} QC=${QC:-8} BASE_TARGET=${BASE_TARGET:-flare-ds41}
. "$Q/qual_core.sh"
case "$QUAL_PROFILE" in
  core) S=${*:-eval tools hotel retrieval longgen gpqa tooleval stress}; [ "${LIVEBENCH:-0}" = 1 ] && [[ " $S " != *" livebench "* ]] && S="$S livebench"
        export LONGGEN_SEEDS=${LONGGEN_SEEDS:-201 202};;
  full) S=${*:-eval tools lavd hotel retrieval longgen eos livebench stress};;
  *) echo "QUAL_PROFILE must be full or core" >&2; exit 2;;
esac
export LONGGEN_SEEDS=${LONGGEN_SEEDS:-201 202 203 204}
note(){ echo "$(date -u +%FT%T) — $*" | tee -a "$O/QUAL.md"; }
has(){ [[ " $S " == *" $1 "* ]] && ! qc_reused "${2:-$1}"; }   # has STAGE [GATE]: wanted and not reused
note "outputs: $O"
note "qualification start: $(ps -eo args | grep 'target/release/[f]lare run' | grep -o 'packages/models/[^ ]*') stages: $S"
note "KV $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}'), power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"
qc_fingerprint; note "profile: $QUAL_PROFILE; longgen seeds $LONGGEN_SEEDS${REUSE_FROM:+; REUSE_FROM $REUSE_FROM ($(qc_reuse_ok && echo same fingerprint || echo 'fingerprint differs: nothing reused'))}"

if has eval; then   # eval-43 (thinking off). Flare clamps temperature 0 up to its fixed 1.0.
  python3 compare_models.py $LABEL > "$O/eval.log" 2>&1
  v=$(python3 -c "import json;d=json.load(open('$O/eval_$LABEL.json'));i=d['intelligence'];print(sum(x['correct'] for x in i),'/',len(i))" 2>/dev/null)
  note "eval-43: $v (temperature clamped to 1.0)"
  n=${v%% *}; if [ "${n:-0}" -ge 42 ] 2>/dev/null; then gate eval PASS "$v"; elif [ "${n:-0}" -ge 41 ] 2>/dev/null; then gate eval WARN "$v"; else gate eval FAIL "${v:-no result}"; fi
fi
if has tools; then
  python3 tool_calls.py 60 8 > "$O/tools.log" 2>&1; v=$(tail -1 "$O/tools.log"); note "$v"
  case "$v" in *"60/60 OK"*) gate tools PASS "$v";; *"59/60"*) gate tools WARN "$v";; *) gate tools FAIL "${v:-no result}";; esac
fi
for p in lavd hotel; do
  g=$p; [ $p = hotel ] && g=hotel-lights
  if has $p $g; then
    name=$p; [ $p = hotel ] && name=hotel-lights
    QUAL_TOP_P=1.0 python3 run_qualification.py $name > "$O/qual-$name.log" 2>&1
    note "$(grep -h 'DONE' "$O/RESULTS.md" | grep "$name-c${QC:-8}" | tail -1 | cut -d'—' -f2-)"; qc_gate_profile $name
  fi
done
if has retrieval; then
  QUAL_TOP_P=1.0 python3 run_qualification.py retrieval > "$O/qual-retrieval.log" 2>&1
  note "$(grep -h 'DONE long retrieval' "$O/RESULTS.md" | tail -1 | cut -d'—' -f2-)"; qc_gate_retrieval
fi
if has longgen; then
  QUAL_TOP_P=1.0 python3 run_qualification.py longgen > "$O/qual-longgen.log" 2>&1
  note "long generation: $(python3 -c "import json;r=json.load(open('$O/long-generation.json'));print(len(r),'runs, fails',sum(x['fail'] for x in r),'| completions',[x['completion'] for x in r],'| repeat_8gram',[x['repeat_8gram'] for x in r])")"
  qc_gate_longgen $(wc -w <<< "$LONGGEN_SEEDS")
fi
if has eos; then   # bench decode with natural EOS: checks the ignore_eos loop theory
  (cd "$LLM_BENCH_DIR" && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
     --concurrency 1,4,8 --contexts 32k --duration 60 --respect-eos \
     --output "$O/eos-bench.json" > "$O/eos-bench.log" 2>&1)
  loops=$(grep -c 'ERROR: loop' "$O/eos-bench.log"); note "respect-eos bench: $(grep -E '│ 32k' "$O/eos-bench.log" | head -1 | tr -s ' ') loops: $loops"
  [ "$loops" = 0 ] && [ -s "$O/eos-bench.json" ] && gate eos PASS "loops 0" || gate eos FAIL "loops $loops"
fi
if has livebench; then   # LiveBench coding, same settings as the .11 run (81.9)
  (cd "$LIVEBENCH_DIR/livebench" && "$LIVEBENCH_DIR/.venv/bin/python" run_livebench.py --venv "$LIVEBENCH_DIR/.venv/bin/activate" --model DeepSeek-V4.1-Flash \
     --model-display-name DeepSeek-V4.1-Flash-$LABEL --api-base http://localhost:15015/v1 --api-key dummy \
     --bench-name live_bench/coding --livebench-release-option 2024-11-25 --parallel-requests 8 --max-tokens 32768 \
     --force-temperature 1.0 > "$O/livebench.log" 2>&1
   "$LIVEBENCH_DIR/.venv/bin/python" show_livebench_result.py --bench-name live_bench/coding --livebench-release-option 2024-11-25 \
     --model-list DeepSeek-V4.1-Flash-$LABEL > "$O/livebench-score.log" 2>&1)
  # score = last number on the last row of the "All Groups" table (pandas truncates long display names, so no name match)
  lb=$(awk '/All Groups/{g=1;next} g && $NF ~ /^[0-9]+(\.[0-9]+)?$/ {s=$NF} END{print s}' "$O/livebench-score.log"); note "LiveBench coding: $lb"
  if [ -z "$lb" ]; then gate livebench FAIL "no score"; elif python3 -c "import sys; sys.exit(0 if float('$lb') >= ${LB_MIN:-80.5} else 1)"; then gate livebench PASS "coding $lb (min ${LB_MIN:-80.5})"
  else gate livebench WARN "coding $lb < ${LB_MIN:-80.5} (temperature 1: re-run once before calling it a regression)"; fi
fi
if has gpqa; then qc_gpqa; fi
if has tooleval; then qc_tooleval; fi
if has stress; then   # .11 protocol: 2 waves x 8 x ~530K unique prompts + pinger, KV oversubscribed
  FLARE_LOG=${FLARE_LOG:?set FLARE_LOG to the server log} python3 stress_kv.py $LABEL-c16 \
    $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}') 530000 16 > "$O/stress.log" 2>&1
  note "stress: $(grep -E 'PASS|FAIL' "$O/stress.log" | tail -1)"; qc_gate_stress $LABEL-c16
fi
qc_summary
note "qualification done"
