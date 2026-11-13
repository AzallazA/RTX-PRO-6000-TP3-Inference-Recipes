#!/bin/bash
# Full qualification of the running Flare GLM-5.3-Flash TP3 package (plan step 6; handbook D.6 tier 5).
# Same stage interface as the DS41 Flare suite (eval tools lavd hotel retrieval longgen eos livebench stress),
# plus the optional tier-1 stages verify and consistency (not in the default list). See README.md.
# Outputs go to $QUAL_OUT (default run-<date>/ next to this script).
# Env: LABEL (result labels; LiveBench caches answers per display name, so give each build its own),
#      FLARE_LOG=<server log> (stress), CKPT=<checkpoint dir> (tokenizer: longgen detokenization, consistency),
#      CAPTURE=<captured-request.san.json> (longgen; default: the copy in the DeepSeek-V4.1-Flash Hyperflare-tp3-20261111
#      folder of this repo), LLM_BENCH_DIR (default $HOME/llm-inference-bench), LIVEBENCH_DIR (default $HOME/LiveBench), MODEL (default GLM-5.3-Flash), FLARE_URL (default http://127.0.0.1:15015),
#      QUAL_TOP_P (0.95), QUAL_TEMPERATURE (1.0), QUAL_EFFORT (max), EVAL_EFFORT (low), TOOLS_EFFORT (high), QC (8),
#      TOOLS_DIR (default ../tools: glm_common.py, tool_calls.py, verify_fix.py), PY (default: $FLARE_DIR/.venv/bin/python
#      when FLARE_DIR=<fork checkout> is set, else python3; consistency needs transformers).
# QUAL_PROFILE=full (default): the stage list above. QUAL_PROFILE=core: eval tools verify hotel retrieval longgen gpqa
#   tooleval stress (LIVEBENCH=1 adds livebench): no lavd, no respect-EOS bench, long generation 2 seeds
#   (LONGGEN_SEEDS), LiveBench replaced by GPQA Diamond (paired with a stored baseline, BASE_TARGET default flare-glm)
#   and tool-eval-bench. Every stage writes a gate line to GATES.tsv and SUMMARY.md at the end; gates, baselines and
#   REUSE_FROM (skip stages already passed on the same package, binary and serve arguments) are in qual_core.sh.
#   Core time at QC=32: eval ~2 min, tools ~1, verify ~3, hotel-lights ~11, retrieval ~3, longgen x2 ~8, GPQA ~20-30,
#   tool-eval ~3-5, stress ~25-35.
set -o pipefail; export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the release folder
cd "$(dirname "$(readlink -f "$0")")"; Q=$PWD
PY=${PY:-$( [ -n "${FLARE_DIR:-}" ] && [ -x "$FLARE_DIR/.venv/bin/python" ] && echo "$FLARE_DIR/.venv/bin/python" || echo python3)}   # venv: consistency needs transformers
TOOLS_DIR=${TOOLS_DIR:-$Q/../tools}; export PYTHONPATH="$TOOLS_DIR${PYTHONPATH:+:$PYTHONPATH}"   # glm_common.py and the shared tools
export CAPTURE=${CAPTURE:-$Q/../../DeepSeek-V4.1-Flash Hyperflare-tp3-20261111/qualification/captured-request.san.json}
LLM_BENCH_DIR=${LLM_BENCH_DIR:-$HOME/llm-inference-bench}; LIVEBENCH_DIR=${LIVEBENCH_DIR:-$HOME/LiveBench}
export LLM_BENCH=${LLM_BENCH:-$LLM_BENCH_DIR/llm_decode_bench.py}   # run_qualification.py (lavd/hotel) uses this path
export MODEL=${MODEL:-GLM-5.3-Flash} FLARE_URL=${FLARE_URL:-http://127.0.0.1:15015}
export QUAL_TOP_P=${QUAL_TOP_P:-0.95} QUAL_TEMPERATURE=${QUAL_TEMPERATURE:-1.0} QUAL_EFFORT=${QUAL_EFFORT:-max}
export QUAL_OUT=${QUAL_OUT:-$Q/run-$(date +%Y%m%d-%H%M)}; mkdir -p "$QUAL_OUT"; O=$QUAL_OUT
PORT=${FLARE_URL##*:}; PORT=${PORT%%/*}
LABEL=${LABEL:-flare-glm53-tp3}
export QC=${QC:-8} BASE_TARGET=${BASE_TARGET:-flare-glm} QCPY=$PY
. "$Q/qual_core.sh"
case "$QUAL_PROFILE" in
  core) S=${*:-eval tools verify hotel retrieval longgen gpqa tooleval stress}; [ "${LIVEBENCH:-0}" = 1 ] && [[ " $S " != *" livebench "* ]] && S="$S livebench"
        export LONGGEN_SEEDS=${LONGGEN_SEEDS:-201 202};;
  full) S=${*:-eval tools lavd hotel retrieval longgen eos livebench stress};;
  *) echo "QUAL_PROFILE must be full or core" >&2; exit 2;;
esac
export LONGGEN_SEEDS=${LONGGEN_SEEDS:-201 202 203 204}
note(){ echo "$(date -u +%FT%T) — $*" | tee -a "$O/QUAL.md"; }
has(){ [[ " $S " == *" $1 "* ]] && ! qc_reused "${2:-$1}"; }   # has STAGE [GATE]: wanted and not reused
kv(){ curl -s "$FLARE_URL/metrics" | awk '/^flare_total_kv_tokens/{print $2}'; }
# Settings are validated once up front (effort must be low/high/max, sampling inside the recipe ranges).
$PY -c "import glm_common as g; print(g.MODEL, g.QUAL_EFFORT, g.EVAL_EFFORT, g.TOOLS_EFFORT, g.TEMPERATURE, g.TOP_P)" >/dev/null || exit 2
curl -sf "$FLARE_URL/health" >/dev/null || { echo "no Flare server at $FLARE_URL" >&2; exit 2; }
SERVED=$(curl -s "$FLARE_URL/v1/models" | $PY -c "import json,sys; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))")
[[ " $SERVED " == *" $MODEL "* ]] || note "WARNING: MODEL=$MODEL is not among the served names ($SERVED)"
note "outputs: $O"
note "qualification start: $(ps -eo args | grep 'target/release/[f]lare run' | grep -o 'packages/models/[^ ]*') stages: $S"
note "model $MODEL, effort $QUAL_EFFORT (eval ${EVAL_EFFORT:-low}, tools ${TOOLS_EFFORT:-high}), temperature $QUAL_TEMPERATURE, top-p $QUAL_TOP_P"
note "KV $(kv), power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
qc_fingerprint; note "profile: $QUAL_PROFILE; longgen seeds $LONGGEN_SEEDS${REUSE_FROM:+; REUSE_FROM $REUSE_FROM ($(qc_reuse_ok && echo same fingerprint || echo 'fingerprint differs: nothing reused'))}"

if has eval; then   # eval-43: thinking cannot be disabled on GLM -> effort low, content graded
  $PY compare_models.py $LABEL > "$O/eval.log" 2>&1
  note "eval-43: $($PY -c "import json;d=json.load(open('$O/eval_$LABEL.json'));s=d['summary'];print(s['intelligence'],'| refused',s['refused'],'| truncated',s['truncated'],'errors',s['errors'],'markup leaks',s['markup_leaks'])") (thinking on, effort ${EVAL_EFFORT:-low})"
  v=$($PY -c "import json;print(json.load(open('$O/eval_$LABEL.json'))['summary']['intelligence'])" 2>/dev/null); n=${v%%/*}; n=${n// /}
  if [ "${n:-0}" -ge 42 ] 2>/dev/null; then gate eval PASS "$v"; elif [ "${n:-0}" -ge 41 ] 2>/dev/null; then gate eval WARN "$v"; else gate eval FAIL "${v:-no result}"; fi
fi
if has tools; then
  $PY "$TOOLS_DIR/tool_calls.py" 60 ${QC:-8} > "$O/tools.log" 2>&1; v=$(tail -1 "$O/tools.log"); note "$v"
  case "$v" in *"60/60 OK"*) gate tools PASS "$v";; *"59/60"*) gate tools WARN "$v";; *) gate tools FAIL "${v:-no result}";; esac
fi
if has verify; then   # D.6 tier 1: decode-vs-prefill corruption count (raw completions)
  $PY "$TOOLS_DIR/verify_fix.py" 20 > "$O/verify_fix.log" 2>&1; v=$(grep 'TRUE corruptions' "$O/verify_fix.log" | tail -1); note "verify_fix: $v"
  case "$v" in *"TRUE corruptions 0,"*|*"TRUE corruptions 0") gate verify PASS "$v";; *) gate verify FAIL "${v:-no TRUE corruptions line}";; esac
fi
if has consistency; then   # D.6 tier 1: token-exact decode vs teacher-forced prefill
  CKPT=${CKPT:?set CKPT to the checkpoint directory} $PY consistency.py 48 4 > "$O/consistency.log" 2>&1
  note "consistency: $(tail -1 "$O/consistency.log")"; gate consistency INFO "$(tail -1 "$O/consistency.log" | cut -c1-200)"
fi
for p in lavd hotel; do
  g=$p; [ $p = hotel ] && g=hotel-lights
  if has $p $g; then
    name=$p; [ $p = hotel ] && name=hotel-lights
    $PY run_qualification.py $name > "$O/qual-$name.log" 2>&1
    note "$(grep -h 'DONE' "$O/RESULTS.md" | grep "$name-c${QC:-8}" | tail -1 | cut -d'—' -f2-)"; qc_gate_profile $name
  fi
done
if has retrieval; then
  $PY run_qualification.py retrieval > "$O/qual-retrieval.log" 2>&1
  note "$(grep -h 'DONE long retrieval' "$O/RESULTS.md" | tail -1 | cut -d'—' -f2-)"; qc_gate_retrieval
fi
if has longgen; then
  CAPTURE=${CAPTURE:?set CAPTURE to captured-request.san.json} CKPT=${CKPT:?set CKPT to the checkpoint directory} \
    $PY run_qualification.py longgen > "$O/qual-longgen.log" 2>&1
  note "long generation: $($PY -c "import json;r=json.load(open('$O/long-generation.json'));print(len(r),'runs, fails',sum(x['fail'] for x in r),'| prompt',[x.get('prompt') for x in r][:1],'| completions',[x['completion'] for x in r],'| repeat_8gram',[x['repeat_8gram'] for x in r])")"
  qc_gate_longgen $(wc -w <<< "$LONGGEN_SEEDS")
fi
if has eos; then   # bench decode with natural EOS (loop check); server-default sampling and effort (template: Max)
  (cd "$LLM_BENCH_DIR" && .venv/bin/python llm_decode_bench.py --port "$PORT" --model "$MODEL" \
     --concurrency 1,4,8 --contexts 32k --duration 60 --respect-eos \
     --output "$O/eos-bench.json" > "$O/eos-bench.log" 2>&1)
  loops=$(grep -c 'ERROR: loop' "$O/eos-bench.log"); note "respect-eos bench: $(grep -E '│ 32k' "$O/eos-bench.log" | head -1 | tr -s ' ') loops: $loops"
  [ "$loops" = 0 ] && [ -s "$O/eos-bench.json" ] && gate eos PASS "loops 0" || gate eos FAIL "loops $loops"
fi
if has livebench; then
  # LiveBench resolves the name GLM-5.3-Flash (case-insensitive) to its glm-5.3-flash config: requests go out as
  # model "glm-5.3-flash" with temperature 1.0, top_p 0.95, reasoning_effort max, max_tokens 131072, streaming
  # (the published z.ai protocol; that config's max_tokens wins over --max-tokens). Flare does not check the
  # model field, which the preflight below confirms before 128 questions are spent.
  LB_MODEL=${LIVEBENCH_MODEL:-GLM-5.3-Flash}
  if curl -sf "$FLARE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
       -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say ok."}],"max_tokens":256,"reasoning_effort":"low"}' >/dev/null; then
    (cd "$LIVEBENCH_DIR/livebench" && "$LIVEBENCH_DIR/.venv/bin/python" run_livebench.py --venv "$LIVEBENCH_DIR/.venv/bin/activate" --model "$LB_MODEL" \
       --model-display-name GLM-5.3-Flash-$LABEL --api-base "$FLARE_URL/v1" --api-key dummy \
       --bench-name live_bench/coding --livebench-release-option 2024-11-25 --parallel-requests ${QC:-8} --max-tokens 131072 \
       --force-temperature 1.0 > "$O/livebench.log" 2>&1
     "$LIVEBENCH_DIR/.venv/bin/python" show_livebench_result.py --bench-name live_bench/coding --livebench-release-option 2024-11-25 \
       --model-list GLM-5.3-Flash-$LABEL > "$O/livebench-score.log" 2>&1)
    note "LiveBench coding: $(grep -i -A3 "average\|$LABEL" "$O/livebench-score.log" | tr -s ' ' | tail -3 | tr '\n' ' ')"
    lb=$(awk '/All Groups/{g=1;next} g && $NF ~ /^[0-9]+(\.[0-9]+)?$/ {s=$NF} END{print s}' "$O/livebench-score.log")
    [ -n "$lb" ] && gate livebench INFO "coding $lb" || gate livebench FAIL "no score"
  else
    note "LiveBench SKIPPED: the server rejected model glm-5.3-flash (set LIVEBENCH_MODEL to a name LiveBench does not know)"
    gate livebench WARN "skipped: server rejected model glm-5.3-flash"
  fi
fi
if has gpqa; then qc_gpqa; fi
if has tooleval; then qc_tooleval; fi
if has stress; then   # 2 waves x 8 x ~530K unique prompts + pinger, KV oversubscribed (GLM KV ~4.2M tokens)
  CAP=$(kv)
  if [ -z "$CAP" ]; then
    note "stress: FAIL (no flare_total_kv_tokens metric)"; gate stress FAIL "no flare_total_kv_tokens metric"
  else
    FLARE_LOG=${FLARE_LOG:?set FLARE_LOG to the server log} $PY stress_kv.py $LABEL-c${QC:-8} \
      "$CAP" ${STRESS_PROMPT_TOKENS:-530000} ${QC:-8} > "$O/stress.log" 2>&1
    note "stress: $(grep -E 'PASS|FAIL' "$O/stress.log" | tail -1)"; qc_gate_stress $LABEL-c${QC:-8}
  fi
fi
qc_summary
note "qualification done"
