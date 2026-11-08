#!/bin/bash
# Full qualification of the running Flare build (2x build: H342 + BRRR tail + MXFP4 Engram in RAM).
# Same protocol as the .11 vLLM release where possible, at concurrency 8 instead of 4 (about 1 h instead of 2.5 h); Flare deltas noted inline.
# Outputs go to $QUAL_OUT (default run-<date>/ next to this script), never into the shipped results/ and RESULTS.md.
# Needs: FLARE_LOG=<server log> (stress stage), CKPT=<checkpoint view> (long generation detokenizes with its tokenizer),
# ~/llm-inference-bench and ~/LiveBench checkouts.
cd "$(dirname "$(readlink -f "$0")")"; Q=$PWD
export QUAL_OUT=${QUAL_OUT:-$Q/run-$(date +%Y%m%d-%H%M)}; mkdir -p "$QUAL_OUT"; O=$QUAL_OUT
S=${*:-eval tools lavd hotel retrieval longgen eos livebench stress}
note(){ echo "$(date -u +%FT%T) — $*" | tee -a "$O/QUAL.md"; }
has(){ [[ " $S " == *" $1 "* ]]; }
note "outputs: $O"
note "qualification start: $(ps -eo args | grep 'target/release/[f]lare run' | grep -o 'packages/models/[^ ]*') stages: $S"
note "KV $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}'), power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"

if has eval; then   # eval-43 (thinking off). Flare clamps temperature 0 up to its fixed 1.0.
  python3 compare_models.py flare-v3-2x > $O/eval.log 2>&1
  note "eval-43: $(python3 -c "import json;d=json.load(open('$O/eval_flare-v3-2x.json'));i=d['intelligence'];print(sum(x['correct'] for x in i),'/',len(i))") (temperature clamped to 1.0)"
fi
if has tools; then
  python3 tool_calls.py 60 8 > $O/tools.log 2>&1; note "$(tail -1 $O/tools.log)"
fi
for p in lavd hotel; do
  if has $p; then
    name=$p; [ $p = hotel ] && name=hotel-lights
    QUAL_TOP_P=1.0 python3 run_qualification.py $name > $O/qual-$name.log 2>&1
    note "$(grep -h 'DONE' $O/RESULTS.md | grep "$name-c${QC:-8}" | tail -1 | cut -d'—' -f2-)"
  fi
done
if has retrieval; then
  QUAL_TOP_P=1.0 python3 run_qualification.py retrieval > $O/qual-retrieval.log 2>&1
  note "$(grep -h 'DONE long retrieval' $O/RESULTS.md | tail -1 | cut -d'—' -f2-)"
fi
if has longgen; then
  QUAL_TOP_P=1.0 python3 run_qualification.py longgen > $O/qual-longgen.log 2>&1
  note "long generation: $(python3 -c "import json;r=json.load(open('$O/long-generation.json'));print(len(r),'runs, fails',sum(x['fail'] for x in r),'| completions',[x['completion'] for x in r],'| repeat_8gram',[x['repeat_8gram'] for x in r])")"
fi
if has eos; then   # bench decode with natural EOS: checks the ignore_eos loop theory
  (cd ~/llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
     --concurrency 1,4,8 --contexts 32k --duration 60 --respect-eos \
     --output $O/eos-bench.json > $O/eos-bench.log 2>&1)
  note "respect-eos bench: $(grep -E '│ 32k' $O/eos-bench.log | head -1 | tr -s ' ') loops: $(grep -c 'ERROR: loop' $O/eos-bench.log)"
fi
if has livebench; then   # LiveBench coding, same settings as the .11 run (81.9)
  (cd ~/LiveBench/livebench && ~/LiveBench/.venv/bin/python run_livebench.py --venv ~/LiveBench/.venv/bin/activate --model DeepSeek-V4.1-Flash \
     --model-display-name DeepSeek-V4.1-Flash-flare-v3-2x --api-base http://localhost:15015/v1 --api-key dummy \
     --bench-name live_bench/coding --livebench-release-option 2024-11-25 --parallel-requests 8 --max-tokens 32768 \
     --force-temperature 1.0 > $O/livebench.log 2>&1
   ~/LiveBench/.venv/bin/python show_livebench_result.py --bench-name live_bench/coding --livebench-release-option 2024-11-25 \
     --model-list DeepSeek-V4.1-Flash-flare-v3-2x > $O/livebench-score.log 2>&1)
  note "LiveBench coding: $(grep -i -A3 'average\|flare-v3-2x' $O/livebench-score.log | tr -s ' ' | tail -3 | tr '\n' ' ')"
fi
if has stress; then   # .11 protocol: 2 waves x 8 x ~530K unique prompts + pinger, KV oversubscribed
  FLARE_LOG=${FLARE_LOG:?set FLARE_LOG to the server log} python3 stress_kv.py flare-v3-2x-c8 \
    $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}') 530000 8 > $O/stress.log 2>&1
  note "stress: $(grep -E 'PASS|FAIL' $O/stress.log | tail -1)"
fi
note "qualification done"
