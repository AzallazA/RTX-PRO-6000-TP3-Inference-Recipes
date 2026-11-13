#!/usr/bin/env python3
"""Quality and long-context qualification of the running Flare GLM-5.3-Flash TP3 server.

Same protocol as the GLM-5.3 Kraken TP3 (vLLM) releases and the DS41 Flare suite, with GLM's sampling defaults
(thinking always on, reasoning_effort QUAL_EFFORT=max, temperature QUAL_TEMPERATURE=1, top-p QUAL_TOP_P=0.95):
  - llm_decode_bench profiles lavd and hotel-lights: 30 runs each, seed 103, 100,000-token cap,
    at concurrency QC (default 8, the package's sequence limit)
  - long retrieval: three exact markers in 128K- and 900K-token prompts
  - long generation after the captured conversation (CAPTURE, ~530K GLM tokens), seeds 201-204,
    natural EOS, 65,536-token cap, scored for 8-gram repetition (LONGGEN_CLEAR_THINKING=1 adds clear_thinking)
Writes RESULTS.md and the raw outputs into $QUAL_OUT (qual_all.sh sets run-<date>/ next to this file).
"""
import datetime, json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_common as g

D = pathlib.Path(os.environ.get('QUAL_OUT', HERE / f'run-{datetime.datetime.now():%Y%m%d-%H%M}'))
D.mkdir(parents=True, exist_ok=True)
# llm-inference-bench checkout (https://github.com/local-inference-lab/llm-inference-bench).
BENCH = os.environ.get('LLM_BENCH', str(pathlib.Path.home() / 'llm-inference-bench/llm_decode_bench.py'))
PY = sys.executable
TOP_P = str(g.TOP_P)
PORT = g.URL.rsplit(':', 1)[-1]
HOST = g.URL.split('://', 1)[-1].rsplit(':', 1)[0]


def note(m):
    print(m, flush=True)
    with (D / 'RESULTS.md').open('a') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()[:19] + ' — ' + m + '\n')


def profile(name, conc):
    tag = f'{name}-c{conc}'
    cmd = [PY, BENCH, '--host', HOST, '--port', PORT, '--model', g.MODEL,
           '--test-profile', name, '--profile-runs', '30', '--profile-concurrency', str(conc),
           '--max-tokens', os.environ.get('QUAL_MAX_TOKENS', '100000'), '--reasoning-effort', g.QUAL_EFFORT,
           '--completion-stats-top-p', TOP_P, '--completion-stats-temperature', str(g.TEMPERATURE),
           '--completion-stats-seed', '103', '--completion-stats-save-text',
           '--completion-stats-no-prefill-scout', '--respect-eos', '--display-mode', 'plain',
           '--output', str(D / f'{tag}.json')]
    note(f'START {tag}')
    with (D / f'{tag}.log').open('w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, input=b'n\n').returncode
    runs = json.load(open(D / f'{tag}.json'))['runs']
    wrong = [(x['run_index'], x['score_detail'], x['completion_tokens']) for x in runs
             if not x.get('correct') and not x.get('hit_max_tokens')]
    note(f'DONE {tag} rc={rc} | exact {sum(1 for x in runs if x.get("correct"))} / wrong {len(wrong)}'
         f' / truncated {sum(1 for x in runs if x.get("hit_max_tokens"))} | {wrong}')


def judge(r):
    q = r['quality']['content']
    c = r['content']
    return dict(empty=not c.strip(), repeat=q.get('repeat_8gram', 0) > 0.05,
                thin=len(c) < 2000 and r['usage']['completion_tokens'] > 1500,
                markup=bool(r.get('markup_leaks', {}).get('content')),
                not_stopped=r.get('finish_reasons', [None])[-1:] != ['stop'])


def main():
    note(f'model {g.MODEL}, effort {g.QUAL_EFFORT}, temperature {g.TEMPERATURE}, top-p {TOP_P}')
    stages = sys.argv[1:] or ['lavd', 'hotel-lights', 'retrieval', 'longgen']
    (D / 'runtime.json').write_text(json.dumps({'server': 'flare', 'cmdline': g.server_cmdline()}))
    if 'lavd' in stages:
        profile('lavd', int(os.environ.get('QC', '8')))
    if 'hotel-lights' in stages:
        profile('hotel-lights', int(os.environ.get('QC', '8')))
    if 'retrieval' in stages:
        note('START long retrieval 128K/900K')
        with (D / 'long-retrieval.log').open('w') as f:
            rc = subprocess.run([PY, str(HERE / 'long_retrieval.py'), '--label', 'release', '--top-p', TOP_P,
                                 '--targets', *os.environ.get('RETRIEVAL_TARGETS', '128000 900000').split()],
                                stdout=f, stderr=subprocess.STDOUT, timeout=7200).returncode
        note(f'DONE long retrieval rc={rc}: ' + ' '.join(l.split(' {')[0] + (' pass' if '"passed": true' in l else ' FAIL')
             for l in (D / 'long-retrieval.log').read_text().splitlines() if l.startswith('DONE')))
    if 'longgen' in stages:
        rows = []
        clear = os.environ.get('LONGGEN_CLEAR_THINKING') == '1'
        for seed in [int(s) for s in os.environ.get('LONGGEN_SEEDS', '201 202 203 204').split()]:
            tag = f'longgen-{seed}'
            with (D / f'{tag}.log').open('w') as f:
                rc = subprocess.run([PY, str(HERE / 'long_context.py'), '--label', tag, '--natural-eos',
                                     '--top-p', TOP_P, '--seed', str(seed),
                                     '--tokens', os.environ.get('LONGGEN_TOKENS', '65536')]
                                    + (['--clear-thinking'] if clear else []),
                                    stdout=f, stderr=subprocess.STDOUT, timeout=5400).returncode
            if rc:
                note(f'{tag} RUNNER rc={rc}')
                rows.append(dict(seed=seed, fail=True, runner_rc=rc, completion=None, repeat_8gram=None))
                (D / 'long-generation.json').write_text(json.dumps(rows, indent=2))
                continue
            r = json.loads((D / 'outputs' / tag / 'result.json').read_text())
            v = judge(r)
            row = dict(seed=seed, content_chars=len(r['content']), reasoning_chars=len(r['reasoning']),
                       completion=r['usage']['completion_tokens'], prompt=r['usage'].get('prompt_tokens'),
                       finish=r.get('finish_reasons'), repeat_8gram=r['quality']['content'].get('repeat_8gram'),
                       reasoning_repeat_8gram=r['quality']['reasoning'].get('repeat_8gram'),
                       ttr=r['quality']['content'].get('worst_window_ttr'), fail=any(v.values()), **v)
            rows.append(row)
            (D / 'long-generation.json').write_text(json.dumps(rows, indent=2))
            note(json.dumps(row))
    note(f'done (top-p {TOP_P})')


if __name__ == '__main__':
    main()
