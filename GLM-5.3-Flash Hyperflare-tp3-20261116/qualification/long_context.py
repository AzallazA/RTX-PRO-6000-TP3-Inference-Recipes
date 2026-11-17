"""Natural-EOS long generation after a long captured conversation; preserves every SSE event and both channels.

GLM-5.3-Flash on Flare. Differences from the DS41 script:
  * the conversation comes from CAPTURE (the sanitized captured request shipped in the DS41 recipes folder,
    captured-request.san.json: originally a GLM-5.3-Flash agent session, 1,428 messages). Under the GLM template it
    is ~530K prompt tokens (~501K with clear_thinking), not the 581,537 DS41 tokens, and not the 826K capture of the
    vLLM GLM releases.
  * reasoning arrives as delta.reasoning_content; content and reasoning are scored separately.
  * --clear-thinking sends chat_template_kwargs.clear_thinking=true (drop reasoning of turns before the last user
    message, the vLLM Kraken release default); Flare's GLM template default is false.
  * the vLLM-only switches of the old script (--trace via /collective_rpc, --ban-tool-start via bad_words) are gone.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_common as g

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get('QUAL_OUT', ROOT)) / 'outputs'
spec = importlib.util.spec_from_file_location("quality", ROOT / "analyze_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
ASK = ('Ignore any pending tool work. Using the full conversation above, write '
       'an exhaustive technical retrospective of at least 6000 words. Cover: '
       '(1) every distinct problem investigated and its resolution, '
       '(2) the systems and components involved and how they interact, '
       '(3) a chronological narrative of how understanding evolved, '
       '(4) unresolved issues and concrete next steps, and '
       '(5) lessons learned. Use detailed prose with headed sections. Be '
       'thorough and specific; do not summarise briefly.')


def idle():
    streak = 0
    while streak < 3:
        rows = [s for s in g.get('/metrics').splitlines() if s.startswith(
            ('flare_active_requests', 'flare_pending_requests'))]
        clean = len(rows) >= 2 and all(float(s.rsplit(' ', 1)[1]) == 0 for s in rows)
        if g.GPU_CHECKS:
            gpu = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits"], text=True)
            clean &= all(float(v) == 0 for s in gpu.splitlines() for v in s.split(',')[1:])
        streak = streak + 1 if clean else 0
        if streak < 3:
            time.sleep(float(os.environ.get('QUAL_IDLE_POLL_S', '5')))


def build_body(capture, a):
    body = json.loads(Path(capture).read_text())
    body.pop('tools', None)
    body.pop('tool_choice', None)
    body['messages'] = (body['messages'] if a.context == 'full' else []) + [{'role': 'user', 'content': ASK}]
    body.update(model=g.MODEL, temperature=a.temperature, top_p=a.top_p,
                seed=a.seed, max_completion_tokens=a.tokens,
                min_tokens=0, ignore_eos=False, repetition_penalty=1.0,
                tool_choice='none', stream=True,
                stream_options={'include_usage': True, 'continuous_usage_stats': True},
                cache_salt='glm53-qualification-' + a.label,
                reasoning_effort=a.effort, return_token_ids=True)
    if a.clear_thinking:
        body['chat_template_kwargs'] = {'clear_thinking': True}
    return body


def read_stream(lines, label, rawlog=None):
    """Accumulate an OpenAI chat SSE stream into channels, usage, finishes, tool calls and token ids."""
    channels = {'content': [], 'reasoning': []}
    state = dict(usage=None, finishes=[], output_ids=[], stop_reasons=[], tool_calls={},
                 first_tool_completion=None, done=False, first=None)
    start = last = time.monotonic()
    for raw in lines:
        if rawlog is not None:
            rawlog.write(raw)
        if not raw.startswith(b'data: '):
            continue
        if raw.strip() == b'data: [DONE]':
            state['done'] = True
            continue
        event = json.loads(raw[6:])
        if event.get('error'):
            raise RuntimeError(event['error'])
        state['usage'] = event.get('usage') or state['usage']
        for choice in event.get('choices', []):
            state['output_ids'].extend(choice.get('token_ids') or [])
            if choice.get('stop_reason') is not None:
                state['stop_reasons'].append(choice['stop_reason'])
            delta = choice.get('delta') or {}
            for tc in delta.get('tool_calls') or []:
                if state['first_tool_completion'] is None and state['usage']:
                    state['first_tool_completion'] = state['usage'].get('completion_tokens')
                entry = state['tool_calls'].setdefault(tc['index'], {'name': '', 'arguments': ''})
                for field in ('name', 'arguments'):
                    entry[field] += (tc.get('function') or {}).get(field) or ''
            for key, value in [('content', delta.get('content')),
                               ('reasoning', delta.get('reasoning_content') or delta.get('reasoning'))]:
                if value:
                    channels[key].append(value)
                    if state['first'] is None:
                        state['first'] = time.monotonic() - start
            if choice.get('finish_reason'):
                state['finishes'].append(choice['finish_reason'])
        if time.monotonic() - last > 30:
            if rawlog is not None:
                rawlog.flush()
            print('PROGRESS', label, state['usage'], 'chars', {k: sum(map(len, v)) for k, v in channels.items()}, flush=True)
            last = time.monotonic()
    record = {k: ''.join(v) for k, v in channels.items()}
    record.update(usage=state['usage'], finish_reasons=state['finishes'], saw_done=state['done'],
                  wall_s=time.monotonic() - start, first_visible_s=state['first'], tool_calls=state['tool_calls'],
                  first_tool_completion=state['first_tool_completion'], output_token_ids=state['output_ids'],
                  stop_reasons=state['stop_reasons'])
    return record


def score(record):
    channels = ('content', 'reasoning')
    record['quality'] = {k: quality.analyze(record[k]) for k in channels}
    record['flags'] = {k: quality.flags(record['quality'][k]) for k in channels}
    record['markup_leaks'] = {k: g.leaked_markup(record[k]) for k in channels}
    usage = record['usage']
    record['qualifying_length'] = bool(usage and usage.get('completion_tokens', 0) >= 25000
        and sum(len(record[k]) for k in channels) >= 25000
        and (record['first_tool_completion'] is None or record['first_tool_completion'] >= 25000))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--top-p', type=float, default=g.TOP_P)
    p.add_argument('--temperature', type=float, default=g.TEMPERATURE)
    p.add_argument('--effort', default=g.QUAL_EFFORT, choices=g.EFFORTS)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--context', choices=('full', 'none'), default='full')
    p.add_argument('--tokens', type=int, default=65536)
    p.add_argument('--natural-eos', action='store_true')
    p.add_argument('--clear-thinking', action='store_true')
    p.add_argument('--payload', type=Path, default=os.environ.get('CAPTURE'))
    a = p.parse_args()
    assert a.natural_eos, 'only the natural-EOS protocol is qualified'
    if a.payload is None:
        raise SystemExit('set CAPTURE (or --payload) to captured-request.san.json')
    out = OUTPUTS / a.label
    out.mkdir(parents=True, exist_ok=False)
    body = build_body(a.payload, a)
    encoded = json.dumps(body, ensure_ascii=False).encode()
    canonical = {k: v for k, v in body.items() if k not in ('cache_salt', 'seed')}
    canonical_sha256 = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    idle()
    powers = ''
    if g.GPU_CHECKS:
        powers = subprocess.check_output(['nvidia-smi', '--query-gpu=power.limit',
                                          '--format=csv,noheader,nounits'], text=True)
        assert all(float(v) <= 600 for v in powers.splitlines()), powers
    runtime = {'server': 'flare', 'cmdline': g.server_cmdline()}
    (out / 'manifest.json').write_text(json.dumps({
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'request_sha256': hashlib.sha256(encoded).hexdigest(),
        'canonical_without_seed_or_cache_salt_sha256': canonical_sha256,
        'capture_sha256': hashlib.sha256(Path(a.payload).read_bytes()).hexdigest(),
        'settings': {k: v for k, v in body.items() if k != 'messages'},
        'appended_prompt': ASK, 'runtime': runtime, 'power_limits': powers}, indent=2))
    (out / 'before.metrics').write_text(g.get('/metrics'))
    print('START', a.label, flush=True)
    request = urllib.request.Request(g.URL + '/v1/chat/completions', data=encoded,
                                     headers={'Content-Type': 'application/json'})
    with (out / 'stream.sse').open('wb') as rawlog, urllib.request.urlopen(request, timeout=1800) as response:
        record = read_stream(response, a.label, rawlog)
    if record['output_token_ids']:
        # Flare has no /detokenize: decode locally with the checkpoint tokenizer (artifact only; scoring uses the stream).
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(g.tokenizer_path())
            raw_text = {'completion': tok.decode(record['output_token_ids'], skip_special_tokens=False)}
        except Exception as error:  # noqa: BLE001
            raw_text = {'unavailable': f'{type(error).__name__}: {error}'}
        (out/'raw-detokenized.json').write_text(json.dumps(raw_text, ensure_ascii=False, indent=2))
    score(record)
    (out/'result.json').write_text(json.dumps(record, indent=2))
    (out/'after.metrics').write_text(g.get('/metrics'))
    assert {'server': 'flare', 'cmdline': g.server_cmdline()} == runtime, 'Service changed during trial'
    print('DONE', a.label, json.dumps({k: v for k, v in record.items()
        if k not in ('content', 'reasoning', 'tool_calls', 'output_token_ids')}), 'tool_count', len(record['tool_calls']), flush=True)
    assert record['saw_done'] and record['usage'], 'Incomplete stream'


if __name__ == '__main__':
    main()
