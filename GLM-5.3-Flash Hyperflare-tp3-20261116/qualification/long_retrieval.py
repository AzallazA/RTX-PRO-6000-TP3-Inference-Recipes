"""Natural-EOS long-context retrieval with exact known answers at three positions (GLM-5.3-Flash on Flare).

Same markers, filler and targets (128K / 900K prompt tokens) as the DS41 suite and the GLM vLLM Kraken releases.
GLM thinks before answering (effort QUAL_EFFORT, default max; the vLLM GLM run used ~90 reasoning tokens at 128K),
so only message.content is parsed; reasoning_content is kept in response.json. A fenced ```json block counts as a
pass (strict_json false), as before.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_common as g

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('QUAL_OUT', HERE)) / 'long-retrieval'
unit = 'This ordinary archived entry contains no requested marker.\n'
expected = {'A': 'ALPHA-R34-7K2M9Q', 'B': 'BETA-R34-8P6V3X', 'C': 'GAMMA-R34-4T9N2Z'}


def wait_idle():
    """Three consecutive 5 s samples with no running or waiting requests."""
    streak = 0
    while streak < 3:
        rows = [s for s in g.get('/metrics').splitlines()
                if s.startswith(('flare_active_requests', 'flare_pending_requests'))]
        clean = len(rows) >= 2 and all(float(s.rsplit(' ', 1)[1]) == 0 for s in rows)
        streak = streak + 1 if clean else 0
        if streak < 3:
            time.sleep(float(os.environ.get('QUAL_IDLE_POLL_S', '5')))


def messages(n):
    text = ('Read this archive and retrieve its three marker values.\n'
            'Marker A = ' + expected['A'] + '\n' + unit * (n//2)
            + 'Marker B = ' + expected['B'] + '\n' + unit * (n-n//2)
            + 'Marker C = ' + expected['C'] + '\n'
            'Return only a JSON object with keys A, B, C and their exact marker values.')
    return [{'role': 'user', 'content': text}]


def parse_answer(text):
    """Strict JSON first, then a single fenced block; returns (value or None, strict)."""
    try:
        return json.loads(text), True
    except ValueError:
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', text.strip(), re.DOTALL)
        try:
            return (json.loads(fenced.group(1)) if fenced else None), False
        except ValueError:
            return None, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--top-p', type=float, default=g.TOP_P)
    parser.add_argument('--targets', nargs='+', type=int, default=[128000, 900000])
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('RETRIEVAL_MAX_TOKENS', '2048')))
    args = parser.parse_args()
    out_root = OUT / args.label
    out_root.mkdir(parents=True, exist_ok=False)
    failed = False
    for target in args.targets:
        wait_idle()
        n = target // 10
        attempts = []
        for _ in range(8):
            msgs = messages(n)
            count = g.count_tokens(msgs)
            attempts.append({'records': n, 'tokens': count})
            if abs(count-target) < 100:
                break
            n = max(1, round(n * target/count))
        assert abs(count-target) < 100, attempts
        body = dict(model=g.MODEL, messages=msgs, temperature=g.TEMPERATURE, top_p=args.top_p,
            seed=target, min_tokens=0, ignore_eos=False, max_completion_tokens=args.max_tokens,
            repetition_penalty=1, tool_choice='none', cache_salt=f'glm53-{args.label}-retrieval-{target}',
            reasoning_effort=g.QUAL_EFFORT)
        request_bytes = json.dumps(body).encode()
        out = out_root / str(target)
        out.mkdir(exist_ok=False)
        (out/'manifest.json').write_text(json.dumps(dict(target=target,
            request_sha256=hashlib.sha256(request_bytes).hexdigest(), attempts=attempts,
            settings={k: v for k, v in body.items() if k != 'messages'}, expected=expected,
            runtime={'server': 'flare', 'cmdline': g.server_cmdline()}), indent=2))
        print('START', target, 'actual prompt', count, flush=True)
        response = g.post('/v1/chat/completions', body, timeout=1800)
        (out/'response.json').write_text(json.dumps(response, indent=2))
        choice = response['choices'][0]
        text = choice['message'].get('content') or ''
        actual, strict_json = parse_answer(text)
        passed = (actual == expected and choice['finish_reason'] == 'stop'
                  and not choice['message'].get('tool_calls')
                  and response['usage']['prompt_tokens'] == count)
        receipt = dict(passed=passed, strict_json=strict_json, actual=actual, expected=expected, usage=response['usage'],
                       finish_reason=choice['finish_reason'],
                       reasoning_chars=len(choice['message'].get('reasoning_content') or ''))
        (out/'result.json').write_text(json.dumps(receipt, indent=2))
        print('DONE', target, json.dumps(receipt), flush=True)
        failed |= not passed
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
