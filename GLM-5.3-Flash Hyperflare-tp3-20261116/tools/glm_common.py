"""Shared settings for the GLM-5.3-Flash Flare TP3 qualification scripts.

Every value comes from the environment, so one checkout serves any package, port or sampling point:
  FLARE_URL         server base URL (default http://127.0.0.1:15015)
  MODEL             served model name (default GLM-5.3-Flash, the --served-model-name of the release serve script)
  QUAL_TOP_P        top-p for the reasoning stages (default 0.95, the recipe default; the server accepts 0.95-1)
  QUAL_TEMPERATURE  temperature (default 1.0; the server accepts 0.95-1)
  QUAL_EFFORT       reasoning effort of lavd/hotel/retrieval/longgen (default max)
  EVAL_EFFORT       eval-43 effort (default low; GLM cannot turn thinking off)
  TOOLS_EFFORT      tool-call stage effort (default high, as the DS41 suite)
  CKPT              checkpoint directory (tokenizer.json, chat_template.jinja) for local detokenization
  QUAL_OUT          output directory
  QUAL_NO_GPU_CHECKS=1  skip nvidia-smi idle/power checks (CPU tests against the mock server only)

GLM template facts this suite depends on (checked by test_glm_qual.py against the checkpoint template):
  * thinking is always on: the generation prompt ends in "<|assistant|><think>"; enable_thinking is ignored.
  * reasoning_effort renders "Reasoning Effort: Low|High|Max"; any other value (medium, xhigh, none) silently
    renders Max, so this suite refuses anything but low/high/max.
  * Flare returns the think span as message.reasoning_content (stream: delta.reasoning_content) and parses
    <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> into OpenAI tool_calls with
    schema-typed JSON arguments.
"""
import json
import os
import urllib.error
import urllib.request

URL = os.environ.get('FLARE_URL', 'http://127.0.0.1:15015').rstrip('/')
MODEL = os.environ.get('MODEL', 'GLM-5.3-Flash')
EFFORTS = ('low', 'high', 'max')
TOP_P_RANGE = (0.95, 1.0)
TEMPERATURE_RANGE = (0.95, 1.0)


def effort(name, default):
    value = os.environ.get(name, default)
    if value not in EFFORTS:
        raise SystemExit(f'{name}={value!r}: the GLM template only knows {EFFORTS}; anything else renders Max')
    return value


def sampling(name, default, bounds):
    value = float(os.environ.get(name, default))
    if not bounds[0] <= value <= bounds[1]:
        raise SystemExit(f'{name}={value}: outside the recipe range {bounds}; Flare would clamp it silently')
    return value


TOP_P = sampling('QUAL_TOP_P', '0.95', TOP_P_RANGE)
TEMPERATURE = sampling('QUAL_TEMPERATURE', '1.0', TEMPERATURE_RANGE)
QUAL_EFFORT = effort('QUAL_EFFORT', 'max')
EVAL_EFFORT = effort('EVAL_EFFORT', 'low')
TOOLS_EFFORT = effort('TOOLS_EFFORT', 'high')
GPU_CHECKS = os.environ.get('QUAL_NO_GPU_CHECKS') != '1'
GLM_MARKUP = ('<tool_call>', '</tool_call>', '<arg_key>', '<arg_value>', '<think>', '</think>',
              '<|assistant|>', '<|user|>', '<|observation|>', '<|system|>', '<|endoftext|>')


def post(path, body, timeout=900, headers=None):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def get(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as response:
        return response.read().decode()


def count_tokens(messages, **extra):
    """Exact prompt tokens through /v1/messages/count_tokens (Flare has no /tokenize)."""
    body = dict(model=MODEL, max_tokens=1, messages=messages, **extra)
    return post('/v1/messages/count_tokens', body, timeout=600, headers={'anthropic-version': '2023-06-01'})['input_tokens']


def metric(name):
    for line in get('/metrics').splitlines():
        if line.startswith(name + ' ') or line.startswith(name + '{'):
            return float(line.rsplit(' ', 1)[1])
    return None


def leaked_markup(text):
    """GLM control markup that must never reach a parsed field (a parser or template failure)."""
    return [tag for tag in GLM_MARKUP if tag in (text or '')]


def server_cmdline():
    import subprocess
    return subprocess.run(['bash', '-c', "ps -eo args | grep 'target/release/[f]lare run' | head -1"],
                          capture_output=True, text=True).stdout.strip()


def tokenizer_path():
    ckpt = os.environ.get('CKPT')
    if not ckpt:
        raise RuntimeError('set CKPT to the checkpoint directory (its tokenizer.json)')
    return os.path.join(ckpt, 'tokenizer.json')
