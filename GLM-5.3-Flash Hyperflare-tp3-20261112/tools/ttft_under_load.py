#!/usr/bin/env python3
"""TTFT of new prompts while other streams decode, against an OpenAI-compatible server (Flare or vLLM).

Used for the TTFT guard A/B of the Hyperflare-tp3 release (qualification/results/ttft-ab-20261110). Stdlib only (threads + http.client).

  probe  N decoder streams (2K cached prompt each, ignore_eos, re-issued back to back); every INTERVAL +- JITTER s
         a probe = shared cached system prefix + unique ~1.9K-token user text, max_tokens 16, thinking off.
         Records probe TTFT (first content/reasoning delta), decoder tok/s in arrival-free windows vs during probe
         prefill, max inter-chunk gaps, idle-server probe TTFT and /metrics deltas.
  agent  N streams loop turns: system prefix + growing history + a new ~1.9K-token "tool result", output capped at
         ~600 tokens. Warm-up, then a measured window: turns finished, TTFT p50/p90/max, output tok/s.

  python3 ttft_under_load.py probe --streams 3 --output r.json
  python3 ttft_under_load.py agent --streams 4 --warmup 30 --duration 300 --output r.json
"""
import argparse
import http.client
import json
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

METRIC_PREFIXES = ("flare_graph_launches_total", "flare_prefill_tokens_total", "flare_generated_tokens_total",
                   "flare_speculative_", "flare_verification_", "flare_active_requests")


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(-(-p * len(ordered) // 1)) - 1))
    return ordered[index]


def summary(values):
    return {"n": len(values), "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "max": max(values) if values else None, "mean": statistics.fmean(values) if values else None}


class Words:
    """Random pseudo-words; ~1.3 tokens per word for BPE vocabularies (the server's usage reports the truth)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        syllables = ["ka", "lo", "mi", "ren", "tas", "vo", "qui", "bar", "nel", "sor", "dun", "fe", "gri", "hal",
                     "jo", "pex", "ru", "sta", "tin", "ub", "wex", "yal", "zor", "cle", "dro", "mon", "pra", "vel"]
        self.vocab = ["".join(self.rng.choice(syllables) for _ in range(self.rng.randint(1, 3))) for _ in range(4000)]

    def text(self, tokens):
        words = max(1, int(tokens / 1.3))
        out = [self.rng.choice(self.vocab) for _ in range(words)]
        for i in range(12, len(out), 13):
            out[i] += "."
        return " ".join(out)


class Client:
    def __init__(self, args):
        self.host, self.port, self.args = args.host, args.port, args
        self.model = args.model or self.models()[0]

    def get(self, path, timeout=10):
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}", timeout=timeout) as response:
            return response.read().decode()

    def models(self):
        return [m["id"] for m in json.loads(self.get("/v1/models"))["data"]]

    def metrics(self):
        try:
            text = self.get("/metrics")
        except Exception as error:  # vLLM and Flare expose different names; missing metrics are not fatal
            return {"error": str(error)}
        values = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.startswith(METRIC_PREFIXES):
                continue
            name, _, value = line.rpartition(" ")
            try:
                values[name] = values.get(name, 0.0) + float(value)
            except ValueError:
                pass
        return values

    def stream(self, messages, max_tokens, ignore_eos=False, stop=None):
        """Streams one chat completion; returns timing and token data. `stop` aborts between chunks."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "temperature": self.args.temperature, "stream_options": {"include_usage": True}}
        if ignore_eos:
            body["ignore_eos"] = True
        if self.args.template_kwargs:  # e.g. vLLM DS41: '{"thinking":false}' (server default thinking=true wins otherwise)
            body["chat_template_kwargs"] = json.loads(self.args.template_kwargs)
        elif not self.args.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}  # DeepSeek frontends reject "thinking"
        result = {"start": time.time(), "first": None, "end": None, "chunks": [], "text": "", "usage": None,
                  "error": None}
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.args.request_timeout)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(body),
                               {"Content-Type": "application/json", "Accept": "text/event-stream"})
            response = connection.getresponse()
            if response.status != 200:
                result["error"] = f"HTTP {response.status}: {response.read()[:300]!r}"
                return result
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("usage"):
                    result["usage"] = event["usage"]
                now = time.time()
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece:
                        if result["first"] is None:
                            result["first"] = now
                        result["chunks"].append(now)
                        if delta.get("content"):
                            result["text"] += delta["content"]
                if stop is not None and stop.is_set():
                    break
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            connection.close()
            result["end"] = time.time()
        return result


def serve_info(args):
    info = {"host": args.host, "port": args.port}
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
        def server(parts):
            flare = len(parts) > 2 and parts[0].split("/")[-1] == "flare" and parts[1] == "run"
            vllm = any(p.split("/")[-1] == "vllm" for p in parts[:2]) and "serve" in parts[:3] \
                or "vllm.entrypoints.openai.api_server" in parts[:4]
            return flare or vllm
        servers = [line for line in ps.splitlines() if server(line.split())]
        info["server_cmdline"] = servers[:2]
        flare = [line.split()[2] for line in servers if line.split()[1:2] == ["run"]]
        info["package"] = flare[0] if flare else None
    except Exception as error:
        info["ps_error"] = str(error)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit,clocks.max.graphics",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        info["gpus"] = [line.strip() for line in smi.splitlines() if line.strip()]
    except Exception as error:
        info["nvidia_smi_error"] = str(error)
    return info


def ttft(result):
    return None if result["first"] is None else result["first"] - result["start"]


def chunk_tokens(result):
    """(timestamp, tokens) per content chunk, spreading the reported completion tokens over the chunks."""
    chunks = result["chunks"]
    if not chunks:
        return []
    total = (result["usage"] or {}).get("completion_tokens") or len(chunks)
    return [(t, total / len(chunks)) for t in chunks]


def window_rate(events, windows):
    duration = sum(b - a for a, b in windows)
    tokens = sum(n for t, n in events if any(a <= t < b for a, b in windows))
    return tokens / duration if duration > 0 else None


def complement(windows, start, end):
    out, cursor = [], start
    for a, b in sorted(windows):
        a, b = max(a, start), min(b, end)
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        out.append((cursor, end))
    return out


def run_probe(args, client, words):
    prefix = words.text(args.prefix_tokens)
    system = {"role": "system", "content": "Reference notes follow. " + prefix}
    log(f"model {client.model}: warming the shared {args.prefix_tokens}-token prefix")
    for _ in range(2):
        warm = client.stream([system, {"role": "user", "content": "Say ok."}], 1)
        if warm["error"]:
            sys.exit(f"warm-up failed: {warm['error']}")

    def probe():
        messages = [system, {"role": "user", "content": "Summarize in one line: " + words.text(args.probe_tokens)}]
        return client.stream(messages, args.probe_max_tokens)

    idle = []
    for _ in range(args.idle_probes):
        result = probe()
        idle.append(ttft(result))
        time.sleep(0.5)
    log(f"idle probe TTFT: {[round(t, 3) for t in idle if t is not None]}")

    stop = threading.Event()
    streams = [{"requests": [], "prompt": [{"role": "user", "content": f"Stream {i}. Continue this text forever: "
                                            + words.text(args.decoder_prompt_tokens)}]} for i in range(args.streams)]

    def decoder(stream):
        while not stop.is_set():
            result = client.stream(stream["prompt"], args.decoder_max_tokens, ignore_eos=True, stop=stop)
            stream["requests"].append(result)
            if result["error"] and not stop.is_set():
                log(f"decoder error: {result['error']}")
                time.sleep(1)

    threads = [threading.Thread(target=decoder, args=(s,), daemon=True) for s in streams]
    for thread in threads:
        thread.start()
    log(f"{args.streams} decoder streams started; warm-up {args.warmup} s")
    time.sleep(args.warmup)
    metrics_start, start = client.metrics(), time.time()
    rng = random.Random(args.seed + 1)
    probes = []
    while time.time() - start < args.duration:
        time.sleep(max(0.0, args.interval + rng.uniform(-args.jitter, args.jitter)))
        result = probe()
        probes.append(result)
        log(f"probe {len(probes)}: TTFT {ttft(result) if result['first'] else 'none'} "
            f"prompt {((result['usage'] or {}).get('prompt_tokens'))} {result['error'] or ''}")
    end = time.time()
    metrics_end = client.metrics()
    stop.set()
    for thread in threads:
        thread.join(timeout=args.request_timeout)

    # Probe-affected windows: arrival to first token (prefill), and arrival to probe end (whole probe).
    prefill_windows = [(p["start"], p["first"] or p["end"]) for p in probes]
    probe_windows = [(p["start"], p["end"] + args.settle) for p in probes]
    events = [e for s in streams for r in s["requests"] for e in chunk_tokens(r)]
    events = [e for e in events if start <= e[0] < end]
    gaps, gaps_during = [], []
    for s in streams:
        for r in s["requests"]:
            chunks = [t for t in r["chunks"] if start <= t < end]
            for a, b in zip(chunks, chunks[1:]):
                gaps.append(b - a)
                if any(pa <= b and a <= pb for pa, pb in prefill_windows):
                    gaps_during.append(b - a)
    ttfts = [ttft(p) for p in probes if p["first"] is not None]
    return {
        "idle_probe_ttft_s": idle,
        "probe_ttft_s": summary(ttfts),
        "probe_errors": [p["error"] for p in probes if p["error"]],
        "probe_prompt_tokens": [(p["usage"] or {}).get("prompt_tokens") for p in probes],
        "probes": [{"arrival": p["start"] - start, "ttft_s": ttft(p), "total_s": p["end"] - p["start"]} for p in probes],
        "decoder_tokens_per_s": {
            "overall": window_rate(events, [(start, end)]),
            "arrival_free": window_rate(events, complement(probe_windows, start, end)),
            "during_probe_prefill": window_rate(events, prefill_windows),
        },
        "decoder_max_gap_s": max(gaps) if gaps else None,
        "decoder_gap_p99_s": percentile(gaps, 0.99),
        "decoder_max_gap_during_probe_prefill_s": max(gaps_during) if gaps_during else None,
        "decoder_requests": sum(len(s["requests"]) for s in streams),
        "decoder_errors": [r["error"] for s in streams for r in s["requests"] if r["error"]][:10],
        "window_s": end - start,
        "metrics": metric_delta(metrics_start, metrics_end),
    }


def run_agent(args, client, words):
    system = {"role": "system", "content": "You are a coding agent. Project notes: " + words.text(args.prefix_tokens)}
    for _ in range(2):
        warm = client.stream([system, {"role": "user", "content": "Say ok."}], 1)
        if warm["error"]:
            sys.exit(f"warm-up failed: {warm['error']}")
    stop = threading.Event()
    begin = time.time()
    start, end = begin + args.warmup, begin + args.warmup + args.duration
    turns, lock = [], threading.Lock()

    def agent(index):
        history, history_tokens, local = [], 0, random.Random(args.seed * 100 + index)
        while not stop.is_set() and time.time() < end:
            tool = words.text(args.turn_tokens + local.randint(-100, 100))
            messages = [system] + history + [{"role": "user", "content": f"Tool result ({index}):\n{tool}\n"
                                                                         "Decide the next step and explain it."}]
            result = client.stream(messages, args.turn_max_tokens, stop=stop)
            usage = result["usage"] or {}
            with lock:
                turns.append({"stream": index, "start": result["start"], "end": result["end"], "ttft_s": ttft(result),
                              "completion_tokens": usage.get("completion_tokens"),
                              "prompt_tokens": usage.get("prompt_tokens"), "error": result["error"]})
            if result["error"]:
                time.sleep(1)
                continue
            history += [messages[-1], {"role": "assistant", "content": result["text"] or "ok"}]
            history_tokens += args.turn_tokens + (usage.get("completion_tokens") or 0)
            if history_tokens > args.max_history_tokens:
                history, history_tokens = [], 0

    threads = [threading.Thread(target=agent, args=(i,), daemon=True) for i in range(args.streams)]
    for thread in threads:
        thread.start()
    log(f"{args.streams} agent streams; warm-up {args.warmup} s, then {args.duration} s")
    time.sleep(max(0.0, start - time.time()))
    metrics_start = client.metrics()
    time.sleep(max(0.0, end - time.time()))
    metrics_end = client.metrics()
    stop.set()
    for thread in threads:
        thread.join(timeout=args.request_timeout)
    counted = [t for t in turns if t["start"] >= start and t["end"] <= end and not t["error"]]
    ttfts = [t["ttft_s"] for t in counted if t["ttft_s"] is not None]
    return {
        "turns_finished": len(counted),
        "turn_ttft_s": summary(ttfts),
        "output_tokens_per_s": sum(t["completion_tokens"] or 0 for t in counted) / args.duration,
        "turn_errors": [t["error"] for t in turns if t["error"]][:10],
        "prompt_tokens": summary([t["prompt_tokens"] for t in counted if t["prompt_tokens"]]),
        "turns": [{**t, "start": t["start"] - start, "end": t["end"] - start} for t in counted],
        "window_s": args.duration,
        "metrics": metric_delta(metrics_start, metrics_end),
    }


def metric_delta(a, b):
    if "error" in a or "error" in b:
        return {"start": a, "end": b}
    return {"delta": {k: b[k] - a.get(k, 0.0) for k in b if b[k] != a.get(k, 0.0)}, "end": b}


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["probe", "agent"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15015)
    parser.add_argument("--model", help="default: first id of /v1/models")
    parser.add_argument("--streams", type=int, help="decoder (probe: default 3) or agent (default 4) streams")
    parser.add_argument("--warmup", type=float, help="seconds before measuring (probe 20, agent 30)")
    parser.add_argument("--duration", type=float, help="measured seconds (probe 120, agent 300)")
    parser.add_argument("--prefix-tokens", type=int, default=16000, help="shared cached system prefix")
    parser.add_argument("--probe-tokens", type=int, default=1900, help="unique probe text per arrival")
    parser.add_argument("--probe-max-tokens", type=int, default=16)
    parser.add_argument("--interval", type=float, default=3.0, help="mean seconds between probe arrivals")
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--idle-probes", type=int, default=3, help="probes on the idle server first")
    parser.add_argument("--decoder-prompt-tokens", type=int, default=2000)
    parser.add_argument("--decoder-max-tokens", type=int, default=2048)
    parser.add_argument("--settle", type=float, default=0.5, help="s after a probe excluded from arrival-free time")
    parser.add_argument("--turn-tokens", type=int, default=1900, help="new tool-result tokens per agent turn")
    parser.add_argument("--turn-max-tokens", type=int, default=600)
    parser.add_argument("--max-history-tokens", type=int, default=100000, help="agent conversation restarts beyond")
    parser.add_argument("--thinking", action="store_true", help="leave thinking on (default: off)")
    parser.add_argument("--template-kwargs", help="JSON chat_template_kwargs sent instead of the --thinking default "
                        "(vLLM DS41 needs '{\"thinking\":false}': its server default thinking=true overrides enable_thinking)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=None, help="default: time-based, so prompts never hit the cache")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", help="JSON result file")
    args = parser.parse_args()
    probe = args.mode == "probe"
    args.streams = args.streams or (3 if probe else 4)
    args.warmup = args.warmup if args.warmup is not None else (20 if probe else 30)
    args.duration = args.duration if args.duration is not None else (120 if probe else 300)
    args.seed = args.seed if args.seed is not None else int(time.time())
    client = Client(args)
    words = Words(args.seed)
    info = serve_info(args)
    result = (run_probe if probe else run_agent)(args, client, words)
    output = {"mode": args.mode, "label": args.label, "model": client.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "args": vars(args), "serve": info, **result}
    if probe:
        t, d = result["probe_ttft_s"], result["decoder_tokens_per_s"]
        line = (f"probe c{args.streams + 1} TTFT p50/p90/max {t['p50']}/{t['p90']}/{t['max']} s (idle "
                f"{percentile([x for x in result['idle_probe_ttft_s'] if x], 0.5)}); decoder tok/s free/during "
                f"{d['arrival_free']}/{d['during_probe_prefill']}; max gap {result['decoder_max_gap_s']} s")
    else:
        t = result["turn_ttft_s"]
        line = (f"agent c{args.streams}: {result['turns_finished']} turns in {args.duration:.0f} s, TTFT p50/p90/max "
                f"{t['p50']}/{t['p90']}/{t['max']} s, {result['output_tokens_per_s']:.1f} output tok/s")
    output["summary_line"] = line
    print(line)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=1, default=str)


if __name__ == "__main__":
    main()
