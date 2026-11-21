#!/usr/bin/env python3
"""Proxy that adds "steering_scale" to every generation request (stdlib only).

For clients that cannot add a request field (for example llm_decode_bench's GPQA profile): point the client at
127.0.0.1:LISTEN and every POST to /v1/chat/completions or /v1/completions whose body is a JSON object is forwarded
to the server with "steering_scale": S at the top level (and in chat_template_kwargs when the client put one there).
Every other request passes through unchanged.

  steer_proxy.py --listen PORT [--upstream PORT] [--upstream-host HOST] --scale S [--log FILE] [--ready FILE]
  steer_proxy.py --self-test

  --upstream / --upstream-host   the server (default 127.0.0.1:15015); the proxy itself listens on 127.0.0.1 only
  --steering-key-file FILE       send X-Steering-Key with the first key in FILE (one key per line or comma-separated,
                                 '#' starts a comment line) when the server has steering keys; the key is never logged

One request per connection: the upstream request carries "Connection: close", the response gets "Connection: close",
the body is re-sent with its new Content-Length, and response bytes are relayed as they arrive (SSE streams keep their
timing). Every connection is its own asyncio task, so concurrent streams do not wait on each other. SIGTERM / SIGINT
stop it. The ready file is written once the socket listens.

--log lines: 'REWRITE <method> <path> steering_scale <value sent> (client sent <value|absent>)' before each rewritten
request is forwarded, and after every event the counters
  'rewritten N passthrough M errors E disconnects D idle I completed C'
  rewritten / passthrough = requests forwarded with / without the field;
  disconnects = the client closed its connection before the response ended (the upstream connection is closed too;
    benchmarks do this at the end of a timed cell), logged 'DISCONNECT <path> after <bytes> bytes'; not an error;
  idle = a client connection closed before sending a request head; not an error;
  errors = everything else, each logged 'ERROR <phase> <method> <path>: <ExceptionType>: <message>'.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
import time

STEERED_PATHS = ("/v1/chat/completions", "/v1/completions")
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "content-length", "te", "upgrade"}


class Stats:
    KEYS = ("rewritten", "passthrough", "errors", "disconnects", "idle", "completed")

    def __init__(self, log):
        for k in self.KEYS:
            setattr(self, k, 0)
        self.log = log

    def line(self):
        return " ".join(f"{k} {getattr(self, k)}" for k in self.KEYS)

    def write(self, text):
        if self.log:
            with open(self.log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    def note(self, what, detail=None):
        setattr(self, what, getattr(self, what) + 1)
        if detail:
            self.write(detail[:600])
        self.write(self.line())


CLIENT_GONE = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


async def read_head(reader):
    data = await reader.readuntil(b"\r\n\r\n")
    lines = data[:-4].decode("latin-1").split("\r\n")
    return lines[0], [tuple(x.split(":", 1)) for x in lines[1:] if ":" in x]


async def read_body(reader, headers):
    h = {k.strip().lower(): v.strip() for k, v in headers}
    if "chunked" in h.get("transfer-encoding", "").lower():
        out = b""
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await reader.readuntil(b"\r\n")) != b"\r\n":
                    pass
                return out
            out += await reader.readexactly(size)
            await reader.readexactly(2)
    n = int(h.get("content-length", "0") or 0)
    return await reader.readexactly(n) if n else b""


def rewrite(path, body, scale):
    """(new body, rewritten?) for one request body"""
    if path.split("?")[0] not in STEERED_PATHS or not body:
        return body, False
    try:
        obj = json.loads(body)
    except ValueError:
        return body, False
    if not isinstance(obj, dict):
        return body, False
    obj["steering_scale"] = scale
    if isinstance(obj.get("chat_template_kwargs"), dict) and "steering_scale" in obj["chat_template_kwargs"]:
        obj["chat_template_kwargs"]["steering_scale"] = scale
    return json.dumps(obj).encode(), True


async def handle(reader, writer, upstream, scale, stats, host="127.0.0.1", key=None):
    up_w = None
    phase, method, path, sent = "request head", "-", "-", 0
    try:
        try:
            start, headers = await read_head(reader)
        except asyncio.IncompleteReadError as e:
            if not e.partial.strip():
                stats.note("idle")
                return
            raise
        method, path = start.split(" ")[:2]
        phase = "request body"
        body = await read_body(reader, headers)
        prev = "-"
        if method == "POST" and path.split("?")[0] in STEERED_PATHS and body:
            try:
                o = json.loads(body)
                prev = repr(o["steering_scale"]) if isinstance(o, dict) and "steering_scale" in o else "absent"
            except ValueError:
                prev = "non-JSON"
        body, changed = rewrite(path, body, scale) if method == "POST" else (body, False)
        keep = [(k.strip(), v.strip()) for k, v in headers if k.strip().lower() not in HOP]
        head = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in keep]
        head += [f"Content-Length: {len(body)}"] if body or method in ("POST", "PUT", "PATCH") else []
        if key and changed:
            head = [x for x in head if not x.lower().startswith("x-steering-key:")] + [f"X-Steering-Key: {key}"]
        head += ["Connection: close"]
        phase = "upstream connect"
        up_r, up_w = await asyncio.open_connection(host, upstream)
        up_w.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
        await up_w.drain()
        if changed:
            stats.write(f"REWRITE {method} {path.split('?')[0]} steering_scale {json.loads(body).get('steering_scale')!r} (client sent {prev})")
        stats.note("rewritten" if changed else "passthrough")
        phase = "upstream response head"
        status, rheaders = await read_head(up_r)
        rkeep = [(k.strip(), v.strip()) for k, v in rheaders if k.strip().lower() not in ("connection", "keep-alive")]
        phase = "relay"
        try:
            writer.write(("\r\n".join([status] + [f"{k}: {v}" for k, v in rkeep] + ["Connection: close"]) + "\r\n\r\n").encode("latin-1"))
            await writer.drain()
        except CLIENT_GONE:
            stats.note("disconnects", f"DISCONNECT {method} {path} before the response head")
            return
        while True:
            chunk = await up_r.read(65536)   # an upstream failure here is an error
            if not chunk:
                break
            if writer.transport.is_closing() or reader.at_eof():
                stats.note("disconnects", f"DISCONNECT {method} {path} after {sent} bytes (client EOF)")
                return
            try:
                writer.write(chunk)
                await writer.drain()
            except CLIENT_GONE:
                stats.note("disconnects", f"DISCONNECT {method} {path} after {sent} bytes")
                return
            sent += len(chunk)
        stats.note("completed")
    except Exception as e:  # noqa: BLE001 - counted and logged with its exception
        stats.note("errors", f"ERROR {phase} {method} {path}: {type(e).__name__}: {e}")
    finally:
        for w in (up_w, writer):
            if w is not None:
                try:
                    w.close()
                except Exception:  # noqa: BLE001
                    pass


async def serve(listen, upstream, scale, log, ready, host="127.0.0.1", key=None):
    stats = Stats(log)
    server = await asyncio.start_server(lambda r, w: handle(r, w, upstream, scale, stats, host, key), "127.0.0.1", listen)
    if ready:
        with open(ready, "w") as f:
            f.write(f"listening {listen} upstream {host}:{upstream} scale {scale} pid {os.getpid()}\n")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()
    if log:
        stats.write("STOP " + stats.line())
    return stats


# ---- self-test: a fake upstream (echo of the received body, SSE-style chunked reply) behind the proxy -----------------
async def _fake_upstream(reader, writer):
    start, headers = await read_head(reader)
    body = await read_body(reader, headers)
    method, path = start.split(" ")[:2]
    h = {k.strip().lower(): v.strip() for k, v in headers}
    if path == "/slow":
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        try:
            for i in range(200):
                part = b"data: tok\n\n"
                writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.01)
        except CLIENT_GONE:
            _UP_ABORTS.append(1)
        writer.close()
        return
    if path == "/metrics":
        payload = b"flare_total_kv_tokens 42\n"
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    else:
        echo = json.dumps({"path": path, "body": json.loads(body) if body else None, "conn": h.get("connection"), "key": h.get("x-steering-key")}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n")
        for part in (b"data: " + echo + b"\n\n", b"data: [DONE]\n\n"):
            writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            await writer.drain()
            await asyncio.sleep(0.01)
        writer.write(b"0\r\n\r\n")
    await writer.drain()
    writer.close()


_UP_ABORTS = []


async def _drop_client(port, after):
    """POST /slow (rewritten path? no: /slow passes through) then close after reading `after` bytes of the stream"""
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(b"POST /slow HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
    await w.drain()
    await r.read(after)
    w.close()


async def _client(port, method, path, body=None, chunked=False):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\nContent-Type: application/json\r\n"
    if chunked:
        w.write((head + "Transfer-Encoding: chunked\r\n\r\n").encode() + f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
    else:
        w.write((head + f"Content-Length: {len(data)}\r\n\r\n").encode() + data)
    await w.drain()
    raw = await r.read()
    w.close()
    return raw


async def _self_test():
    ok = True
    up = await asyncio.start_server(_fake_upstream, "127.0.0.1", 0)
    up_port = up.sockets[0].getsockname()[1]
    tmp = tempfile.mkdtemp(prefix="steer-proxy-selftest-")
    log = os.path.join(tmp, "proxy.log")
    stats = Stats(log)
    px = await asyncio.start_server(lambda r, w: handle(r, w, up_port, 1.0, stats), "127.0.0.1", 0)
    port = px.sockets[0].getsockname()[1]

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("ok   " if cond else "FAIL ") + name)

    raw = await _client(port, "POST", "/v1/chat/completions", {"model": "m", "messages": [], "stream": True})
    head, _, rest = raw.partition(b"\r\n\r\n")
    check("response head says Connection: close (keep-alive dropped)", b"Connection: close" in head and b"keep-alive" not in head)
    check("chunked SSE body piped unchanged (two events + terminator)", rest.count(b"data: ") == 2 and rest.endswith(b"0\r\n\r\n"))
    echo = json.loads(rest.split(b"data: ")[1].split(b"\n\n")[0])
    check("chat body got steering_scale 1.0", echo["body"].get("steering_scale") == 1.0 and echo["body"]["model"] == "m")
    check("upstream request had Connection: close", echo["conn"] == "close")
    raw = await _client(port, "POST", "/v1/completions", {"prompt": [1, 2], "steering_scale": 0.0,
                                                          "chat_template_kwargs": {"steering_scale": 0.0}}, chunked=True)
    echo = json.loads(raw.partition(b"\r\n\r\n")[2].split(b"data: ")[1].split(b"\n\n")[0])
    check("chunked request body decoded; explicit 0.0 overridden at top level and in chat_template_kwargs",
          echo["body"]["steering_scale"] == 1.0 and echo["body"]["chat_template_kwargs"]["steering_scale"] == 1.0)
    raw = await _client(port, "POST", "/v1/embeddings", {"input": "x"})
    echo = json.loads(raw.partition(b"\r\n\r\n")[2].split(b"data: ")[1].split(b"\n\n")[0])
    check("other POST paths pass through unchanged", "steering_scale" not in echo["body"])
    raw = await _client(port, "GET", "/metrics")
    check("GET /metrics passes through", raw.endswith(b"flare_total_kv_tokens 42\n"))
    results = await asyncio.gather(*[_client(port, "POST", "/v1/completions", {"prompt": [i]}) for i in range(16)])
    check("16 concurrent requests all rewritten", all(b'"steering_scale": 1.0' in x for x in results))
    check("counters: rewritten 18, passthrough 2, errors 0, completed 20", (stats.rewritten, stats.passthrough, stats.errors, stats.completed) == (18, 2, 0, 20))
    check("log file written", os.path.exists(log) and "rewritten 18" in open(log).read().splitlines()[-1])
    rw = [x for x in open(log) if " REWRITE " in x]
    check("18 REWRITE lines, each 'steering_scale 1.0'; the explicit 0.0 one says 'client sent 0.0'",
          len(rw) == 18 and all("steering_scale 1.0 (client sent " in x for x in rw) and sum("client sent 0.0)" in x for x in rw) == 1)
    # 16 concurrent streams dropped by the client mid-stream (the bench at a cell end): disconnects, not errors; upstream closed
    await asyncio.gather(*[_drop_client(port, 300) for _ in range(16)])
    for _ in range(300):
        if stats.disconnects >= 16 and len(_UP_ABORTS) >= 16:
            break
        await asyncio.sleep(0.02)
    check("16 mid-stream client drops -> disconnects 16, errors 0", (stats.disconnects, stats.errors) == (16, 0))
    check("the upstream side of every dropped stream was closed", len(_UP_ABORTS) >= 16)
    check("each drop logged as DISCONNECT", sum(1 for x in open(log) if " DISCONNECT POST /slow" in x) == 16)
    r_, w_ = await asyncio.open_connection("127.0.0.1", port)
    w_.close()
    for _ in range(100):
        if stats.idle:
            break
        await asyncio.sleep(0.02)
    check("connection closed before a request head -> idle 1, errors 0", (stats.idle, stats.errors) == (1, 0))
    dead = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    dead_port = dead.sockets[0].getsockname()[1]
    dead.close()
    await dead.wait_closed()
    px2 = await asyncio.start_server(lambda r, w: handle(r, w, dead_port, 1.0, stats), "127.0.0.1", 0)
    await _client(px2.sockets[0].getsockname()[1], "POST", "/v1/completions", {"prompt": [1]})
    check("upstream down -> errors 1, logged with its exception", stats.errors == 1 and any("ERROR upstream connect POST /v1/completions: ConnectionRefusedError" in x for x in open(log)))
    px2.close()
    px3 = await asyncio.start_server(lambda r, w: handle(r, w, up_port, 1.0, stats, "127.0.0.1", "sk-test"), "127.0.0.1", 0)
    r3, w3 = await asyncio.open_connection("127.0.0.1", px3.sockets[0].getsockname()[1])
    w3.write(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nX-Steering-Key: old\r\nContent-Length: 2\r\n\r\n{}")
    await w3.drain()
    raw = await r3.read()
    w3.close()
    echo = json.loads(raw.partition(b"\r\n\r\n")[2].split(b"data: ")[1].split(b"\n\n")[0])
    check("key file: X-Steering-Key replaced by the file's key, key not in the log", echo["key"] == "sk-test" and "sk-test" not in open(log).read())
    px3.close()
    px.close()
    up.close()
    print("steer_proxy self-test " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def read_key(path):
    for line in open(path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(",")[0].strip()
    raise SystemExit("steer_proxy: key file holds no key")


def main():
    if sys.argv[1:] == ["--self-test"]:
        return asyncio.run(_self_test())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", type=int, default=15015)
    p.add_argument("--upstream-host", default="127.0.0.1")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--steering-key-file")
    p.add_argument("--log")
    p.add_argument("--ready")
    a = p.parse_args()
    if not 0.0 <= a.scale <= 1.0:
        p.error("scale must be in [0, 1]")
    if a.listen == a.upstream and a.upstream_host in ("127.0.0.1", "localhost"):
        p.error("listen and upstream ports must differ")
    key = read_key(a.steering_key_file) if a.steering_key_file else None
    asyncio.run(serve(a.listen, a.upstream, a.scale, a.log, a.ready, a.upstream_host, key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
