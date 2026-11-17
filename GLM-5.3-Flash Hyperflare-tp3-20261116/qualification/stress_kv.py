#!/usr/bin/env python3
"""KV-pool stress for the running Flare server: fill and oversubscribe the KV cache with UNIQUE long prompts
(no shared prefix -> no prefix-cache hits), <conc> concurrent (the release uses 8 = the package's slots), in waves
until >= 1.15x the pool capacity has been pushed through, while a short-prompt pinger runs concurrently
(prefill-during-decode). Samples GPU free memory, then scans the server log ($FLARE_LOG) for OOM / mapping-failure /
preemption lines. Writes stress_<preset>.json into $QUAL_OUT (default: this folder).
GLM: thinking cannot be disabled, so long requests and pings run at reasoning_effort STRESS_EFFORT (default low);
only HTTP success and usage are checked, never the text. Temperature/top-p are the recipe defaults (the DS41 0.7
would be clamped to 0.95). STRESS_LONG_TIMEOUT_S (default 1800) bounds each non-streaming long request (queue wait +
prefill): one wave that oversubscribes GLM's ~12M DCP3 KV is ~25-35 min of prefill, so raise it there. QUAL_NO_GPU_CHECKS=1 skips the nvidia-smi free-memory monitor (mock tests only).
usage: FLARE_LOG=<server log> stress_kv.py <preset> <capacity_tokens> [prompt_tok=600000] [conc=4] [max_out=200]
"""
import json, os, sys, time, subprocess, threading, concurrent.futures, urllib.request, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_common as g
BASE=g.URL
EFFORT=g.effort("STRESS_EFFORT", "low")
PRESET=sys.argv[1]; CAP=int(sys.argv[2])
PTOK=int(sys.argv[3]) if len(sys.argv)>3 else 600000
CONC=int(sys.argv[4]) if len(sys.argv)>4 else 4
MAXOUT=int(sys.argv[5]) if len(sys.argv)>5 else 200
WAVES=max(2, math.ceil(CAP*1.15/(CONC*PTOK)))
WORDS=["ledger","berth","dispatch","archive","repeater","cooling","manifest","quorum","lattice","harbor",
       "procurement","safety","shelter","overnight","summary","handoff","schedule","memo","audit","beacon"]
TOK_PER_WORD=None
def _calibrate():
    """Measure tok/word for our synthetic text once via /v1/messages/count_tokens."""
    global TOK_PER_WORD
    rng=random.Random(1); sample=" ".join(rng.choice(WORDS) for _ in range(5000))
    # Flare: no /tokenize; count via the Anthropic endpoint (includes a few template tokens).
    TOK_PER_WORD=g.count_tokens([{"role":"user","content":sample}])/5000.0
def make_prompt(seed):
    if TOK_PER_WORD is None: _calibrate()
    rng=random.Random(seed); n=int(PTOK/TOK_PER_WORD); parts=[f"SESSION-{seed:08x} packet begins."]; count=0
    while count < n:                      # linear: 12 words per line, running count
        parts.append(" ".join(rng.choice(WORDS) for _ in range(12))); count+=12
    return "\n".join(parts)
LONG_TIMEOUT=float(os.environ.get("STRESS_LONG_TIMEOUT_S", "1800"))   # long requests are non-streaming: queue + prefill
def chat(prompt, max_tokens, timeout=1800):
    body=json.dumps({"model":g.MODEL,"messages":[{"role":"user","content":prompt}],
                     "max_tokens":max_tokens,"temperature":g.TEMPERATURE,"top_p":g.TOP_P,"reasoning_effort":EFFORT}).encode()
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t=time.time()
    try:
        d=json.loads(urllib.request.urlopen(r,timeout=timeout).read()); u=d["usage"]
        return ("OK",u["prompt_tokens"],u["completion_tokens"],time.time()-t,"")
    except Exception as e:
        err=""
        if hasattr(e,"read"):
            try: err=e.read().decode()[:200]
            except Exception: pass
        return ("FAIL",0,0,time.time()-t,f"{type(e).__name__}: {e} {err}")
stop=False; minfree=[10**9]; samples=[0]
def monitor():
    while not stop and g.GPU_CHECKS:
        try:
            o=subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10).stdout
            f=min(int(x) for x in o.split()); minfree[0]=min(minfree[0],f); samples[0]+=1
        except Exception: pass
        time.sleep(2)
pings=[]
def pinger():
    i=0
    while not stop:
        pings.append(chat(f"Ping {i}: reply with the word ok and the number {i}.", 16, timeout=600)[0]); i+=1
        time.sleep(float(os.environ.get("STRESS_PING_INTERVAL_S", "5")))
threading.Thread(target=monitor,daemon=True).start(); threading.Thread(target=pinger,daemon=True).start()
t0=time.time()
import os as _os0
_LOG_START=_os0.path.getsize(_os0.environ['FLARE_LOG']); print(f"[{PRESET}] capacity={CAP:,} tok -> {WAVES} waves x {CONC} x ~{PTOK:,} tok unique prompts (+ pinger)", flush=True)
results=[]
for w in range(WAVES):
    seeds=[(hash((PRESET,w,i,time.time()))&0xffffffff) for i in range(CONC)]
    prompts=[make_prompt(s) for s in seeds]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        res=list(ex.map(lambda p: chat(p,MAXOUT,timeout=LONG_TIMEOUT), prompts))
    ok=sum(1 for r in res if r[0]=="OK"); pt=sum(r[1] for r in res)
    print(f"  wave {w+1}/{WAVES}: {ok}/{CONC} ok, {pt:,} prompt tok, {max(r[3] for r in res):.0f}s, minfree so far {minfree[0]} MiB"
          + "".join(f"\n    FAIL: {r[4]}" for r in res if r[0]!="OK"), flush=True)
    results+=res
stop=True; time.sleep(3)
health=urllib.request.urlopen(f"{BASE}/health",timeout=10).status
import os as _os
_logf=_os.environ["FLARE_LOG"]
log=open(_logf,errors="replace").read()[_LOG_START:]
bad=[l for l in log.splitlines() if any(k in l.lower() for k in ("mapping failed","out of memory","cuda error","cublas","traceback","oom","server_error","device error","panic"))]
preempt=[l for l in log.splitlines() if "preempt" in l.lower()]
print(f"(Flare log: {len(preempt)} preemption lines, reported separately)", flush=True)
total_ok=sum(1 for r in results if r[0]=="OK"); total=len(results); pok=pings.count("OK")
verdict="PASS" if (total_ok==total and pok==len(pings) and not bad and health==200) else "FAIL"
print(f"[{PRESET}] {verdict}: {total_ok}/{total} long requests ok, pinger {pok}/{len(pings)} ok, {sum(r[1] for r in results):,} prompt tok pushed, "
      f"min free VRAM {minfree[0]} MiB ({samples[0]} samples), health={health}, bad log lines={len(bad)}, wall {(time.time()-t0)/60:.1f} min", flush=True)
for l in bad[:8]: print("   LOG:", l[:200])
json.dump({"preset":PRESET,"capacity":CAP,"verdict":verdict,"waves":WAVES,"long_ok":total_ok,"long_total":total,"ping_ok":pok,"ping_total":len(pings),
           "prompt_tokens":sum(r[1] for r in results),"min_free_mib":minfree[0],"bad_log_lines":bad[:20],"wall_min":(time.time()-t0)/60},
          open(_os.path.join(_os.environ.get("QUAL_OUT", _os.path.dirname(_os.path.abspath(__file__))), f"stress_{PRESET}.json"),"w"),indent=1)
