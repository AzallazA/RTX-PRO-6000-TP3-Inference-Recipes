# DeepSeek-V4.1-Flash-1017 on NVIDIA Hyperflare at TP3 — 20261121

## DS41 TP3 update

**Already have the Nov 21 package? Keep `gb202-tp3-202611201530`. No re-export is needed.**

- **Server.** Build `flare` from `hyperflare-tp3-20261121.1` if you have the original Nov 21 binary. The maintenance binary fixes steering access and accepts an explicit zero steering scale on Messages and Responses.
- **Export and preparation.** The tools check overlay capabilities, honor explicit export switches, and preserve previous outputs if preparation fails.
- **Release scripts.** Refresh the Nov 21 recipe folder from `main`. The launcher honors the requested Engram storage and feature settings even when the shell exports conflicting values. Custom exports enable the 4,096-row variants only when the selected prefill ladder includes them. Preflight recognizes tagged shallow fork checkouts when run from the fork, checks the adjacent source patch when run from the recipe, and requires three distinct visible GPU indices.

If you already run the `.1` binary, use the updated scripts with it on your next launch. [Release notes](https://github.com/AzallazA/hyperflare-inference/releases/tag/hyperflare-tp3-20261121.1) · [Upgrade instructions](EXPORT.md#ds41-tp3-update)

> **Flare = Hyperflare.** NVIDIA renamed Flare to **Hyperflare** ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
> The binary, the CLI (`flare run`, `flare models`) and the cache paths (`~/.cache/flare`) are still called `flare`.

DeepSeek-V4.1-Flash-1017 served by **Hyperflare at tensor parallel 3 with the full 1M-token context** on three 96 GB
RTX PRO 6000 Blackwell cards. This build supersedes [Hyperflare-tp3-20261116](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md).
It is a new export, a new `flare` binary and a new serve script (see [Coming from 20261116](#coming-from-20261116)).
Changes since then:

- **Optional per-request steering.** The package carries a 0.68 MB steering sidecar (WARMIND v1.1, from upstream PR
  #404). A request sets `steering_scale` between 0 and 1; 0 or omitted is the normal model. The projection runs
  inside the BRRR fused decoder kernel, so steering costs nothing at scale 0 and little at scale 1
  ([Steering](#steering)). An operator can require a key for nonzero scales ([Steering access](#steering-access)).
- **KV cache 5,687,040 tokens** (5.4 × 1M; 20261116: 5,403,008).
- **Long prompts prefill faster.** The indexer splits its rows across the three GPUs for prompts of 32K tokens and more;
  head and index gathers now pull instead of push. Prefill at 512K +15% and at 1M +29% in the first step, then
  +3% more.
- **Decode work in the export.** A blended expert order (code and prose) with 355 hot experts in layers 20-39,
  attention split rules plus a one-launch 24-head attention kernel, encoder-only prefill for chunks that produce no
  output, and smaller items. GPU time per decode round: c8 −6.8%, c16 −5.5% in the attention step.
- **Selected upstream fixes** (tool calls, DSML arguments, 16-bit PNG and GIF input, host-cache metrics, literal
  control-token text). The base stays `70069281`; the picks are listed in [Upstream fixes](#upstream-fixes).
- **Engram read-ahead on by default** (`FLARE_ENGRAM_READ_AHEAD=1`). It does nothing while both Engram tables are in
  host RAM (the default). It helps when a table is served from disk ([Memory](#memory)).

**Qualification:** export-tier checks on the steering package, binary-tier checks on the release binaries, and a release run on the final default (16-stream KV stress, tool-eval-bench, 8-stream agent replay, long retrieval, GPQA at steering scale 1): all gates pass. See [Quality](#quality-and-long-context).

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT), all 48 shards, including DeepSeek's FP8 Engram tables.
Steering sidecar: [jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar)
@ `1bf69d8f`, file `warmind-v1.1-orth4c-steering.safetensors` (sha256 `68551f9b…6a2d`, MIT).
Runtime: [NVIDIA Hyperflare](https://github.com/NVIDIA/hyperflare-inference) `main` @ `70069281` plus this fork
(`source.patch`; the public fork is https://github.com/AzallazA/hyperflare-inference at commit
`607fb7f9`, with runtime from `hyperflare-tp3-20261121.1`), FlashInfer @ `aa6c4221` (upstream's pin, 0.7.3) plus the patches in `tp3-patches/` and the b12x
overlay patches in `tp3-patches/overlay/` (`flashinfer-pcie-twoshot-pull-world3.patch`, `flashinfer-dcp-gather-pull-world3.patch`).
The fused layer tail is adapted from [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0).

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it. Then:

```bash
FLARE=<fork>/flare-cli/target/release/flare ./serve.sh   # port 15015, model DeepSeek-V4.1-Flash, ~51 s
```

Measured on the system in [Test system](#test-system) at **450 W/GPU, +75 MHz core, +6000 MHz memory offset**.
Every measurement is in [MEASUREMENTS.md](MEASUREMENTS.md); qualification results in [qualification/RESULTS.md](qualification/RESULTS.md).

## Headline numbers

| | this build, 450 W | 20261116 (published), 450 W |
| --- | --- | --- |
| Request slots / verification cap / prefix cache | 16 / 32 / 12G | 16 / 32 / 12G |
| KV cache | **5,687,040** (5.4 × 1M) | 5,403,008 |
| Decode 32K c1 / c4 / c8 / c16 (tok/s) | **311-324 / 773-810 / 1,059-1,102 / 1,459-1,467** (range of 3 runs) | 319.5 / 781.2 / 978.8 / 1,331.5 (single 30 s passes) |
| Decode 128K c1 / c4 / c8 / c16 | 321-323 / 758-774 / 1,033-1,063 / 1,396-1,400 | 320.8 / — / — / — |
| Decode 512K c1 | 301-309 | 292.7 |
| Prefill 8K / 32K / 128K / 256K / 512K / 1M (tok/s, cold) | 16,527-16,603 / 18,269-18,336 / 18,791-19,128 / 18,412-18,612 / 17,481-17,671 / 15,840-16,023 | — / 15,766 / — / — / — / — |
| Steps/s 32K c1 / c8 / c16 | 129.8 / 430.9 / 503.4 | 122.8 / 350.7 / 301.0 |
| 8-stream agent replay, 300 s | 658 turns, 530.8 output tok/s, first token p50 / p90 0.395 / 0.627 s | 649 turns, 464.6 output tok/s, 0.42 / 0.72 s |
| Steering scale 1 vs 0, decode 32K | tokens per round 0.994 / 0.979 / 0.946 / 0.992 at c1 / c4 / c8 / c16; prefill level | — |
| Startup | ~51 s | ~55 s |
| eval-43 | **43/43** | 42/43 |
| tool-eval-bench | 93/100 (81 of 91 graded) | 90/100 (77/91) |
| GPQA Diamond, scale 0 (one sample of 198) | **91.4%** (181/198; 5 truncated without an answer) | 91.4% (181/198; 4 truncated) |
| GPQA Diamond, scale 1 | 89.4% (177/198; 3 truncated) | — |
| Refusal set: 30 prompts the stock model tends to refuse | refused 24/30 at scale 0, 3/30 at scale 1; 30 benign prompts: 0/30 refused at both | — |
| KV stress (16 streams) | 32/32 long requests, 168/168 pings, lowest free VRAM 78 / 78 / 78 MiB | PASS: 32/32, 212/212, 64 MiB |
| Long retrieval 128K / 900K | pass / pass | pass / pass |

Speed cells: `llm_decode_bench.py` (llm-inference-bench v0.9.1 with the steps patch in `tools/`), 30 s decode cells
with `--skip-prefill`, and cold `--prefill-only` prefill, each run three times on the final default; the table shows
the **median of the three runs**. The 20261116 column is single 30 s passes, so small differences are inside
run-to-run noise (this build's three 32K c1 runs spread 310.7-323.7). Against 20261116: decode 32K c1 level, c4 +0.8%,
c8 +9.1%, c16 +10.0%; 128K c1 level; 512K c1 +4.7%; prefill 32K +16.1%. Ranges are in
[MEASUREMENTS.md](MEASUREMENTS.md#1-owner-speed-table-release-package-and-binary).

## Contents

1. [Test system](#test-system) · [What this build contains](#what-this-build-contains) · [Memory](#memory)
2. [Steering](#steering) · [Steering access](#steering-access) · [Upstream fixes](#upstream-fixes) · [PCIe note (issue #348)](#pcie-note-issue-348)
3. [Quality and long context](#quality-and-long-context) · [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Coming from 20261116](#coming-from-20261116) · [Building the package](#building-the-package) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

## Test system

Unchanged from [20261116](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#test-system): 3 × RTX PRO 6000
Blackwell Workstation Edition (96 GB) on PCIe Gen5 x16, each on its own CPU root port; 450 W, +75 MHz core,
+6000 MHz memory; Threadripper PRO 7965WX; 256 GB DDR5-6000 (251 GiB visible); NVIDIA driver 615.71.09, CUDA 13.1.

Host RAM is still a hard requirement with both FP8 Engram tables in RAM. The lowest MemAvailable we saw while
qualifying this package was 10.6 GiB (15.0 GiB during the steering speed A/B).

## What this build contains

Cumulative description of the fork's features. The 20261116 [contents table](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#what-this-build-contains)
still applies. The rows below are new since then, in the order they landed. Each was measured against the build
before it; [MEASUREMENTS.md](MEASUREMENTS.md) has the runs.

| Change | Effect when it landed |
| --- | --- |
| Blended expert order (code and prose route counts), 355 hot experts in layers 20-39 (was 342 in 20-33), vision tower arena in host RAM, shared operands, Markov table in VRAM | served KV +2.7% (5,548,672); tok/s c4 / c8 / c16 1.006 / 1.011 / 1.024×, c1 0.935× (flagged; later steps recovered it, see the owner table) |
| Encoder-only prefill (`--scheduler-encoder-prefill true`, export `SKIP_SUFFIX=4096`): prompt chunks that produce no output skip layers 21-39 and the head; also pad-once, decode glue pass 2, Markov rows staged from host RAM | prefill 8K / 32K / 128K +3.9 / +7.1 / +11.2%; decode level |
| Exporter scratch fix: cuBLAS's own pool no longer kept in the package | served KV 5,548,672 → 5,614,208 (+65,536) |
| Pinned q projection, rotary embedding once per step, one merged logprob exchange | GPU time per round c1 32K −0.9%, c1 128K −1.6%, c16 −1.2%; KV unchanged |
| Attention decode split rules by row count plus a one-launch 24-head attention kernel (H24) | GPU time per round c8 32K −6.8%, c16 32K −5.5%, c1 / c4 level; KV 5,687,040 (+72,832) |
| Indexer row split across the GPUs for prompts from 32K tokens (`-split-ctx32768` graphs; shorter prompts keep the replicated graphs) | prefill 128K / 512K / 1M 1.030 / 1.154 / 1.289×, 8K / 32K level; 4096-token append at 512K 345 → 236 ms; KV −7,296 |
| DCP head gather by pull (`GATHER_PULL=1`), prefill rotary embedding shared across layers (`ROTARY_PREFILL=1`) | prefill 128K / 512K / 1M 1.020 / 1.031 / 1.035×; append −4 to −6%; c4 32K +1.0% of round time, c8 / c16 slightly faster |
| Engram disk read-ahead (`FLARE_ENGRAM_READ_AHEAD=1`), default on in `serve.sh` | inert with both tables in RAM (no read-ahead thread, KV and decode unchanged); with a table on disk see [Memory](#memory) |
| Selected upstream fixes ([list](#upstream-fixes)) | decode level (c1 +2.4% on one outlier round, others level; c8 −0.1%) |
| Steering sidecar in the package and steering inside the BRRR kernel ([Steering](#steering)); optional steering keys | release binary + steering package vs the previous default (three fresh-boot rounds each): GPU time per round c1 +1.7%, c8 −2.4% (level); prefill level; KV +7,296 |

## Memory

**GPU (per GPU, release package, served):** 89.63 GiB loaded (weights, graphs, scratch), 5.00 GiB KV cache budget;
15.0 GiB of KV over the 3 GPUs = 5,687,040 tokens. Host model memory 16.6 GiB, host prefix cache 10.8 GiB.

**Host RAM:** 217.6 GiB committed by the load at the default settings (both FP8 Engram tables in RAM, prefix cache 12G):
Engram tables 188.8 GiB, host part of the mixed expert banks 9.5 GiB, mapped host allocations 7.1 GiB, host prefix
cache 12.1 GiB, staging 0.1 GiB. 20261116: 217.3 GiB with the same tables and prefix cache.

**Steering in memory:** the sidecar's directions are baked into the package at export and sit in VRAM as constants:
about 0.68 MB per GPU plus a 99 KB per-GPU arena, not in host RAM. The KV budget is unaffected (5,687,040 tokens with
the sidecar, the same as the export without it).
The serve modes and their RAM are as in [20261116](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#memory).

**Engram read-ahead.** `serve.sh` now sets `FLARE_ENGRAM_READ_AHEAD=1` (`READ_AHEAD=` turns it off). With both tables
in RAM it starts no thread and changes nothing. When a table is read from disk
(`ENGRAM_RAM=layer1` or `ENGRAM_RAM=0`), it loads the rows a prefill chunk will need before the chunk runs.
Measured on an earlier package of this series (cold prompts, empty row cache at boot):

| Disk setup | Without read-ahead (vs both in RAM) | With read-ahead (vs without) |
| --- | --- | --- |
| Layer 14's table on disk, 32 GiB row cache | random 128K prompt 11.15 s vs 7.53 s; text 128K 8.36 s vs 7.14 s | random 128K +20.4%, text 128K +12.6% (two rounds: +20.4 / +21.4%, +12.6 / +12.5%); decode level |
| Layer 14's table on disk, 8 GiB row cache (filled) | random 128K +39%, text 128K +17% slower | random 128K +18.6%, text 128K +11.0%, 32K code −0.7%; decode c1 / c4 / c8 1.008 / 1.014 / 0.992× |
| Both tables on disk, 32 GiB row cache | random 128K +121%, text 128K +94% slower; c1 decode −9% | 32K code +25%, random 128K +38%, text 128K +43% (13.83 → 7.94 s; RAM 7.14 s); c1 decode 1.059×; agent replay output 1.105× |

### Engram tables on disk with a RAM row cache

The two Engram tables are ~94 GiB each in FP8. By default both sit in host RAM. To use less RAM, read one or both from
disk through a host-RAM row cache, which holds the rows recent prompts used. Read-ahead (on by default) fills it before
each prefill chunk.

| Setting | Tables in RAM | Row cache | Use when |
| --- | --- | --- | --- |
| `ENGRAM_RAM=1` (default) | both | none | 256 GB of RAM; fastest prefill |
| `ENGRAM_RAM=layer1` | layer 1 (layer 14 on disk) | `ENGRAM_CACHE`, default 32GiB | less than 256 GB of RAM (177.8 GiB committed in 20261116 at a 32G prefix cache); cold 128K prefill ~4-5% slower, decode level |
| `ENGRAM_RAM=0` | none (both on disk) | `ENGRAM_CACHE`, default 32GiB, shared by the two tables (20261116's `serve.sh` passed no cache size here) | least RAM; cold 128K prefill ~11% slower, decode level |

```bash
ENGRAM_RAM=layer1 ./serve.sh                     # layer 14 from disk, 32 GiB row cache
ENGRAM_RAM=layer1 ENGRAM_CACHE=8GiB ./serve.sh   # smaller cache: more RAM free, a little slower on new prompts
ENGRAM_RAM=0 ENGRAM_CACHE=32GiB ./serve.sh       # both tables from disk
```

- Each table moved to disk frees roughly its size (~94 GiB) minus the row cache. Check with `flare run ... --dry-run`
  (committed host RAM) before serving.
- Put the package on a fast NVMe drive. The slowdowns above are for cold prompts; prompts whose rows are already in the
  row cache run at RAM speed.
- Keep read-ahead on (`READ_AHEAD=` turns it off). Without it, the same setups are 17-121% slower on cold 128K prompts
  (table above).
- Decode is level in every setup: per-token lookups are small and served from the row cache.

## Steering

The package carries one steering sidecar: for each of layers 4-36 a unit direction, a strength and an attention row
gain (0.68 MB per GPU). A request with `steering_scale` s projects that direction out of the attention output and the
shared-expert output, scaled by s. This WARMIND v1.1 sidecar targets refusals. Upstream PR #404 (open) adds the
feature to the unfused layers; on this fork the BRRR fused decoder kernel runs layers 0-39 in decode, so the
projection was added inside that kernel too.

- **Request field:** `steering_scale`, top level of the body or in `chat_template_kwargs`, on Chat Completions and
  Completions. Clamped to [0, 1]; default 0. Conflicting duplicates return HTTP 400.
- **Scale 0 or omitted is the normal model.** It adds no conditioning event, so prefix-cache entries are shared with
  unsteered requests. A nonzero scale adds a position-0 event from the sidecar hash and the exact scale, so different
  scales never reuse each other's KV prefixes.
- **Mixed batches work:** steered and unsteered requests run in the same decode round.
- **Package without a sidecar:** a nonzero scale returns HTTP 400.

Measured on this package (A = the same export without the sidecar on the previous binary; B = this package):

| | c1 | c4 | c8 | c16 |
| --- | --- | --- | --- | --- |
| Scale 0 vs A: GPU time per round (corrected), ms | +0.004 | +0.025 | +0.239 | −0.073 |
| Scale 0 vs A: raw round time | 0.971× | 0.999× | 1.005× | 0.999× |
| Scale 1 vs scale 0: raw round time | 1.010× | 0.994× | 0.958× | 0.997× |
| Scale 1 vs scale 0: tokens per round (draft acceptance) | 0.994× | 0.979× | 0.946× | 0.992× |
| Prefill 32K / 128K / 512K, scale 0 vs A | 0.999 / 0.998 / 1.000× | | | |
| Prefill 32K / 128K / 512K, scale 1 vs 0 | 1.013 / 0.999 / 0.998× | | | |

Three rounds each. In this A/B the analyzer flagged scale 0 at c8 (+0.239 ms corrected, +0.5% raw round time), but
the two arms ran on different binaries. The release binary check then measured the final pair directly (this package
on the release binary vs the previous default, three fresh-boot rounds each): GPU time per round c1 +1.7%, c8 −2.4%,
so scale 0 costs nothing measurable. The cost of
scale 1 is mostly fewer accepted draft tokens at c8, not slower rounds.

**Refusal set.** 30 prompts the stock model tends to refuse and 30 benign prompts, scored by a keyword classifier
(the prompt texts are not published; "partial" = deflected, counted with an additional private phrase list): scale 0 refused 24/30 (3 more partial), scale 1 refused 3/30 (4 more partial);
benign prompts 0/30 refused at both.
**GPQA Diamond** at scale 0: 91.4% (181/198), the same score as 20261116. At scale 1: 89.4%
(177/198; paired against scale 0: 4 questions wrong only at scale 1, 0 only at scale 0, exact McNemar p 0.125).

Scale 1 removes most refusals of this sidecar's target set. What a client does with that is the operator's call; see
[Steering access](#steering-access) to restrict it.

## Steering access

By default any client may set `steering_scale`. To restrict nonzero scales to key holders, give the server one or more
steering keys (all sources add up):

- `flare run --steering-api-keys-file PATH` (preferred: keys stay out of the process list, the environment and the
  logs; one key per line or comma-separated, `#` starts a comment line, at least one key, mode 600);
- `FLARE_STEERING_API_KEYS_FILE=PATH`;
- `FLARE_STEERING_API_KEYS=KEY1,KEY2` (visible to your user and root in the process environment).

There is deliberately no option that takes a key on the command line, where every user could read it with `ps`.

With keys configured, a request that sets a nonzero `steering_scale` (top level or in `chat_template_kwargs`, on any
generation route) must send one key as `X-Steering-Key: KEY`, or, without that header, as `Authorization: Bearer KEY`.
Otherwise the server answers **HTTP 403** with `permission_error` (OpenAI error shape, or Anthropic shape on
`/v1/messages`) before preparing the request, streaming or not. An omitted, null or zero `steering_scale` never needs a
key. If a serving API key is also set, `Authorization` must carry the serving key, so send the steering key in
`X-Steering-Key`. The server compares SHA-256 digests and logs only the number of keys at startup.
`serve.sh`: `STEERING_KEYS_FILE=/path/to/keys ./serve.sh` (sets `FLARE_STEERING_API_KEYS_FILE`; unset by default, so
any client may steer).
Checked on the release binaries with a key file: HTTP 403 in 7 of 7 cases (no key, wrong `X-Steering-Key`, wrong
Bearer, scale in `chat_template_kwargs`, `/v1/completions`, `/v1/messages`, streaming); the right key via
`X-Steering-Key`, Bearer and streaming answered 200 and steered; scale 0 or omitted without a key answered 200 and
reused the cached prefix; the key appeared in no log, process argument or environment (only the file path). The final
binary rejects a key option on the command line as an unknown argument.

**Over a network:** the server speaks plain HTTP, so a key sent across a network can be read by anyone who can see
that traffic. `serve.sh` binds to 127.0.0.1; if you expose the server beyond the machine, put it behind a TLS reverse
proxy (for example Caddy or nginx) or keep it on a trusted LAN or VPN.

### Steering tools

In `tools/`, stdlib Python 3, default server `127.0.0.1:15015` (each has `--help` and `--self-test`):

```bash
# GPQA (or any client that cannot add a request field) at scale 1: every chat/completions request gets steering_scale
python3 tools/steer_proxy.py --listen 15115 --scale 1.0 --log proxy.log      # then point the client at 127.0.0.1:15115
# refusal count of YOUR prompt set (none ship here): {"trigger": [...], "benign": [...]}
python3 tools/refusal_set.py --set my-prompts.json --scale 0 --out s0.json
python3 tools/refusal_set.py --set my-prompts.json --scale 1 --out s1.json
# steering access gate, server started with STEERING_KEYS_FILE=keys.txt ./serve.sh
python3 tools/steering_access_check.py --key-file keys.txt --out access.json
```

- `steer_proxy.py` forwards everything else unchanged, streams responses as they arrive, and logs one `REWRITE` line
  per request plus counters. With steering keys, add `--steering-key-file FILE` (the key is never logged).
- `refusal_set.py` labels the first 400 characters of each answer refused / deflected / answered with a keyword
  classifier. Only generic refusal phrases are built in; add your own deflection (or refusal) phrases with
  `--extra-patterns FILE` (one regex per line). The published "+N partial" counts above also used a private list of
  deflection phrases, so the shipped classifier alone counts no deflections. Prompts and answers go only to `--out`;
  `--key-file` for keyed servers.
- `steering_access_check.py` runs the 13 requests of the access check above (7 must get 403, the rest 200 with the
  expected prefix-cache reuse) and prints `ACCESS PASS|FAIL`; it never prints or stores the key. `--sanity --set FILE`
  checks that scale 1 changes at least one answer of a prompt set.

## Upstream fixes

Picked from upstream `main` onto the fork (base still `70069281`); DS41-relevant ones in plain words:

| Upstream | What it changes for DS41 |
| --- | --- |
| #333 | With parallel tool calls turned off, a second call now finishes after the first instead of failing (HTTP 500 or a stream error) |
| #359, #364 | DSML tool arguments: literal `string="false"` values stay valid JSON; one call is capped at 1 MiB; values nested deeper than 126 levels become strings instead of failing the call |
| #316, #385 | 16-bit PNG input is accepted (reduced to 8-bit); non-animated GIF is accepted; animated GIF is refused with HTTP 400 |
| #400 | Host prefix-cache hit metrics count only successful restores |
| #396 (with #350) | Request text that spells a control token stays plain text (a refactor only for DS41; the fix matters for GLM) |
| #360 | Inductor dtype-view temporaries get their own storage (export correctness) |
| #363 | Test fixtures only |
| `b003ebf5` | No free-memory queries per context during cache allocation |

Not taken: upstream's global-checkpoint-slot family (#389, #346, #354, #357), which builds on code this fork reverted.
Checked on DS41: tool calls (OpenAI streaming and non-streaming, Anthropic `tool_use`, parallel off = one call, 1.1 MiB
history), 16-bit PNG and GIF, eval-43, 8-stream KV stress, decode level.

## PCIe note (issue #348)

Upstream issue #348 reports that concurrent reads from several peers collapse on some PCIe hosts. On this host they do
not: every GPU reading from both peers at once reaches 42.2 GB/s per GPU (single-peer read 50.5 GB/s, copy-engine ring
hop 55.6 GB/s). SM stores pushed to several peers at once do collapse here (2.6 GB/s). Every bandwidth-carrying
exchange in this package pulls (BRRR layer exchange, Engram and drafter all-reduces, prefill all-reduce parts, index
and head gathers); only bookkeeping messages up to 4 KiB push. Example, b12x all-reduce pull vs push: 16 × 5120 rows
12 vs 41 µs, 1024 × 5120 rows 436 vs 823 µs, one 4096-row prefill chunk 1,412 vs 1,834 µs.

## Quality and long context

Only the binary (steering, access gate, upstream fixes) and the package (the 192051 recipe plus the sidecar) changed
since the last qualified local build. The export-tier checks ran on this package when it was exported, the binary-tier
checks on the release binaries, and a release run on the final default covered the 16-stream KV stress,
tool-eval-bench, the 8-stream agent replay, long retrieval and GPQA at steering scale 1. Hotel-lights, the 60-prompt
tool-call run and the consistency check are cited from earlier packages of this series; see
[qualification/RESULTS.md](qualification/RESULTS.md).

| Check | This build | 20261116 |
| --- | --- | --- |
| Smoke (thinking on / off) | 2/2 | 2/2 |
| Images (5 cases), 16-bit PNG and GIF | 5/5; 16-bit PNG (gray, RGB) and GIF served, animated GIF refused (4/4) | 5/5 |
| Corruption check (teacher-forced comparison, 10 runs) | **0** true corruptions | 0 |
| Decode vs prefill consistency | 0.965 top-1 (same export without the sidecar) | 0.951 |
| eval-43 (thinking off) | **43**/43 | 42/43 |
| Needles at 64K / 128K | pass / pass | — |
| Tool calls: 60 prompts at 8 streams | 60/60 on an earlier package of this series (181447) | 60/60 |
| tool-eval-bench (91 graded) | 93/100 (81/91) | 90/100 (77/91) |
| GPQA Diamond, scale 0 / scale 1 | 91.4% / 89.4% | 91.4% |
| hotel-lights, 30 runs at 16 streams | 30/30 on 181447 (not re-run) | 30/30 |
| Long retrieval 128K / 900K | pass / pass | pass / pass |
| Long retrieval 500K / 750K / 1M (same export without the sidecar) | pass / pass / pass | — |
| KV stress, 16 streams | 32/32, 168/168 pings, lowest free VRAM 78 / 78 / 78 MiB | PASS |

## Measured and not adopted

| Idea | Result |
| --- | --- |
| Steering outside the fused kernel (BRRR off on the steered layers, as upstream PR #404 does it) | not built; estimated −6 to −10% at c1 for every request; the in-kernel version costs nothing at scale 0 |
| 8-bit prefill all-reduce (P2) | microbench 1.75× at 768 rows; parked before export (quality gate not run) |
| Fairness / age tuning of the c16 scheduler | both-off gave c16 +5% but 8-stream agent output −15%; middle settings level; defaults kept |
| Indexer merge in layer 20 (decode) | faster at 32K / 128K, slower at 512K / 1M; dropped |
| Upstream global-checkpoint-slot changes (#354 etc.) | not cherry-pickable on this fork; separate experiment |
| Earlier ideas | see the [20261116 list](../DeepSeek-V4.1-Flash%20Hyperflare-tp3-20261116/README.md#measured-and-not-adopted) |

## Known limitations

- **256 GB of host RAM** with both FP8 tables in RAM; `ENGRAM_RAM=layer1` (now with read-ahead) for less.
- **Steering removes refusals for the sidecar's target set at scale 1.** Anyone who can reach the server can use it
  unless steering keys are configured.
- **Scale 0 is not bit-identical run to run** (neither is the unsteered model: FP32 atomics in the fused kernel and
  BF16 atomics in prefill MoE). Long-prompt top-1 agreement with the non-steering package was 1.000.
- **Hardware-specific package:** 3 × 96 GB SM120 only.
- **The KV stress runs with little VRAM headroom** (lowest free VRAM 78 / 78 / 78 MiB; 47-80 MiB
  on the previous packages).
- **Thinking mode and forced tool calls:** HTTP 400, as in 20261116. **top_p 1.0, temperature 1.0** fixed as before.
- **The `--mtp-only` package requires speculation** (`DRAFTS` > 0).
- Not upstreamable as written. This is a personal fork of Hyperflare.

## Coming from 20261116

- **Same base commit** `70069281` plus the [upstream fixes](#upstream-fixes). Fetch fork tag `hyperflare-tp3-20261121.1` or apply
  this folder's `source.patch` to a clean `70069281`.
- **Rebuild the binary** and **export again** (`export.sh` defaults are the new release values: the blended expert
  order in `expert-order/`, `PARK=20-39 HOT=355` and the switches marked "new" in EXPORT.md step 4; add
  `STEERING=<sidecar file>` for steering). Export time: about 35 min with tuning records reused; a from-scratch export also runs tuning (time not measured on this package).
- **Rebuild the b12x overlay** with the DCP gather pull patch (`tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch`; export `GATHER_PULL=1`).
- **Download the steering sidecar** (above) if you want steering; without it the package serves only scale 0.
- **Serve flags:** `serve.sh` now also passes `--scheduler-encoder-prefill true` and sets `FLARE_ENGRAM_READ_AHEAD=1`.
  Everything else (16 slots, cap 32, prefix cache 12G, alpha 0.8, interactive scheduler, tail merge, TTFT order
  shortest, prefill merge 512, stack limit 256, private tables, fast exit) is as in 20261116.
  New `serve.sh` variables: `READ_AHEAD` (default 1; empty = off), `ENCODER_PREFILL` (default true) and
  `STEERING_KEYS_FILE` (unset by default). `ENGRAM_RAM=0` now also passes `--embedding-table-cache-size ${ENGRAM_CACHE:-32GiB}`.
- Same model name `ds41-1017-engram-fp8`; remove old versions with `flare models remove <model>@gb202-tp3-<timestamp>`, then `flare models gc`.

## Building the package

Full guide: **[EXPORT.md](EXPORT.md)**. In short:

```bash
# from this folder
./check_env.sh <fork>                                                              # read-only preflight
./make_checkpoint_view.sh <deepseek snapshot> ~/ds41-1017-engram-fp8               # symlinks only; DeepSeek's snapshot, all 48 shards
# the steering sidecar (optional; without it the package serves scale 0 only)
hf download jcartu/WARMIND-1017-Steering-Sidecar warmind-v1.1-orth4c-steering.safetensors \
  --revision 1bf69d8f31e84ac08deaa6d15eb18a22137e305f --local-dir ~/warmind-sidecar                               # then check sha256 68551f9b…6a2d
# from the fork checkout (export.sh uses its .venv)
cd <fork>
VENV=$PWD/.venv B12X_OVERLAY_PATCHES="$PWD/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch $PWD/tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch" \
  tp3-tools/make-b12x-overlay.sh $PWD/.b12x-overlay-gpull                         # b12x copy with the two-shot and gather pull patches
CKPT=~/ds41-1017-engram-fp8 B12X_OVERLAY=$PWD/.b12x-overlay-gpull \
  STEERING=~/warmind-sidecar/warmind-v1.1-orth4c-steering.safetensors COMPILE_WORKERS=3 \
  /path/to/this-folder/export.sh                                                  # the release build
```

`export.sh` defaults are the switches the release package `202611201530` was exported with (its `tp3-config.json`
records them; EXPORT.md step 4 lists them); leave out `STEERING` for a package without steering. `COMPILE_WORKERS=3` kept the export inside
the test system's power-supply limit. A cold tuning cache needs more free VRAM than serving: if tuning runs out of
memory, export once with `HOT=300`, then the target. The steering package took about 35 min with tuning records reused;
a from-scratch export also runs tuning (time not measured on this package).

The model name is the checkpoint directory's name: this view exports as `ds41-1017-engram-fp8`, which is what
`serve.sh` runs by default.

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `check_env.sh` | read-only preflight: toolchain, memlock, fork commit and patches, disk, idle GPUs (PASS / WARN / FAIL) |
| `versions.txt` | exact versions behind the numbers |
| `MEASUREMENTS.md` | every measurement of this build, with the package and settings of each |
| `ENGRAM-PRECISION.md` | MXFP4 vs FP8 Engram tables (unchanged from 20261116) |
| `serve.sh` / `export.sh` | serve the package / export it from a checkpoint view |
| `make_checkpoint_view.sh` | build the FP8 checkpoint view from DeepSeek's snapshot |
| `source.patch` | the fork's changes against Hyperflare `70069281` (`git diff --binary 70069281 607fb7f9`, runtime tag `hyperflare-tp3-20261121.1`; 721 files, 420 new; applied to a clean `70069281` it gives the fork tree exactly) |
| `expert-order/` | per-layer route counts that set the hot-expert placement: the blended code and prose order (new; `expert-order/SHA256SUMS`) |
| `tp3-patches/` | the four FlashInfer patches, and `overlay/` with the b12x two-shot pull patch and the DCP gather pull patch (`flashinfer-dcp-gather-pull-world3.patch`, new) |
| `tools/` | as in 20261116 (`ttft_under_load.py`, `bench_steps.py`, `verify_fix.py`, `consistency.py`, `llm-bench-flare-steps.patch`), plus the steering tools `steer_proxy.py`, `refusal_set.py`, `steering_access_check.py` (see [Steering tools](#steering-tools)); sha256 of every tool in `tools/SHA256SUMS` |
| `qualification/` | the qualification kit (unchanged from 20261116), `RESULTS.md`, and `results/`: `qual-20261121/` (the release run), `binary-checks-20261120/` (release binary checks and access gate), `owner-speed-20261120/` (the speed table's raw result files) |
| `licenses/` | as in 20261116, plus `WARMIND-1017-Steering-Sidecar-LICENSE` |

**Not in this folder:** the steering sidecar `warmind-v1.1-orth4c-steering.safetensors` (684,060 bytes, sha256
`68551f9b8deca315d0c23f31519f69e240d0d4bf0459676add70762635d86a2d`). Download it from
[jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) at revision
`1bf69d8f` (command in [Building the package](#building-the-package)); like the weights, it is not copied here.

## Licenses

As in 20261116. Steering sidecar: MIT, from the license file of
[jcartu/WARMIND-1017-Steering-Sidecar](https://huggingface.co/jcartu/WARMIND-1017-Steering-Sidecar) @ `1bf69d8f`
(`licenses/WARMIND-1017-Steering-Sidecar-LICENSE`, copied unchanged; its copyright line names DeepSeek, as the base
model's MIT license does). The steering tools in `tools/` are part of this folder and under its license.

## Credits

NVIDIA Hyperflare and FlashInfer b12x; DeepSeek for the model and the FP8 Engram tables; Fraser Price for BRRR;
jcartu for the steering feature (upstream PR #404) and the WARMIND sidecar; the upstream authors of the picked fixes;
Local Inference Lab for `llm-inference-bench` and `tool-eval-bench`.
