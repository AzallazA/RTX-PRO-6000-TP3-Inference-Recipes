# Qualification Summary — DeepSeek-V4.1-Flash Hyperflare-tp2-exl3-20261124

Tested on `2 × RTX PRO 6000 Blackwell 96 GB` (`450 W` PL, `MAXC=8`, `STACK_LIMIT=144 B`), binary `hyperflare-inference@hyperflare-tp2-exl3-20261124` (`5ffe4d54`).

## Full Pre-Publish Minimum Qualification (`gb202-tp2-202611241209`, WARMIND v1.1 `scale=0`, `2,868,992` KV)

| Gate | Result | Detail |
|---|---|---|
| Static graph race check | **PASS** | `0 / 102` racy graphs (`encoder_prefill4096` critical path `3,709`) |
| `verify_fix.py 10` | **PASS** | `10` runs, `1,697` tokens, `0` TRUE corruptions |
| `consistency.py 48 4` (RAM) | **PASS** | `187` positions, `0.984` top-1 agree, `0` missing from top-5, med `0.0042` |
| `consistency.py 48 4` (`ENGRAM=one`) | **PASS** | `192` positions, `0.979` top-1 agree, `0` missing from top-5, med `0.0055` |
| Live WARMIND steering | **PASS** | `scale=None` vs `0.0` exact, monotonic `0.0→0.5→1.0` (`steer_dlp=2.27`), HTTP `400` mismatch, `1.5` clamps to `1.0` |
| Smoke (thinking off/on) | **PASS** | `2 / 2` (`'391'`, `'391'`) |
| Image / vision smoke | **PASS** | `5 / 5` served, `5 / 5` correct (`0.2–0.3s`) |
| `eval-43` (thinking off) | **PASS** | `42 / 43` |
| Tool calls (`c8`) | **PASS** | `60 / 60` OK (`0` chose a different tool, `38s`) |
| `hotel-lights` (`c8`) | **PASS** | `30 / 30` exact, `0` wrong, `0` truncated |
| Long retrieval | **PASS** | `128000 pass`, `900000 pass` |
| `8`-stream KV stress (`1.5×` KV) | **PASS** | `16 / 16` long requests ok, `109 / 109` pings ok, `9,511,722` prompt tokens pushed, min free VRAM `82 MiB` (`393` samples), `0` bad log lines, `13.3 min` wall; post-stress smoke OK, KV unchanged at `2,868,992` |

## Plain Sibling (`gb202-tp2-202611241005`, `2,839,808` KV)

- `0 / 102` racy graphs · `verify_fix 10`: `0` TRUE (`1,950` tok) · `consistency`: `0.979` (RAM) / `0.969` (`ENGRAM=one`, `0` miss) · 54K NIAH both needles verbatim + 46K code PASS · `0` loops across all 12 paired LDB cells + Tetris.
