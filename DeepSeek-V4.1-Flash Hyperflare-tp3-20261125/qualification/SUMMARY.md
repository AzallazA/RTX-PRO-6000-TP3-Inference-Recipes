# qualification qual-tp3-wave7-20261125

package: `ds41-1017-engram-fp8@gb202-tp3-202611251350` (WARMIND v1.1 default, `steering_scale=0`)  
binary: `hyperflare-tp3-20261125` (`ba6a4c11`, sha256 `7aa439a25794...`)  
KV tokens: `5,883,648` (`556` shared snapshot slots, `STACK_LIMIT=48 B`, max kernel stack `32 B`, `LOCAL = 0 B`)  

| gate | result | detail |
|---|---|---|
| static graph race audit | PASS | `0 / 107` racy graphs (`12,669` kernels, `0.6s` boot prep) |
| `verify_fix.py 10` | PASS | `0` TRUE corruptions (`1,667` qual + `943` post-rebuild smoke tokens) |
| consistency (det + 3× stoch) | PASS | det `0.972` (`0` miss, med `0.0040`); 3× stoch `0.972 / 0.972 / 0.957` (`0` miss) |
| WARMIND v1.1 steering | PASS | `scale=None` vs `0.0` exact match (`top1_dlp=0.000000`), shift verified `0.5/1.0`, HTTP 400 mismatch |
| vision / images smoke (5 cases) | PASS | `5/5` served, `5/5` correct (`small` `0.1s`, `1920x900` `0.2s`, `4160x1280` `0.3s`, 2 & 4 images `0.1–0.2s`) |
| `eval-43` (thinking off) | PASS | `42 / 43` (`v10b`: `43 / 43`, temp clamped to `1.0`) |
| tool calls (`60` @ `c8`) | PASS/WARN | `59 / 60` OK (`0` wrong tool, `16s`; `v10b`: `60 / 60` OK, `15s`, temp `1.0`) |
| `hotel-lights` (`30` @ `c16`) | PASS | `30 / 30` exact, `0` wrong, `0` truncated |
| long retrieval (`128K` & `900K`) | PASS | `128000 pass`, `900000 pass` (`73s`) |
| 53.8K NIAH + 46.4K code | PASS | both needles verbatim (`2.40s`); `code32k` `has_def=True`, `0` loops (`3.84s`) |
| partial-tail NIAH (row-shard) | PASS | `3/3` + `22/22` bucket-boundary partial-tail prompts (`127..7681+` tok) verbatim |
| live crash & disconnect sweep | PASS | `12/12` API/4xx/OOB, 8-stream disconnect stress, `16/16` mixed burst HTTP 200 |
| 5-phase sustained stress test (`v10b`) | PASS | `124.2s`, >2.5M prompt tok, `48/48` prefix-cache churn, `112.5K` NIAH + 4× decode, `0` failed, `0` active leaks, post-stress `verify_fix 10` = `0` TRUE |
| alternate modes (`v10b`) | PASS | `ENGRAM_RAM=0` (`0` TRUE, `0.969` / `0` miss), `ENGRAM_RAM=layer1` (`0` TRUE, `0.969` / `0` miss), `BUDGET=96` (`0` TRUE) |
| clean-checkout patches & hygiene | PASS | `b12x-tp3-20261125.patch` & `hyperflare-recipes-tp3-20261125.patch` apply cleanly on `0d48af53` & `39c157a4`; `flare doctor` `7/7` non-root PASS |
