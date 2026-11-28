# qualification qual-tp3-wave15.1-20261128

package: `ds41-1017-engram-fp8@gb202-tp3-202611281539` (Wave 15.1 Global `HOT=375`, WARMIND v1.1 default, `steering_scale=0`)  
binary: `v0.4.1-tp3-20261128` (`7eecfda6`, sha256 `70d03647d8a9...`)  
KV tokens: `2,432,000` (`230` shared snapshot slots with `--history-drafts`; `2,446,592` / `231` with `--no-history-drafts`; `STACK_LIMIT=48 B`, largest kernel stack `32 B`)  

| gate | result | detail |
|---|---|---|
| Rust unit tests (`flare-core`) | PASS | `execution::threaded (21)`, `engram (58)`, `cache (129)`, `runtime_core (195)`, `core (498)`, `model (62)`, `mwaitx (2)`, `heliopause (216)`, `flare-cli (225)` |
| Python unit tests (`ignition` + `recipes`) | PASS | Ignition `456/456` pytest; DS41 recipes `98/98` pytest (`_supports_advisory_prefetch() == True`) |
| `verify_fix.py 10` (RAM) | PASS | `0` TRUE corruptions (`1,617` tokens, `0` retries, `0` skipped) |
| consistency (`48 × 4`, `--no-history-drafts`) | PASS | `192` positions: top-1 `0.969`, `0` missing from prefill top-5, med `\|dlogprob\| 0.0042` |
| consistency (`48 × 4`, `--history-drafts`) | PASS | `192` positions: top-1 `0.964`, `0` missing from prefill top-5, med `\|dlogprob\| 0.0020`, `max 0.8702` |
| 6-case NIAH (`1.8K` – `270K`) | PASS | `6/6` verbatim (`OMEGA-7491`, `DELTA-3825`, `SIGMA-9104`, `ZETA-5519`, `KAPPA-8832`, `ALPHA-6123` @ `253,187` actual prompt tok) |
| vision / images smoke (`5` cases) | PASS | `5/5` served & verified (`Red`, `Blue`, `Circle`, `Square`, `stripes`) |
| WARMIND v1.1 steering | PASS | `scale=0` unsteered vs `scale=1` verified shift |
| disk Engram lookahead prefetch (`#566 + #70`) | PASS | `16 GiB` cache: `40.3K / 40.6K / 39.4K` ON vs `39.7K / 39.8K / 38.9K` OFF; `4 GiB` cold-cache A/B: `flare_engram_prefetch_rows_loaded_total = 1,448`, `0` errors; `40K` disk NIAH PASS (`QUARTZ-91847-COBALT`) |
| GPU stability & thermal soak (`450 W`, `+800 / 2850 MHz`) | PASS | `307s` 6-stage soak (2× 64K/128K/256K prefill + 4×2048 greedy + 50.7K Tetris + 32K/128K C1/C8/C16): **`0` Xids**, `0.0%` SW/HW throttle, pre & post `verify_fix 10` = `0` TRUE (`1,483` / `1,631` tok), `consistency 48×4` = `0.974` (`0` miss, `max 0.5184`) |
| SAM verbatim copy-draft (`HOT=355` Wave 15) | PASS | `C1 short 1,044.0 tok/s`, `C1 long 916.7 tok/s`, `C4 (4×16 P=64) 1,372.9 tok/s`, `100%` SHA-256 verbatim match |
| clean-checkout patches & tarball packaging | PASS | `hyperflare-recipes-tp3-20261128.patch` & `b12x-tp3-20261128.patch` pass `git apply --check` on fresh `39c157a4` & `0d48af53`; `flare-linux-amd64.tgz` (`14 MiB`) built via upstream `packaging/archive.py` and verified |
