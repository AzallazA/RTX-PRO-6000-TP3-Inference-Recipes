PMERGE VERDICT (recomputed 11-16, CPU, unchanged rule): HOLD (std8: WNT gain x1.086 not above its band 11.8%) | every other item met

The original files (AB.md, RULE.txt, per-arm JSONs) are unchanged. The rule is the one in RULE.txt / pmerge_analyze.py.

## Why AB.md shows `WNT x-` at 8 streams
- The data exist. Every 8-stream JSON has `turns`, `out_tok` and `window`, from the same client (ttft_under_load.py /
  mixed_agent.py) as at 16 streams.
- The WNT fit failed instead. On the four OFF 8-stream runs, least squares gives P_std 0.485 s/turn, P_mixed 0.472 s/turn
  and **D = -0.093 ms/token (negative)**. `wnt_fit` rejects a non-positive coefficient, so the analyzer printed
  "WNT unavailable" and the WNT checks counted as NOT MET.
- Why it is negative: at 8 streams the OFF repeats change output a lot with turns flat (std8 off 646 / 640 turns,
  455.8 / 402.6 out tok/s). The output swing is not explained by turn work, so the fit cannot price an output token.
  At 16 streams the fit is positive (P_std 0.293, P_mixed 0.209 s/turn, D 0.98 ms/token).
- So WNT by its own definition does not exist at 8 streams. analyze.err is empty because this is a handled case, not an
  error. The literal rule then gives HOLD ("no WNT").

## WNT at 8 streams, closest admissible forms (not part of the rule; shown for the decision)

| form | std8 off / on (per run) | std8 ON/OFF | mixed8 off / on | mixed8 ON/OFF | band (max same-flag spread at 8, >= 2%) | std8 gain > band? |
|---|---|---|---|---|---|---|
| fit constrained D >= 0 (-> D = 0, WNT = turns) | 1.005 0.995 / 1.129 1.044 | 1.086 | 0.998 1.002 / 1.147 1.019 | 1.083 | 11.8% (mixed8 on: 762 vs 677 turns) | no (8.6% < 11.8%) |
| 16-stream coefficients applied at 8 | 1.078 1.020 / 1.101 1.117 | 1.057 | 0.962 0.935 / 1.015 1.025 | 1.075 | 5.5% | barely (5.7% > 5.5%) |
| 16 streams, own fit (as in AB.md) | 1.024 0.978 / 1.169 1.146 | 1.157 | 1.020 0.977 / 1.108 1.087 | 1.099 | 4.6% | yes |

- **Constrained form, the nearest to the rule:** std8 is +8.6% but inside the noise band. The band comes from ON-arm
  run-to-run spread (on.1 vs on.2: std8 726/671 turns, mixed8 762/677). Mixed8 no-regression: met.
- **16-stream coefficients:** a pass, but with no margin.

## Per-arm numbers (turns / output tok/s / TTFT p50 / p90, 300 s windows)

| workload | off.1 | off.2 | on.1 | on.2 | ON/OFF turns, out, p50, p90 |
|---|---|---|---|---|---|
| std8 | 646 / 455.8 / 0.509 / 0.903 | 640 / 402.6 / 0.532 / 0.900 | 726 / 399.6 / 0.414 / 0.719 | 671 / 471.0 / 0.404 / 0.750 | 1.086, 1.014, 0.786, 0.815 |
| mixed8 | 663 / 508.8 / 0.303 / 0.939 | 666 / 479.6 / 0.303 / 0.995 | 762 / 492.8 / 0.245 / 0.900 | 677 / 562.8 / 0.237 / 0.898 | 1.083, 1.068, 0.796, 0.930 |
| std16 | 592 / 454.5 / 0.533 / 1.004 | 625 / 374.6 / 0.582 / 1.068 | 698 / 497.3 / 0.426 / 0.747 | 637 / 534.1 / 0.404 / 0.714 | 1.097, 1.244, 0.745, 0.705 |
| mixed16 | 687 / 551.3 / 0.312 / 0.990 | 591 / 575.0 / 0.310 / 1.015 | 735 / 606.2 / 0.254 / 0.902 | 702 / 609.0 / 0.261 / 0.952 | 1.124, 1.079, 0.827, 0.925 |

Decode c1 x1.023, c8 x1.011 (level). Errors 0. gpu_check 13/13.

## Verdict by item (unchanged rule)
1. **std, literal turns-or-output >= 1.05:** met at 8 (turns 1.086) and at 16 (1.097 / 1.244).
   - **WNT >= 1.05 and gain > band:** met at 16 (1.157 > 4.6%).
   - At 8: no WNT (negative fit). The constrained form is 1.086 against an 11.8% band, so NOT MET.
2. **mixed, WNT no regression:** met at 16 (1.099). At 8: no WNT. Constrained form 1.083, met.
3. **TTFT p50/p90:** improved everywhere (0.705-0.930). Met.
4. **Decode c1/c8:** level. Met.
5. **Errors 0, gpu_check 13/13:** met.

**Result: HOLD, on one item only.** The 8-stream standard workload shows +8.6% turns, -21% TTFT p50 and -18% p90.
Same-flag repeats at 8 streams spread up to 11.8%, so that gain is not above noise by the rule's band. Every other
item passes, including 16 streams by a wide margin. The rule is not met, so the pre-approval does not apply as written;
whether to adopt on the TTFT and 16-stream evidence is the owner's call.
