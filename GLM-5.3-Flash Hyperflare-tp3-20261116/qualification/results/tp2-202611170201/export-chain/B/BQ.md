| Bq cell (32K, 1 x 30 s, median) | new tok/s | new steps/s | base: 151621 recorded on the same binary (port-g3tp2-1115-0942, 2 passes) tok/s | steps/s | ratio tok / steps | gate |
|---|---|---|---|---|---|---|
| c1 | 246.5 (247) | 105.0 | 245.5 | 103.9 | 1.004 / 1.011x | ok |
| c4 | 519.2 (519) | 276.5 | 516.7 | 280.2 | 1.005 / 0.987x | ok |
| c8 | 762.8 (763) | 314.2 | 748.0 | 310.0 | 1.020 / 1.014x | ok |

Prefill-only tok/s: 8K new 11559 vs 130307 same binary 11403/11385 (1.014x) | proxy 1033 recorded 10702 (1.080x); 32K new 12271 vs 130307 same binary 12086/12140 (1.013x) | proxy 1033 recorded 12026 (1.020x)

Plan rule (INFO): c1 met, c4 met, c8 met

Band (INFO): c1 247 vs recorded 246-252 (band 236-262): inside; c4 519 vs recorded 507-509 (band 497-519): ABOVE; c8 763 vs recorded 699-705 (band 692-712): ABOVE
