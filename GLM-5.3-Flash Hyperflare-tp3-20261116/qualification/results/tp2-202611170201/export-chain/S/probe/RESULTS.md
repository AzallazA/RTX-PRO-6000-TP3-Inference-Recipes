# GLM prefill probe: new-tp2-202611170201 vs new-tp2-202611151621

found = expected items returned in order (longest common subsequence) / asked, over all sends; om/ins/swap/corr = omitted / inserted (incl. duplicates) / out of order / corrupted copies (right animal, wrong digits: hard fail); slips = asked - found + inserted; pos = sends with code word j found (slice order); lp = logprob of the correct first answer token, mean [min,max] (sampling-free direct request). Gate: FAIL on a corrupted copy, on slips significantly above the reference (one-sided conditional binomial test, p < 0.05; without REF against a nominal 3% slip rate), on found rate < 0.75 (reference >= 0.9), or on the lp rule (spread > max(1.0, 2 x ref + 0.5), mean < ref - 1.0); MARGINAL = more slips than the reference, not significant (settle with CASES=... SENDS=12). Count rows are information only.

| case | expected | new-tp2-202611170201 answers | new-tp2-202611151621 answers | found new-tp2-202611170201 | found new-tp2-202611151621 | om/ins/swap/corr new-tp2-202611170201 | new-tp2-202611151621 | pos new-tp2-202611170201 | lp new-tp2-202611170201 | lp new-tp2-202611151621 | gate new-tp2-202611170201 | why | chunks (new-tp2-202611170201) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| needle 128 | ibex-3078 | ibex-3078 / ibex-3078 / ibex-3078 / ibex-3078 | ibex-3078 / ibex-3078 / ibex-3078 / ibex-3078 | 4/4 | 4/4 | 0/0/0/0 | 0/0/0/0 | 4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 2 launches (75 tok) |
| count 128 | 2 | 2 / 2 | 2 / 2 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.02,-0.01] | -0.01 [-0.02,-0.00] | INFO |  | 2 launches (83 tok) |
| needle 256 | lynx-8276 | lynx-8276 / lynx-8276 / lynx-8276 / lynx-8276 | lynx-8276 / lynx-8276 / lynx-8276 / lynx-8276 | 4/4 | 4/4 | 0/0/0/0 | 0/0/0/0 | 4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (184 tok) |
| count 256 | 3 | 3 / 3 | 3 / 3 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.00 [-0.01,-0.00] | -0.01 [-0.01,-0.00] | INFO |  | 3 launches (201 tok) |
| needle 512 | heron-7529 | heron-7529 / heron-7529 / heron-7529 / heron-7529 | heron-7529 / heron-7529 / heron-7529 / heron-7529 | 4/4 | 4/4 | 0/0/0/0 | 0/0/0/0 | 4 | -0.01 [-0.01,-0.00] | -0.00 [-0.01,-0.00] | PASS |  | 2 launches (421 tok) |
| count 512 | 4 | 4 / 4 | 4 / 4 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.01,-0.01] | -0.00 [-0.00,-0.00] | INFO |  | 4 launches (454 tok) |
| needle 1024 | marten-5064,bison-5383 | marten-5064,bison-5383 / marten-5064,bison-5383 / marten-5064,bison-5383 / marten-5064,bison-5383 | marten-5064,bison-5383 / marten-5064,bison-5383 / marten-5064,bison-5383 / marten-5064,bison-5383 | 8/8 | 8/8 | 0/0/0/0 | 0/0/0/0 | 4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 2 launches (922 tok) |
| count 1024 | 5 | 5 / 5 | 5 / 5 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.01,-0.01] | -0.01 [-0.01,-0.01] | INFO |  | 2 launches (945 tok) |
| needle 1536 | 3 codes | 3 codes / 3 codes / 3 codes / 3 codes | 3 codes / 3 codes / 3 codes / 3 codes | 12/12 | 12/12 | 0/0/0/0 | 0/0/0/0 | 4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 2 launches (1434 tok) |
| count 1536 | 6 | 6 / 6 | 6 / 6 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.24 [-0.24,-0.24] | -0.15 [-0.24,-0.06] | INFO |  | 2 launches (1472 tok) |
| needle 2048 | 4 codes | 4 codes / 4 codes / 4 codes / 4 codes | 4 codes / 4 codes / 4 codes / 4 codes | 16/16 | 16/16 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (1936 tok) |
| count 2048 | 2 | 2 / 2 | 2 / 2 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.01,-0.01] | -0.01 [-0.02,-0.01] | INFO |  | 3 launches (1955 tok) |
| needle 2560 | 5 codes | 5 codes / 5 codes / 5 codes / 5 codes | 5 codes / 5 codes / 5 codes / 5 codes | 20/20 | 20/20 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 2 launches (2445 tok) |
| count 2560 | 3 | 3 / 3 | 3 / 3 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.02,-0.01] | -0.03 [-0.06,-0.01] | INFO |  | 2 launches (2467 tok) |
| needle 3072 | 6 codes | 6 codes / 6 codes / 6 codes / 6 codes | 6 codes / 6 codes / 6 codes / 6 codes | 24/24 | 24/24 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (2961 tok) |
| count 3072 | 4 | 4 / 4 | 4 / 4 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.01,-0.01] | -0.01 [-0.02,-0.01] | INFO |  | 3 launches (2982 tok) |
| needle 3584 | 7 codes | 7 codes / 7 codes / 7 codes / 7 codes | 7 codes / 7 codes / 7 codes / 7 codes | 28/28 | 28/28 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (3479 tok) |
| count 3584 | 5 | 5 / 5 | 5 / 5 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.39 [-0.49,-0.30] | -0.28 [-0.40,-0.15] | INFO |  | 3 launches (3479 tok) |
| needle 4096 | 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 32/32 | 32/32 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 2 launches (3969 tok) |
| count 4096 | 6 | 6 / 6 | 6 / 6 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.08 [-0.10,-0.07] | -0.20 [-0.28,-0.11] | INFO |  | 2 launches (4012 tok) |
| needle 6K | 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 32/32 | 32/32 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4,4,4,4 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (5932 tok) |
| count 6K | 2 | 2 / 2 | 2 / 2 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.04 [-0.06,-0.03] | -0.02 [-0.03,-0.02] | INFO |  | 3 launches (5963 tok) |
| needle 12K | 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 32/32 | 32/32 | 0/0/0/0 | 0/0/0/0 | 4,4,4,4,4,4,4,4 | -0.01 [-0.03,-0.00] | -0.01 [-0.01,-0.01] | PASS |  | 5 launches (11735 tok) |
| count 12K | 3 | 3 / 3 | 3 / 3 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | -0.02 [-0.02,-0.02] | -0.01 [-0.01,-0.01] | INFO |  | 5 launches (11748 tok) |
| needle 33K | 8 codes | 8 codes / 8 codes / 8 codes / 8 codes | 8 codes / 8 codes / 8 codes / 7 codes | 32/32 | 31/32 | 0/0/0/0 | 1/0/0/0 | 4,4,4,4,4,4,4,4 | -0.01 [-0.01,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 10 launches (32980 tok) |
| count 33K | 4 | 4 / 4 | 4 / 4 | 2/2 | 2/2 | 0/0/0/0 | 0/0/0/0 | - | n/a | n/a | INFO |  | 11 launches (32930 tok) |
| img1 count | 1 | 1 / 1 / 1 | 1 / 1 / 1 | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | - | -0.00 [-0.01,-0.00] | -0.01 [-0.01,-0.01] | PASS |  | 3 launches (309 tok) |
| img1 letters | A | A / A / A | A / A / A | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | 3 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (313 tok) |
| img2 count | 2 | 2 / 2 / 2 | 2 / 2 / 2 | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.02,-0.01] | -0.01 [-0.01,-0.01] | PASS |  | 3 launches (592 tok) |
| img2 letters | A,A | A,A / A,A / A,A | A,A / A,A / A,A | 6/6 | 6/6 | 0/0/0/0 | 0/0/0/0 | 3,3 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 3 launches (620 tok) |
| img3 count | 3 | 3 / 3 / 3 | 3 / 3 / 3 | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | - | -0.00 [-0.01,-0.00] | -0.01 [-0.01,-0.00] | PASS |  | 2 launches (874 tok) |
| img3 letters | A,A,C | A,A,C / A,A,C / A,A,C | A,A,C / A,A,C / A,A,C | 9/9 | 9/9 | 0/0/0/0 | 0/0/0/0 | 3,3,3 | n/a | n/a | PASS |  | 2 launches (903 tok) |
| img4 count | 4 | 4 / 4 / 4 | 4 / 4 / 4 | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | - | -0.01 [-0.01,-0.01] | -0.01 [-0.01,-0.00] | PASS |  | 4 launches (1155 tok) |
| img4 letters | A,A,C,B | A,A,C,B / A,A,C,B / A,A,C,B | A,A,C,B / A,A,C,B / A,A,C,B | 12/12 | 12/12 | 0/0/0/0 | 0/0/0/0 | 3,3,3,3 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 4 launches (1187 tok) |
| near-max image | B | B / B / B | B / B / B | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | 3 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 4 launches (6734 tok) |
| over-max image | A | A / A / A | A / A / A | 3/3 | 3/3 | 0/0/0/0 | 0/0/0/0 | 3 | -0.00 [-0.00,-0.00] | -0.00 [-0.00,-0.00] | PASS |  | 5 launches (8024 tok) |

## Pooled

| pool | new-tp2-202611170201 found / slips (om/ins/swap/corr) | new-tp2-202611151621 | p (A slips more) | gate |
|---|---|---|---|---|
| needle, all sizes | 248/248 / 0 (0/0/0/0) | 247/248 / 1 (1/0/0/0) | 1.000 | PASS |
| images, all cases | 36/36 / 0 (0/0/0/0) | 36/36 / 0 (0/0/0/0) | 1.000 | PASS |

Pooled needle slips over different sizes are information (a defect is size-specific; the per-size gates decide); the pooled image test is a gate (image cases have 3 sends each, too few per case).

## Verdict per prefill size (new-tp2-202611170201)

| size | needle | count (info) | result | observed prefill |
|---|---|---|---|---|
| 128 | PASS | INFO 2/2 | PASS | 2 launches (75 tok) |
| 256 | PASS | INFO 3/3 | PASS | 3 launches (184 tok) |
| 512 | PASS | INFO 4/4 | PASS | 2 launches (421 tok) |
| 1024 | PASS | INFO 5/5 | PASS | 2 launches (922 tok) |
| 1536 | PASS | INFO 6/6 | PASS | 2 launches (1434 tok) |
| 2048 | PASS | INFO 2/2 | PASS | 3 launches (1936 tok) |
| 2560 | PASS | INFO 3/3 | PASS | 2 launches (2445 tok) |
| 3072 | PASS | INFO 4/4 | PASS | 3 launches (2961 tok) |
| 3584 | PASS | INFO 5/5 | PASS | 3 launches (3479 tok) |
| 4096 | PASS | INFO 6/6 | PASS | 2 launches (3969 tok) |
| 6K | PASS | INFO 2/2 | PASS | 3 launches (5932 tok) |
| 12K | PASS | INFO 3/3 | PASS | 5 launches (11735 tok) |
| 33K | PASS | INFO 4/4 | PASS | 10 launches (32980 tok) |

PREFILL_PROBE new-tp2-202611170201: PASS | new-tp2-202611151621: PASS
