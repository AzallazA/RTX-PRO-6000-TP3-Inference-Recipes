2026-11-08T12:59:41 — outputs: ~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-20261108/qualification/run-20261108-c8
2026-11-08T12:59:41 — qualification start: packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080909 stages: eval tools lavd hotel retrieval longgen eos livebench stress
2026-11-08T12:59:41 — KV 2461184, power 450.00 W 450.00 W 450.00 W 
2026-11-08T13:00:18 — eval-43: 43 / 43 (temperature clamped to 1.0)
2026-11-08T13:00:51 — tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 33 s
2026-11-08T13:10:24 —  DONE lavd-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-08T13:10:24 — 
2026-11-08T13:10:24 — 
2026-11-08T13:10:24 — long generation: 
2026-11-08T13:14:19 — respect-eos bench: │ 32k │ 293.4 37/3 │ 717.1 47/6 │ 963.7 57/8 │ loops: 0
2026-11-08T13:21:54 — LiveBench coding: category coding -- deepseek-v4.1-flash-flare-20261108 83.9 
2026-11-08T13:21:54 — stress: 
2026-11-08T13:21:54 — qualification done
2026-11-08T13:22:41 — outputs: ~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-20261108/qualification/run-20261108-c8
2026-11-08T13:22:41 — qualification start: packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611080909 stages: hotel retrieval longgen stress
2026-11-08T13:22:41 — KV 2461184, power 450.00 W 450.00 W 450.00 W 
2026-11-08T13:32:33 —  DONE hotel-lights-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-08T13:34:40 —  DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
2026-11-08T13:40:41 — long generation: 4 runs, fails 0 | completions [25970, 22882, 15543, 16387] | repeat_8gram [0.0, 0.0, 0.0006, 0.0]
2026-11-08T13:54:53 — stress: [flare-20261108-c8] PASS: 16/16 long requests ok, pinger 120/120 ok, 9,510,861 prompt tok pushed, min free VRAM 78 MiB (419 samples), health=200, bad log lines=0, wall 14.2 min
2026-11-08T13:54:53 — qualification done
