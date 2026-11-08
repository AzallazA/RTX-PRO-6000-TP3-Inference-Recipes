2026-11-08T13:56:32 — outputs: ~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-20261108/qualification/run-20261108.1-c8
2026-11-08T13:56:32 — qualification start: packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611081024 stages: eval tools lavd hotel retrieval longgen eos livebench stress
2026-11-08T13:56:32 — KV 2467200, power 450.00 W 450.00 W 450.00 W 
2026-11-08T13:57:08 — eval-43: 43 / 43 (temperature clamped to 1.0)
2026-11-08T13:57:42 — tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 34 s
2026-11-08T14:06:59 —  DONE lavd-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-08T14:17:18 —  DONE hotel-lights-c8 rc=0 | exact 30 / wrong 0 / truncated 0 | []
2026-11-08T14:19:25 —  DONE long retrieval rc=0: DONE 128000 pass DONE 900000 pass
2026-11-08T14:25:54 — long generation: 4 runs, fails 0 | completions [28196, 20263, 12677, 30297] | repeat_8gram [0.0002, 0.0005, 0.0, 0.0008]
2026-11-08T14:29:50 — respect-eos bench: │ 32k │ 290.1 29/3 │ 716.9 49/6 │ 963.3 57/8 │ loops: 0
2026-11-08T14:37:37 — LiveBench coding: category coding -- deepseek-v4.1-flash-flare-20261108-1 81.6 
2026-11-08T14:42:05 — outputs: ~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-20261108/qualification/run-20261108.1-c8
2026-11-08T14:42:05 — qualification start: packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611081024 stages: stress
2026-11-08T14:42:05 — KV 2467200, power 450.00 W 450.00 W 450.00 W 
2026-11-08T14:56:13 — stress: [flare-20261108-1-c8] PASS: 16/16 long requests ok, pinger 117/117 ok, 9,514,154 prompt tok pushed, min free VRAM 78 MiB (416 samples), health=200, bad log lines=0, wall 14.1 min
2026-11-08T14:56:13 — qualification done
