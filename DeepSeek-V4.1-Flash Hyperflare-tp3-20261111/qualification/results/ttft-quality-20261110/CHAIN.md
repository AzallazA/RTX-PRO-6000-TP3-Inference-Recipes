05:51 ttft quality start: package ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611090700; binary <fork> (07c045df); --scheduler-policy interactive; stages: verify eval tools stress
05:51 power: 450.00 W 450.00 W 450.00 W 
05:52 serving: gb202-tp3-202611090700 --scheduler-policy interactive ; cwd <fork>
05:52 KV 3014656; smoke 391
05:52 verify_fix ok: 10 runs, 1687 tokens: TRUE corruptions 0, natural-uncertainty flags 8
06:07 qual_all eval tools stress exit 0: eval-43: 42 / 43 (temperature clamped to 1.0)|tool calls: 60/60 OK (0 chose a different tool), concurrency 8, 25 s|stress: [ttft-guard-1110-c8] PASS: 16/16 long requests ok, pinger 67/67 ok, 9,512,241 prompt tok pushed, min free VRAM 60 MiB (408 samples), health=200, bad log|
06:07 gates: verify 0 TRUE; eval >= 42/43 (090700 balanced: 42/43); tools 60/60; stress PASS
06:08 default restored (gb202-tp3-202611090700); ttft quality chain done
