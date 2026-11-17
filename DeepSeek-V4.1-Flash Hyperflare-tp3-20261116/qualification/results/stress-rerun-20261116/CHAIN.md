18:50:10 start (server already up, RESTORE=0): rerun of seq235's stress16 + post + scan on 27864a1d79e4 (identity = seq235), KV 5403008 (KV_WANT 5403008), guard 92 C/60 s, GPU_PL 'none', seq seq263, MemAvailable 20170 MiB -> ~/bench-results/ds41-qual-rerun-1116-1850
19:15:46 stress16 run PASS (25.6 min): PASS 32/32 long, ping 212/212, min free 64 MiB, bad lines 0, wall 25.6 min (min free < 480 MiB kit margin, >= 16 MiB floor); guard 40 WARN lines
19:15:46 stress16 PASS (25.6 min): PASS 32/32 long, ping 212/212, min free 64 MiB, bad lines 0, wall 25.6 min (min free < 480 MiB kit margin, >= 16 MiB floor)
19:15:55 post run PASS (0.1 min): smoke 2/2 thinking-off:ok '391' stop 0.06s thinking-on:ok '391' stop 0.12s | served KV 5403008 = launch | verify_fix 10: TRUE corruptions 0 (retried 0, candidates skipped 0)
19:15:55 post PASS (0.1 min)
19:15:55 scan run PASS (0.0 min): serve log 0 fault lines, kernel log 0 NVRM fault lines
19:15:55 scan PASS: serve log 0 fault lines, kernel log 0 NVRM fault lines
19:16:07 restore skipped (RESTORE=0): server stopped by PID, GPUs free
19:16:07 memwatch 891719 stopped
19:16:08 QUALIFICATION DONE preset B tier weights: all gates pass (seq235 cited + stress rerun) (WARN cited: decode32 gpqa); KV 5403008; restore skipped (RESTORE=0): server stopped by PID, GPUs free; 25 min | ~/bench-results/ds41-qual-rerun-1116-1850
