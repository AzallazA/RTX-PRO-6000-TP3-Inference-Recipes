# Release binary checks, 2026-11-20

| Folder | Run | What |
| --- | --- | --- |
| `steersync/` | binary with steering inside BRRR plus the picked upstream fixes | decode A/B against the previous default (three fresh-boot rounds each), boot with a steering keys file, the access gate check (`tools/steering_access_check.py`), key scans of logs, process arguments and environment, post-run smoke and corruption check |
| `nocli/` | final binary (`a7c3a04a`: no key option on the command line) | the same access gate check, the command-line key option rejected as an unknown argument, key scans |

Each folder has the run's `SUMMARY.md`, `GATES.tsv` and `IDENTITY.txt`. Key values never appear in these files (the
check reads the key from a file and records only statuses and token counts); private paths are replaced by
placeholders.
