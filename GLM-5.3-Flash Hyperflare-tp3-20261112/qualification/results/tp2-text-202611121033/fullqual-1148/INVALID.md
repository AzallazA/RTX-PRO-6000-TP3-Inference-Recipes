# Partly invalid run

The GPQA Diamond, tool-eval-bench and KV stress stages of this run are **INVALID**: a queue incident interrupted the run
and the server was killed while GPQA was still running (GPQA shows 189 of 198 questions scored with 9 errors; tool-eval
and stress found no server). They are kept only for completeness. The valid TP2 GPQA and tool-eval results are on the
TP2 image package (`../../tp2-image-202611130307/qual-hotel-gpqa-tooleval/`), the valid TP2 stress run is in
`../../tp2-image-202611130307/lean-chain/S2-qual/`.

Valid in this run: hotel-lights, long retrieval and long generation (all finished before the incident). The prefill
bench and the determinism probe (`prefill.*`, `det-*`) ran on freshly started servers right after it, within the two
minutes before the other chain's export was stopped; the determinism result matches TP3's, but treat the prefill
numbers as informational only.
