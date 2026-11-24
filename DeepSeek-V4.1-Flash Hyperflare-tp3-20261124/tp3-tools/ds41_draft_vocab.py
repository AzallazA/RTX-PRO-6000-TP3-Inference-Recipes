#!/usr/bin/env python3
"""Frequency-ranked draft vocabulary for DS41's DSpark draft head (FR-Spec style; TP3Config.draft_vocab).

Two steps, so held-out numbers are honest and no text ever has to be kept:

  ds41_draft_vocab.py count TOKENIZER_JSON COUNTS_DIR SOURCE.jsonl... [--heldout-share 0.2] [--heldout-only NAME...]
      SOURCE.jsonl rows are {"doc": group id, "text": ...}; the source name is the file stem. Every document group is
      assigned to train or held-out by a hash of its id (sha1, about --heldout-share of the groups held out), so
      versions of one file / one conversation / one question never straddle the split. Writes COUNTS_DIR/<source>.json
      with token-id counts only ({"train": {id: n}, "heldout": {id: n}, ...}); no text.

  ds41_draft_vocab.py build TOKENIZER_JSON COUNTS_DIR OUT_DIR --weights WEIGHTS.json [--sizes 24576,32768,...]
      Ranks tokens by the weighted mixture of the sources' train distributions (WEIGHTS.json: {source: weight};
      sources without a weight are measured but not used for ranking), then keeps per vocabulary-parallel rank the
      special/added tokens of its shard plus its highest-ranked tokens, exactly ``columns`` = ceil(K / 3 / 128) * 128 per
      rank (the trimmed NVFP4 head needs 128-aligned rows; every rank must have the same width). Writes
      OUT_DIR/draft-vocab-<K>.json per size and OUT_DIR/curve.json: held-out miss rate (share of held-out tokens outside
      the kept set) per source and size, plus the weighted miss.

The table lists exactly ``columns`` tokens per rank, so the recipe never fills (``draft_vocab_columns`` refuses a
table that would need filling unless it carries a ``ranking`` to fill from by frequency). The draft head
proposes only these tokens; the target verifies with its full head, so outputs are unchanged and only acceptance on
the missing tokens drops (a target token outside the table always rejects that draft position).
"""
import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

VOCAB = 129280
WORLD = 3
ALIGNMENT = 64   # padded vocabulary partition (weights.load_padded_tensor)
MULTIPLE = 128   # trimmed-head rows per rank (NVFP4 draft head, draft_vocab_columns)


def shard_width(world=WORLD):
    return -(-VOCAB // (world * ALIGNMENT)) * ALIGNMENT


def columns_for(k, world=WORLD):
    per_rank = -(-k // world)
    return min(shard_width(world), -(-per_rank // MULTIPLE) * MULTIPLE)


def special_ids(tokenizer_json):
    return sorted({t["id"] for t in json.loads(Path(tokenizer_json).read_text()).get("added_tokens", []) if t["id"] < VOCAB})


def heldout(doc, share):
    return int(hashlib.sha1(str(doc).encode()).hexdigest()[:8], 16) < share * 2 ** 32


def count(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    out = Path(a.counts)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.sources:
        source = Path(name).stem
        only = source in (a.heldout_only or [])
        counts = {"train": collections.Counter(), "heldout": collections.Counter()}
        docs = {"train": set(), "heldout": set()}
        batch = []

        def flush():
            for split, enc in zip([s for s, _ in batch], tok.encode_batch([t for _, t in batch], add_special_tokens=False)):
                counts[split].update(enc.ids)
            batch.clear()

        for line in open(name, encoding="utf-8"):
            row = json.loads(line)
            split = "heldout" if only or heldout(row["doc"], a.heldout_share) else "train"
            docs[split].add(str(row["doc"]))
            batch.append((split, row["text"]))
            if len(batch) >= 512:
                flush()
        flush()
        record = {"source": source, "heldout_only": only, "heldout_share": a.heldout_share,
                  **{f"{s}_docs": len(docs[s]) for s in docs}, **{f"{s}_tokens": sum(counts[s].values()) for s in counts},
                  **{s: {str(t): n for t, n in sorted(counts[s].items())} for s in counts}}
        (out / f"{source}.json").write_text(json.dumps(record))
        print(f"{source:28s} train {record['train_tokens']:>11,} tok / {record['train_docs']:>6,} docs   held-out "
              f"{record['heldout_tokens']:>10,} tok / {record['heldout_docs']:>5,} docs", file=sys.stderr)


def load_counts(directory):
    records = {}
    for path in sorted(Path(directory).glob("*.json")):
        r = json.loads(path.read_text())
        for s in ("train", "heldout"):
            r[s] = {int(t): n for t, n in r[s].items()}
        records[r["source"]] = r
    return records


def ranking(records, weights):
    """Token ids by weighted mixture probability (train splits only); unseen tokens after, in id (BPE merge) order."""
    score = collections.defaultdict(float)
    for source, w in weights.items():
        r = records[source]
        total = sum(r["train"].values())
        if w <= 0 or not total:
            continue
        for t, n in r["train"].items():
            score[t] += w * n / total
    seen = sorted(score, key=lambda t: (-score[t], t))
    rest = [t for t in range(VOCAB) if t not in score]
    return seen + rest


def select(order, specials, columns, world=WORLD):
    """Per rank: its special tokens, then its highest-ranked tokens, exactly ``columns`` real ids."""
    width = shard_width(world)
    keep = []
    for r in range(world):
        lo, hi = r * width, min((r + 1) * width, VOCAB)
        mine = [t for t in specials if lo <= t < hi]
        if len(mine) > columns or hi - lo < columns:
            raise ValueError(f"rank {r}: {columns} columns cannot hold {len(mine)} special tokens / {hi - lo} real ids")
        have = set(mine)
        for t in order:
            if len(mine) >= columns:
                break
            if lo <= t < hi and t not in have:
                mine.append(t)
                have.add(t)
        keep.extend(mine)
    return sorted(keep)


def miss_rates(records, keep):
    keep = set(keep)
    out = {}
    for source, r in records.items():
        total = sum(r["heldout"].values())
        if total:
            out[source] = sum(n for t, n in r["heldout"].items() if t not in keep) / total
    return out


def weighted(miss, weights):
    used = {s: w for s, w in weights.items() if s in miss and w > 0}
    return sum(miss[s] * w for s, w in used.items()) / sum(used.values())


def build(a):
    records = load_counts(a.counts)
    weights = json.loads(Path(a.weights).read_text())
    gate_weights = {s: w for s, w in weights.items() if not s.startswith("_")}
    missing = [s for s in gate_weights if s not in records]
    if missing:
        raise SystemExit(f"weights name sources without counts: {missing}")
    specials = special_ids(a.tokenizer)
    order = ranking(records, gate_weights)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    for k in (int(x) for x in a.sizes.split(",")):
        columns = columns_for(k)
        keep = select(order, specials, columns)
        miss = miss_rates(records, keep)
        row = {"k": WORLD * columns, "requested_k": k, "columns_per_rank": columns,
               "weighted_heldout_miss": round(weighted(miss, gate_weights), 6),
               "heldout_miss": {s: round(m, 6) for s, m in sorted(miss.items())}}
        curve.append(row)
        if k in [int(x) for x in a.write.split(",") if x] or a.write == "all":
            table = {"k": WORLD * columns, "columns_per_rank": columns, "tokens": keep, "special_tokens": len(specials),
                     "weighted_heldout_miss": row["weighted_heldout_miss"], "heldout_miss": row["heldout_miss"],
                     "weights": gate_weights,
                     "sources": {s: {"train_tokens": r["train_tokens"], "heldout_tokens": r["heldout_tokens"],
                                     "train_docs": r["train_docs"], "heldout_docs": r["heldout_docs"]}
                                 for s, r in records.items()},
                     "method": "ds41_draft_vocab.py build: weighted train-split mixture ranking, per-rank top "
                               "columns_per_rank incl. special tokens; heldout_miss = share of held-out tokens "
                               "(document-level split) outside the table"}
            (out / f"draft-vocab-{WORLD * columns}.json").write_text(json.dumps(table))
        print(f"K {WORLD * columns:>7,} ({columns:>6,}/rank): weighted held-out miss {100 * row['weighted_heldout_miss']:.3f}%  "
              f"worst {max(miss, key=miss.get)} {100 * max(miss.values()):.2f}%", file=sys.stderr)
    (out / "curve.json").write_text(json.dumps({"weights": gate_weights, "curve": curve}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("tokenizer")
    c.add_argument("counts")
    c.add_argument("sources", nargs="+")
    c.add_argument("--heldout-share", type=float, default=0.2)
    c.add_argument("--heldout-only", nargs="*", help="source names used only for measuring (e.g. one-prompt sets)")
    b = sub.add_parser("build")
    b.add_argument("tokenizer")
    b.add_argument("counts")
    b.add_argument("out")
    b.add_argument("--weights", required=True)
    b.add_argument("--sizes", default="24576,32768,40960,49152,65536,98304,129024")  # 129024 = 3 x 43,008 (rank 2's real columns); 129408 aborted (review L3)
    b.add_argument("--write", default="all", help="comma list of requested sizes to write tables for, or 'all'")
    a = ap.parse_args()
    count(a) if a.cmd == "count" else build(a)


if __name__ == "__main__":
    main()
