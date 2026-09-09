#!/usr/bin/env python3
"""Append newly-scored models to all_bfcl_results.tsv (the Excel paste table).

APPEND-ONLY, on purpose. `bfcl evaluate` rewrites score/data_overall.csv with
only the models named on that invocation, so regenerating the whole table from
it would silently drop every historical row -- and would also blank the latency
column for models whose latency was measured on an earlier run. This script
therefore reads data_overall.csv, keeps only models the TSV has never seen,
appends them, and re-sorts by Overall descending.

The TSV is CRLF-terminated so it pastes cleanly into Excel; that is preserved.

Run from the berkeley-function-call-leaderboard directory:
    .venv/bin/python analysis/append_new_results.py
"""
import csv
import re
import sys

SRC = "score/data_overall.csv"
TSV = "analysis/all_bfcl_results.tsv"

# TSV column -> data_overall.csv column. Columns not listed are derived from the
# model's display name.
COL = {
    "Overall": "Overall Acc",
    "NonLive AST": "Non-Live AST Acc",
    "Live": "Live Acc",
    "MultiTurn": "Multi Turn Acc",
    "MT Base": "Multi Turn Base",
    "MT MissFunc": "Multi Turn Miss Func",
    "MT MissParam": "Multi Turn Miss Param",
    "MT LongCtx": "Multi Turn Long Context",
    "Memory": "Memory Acc",
    "Mem KV": "Memory KV",
    "Mem Vector": "Memory Vector",
    "Mem RecSum": "Memory Recursive Summarization",
    "Relevance": "Relevance Detection",
    "Irrelevance": "Irrelevance Detection",
    "Simple": "Non-Live Simple AST",
    "Multiple": "Non-Live Multiple AST",
    "Parallel": "Non-Live Parallel AST",
    "ParallelMult": "Non-Live Parallel Multiple AST",
    "Latency Mean s": "Latency Mean (s)",
}


def val(v):
    v = (v or "").strip()
    return "" if v in ("N/A", "") else v.rstrip("%")


def meta(name):
    """(corpus, epoch, ckpt) from a ModelConfig display_name."""
    m = re.search(r"SFT ([\w-]+) epoch (\d+) \(ckpt-(\d+)\)", name)
    if m:
        return m.group(1).replace("no-reason", "noreason"), m.group(2), m.group(3)
    # GRPO RL runs, all starting from mega ckpt-3850. Corpus names the reward,
    # Ckpt the RL step; Epoch stays blank because RL steps are not epochs.
    m = re.search(r"GRPO (judge|v2|v3) \(step (\d+)\)", name)
    if m:
        return "rl-" + m.group(1), "", "step-" + m.group(2)
    if "RL start point" in name:
        return "mega", "1", "3850"
    if "Instruct" in name:
        return "instruct", "", ""
    if "lenient" in name:
        return "lenient", "", ""
    if "untrained zero point" in name:
        return "base", "", ""
    return "?", "", ""


def main():
    raw = open(TSV, "rb").read().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.rstrip("\r\n").split(eol)
    header, rows = lines[0], lines[1:]
    fields = header.split("\t")
    known = {r.split("\t")[0] for r in rows}

    added = []
    for r in csv.DictReader(open(SRC)):
        name = r["Model"]
        if name in known:
            continue
        vm = re.search(r"\((FC(?: \w+)?|Prompt)\)\s*$", name)
        corpus, epoch, ckpt = meta(name)
        out = {
            "Model": name, "Corpus": corpus, "Epoch": epoch, "Ckpt": ckpt,
            "Variant": vm.group(1) if vm else "",
        }
        out.update({k: val(r.get(src)) for k, src in COL.items()})
        added.append("\t".join(out.get(f, "") for f in fields))

    if not added:
        print("nothing new in", SRC)
        return 0

    rows += added
    rows.sort(key=lambda x: -float(x.split("\t")[5]))
    open(TSV, "wb").write((eol.join([header] + rows) + eol).encode("utf-8"))
    print(f"appended {len(added)} row(s):")
    for a in added:
        print("  ", a.split("\t")[0], "->", a.split("\t")[5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
