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


def read_rows(path):
    """DictReader, but repair rows split by unquoted commas in the Model name.

    bfcl writes score/data_overall.csv unquoted, so a display_name containing
    commas -- e.g. "(ckpt-737, T=1.0, out 16k, ctx 64k)" -- emits extra fields
    and shifts every column after it. Three nemotron T=1.0 rows do this. Model
    is the only free-text column, so any surplus fields belong to it and are
    re-joined with ", " before the row is zipped to the header.
    """
    rdr = csv.reader(open(path))
    header = next(rdr)
    mi = header.index("Model")
    for row in rdr:
        if not row:
            continue
        surplus = len(row) - len(header)
        if surplus > 0:
            row = row[:mi] + [",".join(row[mi:mi + surplus + 1])] + row[mi + surplus + 1:]
        if len(row) != len(header):
            print(f"  SKIP malformed row ({len(row)} fields): {row[mi][:60]!r}")
            continue
        yield dict(zip(header, row))


def val(v):
    v = (v or "").strip()
    return "" if v in ("N/A", "") else v.rstrip("%")


def meta(name):
    """(corpus, epoch, ckpt) from a ModelConfig display_name.

    Ordered most-specific first: the generic "SFT <corpus> (ckpt-N / epoch X"
    form would otherwise swallow the 32B rows and the prompt-probe variants,
    making a probe indistinguishable from a plain re-eval of the same ckpt.
    """
    # --- original format: "SFT <corpus> epoch N (ckpt-M)" ------------------
    m = re.search(r"SFT ([\w-]+) epoch (\d+) \(ckpt-(\d+)\)", name)
    if m:
        return m.group(1).replace("no-reason", "noreason"), m.group(2), m.group(3)

    # same shape with settings appended: "(ckpt-M REPRO / ..." or "(ckpt-M, T=1.0, ..."
    m = re.search(r"SFT ([\w-]+) epoch (\d+) \(ckpt-(\d+)(\s+REPRO)?\s*[,/]", name)
    if m:
        ckpt = m.group(3)
        if m.group(4):
            ckpt += "-repro"
        elif "T=1.0" in name:
            ckpt += "-t1ctx64k"
        return m.group(1), m.group(2), ckpt

    # --- 32B run: tag the corpus so it never merges with the 4B rows -------
    m = re.search(r"olive-32b SFT ([\w-]+) \(ckpt-(\d+)\s*/\s*epoch ([\d.]+)", name)
    if m:
        return m.group(1) + "-32b", m.group(3), m.group(2)

    # --- prompt probes on an already-known checkpoint ----------------------
    m = re.search(
        r"SFT ([\w-]+) \(ckpt-(\d+)[^)]*?"
        r"(parallel-call nudge before tools|parallel-call nudge|neutral system line)", name)
    if m:
        probe = {"parallel-call nudge": "nudge",
                 "parallel-call nudge before tools": "nudge-pre",
                 "neutral system line": "neutral"}[m.group(3)]
        return m.group(1), "", m.group(2) + "-" + probe
    m = re.search(r"SFT ([\w-]+) \(ckpt-(\d+)[^)]*?(parallel-call nudge)", name)
    if m:
        return m.group(1), "", m.group(2) + "-nudge"

    # --- RL runs -----------------------------------------------------------
    m = re.search(r"GRPO ([\w-]+) on olive SFT ckpt-(\d+) \(step (\d+)", name)
    if m:
        return "rl-" + m.group(1), "", "step-" + m.group(3)
    m = re.search(r"GRPO (judge|v2|v3) \(step (\d+)\)", name)
    if m:
        return "rl-" + m.group(1), "", "step-" + m.group(2)
    m = re.search(r"RL v4 from SFT-(\d+) step (\d+)", name)
    if m:
        return "rl-v4", "", "step-" + m.group(2)
    if "RL start point" in name:
        return "mega", "1", "3850"

    # --- generic newer format, incl. the pass@k temperature samples --------
    m = re.search(r"SFT ([\w+-]+(?: [\w+-]+)?) \(ckpt-(\d+)\s*/\s*epoch ([\d.]+)", name)
    if m:
        corpus, ckpt, epoch = m.group(1), m.group(2), m.group(3)
        t1 = re.search(r"T=1\.0 sample (\d+)", name)
        if t1:
            ckpt += "-t1s" + t1.group(1)
        return corpus, epoch, ckpt

    # --- one-offs ----------------------------------------------------------
    if "BASE dual-mode-sft parent" in name:
        return "base-parent", "", ""
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
    for r in read_rows(SRC):
        name = r["Model"]
        if name in known:
            continue
        vm = re.search(r"\((FC(?:[ \w-]+)?|Prompt|Qwen3\.5 XML tool calls)\)\s*$", name)
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
