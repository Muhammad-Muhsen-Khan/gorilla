#!/usr/bin/env python3
"""Rebuild analysis/all_bfcl_results.tsv from score/data_overall.csv.

Unlike append_new_results.py (append-only), this REFRESHES rows that already
exist, because the RoPE defect (see analysis/README.md) invalidated the Live,
Multi-Turn and Memory numbers of every model served before 2026-10-04 and those
models were re-evaluated under their original registry keys.

Two things are preserved, which is why this is not a plain regeneration:

  * Latency. `bfcl evaluate` blanks "Latency Mean (s)" for models whose latency
    was measured on an earlier invocation, so a value already in the TSV is kept
    when the CSV offers none.
  * Rows the CSV no longer carries. Older models whose score/ directories were
    cleaned still belong in the historical table.

Rows superseded by the re-eval are DROPPED: the lenient re-scores, the T=1.0
repeat samples of olive-32k, the prompt-probe one-offs and 32B ckpt-954 were all
generated under the broken serve and cannot be repaired without regenerating.
"""
import csv, re, sys
sys.path.insert(0, "analysis")
from append_new_results import COL, SRC, TSV, read_rows, val, meta

SUPERSEDED = re.compile(
    r"lenient-json|T=1\.0 sample|parallel-call nudge|neutral system line"
    r"|ckpt-954 / epoch 1\.00"
    # the pre-ROPEFIX curriculum-250; the ROPEFIX row carries "/ ROPEFIX /" here
    r"|ckpt-250 / epoch 0\.53 / T="
)
VARIANT = re.compile(r"\((FC(?:[ \w-]+)?|Prompt|Qwen3\.5 XML tool calls)\)\s*$")


def build(name, r, fields, old_latency):
    vm = VARIANT.search(name)
    corpus, epoch, ckpt = meta(name)
    out = {"Model": name, "Corpus": corpus, "Epoch": epoch, "Ckpt": ckpt,
           "Variant": vm.group(1) if vm else ""}
    out.update({k: val(r.get(src)) for k, src in COL.items()})
    if not out.get("Latency Mean s") and old_latency:
        out["Latency Mean s"] = old_latency
    return "\t".join(out.get(f, "") for f in fields)


def main():
    raw = open(TSV, "rb").read().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.rstrip("\r\n").split(eol)
    header, oldrows = lines[0], lines[1:]
    fields = header.split("\t")
    li = fields.index("Latency Mean s")
    old = {}
    for line in oldrows:
        c = line.split("\t")
        old[c[0]] = {"line": line, "lat": c[li] if len(c) > li else ""}

    rows, refreshed, added = [], [], []
    seen = set()
    for r in read_rows(SRC):
        name = r["Model"]
        if SUPERSEDED.search(name):
            continue
        seen.add(name)
        line = build(name, r, fields, old.get(name, {}).get("lat", ""))
        rows.append(line)
        if name in old:
            if line != old[name]["line"]:
                refreshed.append(name)
        else:
            added.append(name)

    kept, dropped = [], []
    for name, d in old.items():
        if name in seen:
            continue
        (dropped if SUPERSEDED.search(name) else kept).append(name)
        if not SUPERSEDED.search(name):
            rows.append(d["line"])

    rows.sort(key=lambda x: -float(x.split("\t")[5] or 0))
    open(TSV, "wb").write((eol.join([header] + rows) + eol).encode("utf-8"))
    print(f"{TSV}: {len(rows)} rows")
    print(f"  refreshed {len(refreshed)}  added {len(added)}  "
          f"kept-from-history {len(kept)}  dropped-superseded {len(dropped)}")
    for n in added:     print("   + ", n[:88])
    for n in dropped:   print("   - ", n[:88])
    return 0


if __name__ == "__main__":
    sys.exit(main())
