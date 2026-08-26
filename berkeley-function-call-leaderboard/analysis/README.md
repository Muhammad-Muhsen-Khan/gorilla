# Analysis artifacts — Qwen3-4B tooling-SFT on BFCL v4

Three self-contained HTML pages written up from the 22-checkpoint suite, plus
the code that generates them. Each page is a single file with no external
assets (Google Fonts aside), so opening it in a browser is enough.

## Pages

| File | What it covers |
|---|---|
| `artifacts/twenty-two-checkpoints.html` | The full write-up: what BFCL v4 measures, the run, all 22 models, five findings, three caveats. |
| `artifacts/reason-vs-no-reason.html` | The two SFT runs cell against cell across ten epochs, plus 20 complete multi-turn transcripts (5 tasks × 4 models) rendered with system prompt, tool responses and reasoning traces. |
| `artifacts/multi-turn-collapse.html` | The failure analysis that came first: why multi-turn sits at the floor, read from transcripts rather than scores. |

## Data

`suite.json` is the compact metric table — 23 rows, one per model, derived from
`score/data_overall.csv`. Everything in the charts and tables comes from it, so
the pages rebuild without the 1.6 GB of `result/` and `score/` (both gitignored).

The transcript renders are the exception: `src/compare/sample.py` reads
`result/*/multi_turn/` directly. Without those directories you can still open
the finished page — the transcripts are baked into the HTML.

## Rebuilding

Each subdirectory of `src/` produces one page. Run scripts from their own
directory; paths resolve relative to the script.

```bash
# reason-vs-no-reason.html  (needs the venv: sample.py imports bfcl_eval)
source ../../../.venv/bin/activate
cd src/compare
python sample.py     # draws 5 multi-turn tasks with a fixed seed -> bundle.json
python render.py     # turns them into transcript cards    -> transcripts.json
python mk.py         # paired table + epoch charts         -> parts.json
python assemble.py   # head.html + body1 + body2 + parts   -> reason_vs_noreason.html

# twenty-two-checkpoints.html
cd src/report
python mkchart.py    # charts + model table -> _charts.html, then splice into r_*.html
```

`sample.py` is seeded (`random.Random(20260826)`), so the pipeline reproduces
`artifacts/reason-vs-no-reason.html` byte for byte.

Intermediates (`bundle.json`, `transcripts.json`, `parts.json`, `_charts.html`,
`_frag.html`) are gitignored — they regenerate.

## One number to carry

`src/lenient_reparse.py` re-scores existing generations with a permissive
parser. The untrained base emits well-formed call objects but never wraps them
in `<tool_call>` tags, so BFCL scores it **0.00** on four categories. Re-parsing
the same generations — no new inference, no new sampling — puts it at **58.67%**
non-live and **40.49%** live. Any comparison against "the base model" that uses
the strict 0.00 is measuring the parser, not the model.
