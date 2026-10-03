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

## Tables

The HTML pages came first and cover the original 22-checkpoint suite. The tables
below are the living record and now carry 39 rows — the four tooling corpora
(ToolACE, ToolMind graphsyn, Nemotron flat, Nemotron prefix), the Mega mix, the
Instruct/base reference points, and five GRPO RL checkpoints trained in the
sibling verl repo on top of Mega ckpt-3850.

| File | What it is |
|---|---|
| `all_bfcl_results.tsv` | Every scored model, one row each, sorted by Overall. **CRLF-terminated** so it pastes straight into Excel — preserve that. |
| `bfcl_best_per_corpus.tsv` | The best checkpoint per corpus, with a note column saying what each row is testing. The paste-into-a-slide version. |

### Updating them

```bash
.venv/bin/python analysis/append_new_results.py    # from the leaderboard dir
```

**Append-only, deliberately.** `bfcl evaluate` rewrites `score/data_overall.csv`
with only the models named on that invocation — it typically holds a handful of
the 34 — so regenerating the table from it would silently drop every historical
row. It would also blank the latency column for models whose latency was
measured on an earlier run; `nemotron-2941-FC`'s 15.92 s is already gone from
the CSV, which is how the trap was found. The script keeps only models the TSV
has never seen, appends them, re-sorts by Overall, and preserves the line
endings. Re-running it when there is nothing new is a no-op.

`bfcl_best_per_corpus.tsv` has a note column that cannot be derived from the
scores, so it is maintained by hand.

`meta()` derives the Corpus/Epoch/Ckpt columns from the display name. RL rows
carry `rl-<reward>` as the corpus and `step-<n>` as the checkpoint; Epoch is
left blank, because an RL step is not an epoch.

## The RL rows

Five rows come from GRPO runs in the verl repo, all starting from Mega
ckpt-3850 and all evaluated FC keepreason with a 4096-token output cap.

| Row | Overall | NonLive AST | Live | MultiTurn | Memory |
|---|---:|---:|---:|---:|---:|
| Mega ckpt-3850 (RL start point) | 34.41 | 83.29 | 78.46 | 22.62 | 17.20 |
| + GRPO judge v1 step 50 | 17.28 | 79.77 | 58.33 | 0.88 | 2.58 |
| + GRPO judge v1 step 100 | 12.36 | 60.69 | 13.32 | 0.12 | 3.66 |
| + GRPO judge v2 step 50 | 31.86 | 69.79 | 77.28 | 16.12 | 20.86 |
| + GRPO judge v2 step 100 | 33.97 | 73.81 | 79.20 | 21.62 | 23.87 |

**v1 is a reward-hacking artifact, not a training curve.** Its reward scored a
tool response correct if any emitted call matched, with no cost for surplus
calls, so the policy learned to spray: mean calls per response went 0.53 to
4.24. MultiTurn collapses to 0.88 and then 0.12 because BFCL grades multi-turn
on final API state, and a sprayed call mutates state irrecoverably.

**v2 replaced that with a hard gate** — wrong call count scores -1 outright.
That killed the spray and recovered MultiTurn to 21.62, but priced "4 of 5
calls correct" identically to prose, leaving no gradient on multi-call rows.
The cost is visible in the per-category numbers: `parallel_multiple` 86.00 to
57.00 and `parallel` 86.50 to 82.00 against the start point, which is nearly
the whole 9.5-point NonLive AST regression. Memory moved the other way, 17.20
to 23.87, driven by memory_kv 5.16 to 19.35 — nothing was optimising for it.

**Two evals of the same weights differ by 1.41 Overall.** `SFT mega epoch 1
(ckpt-3850) (FC keepreason)` and `SFT mega ckpt-3850 (RL start point)` are the
same checkpoint under two paths, scored 35.82 and 34.41. Most of the gap is
Memory (23.23 vs 17.20), which contributes 1.21 of it through the 0.40 Agentic
weight. Treat sub-1.5-point Overall differences between separate runs as noise,
and compare RL checkpoints against the RL start point row rather than the
older one.

## The nemotron-32k rows (abdelrahman-qwen)

Three checkpoints of a second Nemotron tool_calling SFT, run on hgx19
(8x H200) and started from `models/abdelrahman-qwen` rather than
Qwen3-4B-Base. One epoch of 737 steps, per-device batch 8 with
32,768-token packed sequences. Trained with `chat_template_tooling.jinja`
(every turn supervised) and evaluated FC keepreason. `final/` is
byte-identical to checkpoint-737.

| Row | Overall | NonLive AST | Live | MultiTurn | MT LongCtx | Memory |
|---|---:|---:|---:|---:|---:|---:|
| Old Nemotron ckpt-2941 (Qwen3-4B-Base, 16k) | 34.03 | 81.60 | 76.24 | 21.25 | 12.00 | 17.42 |
| nemotron-32k ckpt-500 | 34.07 | 80.25 | 75.80 | 21.12 | 13.00 | 18.92 |
| nemotron-32k ckpt-625 | 33.54 | 80.83 | 76.61 | 19.50 | 10.50 | 17.85 |
| nemotron-32k ckpt-737 | 34.09 | 80.12 | 76.46 | 21.50 | 14.00 | 18.06 |

**No gain from the new base, longer sequences or larger batch.** All three
checkpoints fall within 0.55 Overall of each other and of the old ckpt-2941.
That is inside the 1.41-point noise between two evals of the same weights (see
above), so step 500 had already reached the plateau. NonLive AST comes out about
1 point lower. ckpt-737 is the nominal best and was copied to
`/local/muhsen/verl/models/Qwen3-4B-Base-sft-737` for RL. Its registered path
under `models-sft/` no longer exists.

**The context window was raised to 40,960, and some prompts still exceed it.**
Keepreason puts every turn's `<think>` into the prompt, so long multi-turn
prompts grow past 32k. `max_position_embeddings` was raised from 32,768 to
40,960 in each converted `config.json`, the value official Qwen3-4B ships. The
server used the same `--max-model-len`. Each model still logged 31–35 HTTP 400
context overflows, about 30 of them in `multi_turn_long_context`. The largest
prompt reached 56k tokens. The old keepreason runs at 32,768 logged about 89 per
model. An overflowed entry is scored as a failure, so MT LongCtx for every
keepreason row is partly a context-limit measurement.

These three rows are now in `all_bfcl_results.tsv` (as `nemotron-32k` ckpt
`*-t1ctx64k`), with run logs in `../runlogs/nemotron32k_t1_16k/`.

Their display names contain commas -- `(ckpt-737, T=1.0, out 16k, ctx 64k)` --
and `bfcl` writes `score/data_overall.csv` unquoted, so each of these rows
carries 39 fields against the header's 36 and every column after `Model` is
shifted by three. Read naively they look plausible but wrong (Live 81.50,
MT 66.57 for ckpt-737, which are really Non-Live and Live values). The
`read_rows` helper in `append_new_results.py` detects the field-count surplus
and re-joins it into `Model`, which is the only free-text column. Keep that
repair if the script is rewritten.

## Verifying the eval policy

`verify_official_qwen_template.py` answers "is the `-FC` variant really what
official Qwen3 does?" without taking anyone's word for it. It renders the
`chat_template` field out of `Qwen/Qwen3-4B`'s own `tokenizer_config.json` with
Jinja and diffs it against `QwenFCHandler._format_prompt` — the function that
builds every prompt an `-FC` eval actually sends — across single-turn,
tool-loop, two-user-turn and three-user-turn conversations. All four are
byte-identical.

It also prints the reasoning-in-context census. On a conversation with three
reasoned assistant turns: the official template puts **0** think blocks in the
prompt, `-FC` puts **0**, `-FC-keepreason` puts **3**. Run it after any change
to a handler's `_format_prompt`.

## Transcripts

`render_multi_turn.py` dumps complete multi-turn transcripts as plain text —
system prompt, every tool response, every reasoning trace — into `renders/`.
Those 300 files are committed rather than regenerated, because `result/` is
gitignored and a fresh clone cannot rebuild them.

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
