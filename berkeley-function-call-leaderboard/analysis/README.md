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
below are the living record and now carry 91 rows: the original four tooling
corpora (ToolACE, ToolMind graphsyn, Nemotron flat, Nemotron prefix), the Mega
mix, the Instruct/base reference points, five GRPO RL checkpoints from the
sibling verl repo, and everything added since — the olive corpora (32k, json-full,
remix-regen, lengthbias and four XML variants), the nemotron-32k base, the
curriculum blends including the two toucan hold-outs, the ten leave-one-out
corpus-probe arms, and the 32B.

| File | What it is |
|---|---|
| `all_bfcl_results.tsv` | Every scored model, one row each, sorted by Overall. **CRLF-terminated** so it pastes straight into Excel — preserve that. |
| `bfcl_best_per_corpus.tsv` | The best checkpoint per corpus, with a note column saying what each row is testing. The paste-into-a-slide version. |

> **Rows generated before 2026-10-04 were invalid; the table has been rebuilt.**
> vLLM silently discarded the trained RoPE configuration (see below), so the
> Live, Multi-Turn and Memory numbers of every affected model were wrong — many
> Multi-Turn cells read exactly `0.00`. All 18 affected models were
> re-evaluated, `all_bfcl_results.tsv` was refreshed in place, and the 22 rows
> that cannot be repaired without regenerating were dropped. The table is
> authoritative again as of 2026-10-10 (91 rows).

### Updating them

```bash
.venv/bin/python analysis/append_new_results.py    # add newly scored models
.venv/bin/python analysis/rebuild_results.py       # ALSO refresh rows already present
```

Use `append_new_results.py` for the normal case. Use `rebuild_results.py` when a
model has been **re-evaluated under its existing registry key** — the append
script skips any name the TSV already has, so it would silently keep the stale
numbers. `rebuild_results.py` refreshes those rows, preserves a latency value the
CSV no longer carries, keeps rows the CSV has dropped, and removes the rows
superseded by the RoPE re-eval.

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

## The RoPE defect (2026-10-03) — read this before trusting any older row

`transformers` 5.2.0 writes RoPE **only** as a nested block:

```json
"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}
```

vLLM 0.8.5 — the version BFCL pins in `pyproject.toml:74` — never reads that
key; the string appears nowhere in the package. `vllm/model_executor/models/qwen3.py:160`
does `getattr(config, "rope_theta", 1000000)`, finds nothing, and falls back to
a default. The model then emits coherent text at short context and degenerates
into token noise past roughly 4k prompt tokens.

**Proved, not inferred.** Two hybrid directories were built with the weights
**hard-linked** (byte-identical, `cmp`-verified) and only the config files
swapped for a known-good model's. The degeneration cliff moved:

| hybrid | before | after |
|---|---|---|
| json-full ckpt-448 | collapse at 3,994 tok | clean to 7,775 |
| curriculum ckpt-469 | collapse at 3,994 tok | clean to 26,826 |

**The fix is additive.** A flat `rope_theta` was added to every converted
`config.json` while keeping `rope_parameters`, so transformers 5.x and vLLM
0.8.5 both read the right value from one file. 38 configs patched, 16 already
flat, 54 total under `models-sft/`; each patched file has a `config.json.pre-ropefix`
backup. No `--hf-overrides` is needed any more, and `runlogs/ropefix_reeval/serve.sh`
(max_position_embeddings override only) is the right serve shape. The in-place
patch was then validated by serving a patched directory with no rope override:
clean `OK` at 226/2285/3392/3994/5041/7775 tokens, where the same weights under
the broken serve gave `NO CALL` at 3,994 and `DEGENERATE` beyond.

**What the bug cost.** It understated short-prompt categories hardly at all —
non-live prompts are 226–3,392 tokens, below the cliff — and wrecked everything
long. The worst case:

| curriculum ckpt-469 | broken | correct | delta |
|---|---:|---:|---:|
| Overall | 20.83 | 36.61 | +15.78 |
| Non-Live AST | 51.75 | 69.17 | +17.42 |
| Live | 54.26 | 75.50 | +21.24 |
| Multi-Turn | **0.00** | **31.75** | +31.75 |
| Memory | 6.45 | 20.22 | +13.77 |

Every `0.00` Multi-Turn cell in the old table is this artifact, not a model that
cannot hold a conversation. Correct runs also finish roughly an order of
magnitude faster, because the broken serve spent its time emitting 16k-token
noise blocks that hit the output cap.

18 affected models were re-evaluated on 2026-10-04 (`runlogs/ropefix_reeval/`),
including the 32B at two checkpoints that had never been scored at all.

## Reproducibility bands — what counts as signal

Three evaluations of the **same** nemotron-32k ckpt-737 weights under identical
settings (T=0.001, out 16k, ctx 41k, same handler):

| run | Overall | Multi-Turn | MT LongCtx |
|---|---:|---:|---:|
| ckpt-737 | 34.09 | 21.50 | 14.00 |
| ckpt-737 (T=0.001 row) | 33.94 | 19.50 | 11.50 |
| ckpt-737 REPRO | 33.87 | 21.00 | 16.00 |

So the per-column run-to-run band is about **Overall ±0.1, Multi-Turn ±1.5,
MT LongCtx ±4.5** (11.50 → 16.00 on identical weights). Overall and Non-Live are
the trustworthy columns; `MT LongCtx` is the least trustworthy cell in the suite
and simultaneously the most context-rejected one, so treat any LongCtx delta
under ~5 points as noise.

## The context ceiling understates Multi-Turn as well as Memory

Serving at `--max-model-len 40960` rejects every prompt above it with HTTP 400,
and BFCL records the entry as a failure. Measured across the 2026-10-04 batch:

- **64–284 rejected requests per model**
- **prompt lengths 40,960 – 86,650 tokens**
- affected categories are **multi-turn and memory**, with
  `multi_turn_long_context` the **largest single bucket** in every model checked

The memory ones are `*_prereq` turns — the conversations that populate the
memory store before the scored questions — so dropping them leaves the scored
cases reading an incomplete store. Memory and Multi-Turn are therefore **floors,
not measurements**.

**No ceiling fixes this.** The longest prompts are 2.6× these models' 32k
training length, so a 64k serve would still reject them while pushing positions
far out of distribution. 40,960 is kept for comparability with every existing
`ctx 41k` row. The bias is common-mode across models, so rankings and
between-arm deltas survive; absolute levels do not. Each run under
`runlogs/probe_loo/` and `runlogs/notoucan*/` writes a `<label>.ctxnote.txt`
recording the count, the true prompt range and this caveat.

## `score/data_overall.csv` goes stale while runs are in flight

`bfcl evaluate` rewrites the whole shared CSV from the per-model `score/<key>/`
directories. With several evals finishing concurrently, whichever one writes
last wins, so rows for models scored after it silently keep their **previous**
values. Worse, a category that lands 1–2 entries short makes a strict evaluate
abort with `ValueError` **after** writing some score files, leaving a model's
directory a mix of new and stale per-category scores.

Two consequences, both of which produced wrong tables during this work:

- Check the **exit code**, not just a completion string in the log. 7 of 18
  scoring runs exited `rc=1` while still logging "EVALUATION COMPLETE".
- To get a coherent table: delete `score/<key>/` for every model involved, then
  run **one** `bfcl evaluate` over all of them with `--partial-eval`.

Per-model score JSONs are authoritative; the aggregate CSV is a snapshot.

## The leave-one-out corpus probe (2026-10-05)

Ten arms, one per held-out source, all from the same base (nemotron-32k final),
seed 1, global batch 56, one epoch, final checkpoint only, strict
`-FC-keepreason`. Run logs in `runlogs/probe_loo/`. Negative delta = removing the
source **hurt**, i.e. it contributes.

| arm | Overall | dOverall | dNonLive | dMT |
|---|---:|---:|---:|---:|
| all9 (baseline) | 35.14 | — | — | — |
| no-toucan | **37.30** | **+2.16** | **+6.67** | +2.75 |
| no-toolace | 35.47 | +0.33 | −5.96 | +0.63 |
| no-openseeker | 34.18 | −0.96 | −10.02 | +1.88 |
| no-swe-zero-openhands | 34.14 | −1.00 | −7.12 | +0.26 |
| no-terminal-corpus | 34.12 | −1.02 | −8.77 | +1.50 |
| no-openresearcher | 33.94 | −1.20 | −7.58 | −0.12 |
| no-nemotron-agentic-interactive | 33.86 | −1.28 | −6.87 | +0.26 |
| no-nemotron-agentic-tool | 33.57 | −1.57 | −7.60 | −0.50 |
| no-nemotron-post-training | **33.22** | **−1.92** | **−14.46** | +3.00 |

**Drop toucan.** Removing it improved all nine scored columns. It also had the
lowest train_loss of the ten arms (0.6286 against the baseline's 0.6871), the
signature of easy-to-fit, low-diversity data that is memorised without buying
capability.

**Keep nemotron-post-training** — the largest Overall and Non-Live losses when
held out.

**Everything else is inside noise on Overall**: seven arms span −0.96 to −1.57,
a 0.6-point spread at one seed. Do not rank them against each other. And do not
drop openseeker, swe-zero-openhands, terminal-corpus or openresearcher on these
numbers: ~4% of training tokens were truncated at the 32k max-length,
concentrated in exactly those long trajectories, and their target column is both
the most context-rejected and the noisiest. A null for those four means "neither
the training nor the measurement could see this", not "this source does not
help". The `irrelevance`/`relevance` columns swing without pattern and should be
ignored in both directions.

## The no-toucan follow-ups confirm it, with a caveat

Two full curriculum runs were trained with toucan held out and evaluated on
bm-04 (`runlogs/notoucan/`, `runlogs/notoucan_prop/`):

| model | Non-Live | Live | Multi-Turn | MT LongCtx | Memory |
|---|---:|---:|---:|---:|---:|
| curriculum-469 (with toucan) | 69.17 | 75.50 | **31.75** | 25.50 | 20.22 |
| no-toucan ckpt-150 | 80.69 | 76.02 | 24.62 | 18.50 | 21.29 |
| no-toucan ckpt-403 | 81.73 | 77.28 | 25.12 | 16.00 | 14.62 |
| no-toucan **prop-repeat** ckpt-300 | 81.12 | 76.31 | 28.25 | 23.00 | 22.15 |
| no-toucan **prop-repeat** ckpt-455 | **81.71** | **77.42** | 28.38 | 19.50 | **24.52** |

**The Non-Live gain replicates at full scale**: 81.7 against curriculum-469's
69.17, +12.5 points, and the figure matches the probe's no-toucan arm (81.73)
almost exactly. Non-Live has the ±0.1 band, so this is real.

**The probe overpredicted the Multi-Turn benefit.** It showed +2.75 for removing
toucan; the full runs came in 3.4–6.6 points *below* curriculum-469. Read the
probe's Non-Live column, not its Multi-Turn column.

**Proportional repeat is the better hold-out.** Upsampling the remaining sources
to replace toucan's tokens recovers +3.3 Multi-Turn over the plain hold-out and
gives the best Memory of any 4B model here (24.52), and it does not decay late:
300 → 455 improves or holds every column, where the plain hold-out regressed
(Memory 21.29 → 14.62). Its weak spot is `simple_java` 57.00 / `simple_javascript`
54.00 against `simple_python` 88.00; the parallel cells are strong (85.50 /
87.50).

## The 32B

`olive-32b-olive-tooling-32k-bs3-3ep-30gpu` is the strongest model in the table
once served correctly — and its previously recorded row was pure RoPE artifact
(ckpt-954: 23.12 Overall, 0.50 Multi-Turn).

| checkpoint | Overall | Non-Live | Live | Multi-Turn | Memory |
|---|---:|---:|---:|---:|---:|
| ckpt-1431 (epoch 1.50) | 42.96 | 46.17 | 73.13 | **55.58** | 29.25 |
| ckpt-2385 (epoch 2.49) | **43.61** | 52.40 | 74.91 | 51.93 | **33.33** |

Its margin over the best 4B comes almost entirely from Multi-Turn — 51.9–55.6
against 38.9. Note ckpt-1431 beats the later checkpoint on Multi-Turn while
losing on Non-Live and Memory.

**The run is incomplete.** `trainer_state.json` at step 2385 reports
`max_steps 2871`, `num_train_epochs 3`, `epoch 2.492` — so the directory name
says `3ep` but no 3-epoch checkpoint was ever written, and 2385 is the latest
surviving checkpoint of a run that stopped 486 steps short. Both
`models-sft/` and `checkpoints/` hold the same five (477/954/1431/1908/2385);
the raw DeepSpeed shards and RNG state are intact, so it could be resumed.
The fixed final checkpoint is backed up at
`s3://muhsen-avey-bucket/pi-machine-backup/fixed-models/olive-32b-olive-tooling-32k-bs3-3ep-30gpu/checkpoint-2385/`
(13 objects, 65,543,787,693 bytes, verified file by file).

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
