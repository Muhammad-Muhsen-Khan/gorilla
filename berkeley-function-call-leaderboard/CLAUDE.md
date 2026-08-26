# BFCL v4 evaluation — notes for future agents

This fork carries a local evaluation of Qwen3-4B tooling-SFT checkpoints on
BFCL v4. Everything below was learned the hard way during the first suite; it
is not derivable from the upstream README.

## Environment

BFCL runs from its **own venv** at `berkeley-function-call-leaderboard/.venv`,
deliberately separate from `~/llm-pretrainer` — BFCL pins vllm 0.8.5 / torch
2.6.0, the training project pins vllm 0.11.0 / torch 2.8.0. Letting one
resolver touch the other's env breaks training. Rebuild with
`bash setup_bfcl_env.sh`.

Three gaps in the upstream dependency spec, all handled in that script:

- `qwen-agent` imports `soundfile` without declaring it
- unpinned `transformers` resolves to 5.x, which vllm 0.8.5 cannot use — pin `4.51.3`
- uv venvs ship without `setuptools`, which triton needs

**Two invocation traps:**

- `source .venv/bin/activate` first. Calling `.venv/bin/bfcl` directly fails —
  the handler shells out to `vllm serve` and needs it on PATH.
- Always pass `--backend vllm`. The CLI default is `sglang`, which is not
  installed. The README says the default is vllm; the README is wrong.

## Model registry

Local checkpoints are registered in
`bfcl_eval/constants/custom_model_config.py`, kept out of
`bfcl_eval/constants/model_config.py` so upstream pulls of that file stay
conflict-free — it only gains a 7-line import.

Each entry puts the checkpoint path in `model_name`, so `--local-model-path` is
not needed. Keys look like `qwen3-4b-sft-noreason-448-FC`. Variants: `-FC`
(leaderboard-comparable), `-FC-nativefmt` (registered but unused), and a bare
prompt-mode name.

## Results on disk

| Path | Size | Contents |
|---|---|---|
| `result/` | 752 MB, 3,289 files | raw generations, one dir per model |
| `score/` | 785 MB, 466 files | graded output |
| `score/data_*.csv` | 24 KB total | the six summary tables — the only part worth quoting |

Both `result/` and `score/` are gitignored upstream. Do not try to commit them.
`analysis/suite.json` is the 23-row distillation the write-ups are built from.

Score files hold a header line, then **one record per failure** — an id absent
from a score file passed. They embed the full inference log and both state
dumps, so they run larger than the results they grade.

## Traps in the numbers

**The base model's 0.00 is a parser artifact.** Qwen3-4B-Base emits well-formed
`{"name": ..., "arguments": ...}` objects but never wraps them in `<tool_call>`
tags — 0 of 5,017 generations do. BFCL therefore scores it 0.00 on four
categories. `analysis/src/lenient_reparse.py` re-scores the same generations
leniently: **58.67% non-live, 40.49% live**. Its 100% irrelevance score is the
same artifact inverted — nothing parsed, so it "abstained" everywhere; the real
figure is 17.92%. Never quote the strict 0.00 as evidence the model cannot call
functions.

**Overall Acc is not an average.** It is
`0.10·NonLive + 0.10·Live + 0.10·Irrelevance + 0.30·MultiTurn + 0.40·Agentic`,
where `Agentic = mean(Memory, WebSearch)`. A **missing category scores 0, not
N/A**. We skipped `web_search` (free-tier SerpAPI key, 250 searches against
multi-hop tasks × 22 models), which costs every model up to 20 points of
Overall. Within-suite comparisons stay exact; do not put these numbers beside
the public leaderboard.

**Tool docs reach the model in a shape the SFT corpus never used.** BFCL
serialises them flat — `{"name": ..., "parameters": {"type": "dict"}}` — while
the training corpora used the OpenAI-wrapped form with `"type": "object"`. This
is a real train/eval shift, but it is what every official Qwen3 leaderboard
entry gets, so it was kept deliberately. The surrounding scaffold (system
`# Tools` block, `<tool_call>`, `<tool_response>`) is byte-identical.

**Multi-turn is graded on final API state**, not on the calls. A transcript can
read well and score zero. Prose is a hard stop: a generation with no parseable
call ends the turn, and the next user message arrives against unchanged state.

## Running a suite

`runlogs/driver.py` schedules models round-robin across free GPUs. Three bugs
worth not reintroducing:

- **Wait for GPU memory before spawning.** A killed vLLM server holds ~36 GiB
  for about a minute. Spawning immediately OOMs the next job, and an
  OOM-on-launch used to drain the whole queue in minutes. Guard on free MiB and
  retry the model instead of consuming it.
- **Skip busy GPUs, do not block on them.** Waiting inside the slot loop stalls
  every free GPU behind one busy one.
- **Check for already-running models before queueing.** Two writers appending
  to the same result file produces duplicate rows. `dedupe_results()` keeps the
  last row per id.

Generate everything first, then evaluate. `bfcl evaluate` raises
`ValueError: Length of model result (N) does not match length of test entries`
if a model is even one entry short — generate the missing entries (BFCL skips
existing ones, so it is fast) rather than degrading the whole suite to
`--partial-eval`.

## Write-ups

`analysis/` holds three self-contained HTML pages and the code that builds
them. See `analysis/README.md`.

## Git

`origin` is upstream `ShishirPatil/gorilla` — **do not push there**.
`fork` is `Muhammad-Muhsen-Khan/gorilla`. Work lands on branches of `fork`.

The SerpAPI key lives in `berkeley-function-call-leaderboard/.env`, which is
gitignored (`.gitignore:29`). Keep it out of committed files and out of logs.
