#!/usr/bin/env bash
# Recreate the BFCL v4 evaluation venv from scratch.
#
# Deliberately a SEPARATE venv from ~/llm-pretrainer: BFCL pins vllm==0.8.5
# (which pulls torch 2.6.0), while the training project pins vllm==0.11.0 /
# torch 2.8.0. Letting one resolver touch the other's env breaks training.
#
# Usage:  bash setup_bfcl_env.sh
# Then:   source .venv/bin/activate     # NOT .venv/bin/bfcl directly --
#                                       # the handler shells out to `vllm serve`
#                                       # and needs it on PATH.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"

uv pip install -e ".[oss_eval_vllm]"

# --- Three fixes for gaps in the upstream dependency spec --------------------
# 1. qwen-agent 0.0.34 (an unpinned bfcl dependency) imports `soundfile` at
#    module load, but does not declare it. Without this every `bfcl` command
#    dies with ModuleNotFoundError before doing anything.
# 2. bfcl does not pin `transformers`, so a fresh resolve lands on 5.x, which
#    dropped `all_special_tokens_extended`. vllm 0.8.5 calls it while caching
#    the tokenizer, so the vLLM server exits 1 on startup. 4.51.3 is the
#    version contemporaneous with vllm 0.8.5 and has Qwen3 support.
# 3. uv venvs ship without setuptools; triton imports it when JIT-building
#    kernels, so the vLLM engine core dies on model load.
uv pip install soundfile "transformers==4.51.3" setuptools wheel

echo
echo "Done. Verify with:"
echo "  source .venv/bin/activate && bfcl models | grep qwen3-4b-sft"
