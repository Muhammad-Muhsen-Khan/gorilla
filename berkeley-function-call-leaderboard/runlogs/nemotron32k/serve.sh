#!/usr/bin/env bash
# One vLLM server on ONE GPU for one checkpoint. Usage: serve.sh <gpu> <ckpt-path> <port>
#  --served-model-name = path: BFCL sends model=model_path_or_id.
#  --max-model-len 40960: matches the edited config.json, so keepreason
#    multi-turn prompts past 32k still get the full 4096-token output budget.
#  --override-generation-config: the template's generation_config.json carries
#    max_new_tokens 2048, which vLLM applies as a server-wide cap.
set -u
cd /local/muhsen/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/local/muhsen/hf-cache TMPDIR=/local/muhsen/tmp
exec vllm serve "$2" \
  --served-model-name "$2" \
  --port "$3" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 4096}'
