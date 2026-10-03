#!/usr/bin/env bash
# One vLLM server, data-parallel over 2 GPUs, for one checkpoint. Usage: serve.sh <gpus e.g. 0,1> <ckpt-path> <port>
# Same as runlogs/nemotron32k/serve.sh except:
#  --data-parallel-size 2: two full model replicas behind one endpoint (tp stays 1).
#  --max-model-len 65536 with max_position_embeddings overridden to match (config.json says
#    40960). No rope scaling is added, so prompts under 40960 run exactly as before; positions
#    past 40960 are beyond anything the model was trained on.
#  --override-generation-config max_new_tokens 16384: vLLM applies generation_config.json's
#    max_new_tokens (2048 as shipped) as a server-wide cap over the per-request max_tokens.
#  Invoked via `python -m`: the venv was built under /local/muhsen, so its bin/ shebangs are stale.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
exec "$B/.venv/bin/python" -m vllm.entrypoints.cli.main serve "$2" \
  --served-model-name "$2" \
  --port "$3" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --hf-overrides '{"max_position_embeddings": 65536}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384}'
