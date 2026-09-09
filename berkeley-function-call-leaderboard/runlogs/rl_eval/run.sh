#!/usr/bin/env bash
# One model, one dedicated vLLM server. Usage: run.sh <model-key> <port>
set -u
cd /home/ubuntu/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate
CATS="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,\
irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,\
live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,\
multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum"
export LOCAL_SERVER_PORT="$2"
bfcl generate --model "$1" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 32
echo "GENERATION DONE for $1 rc=$?"
