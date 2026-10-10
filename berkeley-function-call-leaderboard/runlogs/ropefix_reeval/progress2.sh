#!/usr/bin/env bash
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
D=$B/runlogs/ropefix_reeval
cd "$B"
"$B/.venv/bin/python" - <<'PY'
import json,os,glob
B="/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard"
CATS="simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context memory_kv memory_vector memory_rec_sum".split()
def _df(c): return f"{B}/bfcl_eval/data/BFCL_v4_memory.json" if c.startswith("memory_") else f"{B}/bfcl_eval/data/BFCL_v4_{c}.json"
exp={c:sum(1 for _ in open(_df(c))) for c in CATS}
jobs=[l.rstrip("\n").split("\t") for l in open(f"{B}/runlogs/ropefix_reeval/jobs.tsv")]
jobs.append(["hgx11","","","abdelrahman-qwen-sft-jsonfull-448-out16k-FC-keepreason","json","jsonfull-448","",""])
print(f"{'LABEL':<18}{'NODE':<7}{'CASES':>12}  {'PCT':>4}  REMAINING")
for j in jobs:
    node,key,label=j[0],j[3],j[5]
    got=0; miss=[]
    for c in CATS:
        fs=glob.glob(f"{B}/result/{key}/**/BFCL_v4_{c}_result.json",recursive=True)
        n=sum(1 for f in fs for _ in open(f))
        got+=min(n,exp[c])
        if n<exp[c]: miss.append(f"{c}({n}/{exp[c]})")
    tot=sum(exp.values())
    print(f"{label:<18}{node:<7}{got:>6}/{tot:<5}  {got*100//tot:>3}%  {' '.join(miss) if miss else 'COMPLETE'}")
PY
