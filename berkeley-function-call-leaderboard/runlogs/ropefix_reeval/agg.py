"""Aggregate each model's OWN score/<key>/ files. score/data_overall.csv is a shared
snapshot that whichever `bfcl evaluate` finishes last rewrites, so it goes stale while
18 runs are in flight; the per-model score JSONs never do."""
import json,os,sys
B="/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard"
def a(k,sub,c):
    p=f"{B}/score/{k}/{sub}/BFCL_v4_{c}_score.json"
    if not os.path.exists(p): return None
    d=json.loads(open(p).readline()); return d["accuracy"],d["total_count"]
def mean(v): v=[x for x in v if x is not None]; return sum(v)/len(v) if v else None
rows=[l.rstrip("\n").split("\t") for l in open(f"{B}/runlogs/ropefix_reeval/jobs.tsv")]
rows.insert(0,["hgx11","","","abdelrahman-qwen-sft-jsonfull-448-out16k-FC-keepreason","json","jsonfull-448","",""])
print(f"{'MODEL':<18}{'NONLIVE':>9}{'LIVE':>8}{'MT':>8}{'base':>7}{'mfunc':>7}{'mparam':>7}{'long':>7}{'MEM':>8}  n")
out=[]
for r in rows:
    k,label=r[3],r[5]
    nl3=[a(k,"non_live",c) for c in("simple_python","simple_java","simple_javascript")]
    rest=[a(k,"non_live",c) for c in("multiple","parallel","parallel_multiple")]
    simple=mean([x[0] for x in nl3 if x])
    nl=mean([simple]+[x[0] for x in rest if x]) if simple is not None else None
    lv=[a(k,"live",c) for c in("live_simple","live_multiple","live_parallel","live_parallel_multiple")]
    lv=[x for x in lv if x]
    live=(sum(x[0]*x[1] for x in lv)/sum(x[1] for x in lv)) if lv else None
    mt=[a(k,"multi_turn",c) for c in("multi_turn_base","multi_turn_miss_func","multi_turn_miss_param","multi_turn_long_context")]
    mtv=[x[0] if x else None for x in mt]
    mem=[a(k,f"agentic/memory/{d}",c) for d,c in (("kv","memory_kv"),("vector","memory_vector"),("rec_sum","memory_rec_sum"))]
    f=lambda v:"   --  " if v is None else f"{v*100:6.2f} "
    n=sum(x[1] for x in [*[y for y in nl3 if y],*[y for y in rest if y],*lv,*[y for y in mt if y],*[y for y in mem if y]])
    print(f"{label:<18}{f(nl)}{f(live)}{f(mean(mtv))}"+"".join(f"{'  --  ' if v is None else f'{v*100:5.1f}'}" for v in mtv)+f"{f(mean([x[0] for x in mem if x]))}  {n}")
