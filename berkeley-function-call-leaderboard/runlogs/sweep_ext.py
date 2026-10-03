import json, os, sys
sys.path.insert(0, os.getcwd())
from openai import OpenAI
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler
PORT, MODEL = sys.argv[1], sys.argv[2]
h=QwenFCKeepReasonHandler.__new__(QwenFCKeepReasonHandler)
E=[json.loads(l) for l in open("bfcl_eval/data/BFCL_v4_simple_python.json")]
pool=[]
for e in E:
    for f in e["function"]:
        if f["name"] not in {x["name"] for x in pool}: pool.append(f)
print(f"  tool pool available: {len(pool)}")
tgt=E[0]; tfn=tgt["function"][0]; msgs=[m for t in tgt["question"] for m in t]
c=OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")
print(f"{'tools':>6}{'prompt_tok':>11}{'out_tok':>9}  verdict")
for n in (60, 100, 150, 200, 300):
    if n> len(pool): n=len(pool)
    fns=[tfn]+[f for f in pool if f["name"]!=tfn["name"]][:n-1]
    pr=h._format_prompt(msgs, fns)
    try:
        r=c.completions.create(model=MODEL, prompt=pr, temperature=0.001, max_tokens=4096, timeout=900)
    except Exception as ex:
        print(f"{n:>6}{'-':>11}{'-':>9}  ERROR {type(ex).__name__}: {str(ex)[:60]}"); continue
    o=r.choices[0].text; calls=h._extract_tool_calls(o)
    v="OK" if calls else ("DEGENERATE" if r.usage.completion_tokens>3000 else "no call (prose)")
    print(f"{len(fns):>6}{r.usage.prompt_tokens:>11}{r.usage.completion_tokens:>9}  {v}")
    if n>=len(pool): break
