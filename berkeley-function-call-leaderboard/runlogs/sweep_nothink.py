"""Same tool-count sweep, WITH vs WITHOUT the <think> prefill."""
import json, os, sys
sys.path.insert(0, os.getcwd())
from openai import OpenAI
from bfcl_eval.model_handler.local_inference.qwen_xml import (
    QwenXMLHandler, QwenXMLNoThinkPrefillHandler)
PORT, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
E=[json.loads(l) for l in open("bfcl_eval/data/BFCL_v4_simple_python.json")]
pool=[]
for e in E:
    for f in e["function"]:
        if f["name"] not in {x["name"] for x in pool}: pool.append(f)
tgt=E[0]; tfn=tgt["function"][0]; msgs=[m for t in tgt["question"] for m in t]
c=OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")
hs=[("WITH prefill", QwenXMLHandler.__new__(QwenXMLHandler)),
    ("NO   prefill", QwenXMLNoThinkPrefillHandler.__new__(QwenXMLNoThinkPrefillHandler))]
print(f"### {LABEL}  (:{PORT})")
print(f"{'tools':>6}{'ptok':>7}   {'WITH <think> prefill':<26}{'NO prefill (model emits it)':<26}")
for n in (1, 20, 28, 32, 39):
    row=[]; pt=None
    for lbl,h in hs:
        pr=h._format_prompt(msgs, [tfn]+[f for f in pool if f["name"]!=tfn["name"]][:n-1])
        try:
            r=c.completions.create(model=MODEL, prompt=pr, temperature=0.001, max_tokens=4096, timeout=900)
        except Exception as ex:
            row.append(f"ERROR {type(ex).__name__}"); continue
        o=r.choices[0].text; pt=pt or r.usage.prompt_tokens
        calls=h._extract_tool_calls(o, [tfn])
        if calls: v=f"OK ({r.usage.completion_tokens} tok)"
        elif r.usage.completion_tokens>3000: v=f"DEGENERATE ({r.usage.completion_tokens})"
        else: v=f"no call ({r.usage.completion_tokens} tok)"
        row.append(v)
    print(f"{n:>6}{pt:>7}   {row[0]:<26}{row[1]:<26}")
