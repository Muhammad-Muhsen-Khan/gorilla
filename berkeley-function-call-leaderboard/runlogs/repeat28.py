"""Is the 28-tool boundary deterministic? Repeat it 6x per variant."""
import json, os, sys
sys.path.insert(0, os.getcwd())
from openai import OpenAI
from bfcl_eval.model_handler.local_inference.qwen_xml import (
    QwenXMLHandler, QwenXMLNoThinkPrefillHandler)
PORT, MODEL = sys.argv[1], sys.argv[2]
E=[json.loads(l) for l in open("bfcl_eval/data/BFCL_v4_simple_python.json")]
pool=[]
for e in E:
    for f in e["function"]:
        if f["name"] not in {x["name"] for x in pool}: pool.append(f)
tgt=E[0]; tfn=tgt["function"][0]; msgs=[m for t in tgt["question"] for m in t]
fns=[tfn]+[f for f in pool if f["name"]!=tfn["name"]][:27]   # 28 tools
c=OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")
for lbl,h in (("WITH prefill",QwenXMLHandler.__new__(QwenXMLHandler)),
              ("NO   prefill",QwenXMLNoThinkPrefillHandler.__new__(QwenXMLNoThinkPrefillHandler))):
    pr=h._format_prompt(msgs, fns)
    outs=[]
    for _ in range(6):
        r=c.completions.create(model=MODEL, prompt=pr, temperature=0.001, max_tokens=4096, timeout=900)
        n=r.usage.completion_tokens
        calls=h._extract_tool_calls(r.choices[0].text, fns)
        outs.append("OK" if calls else ("DEGEN" if n>3000 else f"nocall({n})"))
    print(f"  {lbl}  28 tools / {r.usage.prompt_tokens} ptok : {outs}")
