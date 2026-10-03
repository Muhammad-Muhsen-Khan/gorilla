"""Long-context degeneration sweep: same question, growing tool-schema block.
Usage: sweep_longctx.py <port> <model-path> <json|xml> <label>"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from openai import OpenAI
from bfcl_eval.model_handler.local_inference.qwen_xml import QwenXMLHandler
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler
PORT, MODEL, FMT, LABEL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
h = (QwenXMLHandler if FMT=="xml" else QwenFCKeepReasonHandler).__new__(
     QwenXMLHandler if FMT=="xml" else QwenFCKeepReasonHandler)
E=[json.loads(l) for l in open("bfcl_eval/data/BFCL_v4_simple_python.json")]
pool=[]
for e in E:
    for f in e["function"]:
        if f["name"] not in {x["name"] for x in pool}: pool.append(f)
tgt=E[0]; tgt_fn=tgt["function"][0]; msgs=[m for t in tgt["question"] for m in t]
c=OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")
print(f"### {LABEL}  ({FMT.upper()}, :{PORT})")
print(f"{'tools':>6}{'prompt_tok':>11}{'out_tok':>9}  verdict")
for n in (1, 20, 28, 32, 39, 60):
    fns=[tgt_fn]+[f for f in pool if f["name"]!=tgt_fn["name"]][:n-1]
    pr=h._format_prompt(msgs, fns)
    try:
        r=c.completions.create(model=MODEL, prompt=pr, temperature=0.001, max_tokens=4096, timeout=900)
    except Exception as ex:
        print(f"{n:>6}{'-':>11}{'-':>9}  ERROR {type(ex).__name__}"); continue
    o=r.choices[0].text
    calls=h._extract_tool_calls(o, fns) if FMT=="xml" else h._extract_tool_calls(o)
    if calls: v="OK"
    elif r.usage.completion_tokens>3000: v="DEGENERATE (noise)"
    else: v="NO CALL (prose)"
    print(f"{n:>6}{r.usage.prompt_tokens:>11}{r.usage.completion_tokens:>9}  {v}")
