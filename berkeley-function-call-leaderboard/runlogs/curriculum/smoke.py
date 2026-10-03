"""Does the XML format still show the tool-calling defect?

Sends N simple_python prompts through QwenFCKeepReasonHandler's own _format_prompt and
classifies each completion:
  OK        parsed into >=1 tool call
  PROSE     no <tool_call> at all
  MALFORMED <tool_call> present but nothing parseable came out
"""
import json, os, re, sys, collections
from openai import OpenAI
sys.path.insert(0, os.getcwd())
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler

PORT, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
MODEL = sys.argv[3]
h = QwenFCKeepReasonHandler.__new__(QwenFCKeepReasonHandler)
cli = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")

entries = [json.loads(l) for l in open("bfcl_eval/data/BFCL_v4_simple_python.json")][:N]
kinds = collections.Counter(); samples = []
for e in entries:
    fns = e["function"]
    msgs = [m for turn in e["question"] for m in turn]
    prompt = h._format_prompt(msgs, fns)
    r = cli.completions.create(model=MODEL, prompt=prompt, temperature=0.001,
                               max_tokens=16384, timeout=600)
    out = r.choices[0].text
    calls = h._extract_tool_calls(out)
    if calls:
        kind = "OK"
    elif "<tool_call>" in out:
        kind = "MALFORMED"
    else:
        kind = "PROSE"
    kinds[kind] += 1
    if len(samples) < 3 or kind != "OK":
        tail = out.split("</think>")[-1].strip()
        samples.append((e["id"], kind, tail[:300], calls[:1]))

print(f"\n=== {MODEL.rsplit('/',1)[-1]}  ({sum(kinds.values())} prompts) ===")
for k in ("OK", "MALFORMED", "PROSE"):
    print(f"   {k:<10} {kinds[k]:3d}  ({kinds[k]/max(sum(kinds.values()),1)*100:5.1f}%)")
print("\n--- samples ---")
for i, (eid, kind, tail, call) in enumerate(samples[:5]):
    print(f"[{kind}] {eid}\n    out : {tail!r}\n    call: {call}")
