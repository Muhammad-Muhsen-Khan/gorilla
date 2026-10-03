"""How often does the 32B emit a structurally-correct <tool_call> whose JSON
will not parse, and why? Compares against the 4B on the same prompts."""
import json, re, sys, requests, collections
sys.path.insert(0, "/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard")
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler as H
from bfcl_eval.utils import load_file
from bfcl_eval.constants.eval_config import PROMPT_PATH
PORT, CKPT, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
h = H.__new__(H)
BLOCK = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
rows = load_file(PROMPT_PATH / "BFCL_v4_simple_python.json")[:N]
stat = collections.Counter(); samples = []
for e in rows:
    msgs = e["question"][0] if isinstance(e["question"][0], list) else [e["question"][0]]
    prompt = h._format_prompt(msgs, e["function"])
    txt = requests.post(f"http://localhost:{PORT}/v1/completions", timeout=900,
          json={"model": CKPT, "prompt": prompt, "temperature": 0.001,
                "max_tokens": 8192}).json()["choices"][0]["text"]
    stat["n"] += 1
    blocks = BLOCK.findall(txt)
    if not blocks:
        stat["no_block"] += 1; continue
    stat["block_found"] += 1
    body = blocks[0].strip()
    try:
        json.loads(body); stat["json_ok"] += 1
    except ValueError as err:
        stat["json_bad"] += 1
        # is it purely a trailing-brace problem?
        stripped = body
        while stripped.endswith("}"):
            try:
                json.loads(stripped); break
            except ValueError:
                stripped = stripped[:-1]
        if stripped and stripped != body:
            try:
                json.loads(stripped); stat["fixable_extra_brace"] += 1
            except ValueError: pass
        if len(samples) < 3: samples.append((e["id"], body[-90:], str(err)[:60]))
print(f"== {LABEL} (n={stat['n']})")
print(f"   <tool_call> block present : {stat['block_found']}")
print(f"   none at all               : {stat['no_block']}")
print(f"   JSON parses               : {stat['json_ok']}")
print(f"   JSON fails                : {stat['json_bad']}")
print(f"   ...fixed by dropping trailing brace(s): {stat['fixable_extra_brace']}")
for i, (eid, tail, err) in enumerate(samples):
    print(f"   sample {i+1} {eid}: ...{tail!r}  -> {err}")
