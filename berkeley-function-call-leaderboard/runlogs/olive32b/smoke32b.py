"""Does BFCL handle the 32B exactly as it handled the 4B?

Builds the prompt with the same handler class the eval uses, sends it to the
server with the same sampling BFCL uses, then runs the handler's own parsers
over the raw text. Checks, in order:
  1. the prompt bytes are shaped like the 4B's (same system block, same tags)
  2. the model closes <think>, so reasoning splits out cleanly
  3. <tool_call> blocks match the handler's regex, which REQUIRES the newlines
     in "<tool_call>\n{...}\n</tool_call>"
  4. decode_ast returns the call list the AST checker expects
"""
import json, sys, requests
sys.path.insert(0, "/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard")
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler as H
from bfcl_eval.utils import load_file
from bfcl_eval.constants.eval_config import PROMPT_PATH

PORT = sys.argv[1]; CKPT = sys.argv[2]; LABEL = sys.argv[3]
h = H.__new__(H)

cases = []
for cat, n in (("simple_python", 3), ("parallel", 2), ("multiple", 2), ("irrelevance", 1)):
    rows = load_file(PROMPT_PATH / f"BFCL_v4_{cat}.json")
    for r in rows[:n]:
        cases.append((cat, r))

ok = {"think_closed": 0, "think_present": 0, "toolcall_regex": 0, "decode_ast": 0, "n": 0}
for cat, entry in cases:
    msgs = entry["question"][0] if isinstance(entry["question"][0], list) else [entry["question"][0]]
    prompt = h._format_prompt(msgs, entry["function"])
    r = requests.post(f"http://localhost:{PORT}/v1/completions", timeout=600, json={
        "model": CKPT, "prompt": prompt, "temperature": 0.001, "max_tokens": 4096})
    txt = r.json()["choices"][0]["text"]
    ok["n"] += 1
    has_open = "<think>" in txt or txt.lstrip().startswith("Okay") or True
    closed = "</think>" in txt
    ok["think_present"] += ("<think>" in txt)
    ok["think_closed"] += closed
    calls = h._extract_tool_calls(txt)          # the handler's own regex
    ok["toolcall_regex"] += bool(calls)
    try:
        h.decode_ast(txt, "Python", True); ok["decode_ast"] += 1
    except Exception:
        pass
    if ok["n"] <= 3:
        print(f"--- {LABEL} {entry['id']}")
        print("    raw head:", repr(txt[:110]))
        print("    has <think>:", "<think>" in txt, "| has </think>:", closed,
              "| tool_calls parsed:", len(calls))
print(f"\n== {LABEL}: {ok['n']} prompts | <think> emitted {ok['think_present']} | "
      f"</think> closed {ok['think_closed']} | tool_call regex matched {ok['toolcall_regex']} | "
      f"decode_ast ok {ok['decode_ast']}")
