import json, sys, requests
sys.path.insert(0, "/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard")
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler as H
from bfcl_eval.utils import load_file
from bfcl_eval.constants.eval_config import PROMPT_PATH
PORT, CKPT = sys.argv[1], sys.argv[2]
h = H.__new__(H)
rows = load_file(PROMPT_PATH / "BFCL_v4_simple_python.json")
e = rows[0]
msgs = e["question"][0] if isinstance(e["question"][0], list) else [e["question"][0]]
prompt = h._format_prompt(msgs, e["function"])
r = requests.post(f"http://localhost:{PORT}/v1/completions", timeout=900,
                  json={"model": CKPT, "prompt": prompt, "temperature": 0.001, "max_tokens": 8192})
txt = r.json()["choices"][0]["text"]
print("=== PROMPT TAIL ==="); print(repr(prompt[-320:]))
print("\n=== FULL OUTPUT (%d chars) ===" % len(txt)); print(txt)
print("\n=== after </think> ==="); print(repr(txt.split("</think>")[-1][:600]))
