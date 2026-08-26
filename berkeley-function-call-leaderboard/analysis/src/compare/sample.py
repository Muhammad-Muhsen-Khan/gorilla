import json, random, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DATA = f"{ROOT}/bfcl_eval/data"
RES  = f"{ROOT}/result"

CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
MODELS = [
    ("base",     "qwen3-4b-base-FC",              "Qwen3-4B-Base (untrained)"),
    ("instruct", "qwen3-4b-instruct-FC",          "Qwen3-4B Instruct"),
    ("noreason", "qwen3-4b-sft-noreason-448-FC",  "SFT no-reason e7"),
    ("reason",   "qwen3-4b-sft-reasoning-258-FC", "SFT reasoning e3"),
]

def jl(p):
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d
    return out

# --- function docs, exactly as populate_test_cases_with_predefined_functions builds them
sys.path.insert(0, ROOT)
from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.utils import load_file

def funcs_for(entry):
    out = []
    for c in entry["involved_classes"]:
        out.extend(load_file(MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[c]))
    if "missed_function" in entry:
        for _, names in entry["missed_function"].items():
            for n in names:
                for i, fd in enumerate(out):
                    if fd["name"] == n:
                        out.pop(i); break
    return out

def sysprompt(function):
    p = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for t in function:
        p += "\n" + json.dumps(t)
    p += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    return p

# --- pick 5 ids, seeded, spread across the four multi-turn categories
tests = {}
for c in CATS:
    tests[c] = jl(f"{DATA}/BFCL_v4_{c}.json")

rng = random.Random(20260826)
picked = []
for i, c in enumerate(CATS):
    ids = sorted(tests[c])
    k = 2 if c == "multi_turn_base" else 1
    picked += [(c, x) for x in rng.sample(ids, k)]
rng.shuffle(picked)
print("picked:", picked)

# --- scores per id per model
def score_map(mdir, cat):
    p = f"{ROOT}/score/{mdir}/multi_turn/BFCL_v4_{cat}_score.json"
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if "id" in d:
                out[d["id"]] = d
    return out

bundle = []
for cat, eid in picked:
    entry = tests[cat][eid]
    fns = funcs_for(entry)
    rec = {
        "id": eid, "cat": cat,
        "involved_classes": entry["involved_classes"],
        "n_functions": len(fns),
        "system": sysprompt(fns),
        "initial_config": entry.get("initial_config", {}),
        "models": {},
    }
    for key, mdir, label in MODELS:
        res = jl(f"{RES}/{mdir}/multi_turn/BFCL_v4_{cat}_result.json").get(eid)
        sc = score_map(mdir, cat).get(eid)
        rec["models"][key] = {
            "label": label, "dir": mdir,
            "log": res.get("inference_log") if res else None,
            "correct": (sc is None),          # score files list only FAILURES
            "error_message": (sc or {}).get("error", {}).get("error_message"),
            "error_type": (sc or {}).get("error", {}).get("error_type"),
        }
    bundle.append(rec)

json.dump(bundle, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle.json"), "w"))
print("wrote bundle.json", os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle.json")))
