"""
Prove the BFCL `-FC` variant renders prompts IDENTICALLY to the official Qwen3
chat template -- i.e. reasoning traces are stripped from context exactly the way
Qwen ships it.

Ground truth = the `chat_template` field of Qwen/Qwen3-4B's own
tokenizer_config.json (the published artifact, NOT the copy pasted into the
handler docstring).

Test = QwenFCHandler._format_prompt, the function that actually builds every
prompt the eval sends.
"""
import json, sys
from jinja2 import Environment
from jinja2.sandbox import ImmutableSandboxedEnvironment

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.constants.custom_model_config import QwenFCKeepReasonHandler

OFFICIAL = "/workspace/hf-cache/Qwen3-4B/tokenizer_config.json"
tmpl_src = json.load(open(OFFICIAL))["chat_template"]

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "separators": (", ", ": ")}
tmpl = env.from_string(tmpl_src)

TOOLS = [
    {"name": "get_weather",
     "description": "Get weather.",
     "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "book_flight",
     "description": "Book a flight.",
     "parameters": {"type": "dict", "properties": {"dest": {"type": "string"}}, "required": ["dest"]}},
]

# Cases modelled on what BFCL actually sends: single-turn FC, a tool loop, and a
# genuine multi-turn conversation where an EARLIER assistant turn reasoned.
CASES = {
 "single_turn": [
   {"role": "user", "content": "Weather in Paris?"},
 ],
 "tool_loop": [
   {"role": "user", "content": "Weather in Paris?"},
   {"role": "assistant", "content": "<think>\nI should call get_weather.\n</think>\n\n",
    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
   {"role": "tool", "content": "{\"temp\": 21}"},
 ],
 "two_user_turns_earlier_reasoning": [
   {"role": "user", "content": "Weather in Paris?"},
   {"role": "assistant", "content": "<think>\nCall get_weather for Paris.\n</think>\n\n",
    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
   {"role": "tool", "content": "{\"temp\": 21}"},
   {"role": "assistant", "content": "<think>\nReport it.\n</think>\n\nIt is 21C in Paris."},
   {"role": "user", "content": "Now book me a flight there."},
   {"role": "assistant", "content": "<think>\nCall book_flight.\n</think>\n\n",
    "tool_calls": [{"function": {"name": "book_flight", "arguments": {"dest": "Paris"}}}]},
   {"role": "tool", "content": "{\"ok\": true}"},
 ],
 "three_user_turns": [
   {"role": "user", "content": "Hi"},
   {"role": "assistant", "content": "<think>\nGreeting.\n</think>\n\nHello!"},
   {"role": "user", "content": "Weather in Paris?"},
   {"role": "assistant", "content": "<think>\nUse the tool.\n</think>\n\n",
    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
   {"role": "tool", "content": "{\"temp\": 21}"},
   {"role": "assistant", "content": "<think>\nAnswer.\n</think>\n\n21C."},
   {"role": "user", "content": "And book a flight."},
 ],
}

h  = QwenFCHandler.__new__(QwenFCHandler)          # no __init__: _format_prompt is pure
kr = QwenFCKeepReasonHandler.__new__(QwenFCKeepReasonHandler)

def official(messages, tools):
    # The official template consumes tool_calls with a `function` sub-dict and
    # renders `arguments | tojson`; BFCL's handler json.dumps() them. Both are
    # compact JSON, so compare after the same normalisation the template applies.
    return tmpl.render(messages=messages, tools=tools, add_generation_prompt=True)

ok = True
for name, msgs in CASES.items():
    want = official(msgs, TOOLS)
    got  = h._format_prompt([dict(m) for m in msgs], TOOLS)
    same = want == got
    ok &= same
    print(f"{'PASS' if same else 'FAIL'}  {name:34s}  official={len(want)}B  handler={len(got)}B")
    if not same:
        import difflib
        for line in difflib.unified_diff(want.splitlines(), got.splitlines(),
                                         "OFFICIAL_QWEN3_TEMPLATE", "BFCL_QwenFCHandler", lineterm="", n=2):
            print("   ", line)

print()
print("=== reasoning-in-context census (case: three_user_turns) ===")
msgs = CASES["three_user_turns"]
p_off = official(msgs, TOOLS)
p_fc  = h._format_prompt([dict(m) for m in msgs], TOOLS)
p_kr  = kr._format_prompt([dict(m) for m in msgs], TOOLS)
n_assistant_reasoned = sum(1 for m in msgs if m["role"] == "assistant" and "<think>" in m["content"])
print(f"assistant turns carrying <think> in the conversation : {n_assistant_reasoned}")
for label, p in (("official Qwen3 template", p_off), ("BFCL -FC  (what we run)", p_fc), ("BFCL -FC-keepreason    ", p_kr)):
    print(f"  <think> blocks rendered into prompt by {label}: {p.count('<think>')}")
print()
print("--- the exact prompt the eval sends (BFCL -FC), tools block elided ---")
print(p_fc.split("</tools>\n\nFor each function call")[-1].split("<|im_end|>\n", 1)[-1])
sys.exit(0 if ok else 1)
