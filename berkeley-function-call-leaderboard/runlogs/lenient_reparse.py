"""Re-score the base model's EXISTING generations with a lenient parser.

The base model emits well-formed {"name":..., "arguments":...} objects but never
the <tool_call> wrapper BFCL's Qwen parser requires, so it scores 0.00. This
rewrites only the presentation: every leading well-formed call object is wrapped
in <tool_call> tags. No new inference; the model's actual choices are untouched.

Two things this intervention does, and both must be stated when quoting the result:
  1. adds the wrapper tags
  2. discards the degenerate trailing repetition (" Дм", " fø" ...) that follows
     the calls, because the model never stops - so this is a LENIENT PARSER
     result, i.e. an upper bound on latent capability, not a strict re-tag.
"""
import json, pathlib, shutil, sys

DEC = json.JSONDecoder()

def extract_calls(text: str):
    """Every well-formed {"name":..,"arguments":..} object, in order."""
    out, i, n = [], 0, len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0: break
        try:
            obj, end = DEC.raw_decode(text, j)
        except ValueError:
            i = j + 1; continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            out.append(obj); i = end
        else:
            i = j + 1
    return out

def relax(s):
    if not isinstance(s, str) or "<tool_call>" in s:
        return s                                    # already tagged: untouched
    calls = extract_calls(s)
    if not calls:
        return s
    return "\n".join('<tool_call>\n%s\n</tool_call>' % json.dumps(c) for c in calls)

def walk(v):
    if isinstance(v, str):  return relax(v)
    if isinstance(v, list): return [walk(x) for x in v]
    return v

src = pathlib.Path("result/qwen3-4b-base-FC")
dst = pathlib.Path("result/qwen3-4b-base-lenient-FC")
if dst.exists(): shutil.rmtree(dst)
changed = total = 0
for f in src.rglob("*_result.json"):
    o = dst / f.relative_to(src); o.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for l in open(f):
        if not l.strip(): continue
        d = json.loads(l); total += 1
        before = json.dumps(d.get("result"))
        d["result"] = walk(d.get("result"))
        if json.dumps(d["result"]) != before: changed += 1
        lines.append(json.dumps(d))
    o.write_text("\n".join(lines) + "\n")
print(f"rewrote {total} entries; {changed} had calls unwrapped ({100*changed/total:.1f}%)")
