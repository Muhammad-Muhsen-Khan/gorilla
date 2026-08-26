import json, os, html
S = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(S, "bundle.json")))

ORDER = [("base", "Base"), ("instruct", "Instruct"), ("noreason", "no-reason e7"), ("reason", "reasoning e3")]
CATLAB = {
    "multi_turn_base": "multi-turn · base",
    "multi_turn_miss_func": "multi-turn · missing function",
    "multi_turn_miss_param": "multi-turn · missing parameter",
    "multi_turn_long_context": "multi-turn · long context",
}
E = lambda s: html.escape(str(s), quote=False)

def pre(text, cls="pl"):
    return f'<pre class="{cls}">{html.escape(str(text))}</pre>'

def msg(role, content, extra=None, reasoning=None, label=None):
    lab = label or role
    out = [f'<div class="msg m-{role}">']
    out.append(f'<span class="who">{E(lab)}</span>')
    out.append('<div class="mbody">')
    if reasoning:
        out.append('<details class="think"><summary>reasoning trace · '
                   f'{len(reasoning.split())} words</summary>{pre(reasoning, "pl think-pl")}</details>')
    txt = content if content is not None else ""
    if str(txt).strip() == "":
        out.append('<p class="empty">— empty response —</p>')
    else:
        out.append(pre(txt))
    if extra:
        out.append(f'<p class="xtra">{E(extra)}</p>')
    out.append("</div></div>")
    return "".join(out)

def render_log(log):
    if not log:
        return None
    out = []
    turn_no = 0
    state_no = 0
    for block in log:
        if isinstance(block, list):
            state_no += 1
            inner = "".join(
                f'<h4>{E(s.get("class_name","?"))}</h4>{pre(json.dumps(s.get("content"), indent=1, default=str))}'
                for s in block if s.get("role") == "state_info")
            if inner:
                out.append(f'<details class="state"><summary>API state · checkpoint {state_no}'
                           f' <span class="cnt">{len(block)} instance(s)</span></summary>{inner}</details>')
            continue
        turn_no += 1
        out.append(f'<div class="turn"><div class="tno">Turn {turn_no}</div>')
        for m in block.get("begin_of_turn_query", []):
            out.append(msg("user", m.get("content"), label="user"))
        for key in sorted([k for k in block if k.startswith("step_")],
                          key=lambda k: int(k.split("_")[1])):
            steps = block[key]
            out.append(f'<div class="step"><span class="sno">{E(key.replace("_", " "))}</span>')
            for m in steps:
                r = m.get("role")
                if r == "assistant":
                    out.append(msg("assistant", m.get("content"), reasoning=m.get("reasoning_content"),
                                   label="assistant"))
                elif r == "tool":
                    out.append(msg("tool", m.get("content"), label="tool response"))
                elif r == "handler_log":
                    out.append(f'<div class="hl">{E(m.get("content"))}</div>')
                else:
                    out.append(msg("user", m.get("content"), label=r))
            out.append("</div>")
        out.append("</div>")
    return "\n".join(out)

cards = []
for i, rec in enumerate(B):
    rid = f"tc{i}"
    cat = CATLAB.get(rec["cat"], rec["cat"])
    radios, labels, panels = [], [], []
    for j, (key, lab) in enumerate(ORDER):
        m = rec["models"][key]
        checked = " checked" if j == 0 else ""
        radios.append(f'<input class="tabin" type="radio" name="{rid}" id="{rid}-{key}"{checked}>')
        badge = '<span class="bdg ok">pass</span>' if m["correct"] else '<span class="bdg no">fail</span>'
        labels.append(f'<label class="tab t-{key}" for="{rid}-{key}">{E(lab)}{badge}</label>')
        body = render_log(m["log"])
        if body is None:
            body = ('<div class="msg m-err"><span class="who">harness</span><div class="mbody">'
                    '<p class="xtra">No transcript: the request exceeded the served context window, '
                    'so this entry never produced a conversation.</p>'
                    f'{pre(m.get("error_message") or "inference error")}</div></div>')
        err = ("" if m["correct"] else
               f'<div class="verdict"><span class="et">{E(m["error_type"])}</span>'
               f'<span class="em">{E((m["error_message"] or "")[:400])}</span></div>')
        panels.append(f'<div class="panel p-{key}">{err}<div class="convo">{body}</div></div>')

    cards.append(f'''<article class="tcard" id="{rid}">
<header class="thead">
  <div>
    <span class="tid">{E(rec["id"])}</span>
    <h3>{E(cat)}</h3>
    <p class="tmeta">{E(" + ".join(rec["involved_classes"]))} · {rec["n_functions"]} functions in scope</p>
  </div>
</header>
<details class="sysd"><summary>System prompt <span class="cnt">{len(rec["system"]):,} characters — the identical prompt all four models received</span></summary>{pre(rec["system"], "pl sys-pl")}</details>
<details class="sysd"><summary>Initial API state <span class="cnt">the world before turn 1</span></summary>{pre(json.dumps(rec["initial_config"], indent=1), "pl sys-pl")}</details>
{"".join(radios)}
<div class="tabbar">{"".join(labels)}</div>
<div class="panels">{"".join(panels)}</div>
</article>''')

css_rules = []
for i in range(len(B)):
    for key, _ in ORDER:
        css_rules.append(f'#tc{i}-{key}:checked ~ .tabbar .t-{key}{{background:var(--surface);color:var(--ink);'
                         f'border-bottom-color:transparent}}')
        css_rules.append(f'#tc{i}-{key}:checked ~ .panels .p-{key}{{display:block}}')

json.dump({"cards": "\n".join(cards), "tabcss": "\n".join(css_rules)},
          open(os.path.join(S, "transcripts.json"), "w"))
print("rendered", len(cards), "cards;", sum(len(c) for c in cards), "bytes")
