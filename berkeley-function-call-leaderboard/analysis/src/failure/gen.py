import json, html, os

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, 'transcripts.json')))
E = html.escape

META = {
 "multi_turn_base_0":  ("Filesystem", "Never enters the directory, then repeats itself"),
 "multi_turn_base_54": ("Vehicle control", "Told exactly how to fix it; explains instead"),
 "multi_turn_base_65": ("Vehicle control", "“Recovers” by calling something unrelated"),
}

def render(tid):
    turns = T[tid]
    o = [f'<div class="tx" id="tx-{tid}" role="tabpanel" aria-labelledby="tab-{tid}">']
    for i, t in enumerate(turns):
        o.append('<div class="turn">')
        o.append(f'<div class="turn-h"><span class="tnum">turn {i}</span></div>')
        o.append(f'<div class="msg user"><span class="who">user</span><p>{E(t["user"])}</p></div>')
        for st in t["steps"]:
            o.append(f'<div class="step"><span class="snum">step {st["n"]}</span><div class="sbody">')
            for ev in st["events"]:
                if ev["t"] == "assistant":
                    c = ev["c"]
                    if "<tool_call>" in c:
                        o.append('<div class="msg calls"><span class="who">model</span><pre>' + E(c) + '</pre></div>')
                    else:
                        o.append('<div class="msg prose"><span class="who">model</span><p>' + E(c) + '</p></div>')
                elif ev["t"] == "tool":
                    cls = "tool err" if ev["err"] else "tool"
                    lbl = "tool error" if ev["err"] else "tool"
                    o.append(f'<div class="{cls}"><span class="who">{lbl}</span><pre>' + E(ev["c"]) + '</pre></div>')
                else:
                    o.append('<div class="note">' + E(ev["c"]) + '</div>')
            o.append('</div></div>')
        o.append('</div>')
    o.append('</div>')
    return "\n".join(o)

tabs = "\n".join(
    f'<button class="tab{" on" if i==0 else ""}" id="tab-{k}" data-t="{k}" role="tab" '
    f'aria-selected="{"true" if i==0 else "false"}">'
    f'<span class="tid">{k.replace("multi_turn_","")}</span>'
    f'<span class="tsub">{E(META[k][1])}</span></button>'
    for i, k in enumerate(META))

panels = "\n".join(render(k) for k in META)
open(os.path.join(HERE, '_frag.html'), 'w').write(
    '<!--TABS-->\n' + tabs + '\n<!--PANELS-->\n' + panels)
print("fragment bytes:", len(tabs)+len(panels))
