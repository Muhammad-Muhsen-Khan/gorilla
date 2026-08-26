import json, os
S = os.path.dirname(os.path.abspath(__file__))
r = lambda n: open(os.path.join(S, n)).read()
parts = json.load(open(os.path.join(S, "parts.json")))
tr = json.load(open(os.path.join(S, "transcripts.json")))

head = r("head.html").replace("/*TABCSS*/", tr["tabcss"])
body = r("body1.html") + r("body2.html")
for k, v in parts.items():
    body = body.replace(f"<!--{k}-->", v)
body = body.replace("<!--CARDS-->", tr["cards"])

out = head + "\n" + body
p = os.path.join(S, "reason_vs_noreason.html")
open(p, "w").write(out)
print("wrote", p, len(out), "bytes")

leftover = [x for x in ("<!--PAIRED_HEAD-->", "<!--PAIRED_BODY-->", "<!--PAIRED_FOOT-->",
                        "<!--BASELINES-->", "<!--C1-->", "<!--C2-->", "<!--CARDS-->", "/*TABCSS*/")
            if x in out]
print("unfilled placeholders:", leftover or "none")
