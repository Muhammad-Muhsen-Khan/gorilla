import json, os, html, re
S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "reason_vs_noreason.html")

D = json.load(open(os.path.join(S, "..", "..", "suite.json")))
NR = sorted([r for r in D if r["grp"] == "noreason"], key=lambda x: x["epoch"])
RE = sorted([r for r in D if r["grp"] == "reasoning"], key=lambda x: x["epoch"])
BASE = [r for r in D if r["grp"] == "base"][0]
LEN  = [r for r in D if r["grp"] == "lenient"][0]
INS  = [r for r in D if r["grp"] == "instruct"][0]

METRICS = [("overall", "Overall"), ("nonlive", "Non-Live"), ("live", "Live"),
           ("irrel", "Irrel."), ("mt", "Multi-Turn"), ("mem", "Memory")]

def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)

# ---------------------------------------------------------------- paired table
def paired_table():
    head1 = '<tr><th rowspan="2" class="ep">Epoch</th>' + "".join(
        f'<th colspan="2" class="grp">{lab}</th>' for _, lab in METRICS) + "</tr>"
    head2 = "<tr>" + "".join(
        '<th class="sub s1">no-r</th><th class="sub s2">reas</th>' for _ in METRICS) + "</tr>"
    body = []
    for i in range(10):
        a, b = NR[i], RE[i]
        cells = []
        for k, _ in METRICS:
            va, vb = a[k], b[k]
            wa = " win" if va > vb else ""
            wb = " win" if vb > va else ""
            cells.append(f'<td class="n{wa}">{va:.2f}</td><td class="n r{wb}">{vb:.2f}</td>')
        body.append(f'<tr><td class="ep">{i+1}</td>{"".join(cells)}</tr>')
    foot = ['<tr class="mean"><td class="ep">mean</td>']
    for k, _ in METRICS:
        ma, mb = mean(NR, k), mean(RE, k)
        foot.append(f'<td class="n">{ma:.2f}</td><td class="n r">{mb:.2f}</td>')
    foot.append("</tr>")
    foot.append('<tr class="delta"><td class="ep">Δ</td>')
    for k, _ in METRICS:
        d = mean(RE, k) - mean(NR, k)
        cls = " pos" if d > 0 else ""
        foot.append(f'<td class="n dl{cls}" colspan="2">{d:+.2f}</td>')
    foot.append("</tr>")
    return head1 + head2, "\n".join(body), "\n".join(foot)

def wins():
    n = 0
    for i in range(10):
        for k, _ in METRICS:
            if RE[i][k] > NR[i][k]:
                n += 1
    return n

def baseline_table():
    rows = []
    for r, lab, note, cls in (
        (BASE, "Qwen3-4B-Base", "untrained, strict parse", "b-base"),
        (LEN,  "Qwen3-4B-Base", "untrained, lenient re-parse", "b-len"),
        (max(NR, key=lambda x: x["overall"]), "SFT no-reason e7", "best of the no-reason run", "b-nr"),
        (max(RE, key=lambda x: x["overall"]), "SFT reasoning e3", "best of the reasoning run", "b-re"),
        (INS,  "Qwen3-4B Instruct", "official reference", "b-ins")):
        cells = "".join(f'<td class="n">{r[k]:.2f}</td>' for k, _ in METRICS)
        rows.append(f'<tr class="{cls}"><td class="ep"><span class="dot"></span>{lab}'
                    f'<span class="note">{note}</span></td>{cells}</tr>')
    return "\n".join(rows)

# ---------------------------------------------------------------- line charts
def linechart(field, ylo, yhi, ticks, refval=None, reflabel=None):
    W, H, L, R, T, B = 640, 250, 44, 16, 16, 34
    pw, ph = W - L - R, H - T - B
    x = lambda e: L + pw * (e - 1) / 9
    y = lambda v: T + ph * (1 - (v - ylo) / (yhi - ylo))
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{field} by epoch">']
    for t in ticks:
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        o.append(f'<text class="ax" x="{L-8}" y="{y(t)+4:.1f}" text-anchor="end">{t}</text>')
    for e in range(1, 11):
        o.append(f'<text class="ax" x="{x(e):.1f}" y="{H-12}" text-anchor="middle">{e}</text>')
    if refval is not None and ylo <= refval <= yhi:
        o.append(f'<line class="refline" x1="{L}" x2="{W-R}" y1="{y(refval):.1f}" y2="{y(refval):.1f}"/>')
        o.append(f'<text class="reftxt" x="{L+6}" y="{y(refval)-7:.1f}" text-anchor="start">{reflabel}</text>')
    for series, cls in ((NR, "s1"), (RE, "s2")):
        pts = " ".join(f"{x(r['epoch']):.1f},{y(r[field]):.1f}" for r in series)
        o.append(f'<polyline class="ln {cls}" points="{pts}"/>')
        for r in series:
            o.append(f'<circle class="pt {cls}" cx="{x(r["epoch"]):.1f}" cy="{y(r[field]):.1f}" r="4.5">'
                     f'<title>epoch {r["epoch"]} · {r[field]}%</title></circle>')
    o.append(f'<text class="dlab s1" x="{x(10)-6:.1f}" y="{y(NR[-1][field])-11:.1f}" text-anchor="end">no-reason</text>')
    o.append(f'<text class="dlab s2" x="{x(10)-6:.1f}" y="{y(RE[-1][field])+18:.1f}" text-anchor="end">reasoning</text>')
    o.append(f'<text class="axtitle" x="{L+pw/2}" y="{H-1}" text-anchor="middle">training epoch</text>')
    o.append("</svg>")
    return "\n".join(o)

json.dump({"wins": wins(),
           "means": {k: [round(mean(NR, k), 2), round(mean(RE, k), 2)] for k, _ in METRICS}},
          open(os.path.join(S, "stats.json"), "w"), indent=1)

PARTS = {
    "PAIRED_HEAD": paired_table()[0],
    "PAIRED_BODY": paired_table()[1],
    "PAIRED_FOOT": paired_table()[2],
    "BASELINES":   baseline_table(),
    "C1": linechart("nonlive", 78, 90, [78, 81, 84, 87, 90], INS["nonlive"], "Instruct 87.23"),
    "C2": linechart("mt", 0, 4, [0, 1, 2, 3, 4]),
}
json.dump(PARTS, open(os.path.join(S, "parts.json"), "w"))
print("tables + charts built; reasoning wins", wins(), "of 60 cells")
print(json.dumps(json.load(open(os.path.join(S, "stats.json"))), indent=1))
