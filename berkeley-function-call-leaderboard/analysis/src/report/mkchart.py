import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, '..', '..', 'suite.json')
D=json.load(open(SUITE))
NR=sorted([r for r in D if r['grp']=='noreason'], key=lambda x:x['epoch'])
RE=sorted([r for r in D if r['grp']=='reasoning'], key=lambda x:x['epoch'])
BASE=[r for r in D if r['grp']=='base'][0]
LEN=[r for r in D if r['grp']=='lenient'][0]
INS=[r for r in D if r['grp']=='instruct'][0]

def linechart(cid, field, ylo, yhi, ticks, refval=None, reflabel=None):
    W,H,L,R,T,B = 640,250,44,16,16,34
    pw,ph = W-L-R, H-T-B
    x=lambda e:L+pw*(e-1)/9
    y=lambda v:T+ph*(1-(v-ylo)/(yhi-ylo))
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{cid}">']
    for t in ticks:
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        o.append(f'<text class="ax" x="{L-8}" y="{y(t)+4:.1f}" text-anchor="end">{t}</text>')
    for e in range(1,11):
        o.append(f'<text class="ax" x="{x(e):.1f}" y="{H-12}" text-anchor="middle">{e}</text>')
    if refval is not None and ylo <= refval <= yhi:
        o.append(f'<line class="refline" x1="{L}" x2="{W-R}" y1="{y(refval):.1f}" y2="{y(refval):.1f}"/>')
        o.append(f'<text class="reftxt" x="{L+6}" y="{y(refval)-7:.1f}" text-anchor="start">{reflabel}</text>')
    for series,cls in ((NR,'s1'),(RE,'s2')):
        pts=" ".join(f"{x(r['epoch']):.1f},{y(r[field]):.1f}" for r in series)
        o.append(f'<polyline class="ln {cls}" points="{pts}"/>')
        for r in series:
            o.append(f'<circle class="pt {cls}" cx="{x(r["epoch"]):.1f}" cy="{y(r[field]):.1f}" r="4.5">'
                     f'<title>epoch {r["epoch"]} · {r[field]}%</title></circle>')
    last=NR[-1]; o.append(f'<text class="dlab s1" x="{x(10)-6:.1f}" y="{y(last[field])-11:.1f}" text-anchor="end">no-reason</text>')
    last=RE[-1]; o.append(f'<text class="dlab s2" x="{x(10)-6:.1f}" y="{y(last[field])+18:.1f}" text-anchor="end">reasoning</text>')
    o.append(f'<text class="axtitle" x="{L+pw/2}" y="{H-1}" text-anchor="middle">training epoch</text>')
    o.append('</svg>')
    return "\n".join(o)

def barchart():
    secs=[("Non-Live",'nonlive'),("Live",'live'),("Irrelevance",'irrel'),("Multi-Turn",'mt'),("Memory",'mem')]
    models=[("Base",BASE,'m-base'),("SFT best",max(NR,key=lambda r:r['overall']),'m-sft'),("Instruct",INS,'m-ins')]
    W,H,L,R,T,B=640,270,44,16,16,52
    pw,ph=W-L-R,H-T-B
    gw=pw/len(secs); bw=min(26,(gw-18)/3)
    y=lambda v:T+ph*(1-v/100)
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="section comparison">']
    for t in (0,25,50,75,100):
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        o.append(f'<text class="ax" x="{L-8}" y="{y(t)+4:.1f}" text-anchor="end">{t}</text>')
    for i,(sname,key) in enumerate(secs):
        cx=L+gw*i+gw/2
        for j,(mname,rec,cls) in enumerate(models):
            v=rec[key] or 0
            bx=cx-(3*bw+4)/2+j*(bw+2)
            h=max(ph*v/100,1.5)
            o.append(f'<rect class="bar {cls}" x="{bx:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{h:.1f}" rx="4">'
                     f'<title>{mname} · {sname} · {v}%</title></rect>')
            if v>=1:
                o.append(f'<text class="blab" x="{bx+bw/2:.1f}" y="{y(v)-5:.1f}" text-anchor="middle">{v:.0f}</text>')
        o.append(f'<text class="ax" x="{cx:.1f}" y="{H-26}" text-anchor="middle">{sname}</text>')
    o.append('</svg>')
    return "\n".join(o)

def table():
    order=[BASE,LEN,INS]+NR+RE
    rows=[]
    for r in order:
        n=r['model'].replace('Qwen3-4B-Base SFT ','').replace(' (FC)','').replace('Qwen3-4B','Qwen3-4B')
        cls=' class="ref"' if r['grp'] in ('base','instruct','lenient') else ''
        rows.append("<tr%s><td>%s</td>%s</tr>"%(cls,n,"".join(
            f"<td>{r[k] if r[k] is not None else '—'}</td>"
            for k in ('overall','nonlive','live','irrel','mt','mem','lat'))))
    return "\n".join(rows)

open(os.path.join(HERE, '_charts.html'), 'w').write("<!--C1-->\n"+linechart('nonlive','nonlive',78,90,[78,81,84,87,90],INS['nonlive'],'Instruct 87.23')
  +"\n<!--C2-->\n"+linechart('mt','mt',0,4,[0,1,2,3,4])
  +"\n<!--C3-->\n"+barchart()
  +"\n<!--TBL-->\n"+table())
print("charts generated")
