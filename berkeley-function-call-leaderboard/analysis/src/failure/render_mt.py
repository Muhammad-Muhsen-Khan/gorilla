import json, sys, re

def load(cat):
    p=f"result/qwen3-4b-sft-noreason-128-FC/multi_turn/BFCL_v4_{cat}_result.json"
    return [json.loads(l) for l in open(p)]

def transcript(d, max_chars=500):
    out=[f"═══ {d['id']} ═══"]
    for blk in d['inference_log']:
        if not isinstance(blk, dict): continue
        for m in blk.get('begin_of_turn_query', []):
            out.append(f"\n👤 USER: {m['content'][:max_chars]}")
        for k in sorted(x for x in blk if x.startswith('step_')):
            out.append(f"\n  ── {k} ──")
            for m in blk[k]:
                r=m.get('role'); c=m.get('content')
                if r=='assistant':
                    out.append(f"  🤖 {str(c)[:max_chars]}")
                elif r=='tool':
                    out.append(f"  🔧 -> {str(c)[:220]}")
                elif r=='handler_log':
                    if 'Successfully' not in str(c):
                        out.append(f"  ⚠️  {str(c)[:200]}")
    return "\n".join(out)

if __name__=="__main__":
    cat=sys.argv[1]; ids=sys.argv[2:]
    for d in load(cat):
        if not ids or d['id'] in ids:
            print(transcript(d)); print()
