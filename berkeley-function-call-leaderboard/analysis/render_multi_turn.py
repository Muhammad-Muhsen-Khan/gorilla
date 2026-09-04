#!/usr/bin/env python3
"""Render BFCL multi-turn completions for the three ToolMind checkpoints as text.

    python3 analysis/render_multi_turn.py [--n 100] [--out analysis/renders/multi_turn]

One file per (checkpoint, example), holding the whole episode turn by turn: the
system prompt the model was actually served, the user query that opens each turn, then every step within it as the model's
reasoning, the <tool_call> it emitted, what BFCL decoded that call to, and the
result the executed API handed back. Ground truth and the grader's verdict sit
in the header so a transcript can be read against what was wanted.

The SAME example ids are rendered for all three checkpoints, so the three
folders line up file-for-file and a diff is meaningful.

Transcripts come from result/ (all 200 per category); verdicts come from score/,
which records only the failures -- an id absent from the score file passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfcl_eval.utils import (add_language_specific_hint_to_function_doc,
                             populate_test_cases_with_predefined_functions)
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["base", "long_context", "miss_func", "miss_param"]
CHECKPOINTS = [
    ("qwen3-4b-sft-toolmind-916-FC", "epoch 1 (ckpt-916)"),
    ("qwen3-4b-sft-toolmind-1832-FC", "epoch 2 (ckpt-1832)"),
    ("qwen3-4b-sft-toolmind-2748-FC", "epoch 3 (ckpt-2748)"),
]
RULE = "=" * 100
BAR = "#" * 100
SUB = "-" * 100


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_category(cat: str):
    """(prompts by id, ground truth by id, {model: {id: result}}, {model: {id: score}})"""
    entries = read_jsonl(ROOT / f"bfcl_eval/data/BFCL_v4_multi_turn_{cat}.json")
    # Same two passes BFCL applies before inference, so entry["function"] is
    # byte-for-byte the list the served prompt was built from. Verified: the
    # reconstruction below reproduces the recorded input_token_count exactly.
    populate_test_cases_with_predefined_functions(entries)
    add_language_specific_hint_to_function_doc(entries)
    data = {r["id"]: r for r in entries}
    gt = {r["id"]: r for r in
          read_jsonl(ROOT / f"bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_{cat}.json")}
    results, scores = {}, {}
    for model, _ in CHECKPOINTS:
        rp = ROOT / f"result/{model}/multi_turn/BFCL_v4_multi_turn_{cat}_result.json"
        results[model] = {r["id"]: r for r in read_jsonl(rp)}
        sp = ROOT / f"score/{model}/multi_turn/BFCL_v4_multi_turn_{cat}_score.json"
        rows = read_jsonl(sp) if sp.exists() else []
        # Line 0 of a score file is the {"accuracy": ...} header, not an entry.
        scores[model] = {r["id"]: r for r in rows if "id" in r}
    return data, gt, results, scores


_HANDLER = QwenFCHandler.__new__(QwenFCHandler)


def system_prompt(functions: list) -> str:
    """The system turn exactly as QwenFCHandler served it.

    OSSHandler.inference routes even an is_fc_model handler down the *prompting*
    path, but QwenFCHandler overrides _format_prompt, so BFCL's default
    "You are an expert in composing functions" system prompt never appears --
    the tool docs are carried once, by Qwen3's own # Tools block.
    """
    rendered = _HANDLER._format_prompt([{"role": "user", "content": ""}], functions)
    return rendered.split("<|im_end|>", 1)[0].removeprefix("<|im_start|>system\n")


def fmt_block(text: str, indent: str = "    ") -> str:
    if text is None:
        return indent + "(none)"
    text = str(text)
    if not text.strip():
        return indent + "(empty)"
    return "\n".join(indent + ln for ln in text.splitlines())


def render(example_id, cat, model, label, prompt, gt, result, score) -> str:
    L = [RULE,
         f"id           : {example_id}",
         f"category     : multi_turn_{cat}",
         f"checkpoint   : {model}   [{label}]"]
    if score is None:
        L.append("verdict      : PASS")
    else:
        err = score.get("error") or {}
        # error_message is a list for some error types and a bare string for
        # others (instance_state_mismatch); iterating the string yields chars.
        msgs = err.get("error_message") or []
        if isinstance(msgs, str):
            msgs = [msgs]
        L.append("verdict      : FAIL")
        L.append(f"error_type   : {err.get('error_type', '?')}")
        for m in msgs:
            L.append(f"error        : {m}")
    n_model = len(result.get("result") or [])
    L.append(f"turns        : {len(gt.get('ground_truth') or [])} ground-truth / {n_model} model")
    L.append(RULE)
    L.append("")
    L.append("GROUND TRUTH — the call sequence each turn was supposed to produce")
    L.append(SUB)
    for i, turn in enumerate(gt.get("ground_truth") or [], start=1):
        L.append(f"  turn {i}:")
        for call in turn:
            L.append(f"      {call}")
    L.append("")

    funcs = list(prompt.get("function") or [])
    L += [SUB, f"SYSTEM PROMPT  ({len(funcs)} tool docs)", SUB,
          fmt_block(system_prompt(funcs), "  "), ""]
    holdout = prompt.get("missed_function") or {}
    if holdout:
        L.append("NOTE: miss_func — tool docs below are appended mid-episode and the")
        L.append("      system prompt is recompiled at that turn.")
        for t, fs in sorted(holdout.items(), key=lambda kv: int(kv[0])):
            L.append(f"      before turn {int(t) + 1}: {[f['name'] for f in fs]}")
        L.append("")

    il = result.get("inference_log") or []
    turn_no = 0
    for entry in il:
        if isinstance(entry, list):          # state snapshot between turns
            L += [SUB, "STATE" if turn_no == 0 else f"STATE after turn {turn_no}", SUB]
            for st in entry:
                L.append(f"  [{st.get('class_name')}]")
                L.append(fmt_block(st.get("content"), "      "))
            L.append("")
            continue

        turn_no += 1
        L += ["", BAR, f"TURN {turn_no}", BAR, ""]
        for q in entry.get("begin_of_turn_query") or []:
            L += [f"---- {q.get('role', 'user')} ----", fmt_block(q.get("content")), ""]

        steps = sorted((k for k in entry if k.startswith("step_")),
                       key=lambda k: int(k.split("_")[1]))
        for sk in steps:
            n = sk.split("_")[1]
            for m in entry[sk]:
                role = m.get("role")
                if role == "assistant":
                    L.append(f"---- step {n} : assistant ----")
                    if m.get("reasoning_content"):
                        L += ["  <reasoning>", fmt_block(m["reasoning_content"], "      "),
                              "  </reasoning>"]
                    L += ["  <content>", fmt_block(m.get("content"), "      "), "  </content>", ""]
                elif role == "handler_log":
                    dec = m.get("model_response_decoded")
                    L.append(f"---- step {n} : handler ----")
                    L.append(f"    note    : {m.get('content')}")
                    if dec is not None:
                        L.append(f"    decoded : {json.dumps(dec, ensure_ascii=False)}")
                    L.append("")
                elif role == "tool":
                    L += [f"---- step {n} : tool result ----",
                          fmt_block(m.get("content"), "      "), ""]
                else:
                    L += [f"---- step {n} : {role} ----", fmt_block(m.get("content")), ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="examples total, spread evenly")
    ap.add_argument("--out", default="analysis/renders/multi_turn")
    args = ap.parse_args()

    per_cat = args.n // len(CATEGORIES)
    out_root = ROOT / args.out
    index = [f"BFCL multi-turn completions — {args.n} examples x {len(CHECKPOINTS)} checkpoints",
             "Same ids in every checkpoint folder, so the three line up file-for-file.",
             "",
             f"{'id':30}{'ep1/916':10}{'ep2/1832':10}{'ep3/2748':10}failure mode (epoch 2)",
             "-" * 100]
    stats = {m: {"pass": 0, "fail": 0} for m, _ in CHECKPOINTS}

    for cat in CATEGORIES:
        data, gt, results, scores = load_category(cat)
        # Stride the whole category rather than taking the first N: BFCL orders
        # multi-turn ids by scenario, so a head slice samples a few scenarios
        # deeply instead of the category broadly.
        ordered = sorted(results[CHECKPOINTS[0][0]], key=lambda i: int(i.rsplit("_", 1)[1]))
        step = max(1, len(ordered) // per_cat)
        ids = ordered[::step][:per_cat]
        for eid in ids:
            row = [f"{eid:30}"]
            for model, label in CHECKPOINTS:
                res = results[model].get(eid)
                if res is None:
                    row.append(f"{'MISSING':10}")
                    continue
                sc = scores[model].get(eid)
                d = out_root / model
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{eid}.txt").write_text(
                    render(eid, cat, model, label, data.get(eid, {}), gt.get(eid, {}), res, sc),
                    encoding="utf-8")
                verdict = "PASS" if sc is None else "FAIL"
                stats[model]["pass" if sc is None else "fail"] += 1
                row.append(f"{verdict:10}")
            mid = scores[CHECKPOINTS[1][0]].get(eid)
            row.append((mid or {}).get("error", {}).get("error_type", "") if mid else "")
            index.append("".join(row))

    index += ["-" * 100]
    for model, label in CHECKPOINTS:
        s = stats[model]
        tot = s["pass"] + s["fail"]
        index.append(f"{model:34} {label:22} {s['pass']}/{tot} pass "
                     f"({s['pass'] / max(1, tot):.1%})")
    (out_root / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("\n".join(index[-5:]))
    print(f"\nwrote to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
