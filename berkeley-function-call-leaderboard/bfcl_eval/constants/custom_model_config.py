"""
Local (non-upstream) model registrations for the Qwen3-4B tooling-SFT experiment.

Kept in its own module so that `git pull` on the upstream gorilla repo does not
conflict with our entries; `model_config.py` only gains a single import and a
single dict-merge.

`model_name` holds an absolute checkpoint directory rather than a Hugging Face
repo id. `OSSHandler.spin_up_local_server` passes that value straight to
`AutoTokenizer/AutoConfig.from_pretrained` and to the vLLM server, so a local
directory works exactly like a hub id and `--local-model-path` is not needed.

Two flavours are registered per checkpoint:

  *-FC              Upstream `QwenFCHandler`. Identical treatment to the
                    official Qwen3-8B/14B/32B leaderboard entries, so the
                    numbers stay comparable to the public leaderboard. Tool
                    docs are serialised in BFCL's flat form
                    ({"name": ..., "parameters": {"type": "dict", ...}}).

  *-FC-nativefmt    `QwenFCNativeToolsHandler` below. Same in every respect
                    except the tool docs are serialised the way the SFT corpus
                    rendered them ({"type": "function", "function": {...}} with
                    "type": "object"). Not leaderboard-comparable; run it to
                    measure how much of the score is prompt-format sensitivity
                    rather than tool-calling ability.

  (no suffix)       Prompt mode via upstream `QwenHandler`. Needed for the v4
                    `format_sensitivity` category, which only applies to
                    prompting-mode models.
"""

import json
import os
import re

from overrides import override

from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.local_inference.qwen_xml import (
    QwenXMLHandler,
    QwenXMLNoThinkPrefillHandler,
)


def _system_suffix() -> str:
    """Extra instruction appended AFTER the tool-block system prompt, or "".

    Set BFCL_SYSTEM_SUFFIX to probe how much of a behaviour is a prompt-level
    habit rather than a trained-in policy. Unset (the default) reproduces the
    stock prompt byte for byte, so existing rows stay comparable.

    Note this lands *outside* the string the SFT template always ended with
    (`...</tool_call>` then `<|im_end|>`). A model overfit to that boundary can
    fall out of tool-calling mode entirely; use _system_prefix() to place the
    same sentence in the slot a training-time system message occupied.
    """
    suffix = os.environ.get("BFCL_SYSTEM_SUFFIX", "").strip()
    return "\n\n" + suffix if suffix else ""


def _system_prefix() -> str:
    """Instruction placed BEFORE "# Tools", where a system message was trained.

    `chat_template_tooling.jinja` renders messages[0].content followed by a
    blank line and then the tool block, so this is the in-distribution slot.
    """
    prefix = os.environ.get("BFCL_SYSTEM_PREFIX", "").strip()
    return prefix + "\n\n" if prefix else ""


class QwenFCNativeToolsHandler(QwenFCHandler):
    """QwenFCHandler, but tool docs rendered in the shape the SFT corpus used."""

    @override
    def _format_prompt(self, messages, function):
        return super()._format_prompt(messages, [_to_openai_tool(f) for f in function])


class QwenFCKeepReasonHandler(QwenFCHandler):
    """QwenFCHandler that keeps the <think> block of EVERY assistant turn in context.

    Upstream `QwenFCHandler` faithfully reproduces Qwen3's official *inference*
    policy: `last_query_index` is the index of the most recent genuine user
    message, and reasoning is re-emitted only for assistant turns after it.
    Everything earlier renders as bare content. Verified 2026-09-01: across two
    user turns, only the current turn's <think> survives.

    That is correct for stock Qwen3, and wrong for a model fine-tuned on a corpus
    that reasons on every assistant turn (Nemotron tool_calling: 1,422,336 of
    1,422,358 turns). Under `chat_template_tooling.jinja` training always showed
    reasoned history; under the upstream handler evaluation shows bare history
    for everything before the last user message. This class removes that skew.

    The method below is a mechanical copy of QwenFCHandler._format_prompt with a
    single substitution -- see the comment at `last_query_index`. Not
    leaderboard-comparable (the published Qwen3 rows all ran the upstream
    policy); run it against the `-FC` rows to measure what the reasoning is worth.
    """

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"
            formatted_prompt += _system_prefix()

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            formatted_prompt += _system_suffix() + "<|im_end|>\n"

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        # THE ONLY CHANGE vs QwenFCHandler: upstream scans backwards for the last real
        # user message and strips reasoning from every assistant turn at or before it.
        # Pinning the cutoff to -1 makes `idx > last_query_index` true for EVERY
        # assistant turn, so every <think> block is rendered into the prompt.
        last_query_index = -1

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"

                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"

                        if "function" in tool_call:
                            tool_call = tool_call["function"]

                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '

                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])

                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt


def _to_openai_tool(func_doc: dict) -> dict:
    """Flat BFCL func doc -> OpenAI-style tool object, with dict->object types."""
    return {
        "type": "function",
        "function": {
            "name": func_doc["name"],
            "description": func_doc.get("description", ""),
            "parameters": _dict_type_to_object(func_doc.get("parameters", {})),
        },
    }


def _dict_type_to_object(schema):
    """BFCL writes JSON-Schema `"type": "dict"`; Qwen's corpus uses `"object"`."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "type" and value == "dict":
                out[key] = "object"
            else:
                out[key] = _dict_type_to_object(value)
        return out
    if isinstance(schema, list):
        return [_dict_type_to_object(item) for item in schema]
    return schema


M = "/home/ubuntu/llm-pretrainer/models"

# Two SFT runs, 10 epochs each. Steps-per-epoch come from the training logs:
# no-reason 64/epoch (640 steps total), reasoning 86/epoch (864 total).
# Registry keys stay step-based (they match the checkpoint dir names and keep
# already-generated results valid); the epoch lives in display_name.
_RUNS = {
    "noreason":  (64, f"{M}/sft-tooling-noreason/Qwen3-4B-Base-sft-%d",  "no-reason"),
    "reasoning": (86, f"{M}/sft-tooling-reasoning/Qwen3-4B-Base-sft-%d", "reasoning"),
}

_CHECKPOINTS = {
    "qwen3-4b-base":     (f"{M}/Qwen3-4B-Base", "Qwen3-4B-Base (untrained zero point)"),
    # Scoring-only entry: the SAME base generations, re-parsed leniently so that
    # bare {"name":..,"arguments":..} objects count even without <tool_call> tags.
    # Never generated against; exists so `bfcl evaluate` can score the rewrite.
    "qwen3-4b-base-lenient": (f"{M}/Qwen3-4B-Base", "Qwen3-4B-Base (lenient parse)"),
    "qwen3-4b-instruct": ("Qwen/Qwen3-4B",      "Qwen3-4B Instruct (reference)"),
}
for _run, (_per_epoch, _tmpl, _label) in _RUNS.items():
    for _epoch in range(1, 11):
        _step = _per_epoch * _epoch
        _CHECKPOINTS[f"qwen3-4b-sft-{_run}-{_step}"] = (
            _tmpl % _step,
            f"Qwen3-4B-Base SFT {_label} epoch {_epoch} (ckpt-{_step})",
        )

# ---------------------------------------------------------------------------
# ToolMind graphsyn SFT (2026-08-31). 3 epochs, per-device batch 4 on 8x A100,
# 916 steps/epoch, so ckpt-916/-1832/-2748 are epochs 1/2/3. The run's `final/`
# save is byte-identical to checkpoint-2748 (same md5), so there are only three
# distinct models. Trained under chat_template_toolmind.jinja, which supervises
# only the final assistant turn -- see llm-pretrainer README.
# Paths are absolute on this box; the /home/ubuntu paths above are from the
# machine the first 22-checkpoint suite ran on and do not resolve here.
# ---------------------------------------------------------------------------
_TOOLMIND = "/workspace/toolmind/models-sft/Qwen3-4B-Base-sft-%d"
for _epoch, _step in enumerate((916, 1832, 2748), start=1):
    _CHECKPOINTS[f"qwen3-4b-sft-toolmind-{_step}"] = (
        _TOOLMIND % _step,
        f"Qwen3-4B-Base SFT toolmind epoch {_epoch} (ckpt-{_step})",
    )

# ---------------------------------------------------------------------------
# Nemotron tool_calling SFT (2026-09-01). ONE epoch, per-device batch 4 on 8x
# A100, 2,941 steps, 1.533 B tokens. Trained under chat_template_tooling.jinja,
# which supervises EVERY assistant turn including its <think> block -- unlike
# ToolMind, which supervises only the last. `final/` is byte-identical to
# checkpoint-2941 (same md5), so only three distinct models exist.
# Because this corpus reasons on every turn, the `-FC-keepreason` variant is the
# faithful evaluation; plain `-FC` measures it under Qwen3's stripping policy.
# ---------------------------------------------------------------------------
_NEMOTRON = "/workspace/nemotron/models-sft/Qwen3-4B-Base-sft-%d"
for _step in (2000, 2500, 2941):
    _CHECKPOINTS[f"qwen3-4b-sft-nemotron-{_step}"] = (
        _NEMOTRON % _step,
        f"Qwen3-4B-Base SFT nemotron epoch 1 (ckpt-{_step})",
    )


# ---------------------------------------------------------------------------
# Mega = toolmind-graphsyn + nemotron-tooling, concatenated and shuffled
# (seed 42), 470,387 rows, one epoch = 3,850 steps, train_loss 0.6288.
# Trained with chat_template_mega.jinja, which supervises every assistant turn
# whose <think> is non-empty -- "every turn" on nemotron rows, "the last turn"
# on toolmind rows. `final/` and `checkpoint-3850` are the same weights (both
# 8044982080 bytes), so only three distinct models exist.
# The corpus mixes per-turn and last-turn reasoning, so BOTH variants are
# informative: `-FC` under Qwen3's stripping policy, `-FC-keepreason` with the
# reasoning kept in context.
# ---------------------------------------------------------------------------
_MEGA = "/workspace/mega/models-sft/Qwen3-4B-Base-sft-%d"
for _step in (3000, 3500, 3850):
    _CHECKPOINTS[f"qwen3-4b-sft-mega-{_step}"] = (
        _MEGA % _step,
        f"Qwen3-4B-Base SFT mega epoch 1 (ckpt-{_step})",
    )


# ---------------------------------------------------------------------------
# Prefix = nemotron-tooling-prefix-sft (2026-09-03/04). Prefix-expanded cut of
# nemotron-tooling: every assistant turn of a source conversation becomes its
# own row, truncated at that turn, with <think> STRIPPED from every context
# turn and kept only on the final target turn (measured: 31.5% of assistant
# turns carry a non-empty <think>, and that count equals the row count exactly).
# 1,412,753 rows, one epoch = 6,493 steps, per-device batch 4 on 8x A100,
# max_length 16384, trained under chat_template_toolmind.jinja (supervises the
# last assistant turn only).
#
# Because the corpus is reasoning-free in context by construction, the FAITHFUL
# evaluation is plain `-FC` -- upstream QwenFCHandler, i.e. Qwen3's official
# inference policy, which strips <think> from every assistant turn at or before
# the last real user message. This is the same handler the ToolMind checkpoints
# were scored under. `-FC-keepreason` is NOT the right variant here: it would
# feed reasoned history the model never saw in training.
# ---------------------------------------------------------------------------
_PREFIX = "/workspace/prefix/models-sft/Qwen3-4B-Base-sft-%d"
for _step in (1400, 2800, 4200, 5600, 6493):
    _CHECKPOINTS[f"qwen3-4b-sft-prefix-{_step}"] = (
        _PREFIX % _step,
        f"Qwen3-4B-Base SFT prefix epoch 1 (ckpt-{_step})",
    )


# ---------------------------------------------------------------------------
# GRPO RL on nemotron_pivot (2026-09-06), starting FROM mega ckpt-3850 -- the
# highest scorer of the SFT suite at 35.82% Overall under FC keepreason.
# Run: rl_scripts/nemotron_pivot_grpo_judge.sh in the verl repo. Reward =
# binary LLM judge (minimax-m3) on prose-GT rows + smooth [-1,1] argument-level
# score on tool-GT rows. Checkpoints merged from FSDP with
# `python -m verl.model_merger merge --backend fsdp`.
#
# step 50  : before tool-call spraying took hold (mean 0.97 calls/response at
#            train step 25, ~2 by step 41). Held-out tool accuracy 0.209 -> 0.498.
# step 100 : after spraying (mean ~4.2 calls/response, single-call responses
#            0.5%). Reported accuracy keeps rising but first-call accuracy is
#            flat, so this checkpoint tests whether the spray hurts BFCL.
#
# The SFT starting point is re-registered here under its LOCAL path: the
# _MEGA entry above points at /workspace/mega/... from the training box, which
# does not resolve on this machine.
#
# All three carry chat_template_tooling.jinja (verified identical), which
# supervises every assistant turn's <think> -- so `-FC-keepreason` is the
# faithful variant, matching how the RL run itself rendered prompts.
# ---------------------------------------------------------------------------
_VERL = "/home/ubuntu/verl/models"
_CHECKPOINTS["qwen3-4b-rl-sft-base-3850"] = (
    f"{_VERL}/Qwen3-4B-Base-sft-3850",
    "Qwen3-4B-Base SFT mega ckpt-3850 (RL start point)",
)
_CHECKPOINTS["qwen3-4b-rl-judge-50"] = (
    f"{_VERL}/rl-judge-step50",
    "Qwen3-4B-Base SFT mega-3850 + GRPO judge (step 50)",
)
_CHECKPOINTS["qwen3-4b-rl-judge-100"] = (
    f"{_VERL}/rl-judge-step100",
    "Qwen3-4B-Base SFT mega-3850 + GRPO judge (step 100)",
)


# ---------------------------------------------------------------------------
# GRPO RL v2 on nemotron_pivot (2026-09-07), same start point (mega ckpt-3850).
# Reward differences vs the "GRPO judge" entries above:
#   * prose rows: the LLM judge no longer sees the gold reply. It only asks
#     whether the response is coherent prose addressing the last user message,
#     i.e. a leniency gate rather than a correctness check.
#   * tool rows: the emitted call count must EQUAL the ground truth count
#     (always 1 in this corpus) or the reward is -1 with nothing else examined.
#     This closed the call-spraying that the v1 run degenerated into.
#
# In-domain at step 74: first-call accuracy 0.189 -> 0.364 with 0.89 calls per
# response (v1 reached 4.24), but prose was answered with a tool call on 92.6%
# of prose-GT rows, so abstention is expected to be weak.
#
# Same chat_template_tooling.jinja as every other entry here, so -FC-keepreason
# remains the faithful variant.
# ---------------------------------------------------------------------------
_CHECKPOINTS["qwen3-4b-rl-v2-50"] = (
    "/home/ubuntu/verl/models/rl-v2-step50",
    "Qwen3-4B-Base SFT mega-3850 + GRPO v2 (step 50)",
)
_CHECKPOINTS["qwen3-4b-rl-v2-100"] = (
    "/home/ubuntu/verl/models/rl-v2-step100",
    "Qwen3-4B-Base SFT mega-3850 + GRPO v2 (step 100)",
)


# ---------------------------------------------------------------------------
# Nemotron tool_calling SFT from models/abdelrahman-qwen (2026-09-15/16), on
# hgx19 (8x H200). ONE epoch, per-device batch 8 x 32768 packed, 737 steps,
# chat_template_tooling.jinja (every assistant turn supervised, <think> included).
# Run: llm-pretrainer/run_abdelrahman_nemotron_sft.sh. `final/` is byte-identical
# to checkpoint-737 (cmp), so only three distinct models exist.
#
# Converted with llm-pretrainer/src/sft_checkpoints_to_pt.sh, then
# max_position_embeddings raised 32768 -> 40960 in each config.json (the value
# official Qwen3-4B ships, same rope_theta). The handler sizes max_tokens from
# that field, and keepreason multi-turn prompts carry every turn's reasoning, so
# 32768 would squeeze or reject the longest long_context prompts.
# Reasons on every turn, so `-FC-keepreason` is the faithful variant.
# ---------------------------------------------------------------------------
_NEM32K = "/local/muhsen/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-%d"
for _step in (500, 625, 737):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-nemotron32k-{_step}"] = (
        _NEM32K % _step,
        f"abdelrahman-qwen SFT nemotron-32k epoch 1 (ckpt-{_step})",
    )


# Same three checkpoints re-evaluated on hgx11 (2026-09-17) with temperature 1.0, a 16384-token
# output cap and a 65536-token context (runlogs/nemotron32k_t1_16k/). Separate keys so results
# land in their own result/ dirs: BFCL skips ids already present, so reusing the keys above
# would silently keep the 4096-token, temperature-0.001 generations.
# ckpt-737 is served from Qwen3-4B-Base-sft-final, which is byte-identical to checkpoint-737.
_NEM32K_HGX11 = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-%s"
for _step, _dir in ((500, "500"), (625, "625"), (737, "final")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-nemotron32k-{_step}-t1-out16k-ctx64k"] = (
        _NEM32K_HGX11 % _dir,
        f"abdelrahman-qwen SFT nemotron-32k epoch 1 (ckpt-{_step}, T=1.0, out 16k, ctx 64k)",
    )


# ckpt-737 again on hgx11 (2026-09-17), isolating the output cap: the original
# near-greedy settings (temperature 0.001, 40960-token context) with only the
# output cap raised 4096 -> 16384 (runlogs/nemotron32k_t0_16k/).
# No commas in the display name: bfcl writes score/data_*.csv unquoted, so a
# comma there shifts every later column (the T=1.0 rows above have this problem).
_CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k"] = (
    _NEM32K_HGX11 % "final",
    "abdelrahman-qwen SFT nemotron-32k epoch 1 (ckpt-737 / T=0.001 / out 16k / ctx 41k)",
)


# GRPO v4 RL checkpoint, step 450, started from nemotron-32k SFT ckpt-737
# (llm-pretrainer/models/qwen3-4b-rl-v4-sft737-step450). Evaluated on avey-bm-02
# (2026-09-19) with the original near-greedy settings (temperature 0.001, the
# 40960 context from its config.json) and a 16384-token output cap, so it compares
# directly with abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k
# (runlogs/rl_v4_step450_out16k/).
_CHECKPOINTS["qwen3-4b-rl-v4-sft737-step450-out16k"] = (
    "/mnt/data01/muhsen/tooling/llm-pretrainer/models/qwen3-4b-rl-v4-sft737-step450",
    "Qwen3-4B RL v4 from SFT-737 step 450 (T=0.001 / out 16k / ctx 41k)",
)


# Olive tooling SFT from models/abdelrahman-qwen (2026-09-21/22), 3 epochs, 1074
# steps, batch 8 x 30 GPUs x 32768 packed, chat_template_tooling.jinja (every turn
# supervised, <think> on every assistant turn -> keepreason). Converted with
# llm-pretrainer/src/sft_checkpoints_to_pt.sh (weights MOVED out of the training
# checkpoints). Evaluated on avey-bm-04 with the same settings as the ckpt-737
# out-16k row: temperature 0.001, 40960 context, 16384-token output cap
# (runlogs/olive32k_out16k/).
_OLIVE32K = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-32k-bs8-3ep-30gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((875, "2.44"), (1000, "2.79"), (1074, "3")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-olive32k-{_step}-out16k"] = (
        _OLIVE32K % _step,
        f"abdelrahman-qwen SFT olive-32k (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )



class QwenFCLenientHandler(QwenFCKeepReasonHandler):
    """QwenFCKeepReasonHandler that tolerates trailing junk after the tool-call JSON.

    olive-32b has no tool-calling prior (its base chat template has no tools
    block at all), and while learning the syntax it closes the object one brace
    too many:

        {"name": "math.factorial", "arguments": {"number": 5}}}

    Upstream `_extract_tool_calls` runs json.loads on that, raises, and swallows
    the error with a bare except -- so a well-formed call with the right name and
    arguments is reported as no call at all, and every AST category scores ~0 for
    a reason that has nothing to do with tool-calling ability.

    raw_decode parses the leading JSON value and ignores whatever follows, which
    repairs exactly this case and nothing else: a body that is malformed *before*
    its first complete value still fails, as it should. Rows scored with this
    handler are NOT comparable to the strict rows -- they answer "can the model
    pick the right call?", not "does the model emit valid JSON?".
    """

    _DECODER = json.JSONDecoder()

    @staticmethod
    @override
    def _extract_tool_calls(input_string):
        matches = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", input_string, re.DOTALL)
        result = []
        for match in matches:
            body = match.strip()
            try:
                result.append(json.loads(body))
                continue
            except ValueError:
                pass
            try:
                obj, _ = QwenFCLenientHandler._DECODER.raw_decode(body)
            except ValueError:
                continue
            result.append(obj)
        return result



# --- prompt-nudge probe --------------------------------------------------
# Same weights as the rows above; the only difference is BFCL_SYSTEM_SUFFIX,
# set by runlogs/parallel_nudge/eval.sh. olive-1074 emits exactly one tool call
# on 44% of parallel prompts although its own reasoning says it needs several;
# these rows measure whether one sentence in the system prompt recovers that,
# i.e. whether the habit is prompt-level or trained in. Run on parallel and
# parallel_multiple only. Not leaderboard-comparable.
_CHECKPOINTS["abdelrahman-qwen-sft-olive32k-1074-out16k-nudge"] = (
    _OLIVE32K % 1074,
    "abdelrahman-qwen SFT olive-32k (ckpt-1074 / epoch 3 / parallel-call nudge / T=0.001 / out 16k / ctx 41k)",
)
_CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-out16k-nudge"] = (
    _CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k"][0],
    "abdelrahman-qwen SFT nemotron-32k (ckpt-737 / parallel-call nudge / T=0.001 / out 16k / ctx 41k)",
)


_CHECKPOINTS["abdelrahman-qwen-sft-olive32k-1074-out16k-nudge2"] = (
    _OLIVE32K % 1074,
    "abdelrahman-qwen SFT olive-32k (ckpt-1074 / epoch 3 / parallel-call nudge before tools / T=0.001 / out 16k / ctx 41k)",
)
_CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-out16k-nudge2"] = (
    _CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k"][0],
    "abdelrahman-qwen SFT nemotron-32k (ckpt-737 / parallel-call nudge before tools / T=0.001 / out 16k / ctx 41k)",
)


# Probe C: a neutral system line carrying no instruction about call counts.
# Both nudges raised olive's terse-register share (44% -> 92% / 71%), so the
# trigger may be the mere presence of a leading system message -- the shape the
# olive corpus always had -- rather than what it says. This row separates the two.
_CHECKPOINTS["abdelrahman-qwen-sft-olive32k-1074-out16k-neutral"] = (
    _OLIVE32K % 1074,
    "abdelrahman-qwen SFT olive-32k (ckpt-1074 / epoch 3 / neutral system line / T=0.001 / out 16k / ctx 41k)",
)


# pass@8 sampling rows: seven independent T=1.0 draws of the same weights as
# the scored ckpt-1074 row, live + non-live only. Separate registry keys so each
# draw lands in its own result/ and score/ directory; runlogs/olive_pass8/.
for _s in range(1, 8):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-olive32k-1074-t1-s{_s}"] = (
        _OLIVE32K % 1074,
        f"abdelrahman-qwen SFT olive-32k (ckpt-1074 / epoch 3 / T=1.0 sample {_s} / out 16k / ctx 41k)",
    )


# GRPO RL on top of the olive SFT ckpt-1074 (verl run grpo_olive_rl300k_olivesft1074,
# dataset rl-data/olive_rl300k). FSDP shards merged to HF format with
# `python -m verl.model_merger merge --backend fsdp` inside the verlai/verl image,
# run as our own uid so the root-owned training checkpoints stay unwritable.
# Step 50 was pruned by max_actor_ckpt_to_keep=2 before the run was restarted at
# 11:51 on 2026-09-24 with max_actor_ckpt_to_keep=null; every step from 100 on is kept.
# Same eval settings as the olive SFT rows: T=0.001, ctx 40960, 16384-token output.
_GRPO_OLIVE = "/mnt/data01/muhsen/tooling/models-sft/grpo-olive-rl300k-olivesft1074/global_step_%d"
for _step in (100, 150, 200, 250, 300, 350):
    _CHECKPOINTS[f"grpo-olive-rl300k-sft1074-step{_step}-out16k"] = (
        _GRPO_OLIVE % _step,
        f"GRPO olive-rl300k on olive SFT ckpt-1074 (step {_step} / T=0.001 / out 16k / ctx 41k)",
    )


# olive-32b SFT on the same olive-tooling-sft corpus as the 4B rows above, so the
# two are a clean 4B-vs-32B comparison. The checkpoints already hold consolidated
# HF safetensors; they were copied (never moved) out of the LIVE training run and
# given the base model's tokenizer files, whose vocab/merges/added tokens are
# byte-identical to the trainer's.
#
# Two differences from the 4B rows that the serve script has to cancel out:
#   * this generation_config sets do_sample/top_k 20/top_p 0.8/repetition_penalty
#     1.05, none of which applied to the 4B runs -- vLLM would apply them server
#     wide and make the numbers incomparable.
#   * max_position_embeddings is already 131072, so no --hf-overrides is needed;
#     --max-model-len 40960 alone matches the 4B context.
_OLIVE32B = "/mnt/data01/muhsen/tooling/models-sft/olive-32b-olive-tooling-32k-bs3-3ep-30gpu/checkpoint-%d"
for _step, _epoch in ((477, "0.50"), (954, "1.00"), (1431, "1.50"),
                      (1908, "1.99"), (2385, "2.49")):
    _CHECKPOINTS[f"olive32b-olive-tooling-{_step}-out16k"] = (
        _OLIVE32B % _step,
        f"olive-32b SFT olive-tooling (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# abdelrahman-qwen 4B SFT on the olive-tooling *length-bias* remix
# (llm-pretrainer/sft-datasets/olive-tooling-sft-length-bias), 32k seq, bs8,
# 3 epochs on 32 GPUs.  1020 steps total = 340 steps/epoch, but checkpoints land
# every 125 steps, so no checkpoint sits exactly on an epoch boundary; these are
# the first checkpoint at or past each one.  config.json is a genuine Qwen3-4B
# (36L / 2560h / 151936 vocab) with max_position_embeddings 32768 and a
# generation_config that caps max_new_tokens at 2048, so serving needs the same
# --hf-overrides / --override-generation-config as the olive-32k rows.  Same
# settings as those rows so the numbers stay comparable: T=0.001, ctx 40960,
# 16384-token output cap (runlogs/lengthbias/).
_LENGTHBIAS = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-lengthbias-32k-bs8-3ep-32gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((375, "1.10"), (750, "2.21"), (1020, "3.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-lengthbias-{_step}-out16k"] = (
        _LENGTHBIAS % _step,
        f"abdelrahman-qwen SFT olive-lengthbias (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )

# abdelrahman-qwen 4B SFT on the olive regen-xml 8-source corpus, 32k, bs8,
# 1 epoch, 24 GPUs.  Trained with chat_template_tooling_xml.jinja -- Qwen3.5's
# template with reasoning preserved on every assistant turn -- so tool calls are
# XML (<function=name><parameter=p>value</parameter></function>), not JSON.
# Served through QwenXMLHandler, whose _format_prompt is a byte-exact port of
# that template and whose decoder mirrors vLLM 0.12's qwen3xml tool parser
# (runlogs/regenxml/).
_REGENXML = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-regen-xml-8src-32k-bs8-1ep-24gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((125, "0.48"), (261, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-regenxml-{_step}-out16k"] = (
        _REGENXML % _step,
        f"abdelrahman-qwen SFT olive-regen-xml (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# abdelrahman-qwen 4B SFT on olive-tooling-sft-xml-8-source-term-corp, 32k, bs8,
# 1 epoch, 24 GPUs (effective batch 192, 262 steps).  Same
# chat_template_tooling_xml.jinja as the regen-xml run (md5 identical), so the
# same QwenXMLHandler serves it; the two differ only in corpus, which makes them
# a clean A/B on the data (runlogs/olivexml/).
_OLIVEXML = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-xml-8src-32k-bs8-1ep-24gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((125, "0.48"), (262, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-olivexml-{_step}-out16k"] = (
        _OLIVEXML % _step,
        f"abdelrahman-qwen SFT olive-xml-8src (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# The shared PARENT of every tooling SFT run above: models/abdelrahman-qwen, a
# 2-epoch "dual-mode-sft" of Qwen3-4B-Base (lr 1e-5, max_length 16384, Aug 05).
# Evaluated here as a baseline to separate what the tooling SFT added from what
# it destroyed -- it is coherent at prompt lengths where the XML children
# degenerate.  Scored under both the JSON (-FC-keepreason) and XML (-XML)
# variants off the same server, since the format lives in the handler
# (runlogs/basemodel/).
_CHECKPOINTS["abdelrahman-qwen-base-out16k"] = (
    "/mnt/data01/muhsen/tooling/llm-pretrainer/models/abdelrahman-qwen",
    "abdelrahman-qwen BASE dual-mode-sft parent (T=0.001 / out 16k / ctx 41k)",
)


# Two 1-epoch JSON-template runs (chat_template_tooling.jinja, md5 identical to
# the olive-32k run, so -FC-keepreason / -FC-keepreason-lenient apply):
#   json-full   : sft-datasets/olive-tooling-sft (the FULL 339,629-row corpus)
#                 global batch 192, 24 GPUs, 448 steps
#   remix-regen : sft-datasets/olive-tooling-sft-remix-reasoning-regen (299,996 rows)
#                 global batch 132, 22 GPUs, 455 steps
# Middle (step 250, ~epoch 0.56/0.55) and final checkpoints evaluated
# (runlogs/jsonfull_remix/).
_JSONFULL = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-json-full-32k-gb192-1ep-24gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.56"), (448, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-jsonfull-{_step}-out16k"] = (
        _JSONFULL % _step,
        f"abdelrahman-qwen SFT olive-json-full (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )
_REMIXREGEN = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-remix-regen-32k-gb132-1ep-22gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.55"), (455, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-remixregen-{_step}-out16k"] = (
        _REMIXREGEN % _step,
        f"abdelrahman-qwen SFT olive-remix-regen (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# olive-xml-FULL: the XML template (chat_template_tooling_xml.jinja, md5
# fbbd150cc817bff2582096482c5d9482 -- verified against the copy shipped in each
# converted checkpoint) applied to the FULL olive corpus rather than the
# 8-source cut, 32k, 1 epoch, 15 GPUs (bs 3x4), 482 steps.  No raw run dir or
# training_config survives, so the template was identified from the converted
# model's own chat_template.jinja.  Served with QwenXMLHandler (-XML).
# Middle (step 250, ~epoch 0.52) and final (482) evaluated
# (runlogs/xmlfull/).
_XMLFULL = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-xml-full-32k-bs3x4-1ep-15gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.52"), (482, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-xmlfull-{_step}-out16k"] = (
        _XMLFULL % _step,
        f"abdelrahman-qwen SFT olive-xml-full (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# remix-regen XML: the reasoning-regen remix corpus
# (sft-datasets/olive-tooling-sft-remix-reasoning-regen-xml) rendered with the
# XML template, 32k, 1 epoch, 23 GPUs (global batch 138), 442 steps.  This is
# the cell that combines XML's better Live with regen's better Non-Live --
# trained earlier but never converted until now.  Converted with cp from
# checkpoints/.../checkpoint-{250,442}; weights byte-identical to source, the
# four tokenizer files the trainer does not save taken from models/abdelrahman-qwen.
# Middle (250 / epoch 0.57) and final (442 / epoch 1.00) (runlogs/remixregenxml/).
_REMIXREGENXML = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-remix-regen-xml-32k-gb138-1ep-23gpu/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.57"), (442, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-remixregenxml-{_step}-out16k"] = (
        _REMIXREGENXML % _step,
        f"abdelrahman-qwen SFT olive-remix-regen-xml (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# REPRODUCIBILITY CHECK of the nemotron-737 row (Overall 33.94 / Non-Live 81.56 /
# Live 75.35 / MultiTurn 19.50).  That 19.50% is the only 1-epoch multi-turn
# result in the whole study and a load-bearing input to the scale-up decision,
# and the degeneration boundary was shown to be stochastic under vLLM
# data-parallel batching -- so it is worth re-running.  Same weights, same
# handler, same settings as the original row; a separate registry key only so
# the generations land in their own result/ dir instead of being skipped
# (runlogs/nemotron_repro/).
_CHECKPOINTS["abdelrahman-qwen-sft-nemotron32k-737-repro"] = (
    "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-final",
    "abdelrahman-qwen SFT nemotron-32k epoch 1 (ckpt-737 REPRO / T=0.001 / out 16k / ctx 41k)",
)


# CURRICULUM run, trained on another machine and pulled from
# s3://muhsen-avey-bucket/pi-machine-backup/models/abdelrahman-qwen-curriculum-nemotron-olive-remix-regen-json/
# (40 objects, verified byte-for-byte against S3 on download).  Name says it
# blends nemotron with the olive remix reasoning-regen corpus under a curriculum
# -- i.e. the three levers the analysis pointed at: nemotron's shallow-
# concentrated depth profile, regen's multi-call emission, and explicit
# shallow->deep ordering.  JSON template (chat_template.jinja md5
# 0fd9d91c2abadf39f998d6eb112a557a) so -FC-keepreason / -FC-keepreason-lenient
# apply.  Unlike the other conversions this one already carries
# max_position_embeddings 40960.  Middle (250) and final (469) evaluated
# (runlogs/curriculum/).
_CURRIC = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-curriculum-nemotron-olive-remix-regen-json/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.53"), (469, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-curriculum-{_step}-out16k"] = (
        _CURRIC % _step,
        f"abdelrahman-qwen SFT curriculum nemotron+olive-remix-regen (ckpt-{_step} / epoch {_epoch} / T=0.001 / out 16k / ctx 41k)",
    )


# CURRICULUM re-eval with the RoPE config repaired.
#
# transformers 5.2.0 writes RoPE as a nested block,
#   "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}
# and vLLM 0.8.5 -- the version BFCL pins -- never reads that key; the string
# appears nowhere in the package. Every 5.2.0-converted checkpoint was therefore
# served with its trained RoPE silently discarded, which produced coherent text
# at short context and token noise past ~4k prompt tokens. That, not the data,
# is why the first curriculum eval scored 0.00% multi-turn.
#
# models-sft/curriculum-oldcfg/ holds the same weights (hard-linked, verified
# byte-identical) with config/generation_config/tokenizer copied byte-for-byte
# from the 4.x-converted olive-32k ckpt-1074, which carries a flat rope_theta.
# Proven on a 1-GPU sweep: these weights go from collapsing at 3,994 prompt
# tokens to clean at 26,826, matching their nemotron-737 parent.
# Separate registry keys so the invalid generations are not reused
# (runlogs/curriculum/, ropefix run).
_CURRIC_FIX = "/mnt/data01/muhsen/tooling/models-sft/curriculum-oldcfg/Qwen3-4B-Base-sft-%d"
for _step, _epoch in ((250, "0.53"), (469, "1.00")):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-curriculum-{_step}-ropefix-out16k"] = (
        _CURRIC_FIX % _step,
        f"abdelrahman-qwen SFT curriculum nemotron+olive-remix-regen (ckpt-{_step} / epoch {_epoch} / ROPEFIX / T=0.001 / out 16k / ctx 41k)",
    )


# Leave-one-out corpus probe (llm-pretrainer, requested by the training session).
# One arm per held-out source, all from the SAME base (nemotron-32k sft-final),
# one epoch, seed 1, global batch 56 on 7 GPUs -- so the arms are directly
# comparable and the delta from `all9` is that source's contribution.
# Converted with the flat rope_theta already on disk, so runlogs/ropefix_reeval/
# serve.sh (max_position_embeddings override only) is the right serve shape.
# JSON tool calls + reasoning on every assistant turn => -FC-keepreason, not -FC.
# NOTE: only one all9 run (seed 1) exists, so there is no seed noise band.
_PROBE = "/mnt/data01/muhsen/tooling/models-sft/probe-%s-seed1-32k-gb56-1ep-7gpu/Qwen3-4B-Base-sft-final"
for _arm in ("all9", "no-toucan", "no-swe-zero-openhands", "no-nemotron-post-training",
             "no-nemotron-agentic-tool", "no-openresearcher", "no-terminal-corpus",
             "no-openseeker", "no-nemotron-agentic-interactive", "no-toolace"):
    _CHECKPOINTS[f"probe-{_arm}-out16k"] = (
        _PROBE % _arm,
        f"probe LOO {_arm} (seed 1 / gb56 / 1ep / final / T=0.001 / out 16k / ctx 41k)",
    )


# Curriculum nemotron -> olive-remix-regen WITHOUT toucan, JSON tool calls.
# Same curriculum shape as abdelrahman-qwen-curriculum-nemotron-olive-remix-regen-json
# but with the toucan source held out -- the LOO probe (2026-10-05) showed removing
# toucan improved every column (Overall +2.16, Non-Live +6.67, Live +2.37, MT +2.75),
# and toucan had the lowest train_loss of all ten arms (0.6286 vs 0.6871 baseline),
# the signature of easy-to-fit, low-diversity data. Downloaded from
# s3://muhsen-avey-bucket/pi-machine-backup/models/abdelrahman-qwen-curriculum-nemotron-olive-remix-notoucan-json/
# Checkpoints 150 / 300 / 403; first (150) and last (403) evaluated (runlogs/notoucan/).
_NOTOUCAN = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-curriculum-nemotron-olive-remix-notoucan-json/Qwen3-4B-Base-sft-%d"
for _step in (150, 300, 403):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-notoucan-{_step}-out16k"] = (
        _NOTOUCAN % _step,
        f"abdelrahman-qwen SFT curriculum nemotron+olive-remix NO-TOUCAN (ckpt-{_step} / T=0.001 / out 16k / ctx 41k)",
    )


# Curriculum nemotron -> olive-remix, toucan held out, with PROPORTIONAL REPEAT of the
# remaining sources (prop-repeat) so the held-out tokens are made up by upsampling the
# rest rather than shrinking the corpus. JSON tool calls, chat_template_tooling.jinja.
# Downloaded from s3://muhsen-avey-bucket/pi-machine-backup/fixed-models/
#   abdelrahman-qwen-curriculum-nemotron-olive-remix-notoucan-prop-repeat-json/
# Checkpoints 150 / 300 / 450 / 455; middle (300) and last (455) evaluated
# (runlogs/notoucan_prop/).  Compare against the plain no-toucan run (403 steps,
# Non-Live 81.73 / MT 25.12) to see whether upsampling recovers the multi-turn that
# the plain hold-out lost versus curriculum-469 (MT 31.75).
_NOTOUCAN_PROP = "/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-curriculum-nemotron-olive-remix-notoucan-prop-repeat-json/Qwen3-4B-Base-sft-%d"
for _step in (150, 300, 450, 455):
    _CHECKPOINTS[f"abdelrahman-qwen-sft-notoucanprop-{_step}-out16k"] = (
        _NOTOUCAN_PROP % _step,
        f"abdelrahman-qwen SFT curriculum nemotron+olive-remix NO-TOUCAN prop-repeat (ckpt-{_step} / T=0.001 / out 16k / ctx 41k)",
    )


_VARIANTS = [
    # registry suffix, handler,                   is_fc_model, display suffix
    ("-FC", QwenFCHandler, True, " (FC)"),
    ("-FC-nativefmt", QwenFCNativeToolsHandler, True, " (FC nativefmt)"),
    ("-FC-keepreason", QwenFCKeepReasonHandler, True, " (FC keepreason)"),
    ("-FC-keepreason-lenient", QwenFCLenientHandler, True, " (FC keepreason lenient-json)"),
    ("-XML", QwenXMLHandler, True, " (Qwen3.5 XML tool calls)"),
    ("-XML-nothinkprefill", QwenXMLNoThinkPrefillHandler, True, " (XML, model emits <think> itself)"),
    ("", QwenHandler, False, " (Prompt)"),
]


def _build() -> dict:
    from bfcl_eval.constants.model_config import ModelConfig

    mapping = {}
    for stem, (path, display) in _CHECKPOINTS.items():
        for suffix, handler, is_fc, display_suffix in _VARIANTS:
            mapping[f"{stem}{suffix}"] = ModelConfig(
                model_name=path,
                display_name=f"{display}{display_suffix}",
                url="https://huggingface.co/Qwen/Qwen3-4B-Base",
                org="Qwen (local SFT)",
                license="apache-2.0",
                model_handler=handler,
                input_price=None,
                output_price=None,
                is_fc_model=is_fc,
                underscore_to_dot=False,
            )
    return mapping
