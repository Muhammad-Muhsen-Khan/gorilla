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

from overrides import override

from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


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

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

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


_VARIANTS = [
    # registry suffix, handler,                   is_fc_model, display suffix
    ("-FC", QwenFCHandler, True, " (FC)"),
    ("-FC-nativefmt", QwenFCNativeToolsHandler, True, " (FC nativefmt)"),
    ("-FC-keepreason", QwenFCKeepReasonHandler, True, " (FC keepreason)"),
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
