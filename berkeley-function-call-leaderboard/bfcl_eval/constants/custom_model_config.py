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

_VARIANTS = [
    # registry suffix, handler,                   is_fc_model, display suffix
    ("-FC", QwenFCHandler, True, " (FC)"),
    ("-FC-nativefmt", QwenFCNativeToolsHandler, True, " (FC, native tool fmt)"),
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
