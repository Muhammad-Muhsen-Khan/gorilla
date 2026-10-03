"""Handler for models trained on `chat_template_tooling_xml.jinja`.

That template is Qwen3.5's chat template with reasoning preserved on every
assistant turn, and its tool calls are XML rather than JSON:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

Two things follow from the format, and both are handled below.

* There is no JSON object around the call, so the surplus-closing-brace failure
  that the JSON template produced (`{"name": ..., "arguments": {...}}}`) cannot
  occur -- there is no brace to duplicate.
* XML carries no types: every value arrives as text. The reference decoder
  recovers the type from the tool schema, so this handler does too. The
  conversion rules in `_convert_param_value` are a port of
  `vllm/entrypoints/openai/tool_parsers/qwen3xml_tool_parser.py` (vLLM 0.12.0);
  BFCL pins vLLM 0.8.5, which predates that parser, and BFCL parses the raw
  completion itself rather than letting the server do it, so the logic has to
  live here either way.

`decode_ast`/`decode_execute` take an optional `function=` schema. The eval
runner passes it only for handlers that set `WANTS_FUNCTION_SCHEMA`; without it
every value degenerates to `str`, which the AST checker's strict
`type(value) == expected` comparison would reject.
"""

import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

# <function=NAME> or <function name="NAME">
_FUNCTION_RE = re.compile(
    r"<function(?:=([^>\s]+)|\s+name\s*=\s*[\"']([^\"']+)[\"'])\s*>(.*?)</function>",
    re.DOTALL,
)
# <parameter=NAME> or <parameter name="NAME">
_PARAM_RE = re.compile(
    r"<parameter(?:=([^>\s]+)|\s+name\s*=\s*[\"']([^\"']+)[\"'])\s*>(.*?)</parameter>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_STRING_TYPES = {"string", "str", "text", "varchar", "char", "enum"}
_BOOL_TYPES = {"boolean", "bool", "binary"}
_STRUCT_TYPES = {"object", "array", "arr", "sequence"}


def _strip_one_newline(value: str) -> str:
    """Undo the template's `<parameter=x>\\n` ... `\\n</parameter>` padding.

    Exactly one newline on each side is the template's own framing; anything
    further is part of the value (the system prompt advertises multi-line
    values), so only one is removed.
    """
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _param_types(function: list | None, func_name: str) -> dict:
    """{param_name: json-schema type} for `func_name`, or {} when unavailable."""
    for tool in function or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        if spec.get("name") != func_name:
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties", params) if isinstance(params, dict) else {}
        out = {}
        for name, cfg in props.items() if isinstance(props, dict) else []:
            if isinstance(cfg, dict):
                t = cfg.get("type", "string")
                out[name] = str(t[0] if isinstance(t, list) and t else t)
        return out
    return {}


def _convert_param_value(raw: str, param_type: str) -> Any:
    """Port of Qwen3XMLToolParser._convert_param_value (vLLM 0.12.0).

    The one addition: object/array values are `json.loads`-ed here. The
    reference splices their raw text into the OpenAI `arguments` JSON string,
    so the caller's own `json.loads` does the same job one level up; BFCL wants
    real Python values, so the parse happens here instead.
    """
    if raw.lower() == "null":
        return None

    param_type = param_type.strip().lower()

    if param_type in _STRING_TYPES:
        return raw

    if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw

    if param_type.startswith(("num", "float", "double", "decimal")):
        try:
            as_float = float(raw)
        except (ValueError, TypeError):
            return raw
        # Reference behaviour: an integral float degenerates to int. The AST
        # checker auto-widens int -> float for Python, so this stays correct.
        return as_float if as_float - int(as_float) != 0 else int(as_float)

    if param_type in _BOOL_TYPES:
        return raw.lower() == "true"

    if param_type in _STRUCT_TYPES or param_type.startswith(("dict", "list", "tuple")):
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    # repair_param_type(): anything unrecognised is treated as a string.
    return raw


class QwenXMLHandler(QwenFCHandler):
    """Qwen3.5 XML tool-calling, with every turn's <think> kept in context."""

    #: Opt in to receiving the tool schema in decode_ast/decode_execute.
    WANTS_FUNCTION_SCHEMA = True

    # ------------------------------------------------------------------ parse

    @staticmethod
    @override
    def _extract_tool_calls(input_string, function=None):
        """[{"name": ..., "arguments": {...}}] for each well-formed call."""
        result = []
        for block in _TOOL_CALL_RE.findall(input_string or ""):
            for eq_name, attr_name, body in _FUNCTION_RE.findall(block):
                name = (eq_name or attr_name).strip()
                if not name:
                    continue
                types = _param_types(function, name)
                args = {}
                for p_eq, p_attr, raw in _PARAM_RE.findall(body):
                    p_name = (p_eq or p_attr).strip()
                    if not p_name:
                        continue
                    args[p_name] = _convert_param_value(
                        _strip_one_newline(raw), types.get(p_name, "string")
                    )
                result.append({"name": name, "arguments": args})
        return result

    @override
    def decode_ast(self, result, language, has_tool_call_tag, function=None):
        tool_calls = self._extract_tool_calls(result, function)
        if type(tool_calls) != list or any(type(i) != dict for i in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{c["name"]: dict(c["arguments"])} for c in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag, function=None):
        tool_calls = self._extract_tool_calls(result, function)
        if type(tool_calls) != list or any(type(i) != dict for i in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call(
            [{c["name"]: c["arguments"]} for c in tool_calls]
        )

    # ----------------------------------------------------------------- render

    #: Verbatim from chat_template_tooling_xml.jinja. Kept as one literal so a
    #: drift test can compare it against the template file.
    _CALL_FORMAT_INSTRUCTIONS = (
        "\n\nIf you choose to call a function ONLY reply in the following format"
        " with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format:"
        " an inner <function=...></function> block must be nested within"
        " <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural"
        " language BEFORE the function call, but NOT after\n- If there is no function"
        " call available, answer the question like normal with your current knowledge"
        " and do not tell the user about function calls\n</IMPORTANT>"
    )

    @staticmethod
    def _render_argument(value):
        """The template's `| tojson` / `| string` split for a parameter value."""
        if isinstance(value, dict) or (
            isinstance(value, (list, tuple)) and not isinstance(value, str)
        ):
            return json.dumps(value, ensure_ascii=False)
        if value is True:
            return "True"
        if value is False:
            return "False"
        if value is None:
            return "None"
        return str(value)

    @override
    def _format_prompt(self, messages, function):
        """Port of chat_template_tooling_xml.jinja.

        Differences from the Jinja source, both deliberate:
          * `preserve_thinking` is left at its default, so every assistant turn
            keeps its <think> block -- the property the corpus was built for.
          * the vision/`render_content` branches are dropped; BFCL messages are
            always plain strings.
        """
        out = ""

        if function:
            out += "<|im_start|>system\n"
            out += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
            for tool in function:
                out += "\n" + json.dumps(tool, ensure_ascii=False)
            out += "\n</tools>"
            out += self._CALL_FORMAT_INSTRUCTIONS
            if messages and messages[0]["role"] == "system":
                sys_content = (messages[0]["content"] or "").strip()
                if sys_content:
                    out += "\n\n" + sys_content
            out += "<|im_end|>\n"
        elif messages and messages[0]["role"] == "system":
            out += (
                "<|im_start|>system\n"
                + (messages[0]["content"] or "").strip()
                + "<|im_end|>\n"
            )

        for idx, message in enumerate(messages):
            role = message["role"]
            content = (message.get("content") or "").strip()

            if role == "system":
                continue  # already emitted above

            if role == "user":
                out += f"<|im_start|>user\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning = message.get("reasoning_content")
                if not isinstance(reasoning, str):
                    reasoning = ""
                    if "</think>" in content:
                        head, _, tail = content.partition("</think>")
                        reasoning = head.rstrip("\n").split("<think>")[-1].lstrip("\n")
                        content = tail.lstrip("\n")
                reasoning = reasoning.strip()

                out += "<|im_start|>assistant\n"
                out += "<think>\n" + reasoning + "\n</think>\n\n" + content

                for n, tool_call in enumerate(message.get("tool_calls") or []):
                    call = tool_call.get("function", tool_call)
                    if n == 0:
                        out += "\n\n" if content.strip() else ""
                    else:
                        out += "\n"
                    out += "<tool_call>\n<function=" + call["name"] + ">\n"
                    args = call.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    if isinstance(args, dict):
                        for k, v in args.items():
                            out += f"<parameter={k}>\n{self._render_argument(v)}\n</parameter>\n"
                    elif isinstance(args, list):
                        for param in args:
                            out += (
                                f"<parameter={param['name']}>\n"
                                f"{param['value']}\n</parameter>\n"
                            )
                    out += "</function>\n</tool_call>"

                out += "<|im_end|>\n"

            elif role == "tool":
                prev = messages[idx - 1]["role"] if idx > 0 else None
                nxt = messages[idx + 1]["role"] if idx < len(messages) - 1 else None
                if idx > 0 and prev != "tool":
                    out += "<|im_start|>user"
                out += f"\n<tool_response>\n{content}\n</tool_response>"
                if idx == len(messages) - 1 or nxt != "tool":
                    out += "<|im_end|>\n"

        # add_generation_prompt with thinking enabled: the turn opens inside
        # <think>, so the completion begins mid-reasoning.
        out += "<|im_start|>assistant\n<think>\n"
        return out

class QwenXMLNoThinkPrefillHandler(QwenXMLHandler):
    """QwenXMLHandler that stops the prompt at `<|im_start|>assistant\n`.

    The template's add_generation_prompt appends `<think>\n`, so the model is
    normally handed the opening of its own reasoning block. But the training
    loss span *starts* at `<think>` (token 151667) -- see the mask: the three
    header tokens are excluded and 151667 is the first supervised token. So the
    model was trained to emit that token itself, and pre-filling it removes the
    only decision point it ever practised at the start of a turn.

    Teacher forcing makes the two equivalent in principle: after
    `<|im_start|>assistant\n` the target is `<think>`, and continuing from an
    already-present `<think>\n` is the same trajectory. This variant exists to
    test that empirically against the degeneration cliff.
    """

    _PREFILL = "<|im_start|>assistant\n<think>\n"

    @override
    def _format_prompt(self, messages, function):
        out = super()._format_prompt(messages, function)
        assert out.endswith(self._PREFILL), out[-40:]
        return out[: -len("<think>\n")]
