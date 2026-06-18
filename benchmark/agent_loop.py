"""Observed agent-loop execution for local repo development tasks."""
from __future__ import annotations

from copy import deepcopy
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from benchmark.repo_patch import (
    _LAUNCH_ERROR,
    _TIMEOUT_ERROR,
    _make_sentinel_hidden_test,
    _resolve_fixture_path,
    _run_test_command,
    _safe_target,
    _sentinel_completed,
    _is_protected_harness_path,
    _truncate,
    _validate_test_command,
    _write_hidden_tests,
)
from benchmark.responses_api import (
    messages_to_responses_input,
    response_output_text,
    response_usage_metadata,
    usage_metadata,
)


_AGENT_SYSTEM = """You are working in a temporary repository workspace.
Respond with exactly one tool action per turn, with no markdown.
Available tools:
  {"tool":"list_files","args":{"path":"."}}
  {"tool":"read_file","args":{"path":"relative/path.py"}}
  {"tool":"write_file","args":{"path":"relative/path.py","content":"complete file contents"}}
  {"tool":"run_tests","args":{}}
  {"tool":"final","args":{"summary":"short summary"}}
Simple function-call forms like list_files(path=".") are also accepted.
For multi-line write_file content, use either JSON-escaped newlines or triple-quoted content.
Use relative paths only. Hidden tests are not visible during the loop."""

_ALLOWED_TOOLS = {"list_files", "read_file", "write_file", "run_tests", "final"}

_AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a relative directory in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path. Use . for the workspace root.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from a relative workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to read."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace a relative workspace file with complete UTF-8 contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to write."},
                    "content": {"type": "string", "description": "Complete replacement file contents."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the visible test command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final",
            "description": "Finish the loop when the workspace is ready for hidden tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Short summary of the completed fix."},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]


def run_agent_loop(
    *,
    client,
    model_id: str,
    task: dict,
    backend_name: str,
    bench_config: dict,
    use_responses_api: bool = False,
    responses_params: dict | None = None,
    chat_params: dict | None = None,
) -> dict:
    """Run an observed tool-use loop against a copied fixture repo."""
    scoring = task.get("scoring", {})
    fixture_raw = scoring.get("repo_fixture")
    if not fixture_raw:
        return _result(task, backend_name, 0.0, "agent_loop: no repo_fixture configured")

    fixture = _resolve_fixture_path(str(fixture_raw))
    if fixture is None or not fixture.exists() or not fixture.is_dir():
        return _result(task, backend_name, 0.0, f"agent_loop: fixture not found: {fixture_raw}")

    command = scoring.get("test_command")
    if not command:
        return _result(task, backend_name, 0.0, "agent_loop: no test_command configured")
    ok, detail = _validate_test_command(command)
    if not ok:
        return _result(task, backend_name, 0.0, detail.replace("repo_patch:", "agent_loop:", 1))

    t_start = time.perf_counter()
    completion_tokens = 0
    reasoning_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    api_cost = 0.0
    trace = {
        "surface": task.get("execution_surface", "observed_agent_loop"),
        "scoring_type": "agent_loop",
        "events": [{"event": "request_start", "elapsed_ms": 0.0}],
        "tool_calls": [],
    }

    with tempfile.TemporaryDirectory(prefix="llm-bench-agent-") as tmp:
        workspace = Path(tmp) / "workspace"
        shutil.copytree(
            fixture,
            workspace,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"),
        )

        messages = _initial_messages(task)
        max_steps = int(scoring.get("max_steps", 8))
        final_summary = ""

        for step in range(1, max_steps + 1):
            try:
                content, usage_meta, native_tool_context = _call_model(
                    client,
                    model_id,
                    messages,
                    task,
                    scoring,
                    bench_config,
                    use_responses_api=use_responses_api,
                    responses_params=responses_params or {},
                    chat_params=chat_params or {},
                )
            except Exception as exc:
                return _finish_result(
                    task, backend_name, trace, t_start, completion_tokens,
                    reasoning_tokens, prompt_tokens, total_tokens, api_cost,
                    0.0, f"agent_loop API error: {exc}", final_summary,
                    termination="api_error",
                )
            completion_tokens += usage_meta["completion_tokens"]
            reasoning_tokens += usage_meta["reasoning_tokens"]
            prompt_tokens += usage_meta["prompt_tokens"]
            total_tokens += usage_meta["total_tokens"]
            api_cost += usage_meta["api_cost"] or 0.0

            action = _extract_action(content)
            if action is None:
                preview = _truncate(content, 1000)
                trace["events"].append({
                    "event": "invalid_action",
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "preview": preview,
                })
                return _finish_result(
                    task, backend_name, trace, t_start, completion_tokens,
                    reasoning_tokens, prompt_tokens, total_tokens, api_cost,
                    0.0, f"agent_loop: invalid action JSON: {preview}", preview,
                    termination="invalid_action",
                )

            tool = str(action.get("tool", "")).strip()
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            ok, observation, should_stop = _execute_tool(tool, args, workspace, command, float(scoring.get("timeout", 60)))
            trace["tool_calls"].append({
                "step": step,
                "tool": tool or "?",
                "args": _safe_args_for_trace(args),
                "ok": ok,
                "observation": _truncate(observation, 1000),
            })

            if tool == "final":
                final_summary = str(args.get("summary") or "").strip()
                score, final_detail = _score_final_workspace(workspace, scoring, command)
                return _finish_result(
                    task, backend_name, trace, t_start, completion_tokens,
                    reasoning_tokens, prompt_tokens, total_tokens, api_cost,
                    score, final_detail, final_summary,
                    termination="final",
                )

            if native_tool_context is not None:
                if native_tool_context.get("api") == "responses":
                    messages.extend(native_tool_context["response_items"])
                    messages.append({
                        "type": "function_call_output",
                        "call_id": native_tool_context["tool_call_id"],
                        "output": observation,
                    })
                else:
                    messages.append(native_tool_context["assistant_message"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": native_tool_context["tool_call_id"],
                        "content": observation,
                    })
            else:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}\n\nReturn the next JSON action."})

            if should_stop:
                return _finish_result(
                    task, backend_name, trace, t_start, completion_tokens,
                    reasoning_tokens, prompt_tokens, total_tokens, api_cost,
                    0.0, observation, final_summary,
                    termination="stopped",
                )

        return _finish_result(
            task, backend_name, trace, t_start, completion_tokens,
            reasoning_tokens, prompt_tokens, total_tokens, api_cost,
            0.0, f"agent_loop: max steps ({max_steps}) reached without final", final_summary,
            termination="max_steps",
        )


def disabled_agent_loop_result(task: dict, backend_name: str) -> dict:
    return _result(
        task,
        backend_name,
        0.0,
        "agent_loop disabled — rerun with --allow-code-exec",
        trace={"surface": task.get("execution_surface", "observed_agent_loop"), "scoring_type": "agent_loop", "events": [], "tool_calls": []},
    )


def _initial_messages(task: dict) -> list[dict]:
    messages = [{"role": "system", "content": task.get("system") or _AGENT_SYSTEM}]
    if task.get("system"):
        messages.append({"role": "system", "content": _AGENT_SYSTEM})
    messages.append({
        "role": "user",
        "content": (
            f"{task['prompt']}\n\n"
            "Inspect and edit the repository using JSON tool actions. "
            "Call run_tests before final when useful. Call final only when ready."
        ),
    })
    return messages


def _call_model(
    client,
    model_id: str,
    messages: list[dict],
    task: dict,
    scoring: dict,
    bench_config: dict,
    *,
    use_responses_api: bool = False,
    responses_params: dict | None = None,
    chat_params: dict | None = None,
) -> tuple[str, dict, dict | None]:
    if use_responses_api:
        response = client.responses.create(
            **_responses_kwargs(
                model_id,
                messages,
                task,
                scoring,
                bench_config,
                responses_params or {},
            )
        )
        native_tool_context = _response_native_tool_context(response)
        content = _first_nonblank(
            response_output_text(response),
            native_tool_context["action_text"] if native_tool_context else None,
        )
        return content, response_usage_metadata(response), native_tool_context

    response = client.chat.completions.create(
        **_chat_completion_kwargs(
            model_id,
            messages,
            task,
            scoring,
            bench_config,
            chat_params or {},
        )
    )
    message = response.choices[0].message
    native_tool_context = _message_native_tool_context(message)
    content = _first_nonblank(
        message.content,
        native_tool_context["action_text"] if native_tool_context else None,
        getattr(message, "reasoning_content", None),
        getattr(message, "thinking", None),
        getattr(message, "reasoning", None),
    )
    usage = getattr(response, "usage", None)
    return content, usage_metadata(usage), native_tool_context


def _chat_completion_kwargs(
    model_id: str,
    messages: list[dict],
    task: dict,
    scoring: dict,
    bench_config: dict,
    chat_params: dict,
) -> dict:
    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": task.get("temperature", bench_config.get("temperature", 0.0)),
        "max_tokens": int(
            scoring.get("action_max_tokens", min(bench_config.get("max_tokens", 4096), 2048))
        ),
        "stream": False,
        "timeout": bench_config.get("timeout", 180),
    }
    kwargs.update(chat_params)
    if _native_tools_enabled(task, scoring, bench_config):
        kwargs["tools"] = _agent_tool_schemas()
        kwargs["tool_choice"] = "auto"
    return kwargs


def _native_tools_enabled(task: dict, scoring: dict, bench_config: dict) -> bool:
    if "native_tools" in scoring:
        return bool(scoring["native_tools"])
    if "agent_loop_native_tools" in task:
        return bool(task["agent_loop_native_tools"])
    return bool(bench_config.get("agent_loop_native_tools", False))


def _agent_tool_schemas() -> list[dict]:
    return deepcopy(_AGENT_TOOL_SCHEMAS)


def _responses_kwargs(
    model_id: str,
    messages: list[dict],
    task: dict,
    scoring: dict,
    bench_config: dict,
    responses_params: dict,
) -> dict:
    kwargs = {
        "model": model_id,
        "input": messages_to_responses_input(messages),
        "timeout": bench_config.get("timeout", 180),
    }
    kwargs.update(responses_params)
    if _native_tools_enabled(task, scoring, bench_config):
        kwargs["tools"] = _responses_tool_schemas()
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = False
    return kwargs


def _responses_tool_schemas() -> list[dict]:
    tools = []
    for schema in _AGENT_TOOL_SCHEMAS:
        function = schema["function"]
        tools.append({
            "type": "function",
            "name": function["name"],
            "description": function.get("description"),
            "parameters": deepcopy(function.get("parameters") or {}),
            "strict": False,
        })
    return tools


def _first_nonblank(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def _extract_action(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    candidates = [stripped]
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char == "{":
            candidates.append(stripped[idx:])
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("tool"), str):
            return value
        if isinstance(value, dict) and "summary" in value and "tool" not in value:
            return {"tool": "final", "args": {"summary": str(value.get("summary") or "")}}
    triple_quoted = _extract_triple_quoted_write_file_action(text)
    if triple_quoted is not None:
        return triple_quoted
    tool_colon = _extract_tool_colon_json_action(text)
    if tool_colon is not None:
        return tool_colon
    kimi_tool_call = _extract_kimi_tool_call_section_action(text)
    if kimi_tool_call is not None:
        return kimi_tool_call
    function_call = _extract_function_call_action(text)
    if function_call is not None:
        return function_call
    lenient_write = _extract_lenient_write_file_function_action(text)
    if lenient_write is not None:
        return lenient_write
    return _extract_tool_call_tag_action(text)


def _message_tool_call_action_text(message) -> str | None:
    context = _message_native_tool_context(message)
    return context["action_text"] if context else None


def _response_native_tool_context(response) -> dict | None:
    for item in getattr(response, "output", None) or []:
        if _get_attr_or_key(item, "type") != "function_call":
            continue
        raw_name = _get_attr_or_key(item, "name")
        if not raw_name:
            continue
        tool = str(raw_name).rsplit(".", 1)[-1]
        if tool not in _ALLOWED_TOOLS:
            continue
        raw_args = _get_attr_or_key(item, "arguments") or "{}"
        try:
            args = json.loads(str(raw_args))
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = _get_attr_or_key(item, "call_id") or _get_attr_or_key(item, "id") or f"call_{tool}"
        response_item = {
            "type": "function_call",
            "call_id": str(call_id),
            "name": str(raw_name),
            "arguments": str(raw_args),
        }
        item_id = _get_attr_or_key(item, "id")
        if item_id:
            response_item["id"] = str(item_id)
        return {
            "api": "responses",
            "action_text": json.dumps({"tool": tool, "args": args}),
            "tool_call_id": str(call_id),
            "response_items": [response_item],
        }
    return None


def _message_native_tool_context(message) -> dict | None:
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        function = _get_attr_or_key(call, "function")
        if function is None:
            continue
        raw_name = _get_attr_or_key(function, "name")
        if not raw_name:
            continue
        tool = str(raw_name).rsplit(".", 1)[-1]
        if tool not in _ALLOWED_TOOLS:
            continue
        raw_args = _get_attr_or_key(function, "arguments") or "{}"
        try:
            args = json.loads(str(raw_args))
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        action_text = json.dumps({"tool": tool, "args": args})
        call_id = _get_attr_or_key(call, "id") or f"call_{tool}"
        call_type = _get_attr_or_key(call, "type") or "function"
        return {
            "action_text": action_text,
            "tool_call_id": str(call_id),
            "assistant_message": {
                "role": "assistant",
                "content": _message_content_for_protocol(message),
                "tool_calls": [
                    {
                        "id": str(call_id),
                        "type": str(call_type),
                        "function": {
                            "name": str(raw_name),
                            "arguments": str(raw_args),
                        },
                    }
                ],
            },
        }
    return None


def _message_content_for_protocol(message) -> str | None:
    content = getattr(message, "content", None)
    if content is None:
        return None
    return str(content)


def _get_attr_or_key(value, key: str):
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _extract_triple_quoted_write_file_action(text: str) -> dict | None:
    match = re.search(
        r'"tool"\s*:\s*"write_file".*?'
        r'"path"\s*:\s*"(?P<path>[^"]+)".*?'
        r'"content"\s*:\s*(?P<quote>"""|\'\'\')(?P<content>.*?)(?P=quote)',
        text,
        re.DOTALL,
    )
    if not match:
        return None
    return {
        "tool": "write_file",
        "args": {
            "path": match.group("path"),
            "content": match.group("content"),
        },
    }


def _extract_tool_call_tag_action(text: str) -> dict | None:
    match = re.search(
        r"<\|tool_call\|>\s*call:(?:tool:)?(?P<tool>[A-Za-z_]\w*)\{(?P<body>.*?)\}<tool_call\|>",
        text,
        re.DOTALL,
    )
    if not match:
        match = re.search(
            r"call:(?:tool:)?(?P<tool>[A-Za-z_]\w*)\{(?P<body>.*?)\}",
            text,
            re.DOTALL,
        )
    if not match:
        return None
    args: dict[str, str] = {}
    body = match.group("body")
    for item in re.finditer(r"(?P<key>[A-Za-z_]\w*)\s*:\s*(?P<value>\"[^\"]*\"|[^,}]+)", body):
        value = item.group("value").strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        args[item.group("key")] = value
    tool = match.group("tool")
    if tool not in _ALLOWED_TOOLS:
        return None
    return {"tool": tool, "args": args}


def _extract_kimi_tool_call_section_action(text: str) -> dict | None:
    match = re.search(
        r"<\|tool_call_begin\|>"
        r"(?:(?:functions|tools?)\.)?(?P<tool>[A-Za-z_]\w*)"
        r"(?::\d+)?"
        r"<\|tool_call_argument_begin\|>"
        r"(?P<body>.*?)"
        r"<\|tool_call_end\|>",
        text,
        re.DOTALL,
    )
    if not match:
        return None
    tool = match.group("tool")
    if tool not in _ALLOWED_TOOLS:
        return None
    args = _decode_json_object_at(match.group("body").strip(), 0) or {}
    return {"tool": tool, "args": args}


def _extract_tool_colon_json_action(text: str) -> dict | None:
    tool_pattern = "|".join(re.escape(tool) for tool in sorted(_ALLOWED_TOOLS, key=len, reverse=True))
    for match in re.finditer(rf"\btool\s*:\s*(?P<tool>{tool_pattern})\s*(?P<body>\{{)", text):
        args = _decode_json_object_at(text, match.start("body"))
        if isinstance(args, dict):
            return {"tool": match.group("tool"), "args": args}
    return None


def _extract_function_call_action(text: str) -> dict | None:
    stripped = text.strip()
    tool_pattern = "|".join(re.escape(tool) for tool in sorted(_ALLOWED_TOOLS, key=len, reverse=True))
    for match in re.finditer(rf"(?<!\w)(?:(?:[A-Za-z_]\w*)\.)?(?P<tool>{tool_pattern})\s*\(", stripped):
        tool = match.group("tool")
        open_paren = stripped.find("(", match.start())
        close_paren = _find_matching_paren(stripped, open_paren)
        if close_paren is None:
            continue
        body = stripped[open_paren + 1:close_paren]
        args = _parse_function_call_args(body)
        if args is not None:
            return {"tool": tool, "args": args}
    return None


def _parse_function_call_args(body: str) -> dict[str, str] | None:
    stripped = body.strip()
    if stripped.startswith("{"):
        value = _decode_json_object_at(stripped, 0)
        return value if isinstance(value, dict) else None
    return _parse_keyword_args(body)


def _decode_json_object_at(text: str, start: int) -> dict | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_lenient_write_file_function_action(text: str) -> dict | None:
    """Parse common write_file(...) calls whose content string contains raw quotes."""
    stripped = text.strip()
    for match in re.finditer(r"(?<![\w.])write_file\s*\(", stripped):
        body_start = match.end()
        content_match = re.search(r"\bcontent\s*=\s*(?P<quote>['\"])", stripped[body_start:], re.DOTALL)
        if not content_match:
            continue

        content_key_start = body_start + content_match.start()
        content_start = body_start + content_match.end()
        quote = content_match.group("quote")
        prefix = stripped[body_start:content_key_start]
        path = _extract_quoted_keyword(prefix, "path")
        if path is None:
            continue

        close_quote = _find_last_function_content_quote(stripped, content_start, quote)
        if close_quote is None:
            continue
        content = _decode_common_string_escapes(stripped[content_start:close_quote])
        return {"tool": "write_file", "args": {"path": path, "content": content}}
    return None


def _extract_quoted_keyword(text: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)", text, re.DOTALL)
    if not match:
        return None
    return _decode_common_string_escapes(match.group("value"))


def _find_last_function_content_quote(text: str, start: int, quote: str) -> int | None:
    close_quote: int | None = None
    idx = start
    while idx < len(text):
        if text[idx] == "\\":
            idx += 2
            continue
        if text[idx] == quote:
            after = idx + 1
            while after < len(text) and text[after].isspace():
                after += 1
            if after < len(text) and text[after] == ")":
                close_quote = idx
        idx += 1
    return close_quote


def _find_matching_paren(text: str, open_paren: int) -> int | None:
    idx = open_paren + 1
    depth = 1
    quote: str | None = None
    triple_quote: str | None = None
    while idx < len(text):
        if triple_quote:
            if text.startswith(triple_quote, idx):
                idx += len(triple_quote)
                triple_quote = None
                continue
            idx += 1
            continue
        if quote:
            if text[idx] == "\\":
                idx += 2
                continue
            if text[idx] == quote:
                quote = None
            idx += 1
            continue
        if text.startswith('"""', idx) or text.startswith("'''", idx):
            triple_quote = text[idx:idx + 3]
            idx += 3
            continue
        if text[idx] in {'"', "'"}:
            quote = text[idx]
            idx += 1
            continue
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return None


def _parse_keyword_args(body: str) -> dict[str, str] | None:
    args: dict[str, str] = {}
    idx = 0
    length = len(body)
    while idx < length:
        while idx < length and body[idx] in " \t\r\n,":
            idx += 1
        if idx >= length:
            break
        key_match = re.match(
            r"\"(?P<double>[A-Za-z_]\w*)\"|'(?P<single>[A-Za-z_]\w*)'|(?P<bare>[A-Za-z_]\w*)",
            body[idx:],
        )
        if not key_match:
            return None
        key = key_match.group("double") or key_match.group("single") or key_match.group("bare")
        idx += len(key_match.group(0))
        while idx < length and body[idx].isspace():
            idx += 1
        if idx >= length or body[idx] not in {"=", ":"}:
            return None
        idx += 1
        while idx < length and body[idx].isspace():
            idx += 1
        value, idx = _parse_arg_value(body, idx)
        if value is None:
            return None
        args[key] = value
        while idx < length and body[idx].isspace():
            idx += 1
        if idx < length and body[idx] == ",":
            idx += 1
    return args


def _parse_arg_value(body: str, idx: int) -> tuple[str | None, int]:
    for quote in ('"""', "'''"):
        if body.startswith(quote, idx):
            end = body.find(quote, idx + len(quote))
            if end == -1:
                return None, idx
            return body[idx + len(quote):end], end + len(quote)
    if idx < len(body) and body[idx] in {'"', "'"}:
        quote = body[idx]
        idx += 1
        chars = []
        while idx < len(body):
            char = body[idx]
            if char == "\\" and idx + 1 < len(body):
                chars.append(_decode_common_escape(body[idx + 1]))
                idx += 2
                continue
            if char == quote:
                return "".join(chars), idx + 1
            chars.append(char)
            idx += 1
        return None, idx
    start = idx
    while idx < len(body) and body[idx] != ",":
        idx += 1
    return body[start:idx].strip(), idx


def _decode_common_string_escapes(value: str) -> str:
    chars: list[str] = []
    idx = 0
    while idx < len(value):
        if value[idx] == "\\" and idx + 1 < len(value):
            chars.append(_decode_common_escape(value[idx + 1]))
            idx += 2
            continue
        chars.append(value[idx])
        idx += 1
    return "".join(chars)


def _decode_common_escape(char: str) -> str:
    return {
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }.get(char, char)


def _execute_tool(tool: str, args: dict, workspace: Path, command, timeout_s: float) -> tuple[bool, str, bool]:
    if tool == "list_files":
        return _tool_list_files(workspace, str(args.get("path") or "."))
    if tool == "read_file":
        return _tool_read_file(workspace, str(args.get("path") or ""))
    if tool == "write_file":
        return _tool_write_file(workspace, str(args.get("path") or ""), str(args.get("content") or ""))
    if tool == "run_tests":
        proc = _run_test_command(command, workspace, timeout_s)
        if proc.returncode == _TIMEOUT_ERROR:
            return False, f"agent_loop: visible tests timed out after {timeout_s:g}s", False
        if proc.returncode == _LAUNCH_ERROR:
            return False, f"agent_loop: visible tests failed to launch: {_truncate(proc.stderr or proc.stdout, 500)}", False
        output = _truncate((proc.stdout or "") + (proc.stderr or ""), 1000)
        return proc.returncode == 0, f"exit {proc.returncode}: {output}", False
    if tool == "final":
        return True, "agent_loop: final received", True
    return False, f"agent_loop: unknown tool: {tool}", True


def _tool_list_files(workspace: Path, rel: str) -> tuple[bool, str, bool]:
    target = _safe_target(workspace, rel)
    if target is None or not target.exists() or not target.is_dir():
        return False, f"agent_loop: invalid directory: {rel}", False
    paths = []
    for path in sorted(target.rglob("*")):
        if any(part in {"__pycache__", ".pytest_cache", ".llm_bench_sentinel"} for part in path.parts):
            continue
        if path.is_file():
            paths.append(str(path.relative_to(workspace)).replace("\\", "/"))
        if len(paths) >= 200:
            paths.append("...truncated...")
            break
    return True, "\n".join(paths), False


def _tool_read_file(workspace: Path, rel: str) -> tuple[bool, str, bool]:
    target = _safe_target(workspace, rel)
    if target is None or not target.exists() or not target.is_file():
        return False, f"agent_loop: invalid file: {rel}", False
    return True, target.read_text(encoding="utf-8")[:20000], False


def _tool_write_file(workspace: Path, rel: str, content: str) -> tuple[bool, str, bool]:
    target = _safe_target(workspace, rel)
    if target is None:
        return False, f"agent_loop: unsafe path rejected: {rel}", False
    if _is_protected_harness_path(rel):
        return False, f"agent_loop: protected path rejected: {rel}", False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return True, f"agent_loop: wrote {rel}", False


def _score_final_workspace(workspace: Path, scoring: dict, command) -> tuple[float, str]:
    sentinel = _make_sentinel_hidden_test()
    hidden_tests = list(scoring.get("hidden_tests", [])) + [sentinel["test"]]
    ok, detail = _write_hidden_tests(workspace, hidden_tests)
    if not ok:
        return 0.0, detail.replace("repo_patch:", "agent_loop:", 1)

    timeout_s = float(scoring.get("timeout", 60))
    proc = _run_test_command(command, workspace, timeout_s)
    if proc.returncode == _TIMEOUT_ERROR:
        return 0.0, f"agent_loop: tests timed out after {timeout_s:g}s"
    if proc.returncode == _LAUNCH_ERROR:
        return 0.0, f"agent_loop: test command failed to launch: {_truncate(proc.stderr or proc.stdout, 500)}"
    if proc.returncode == 0:
        if not _sentinel_completed(workspace, sentinel):
            return 0.0, "agent_loop: test sentinel did not run; rejecting zero-exit result"
        return 1.0, "agent_loop: tests passed"
    return 0.0, (
        f"agent_loop: tests failed (exit {proc.returncode}): "
        f"{_truncate((proc.stdout or '') + (proc.stderr or ''), 500)}"
    )


def _safe_args_for_trace(args: dict) -> dict:
    safe = dict(args)
    if "content" in safe:
        safe["content"] = f"<{len(str(safe['content']))} chars>"
    return safe


def _finish_result(
    task: dict,
    backend_name: str,
    trace: dict,
    t_start: float,
    completion_tokens: int,
    reasoning_tokens: int,
    prompt_tokens: int,
    total_tokens: int,
    api_cost: float,
    score: float,
    detail: str,
    summary: str,
    *,
    termination: str,
) -> dict:
    total_ms = (time.perf_counter() - t_start) * 1000
    progress = _agent_loop_progress(trace, score, termination)
    trace["progress"] = progress
    trace["events"].append({
        "event": "agent_loop_complete",
        "elapsed_ms": round(total_ms, 1),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "api_cost": round(api_cost, 8) if api_cost else None,
        "score": score,
        "termination": termination,
        "progress_score": progress["score"],
    })
    return _result(
        task,
        backend_name,
        score,
        detail,
        response=summary,
        total_ms=round(total_ms, 1),
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        api_cost=round(api_cost, 8) if api_cost else None,
        trace=trace,
        progress=progress,
    )


def _agent_loop_progress(trace: dict, hidden_score: float, termination: str) -> dict:
    calls = trace.get("tool_calls") or []
    checks = {
        "valid_action": bool(calls),
        "inspected_workspace": any(
            call.get("ok") is True and call.get("tool") in {"list_files", "read_file"}
            for call in calls
        ),
        "wrote_file": any(
            call.get("ok") is True and call.get("tool") == "write_file"
            for call in calls
        ),
        "ran_visible_tests": any(call.get("tool") == "run_tests" for call in calls),
        "visible_tests_passed": any(
            call.get("ok") is True and call.get("tool") == "run_tests"
            for call in calls
        ),
        "final_called": any(call.get("tool") == "final" for call in calls),
        "hidden_tests_passed": hidden_score >= 1.0,
    }
    passed = sum(1 for ok in checks.values() if ok)
    total = len(checks)
    return {
        "score": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "termination": termination,
        "checks": checks,
    }


def _result(
    task: dict,
    backend_name: str,
    score: float,
    detail: str,
    response: str = "",
    total_ms: float | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    total_tokens: int = 0,
    api_cost: float | None = None,
    trace: dict | None = None,
    progress: dict | None = None,
) -> dict:
    progress = progress or (trace or {}).get("progress") or {}
    return {
        "task_id": task["id"],
        "response": response,
        "error": None,
        "ttft_ms": None,
        "total_ms": total_ms,
        "tps": round(completion_tokens / (total_ms / 1000), 1) if completion_tokens and total_ms else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "api_cost": api_cost,
        "backend": backend_name,
        "agent_loop_score": score,
        "agent_loop_detail": detail,
        "agent_loop_progress_score": progress.get("score"),
        "agent_loop_progress_passed": progress.get("passed"),
        "agent_loop_progress_total": progress.get("total"),
        "agent_loop_termination": progress.get("termination"),
        "execution_trace": trace or {
            "surface": task.get("execution_surface", "observed_agent_loop"),
            "scoring_type": "agent_loop",
            "events": [],
            "tool_calls": [],
        },
    }
