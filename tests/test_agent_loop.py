import json
from copy import deepcopy
from types import SimpleNamespace

from benchmark.agent_loop import run_agent_loop


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("fake client has no response left")
        response_item = self.responses.pop(0)
        if isinstance(response_item, dict):
            content = response_item.get("content")
            reasoning_content = response_item.get("reasoning_content")
            thinking = response_item.get("thinking")
            reasoning = response_item.get("reasoning")
            tool_calls = response_item.get("tool_calls")
            reasoning_tokens = response_item.get("reasoning_tokens")
            prompt_tokens = response_item.get("prompt_tokens")
            total_tokens = response_item.get("total_tokens")
            cost = response_item.get("cost")
        else:
            content = response_item
            reasoning_content = None
            thinking = None
            reasoning = None
            tool_calls = None
            reasoning_tokens = None
            prompt_tokens = None
            total_tokens = None
            cost = None
        usage_text = content or reasoning_content or thinking or reasoning or ""
        usage = SimpleNamespace(
            completion_tokens=len(usage_text.split()),
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
        if reasoning_tokens is not None:
            usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
        message = SimpleNamespace(content=content)
        if reasoning_content is not None:
            message.reasoning_content = reasoning_content
        if thinking is not None:
            message.thinking = thinking
        if reasoning is not None:
            message.reasoning = reasoning
        if tool_calls is not None:
            message.tool_calls = [
                SimpleNamespace(
                    id=call.get("id", "call_0"),
                    type="function",
                    function=SimpleNamespace(name=call["name"], arguments=call["arguments"]),
                )
                for call in tool_calls
            ]
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeResponsesClient:
    def __init__(self, responses):
        self.responses_list = list(responses)
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if not self.responses_list:
            raise AssertionError("fake responses client has no response left")
        output_text = self.responses_list.pop(0)
        usage = SimpleNamespace(output_tokens=len(output_text.split()))
        return SimpleNamespace(output_text=output_text, output=[], usage=usage, status="completed")


def _task(tmp_path, **scoring_updates):
    fixture = tmp_path / "fixture"
    package = fixture / "calc"
    tests = fixture / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "stats.py").write_text(
        "def mean(values):\n"
        "    return sum(values) // len(values)\n",
        encoding="utf-8",
    )
    (tests / "test_stats.py").write_text(
        "from calc.stats import mean\n\n"
        "def test_mean_integer_values():\n"
        "    assert mean([2, 4, 6]) == 4\n",
        encoding="utf-8",
    )
    scoring = {
        "type": "agent_loop",
        "repo_fixture": str(fixture),
        "test_command": ["{python}", "-m", "pytest", "-q"],
        "max_steps": 6,
        "hidden_tests": [
            {
                "path": "tests/test_hidden_stats.py",
                "content": (
                    "from calc.stats import mean\n\n"
                    "def test_mean_float_result():\n"
                    "    assert mean([1, 2]) == 1.5\n"
                ),
            }
        ],
    }
    scoring.update(scoring_updates)
    return {
        "id": "agent_loop_mean",
        "prompt": "Fix calc.stats.mean so it returns the arithmetic mean.",
        "category": "agent_loop",
        "execution_surface": "observed_agent_loop",
        "scoring": scoring,
    }


def _run(client, task):
    return run_agent_loop(
        client=client,
        model_id="fake-model",
        task=task,
        backend_name="fake",
        bench_config={"temperature": 0.0, "timeout": 30},
    )


def test_agent_loop_executes_tools_and_scores_hidden_tests(tmp_path):
    task = _task(tmp_path)
    client = FakeClient([
        '{"tool": "read_file", "args": {"path": "calc/stats.py"}}',
        (
            '{"tool": "write_file", "args": {"path": "calc/stats.py", '
            '"content": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'
        ),
        '{"tool": "run_tests", "args": {}}',
        '{"tool": "final", "args": {"summary": "fixed mean"}}',
    ])

    result = _run(client, task)

    assert result["error"] is None
    assert result["agent_loop_score"] == 1.0
    assert result["response"] == "fixed mean"
    assert "tests passed" in result["agent_loop_detail"]
    trace = result["execution_trace"]
    assert trace["surface"] == "observed_agent_loop"
    assert [event["tool"] for event in trace["tool_calls"]] == [
        "read_file",
        "write_file",
        "run_tests",
        "final",
    ]
    assert "tools" not in client.requests[0]
    assert "def mean" in client.requests[1]["messages"][-1]["content"]


def test_agent_loop_rejects_malformed_action(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(["not json"])

    result = _run(client, task)

    assert result["agent_loop_score"] == 0.0
    assert "invalid action" in result["agent_loop_detail"].lower()
    assert result["response"] == "not json"
    assert result["execution_trace"]["events"][-2]["event"] == "invalid_action"


def test_agent_loop_reports_max_steps_without_final(tmp_path):
    task = _task(tmp_path, max_steps=1)
    client = FakeClient(['{"tool": "read_file", "args": {"path": "calc/stats.py"}}'])

    result = _run(client, task)

    assert result["agent_loop_score"] == 0.0
    assert "max steps" in result["agent_loop_detail"].lower()


def test_agent_loop_rejects_unsafe_paths(tmp_path):
    task = _task(tmp_path, max_steps=2)
    outside = tmp_path / "outside.py"
    client = FakeClient([
        '{"tool": "write_file", "args": {"path": "../outside.py", "content": "bad"}}',
        '{"tool": "final", "args": {"summary": "done"}}',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 0.0
    assert not outside.exists()
    assert any(
        event["tool"] == "write_file" and event["ok"] is False
        for event in result["execution_trace"]["tool_calls"]
    )


def test_agent_loop_rejects_hidden_test_collection_sabotage(tmp_path):
    task = _task(tmp_path, max_steps=2)
    conftest = (
        "def pytest_ignore_collect(collection_path, config):\n"
        "    return collection_path.name.startswith('test_hidden')\n"
    )
    client = FakeClient([
        '{"tool": "write_file", "args": {"path": "conftest.py", "content": ' + json.dumps(conftest) + '}}',
        '{"tool": "final", "args": {"summary": "tests pass"}}',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 0.0
    assert any(
        event["tool"] == "write_file" and event["ok"] is False and "protected" in event["observation"]
        for event in result["execution_trace"]["tool_calls"]
    )


def test_agent_loop_hides_hidden_tests_until_final(tmp_path):
    task = _task(tmp_path)
    client = FakeClient([
        '{"tool": "list_files", "args": {"path": "tests"}}',
        '{"tool": "final", "args": {"summary": "no change"}}',
    ])

    result = _run(client, task)

    first_observation = client.requests[1]["messages"][-1]["content"]
    assert "test_hidden_stats.py" not in first_observation
    assert result["agent_loop_score"] == 0.0
    assert "tests failed" in result["agent_loop_detail"]


def test_agent_loop_accepts_common_tool_call_tag_syntax(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        '<|tool_call|>call:tool:list_files{path:"."}<tool_call|>',
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["ok"] is True
    assert result["agent_loop_score"] == 0.0


def test_agent_loop_reads_nonstream_reasoning_action_when_content_is_empty(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {"content": "", "reasoning_content": '{"tool": "list_files", "args": {"path": "."}}'},
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["ok"] is True


def test_agent_loop_reads_nonstream_reasoning_field_action_when_content_is_empty(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {"content": "", "reasoning": '{"tool": "list_files", "args": {"path": "."}}'},
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["ok"] is True


def test_agent_loop_reads_native_tool_call_when_content_is_empty(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {
            "content": "   ",
            "reasoning": "Let me inspect first.",
            "tool_calls": [{"name": "list_files", "arguments": '{"path": "."}'}],
        },
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["args"] == {"path": "."}


def test_agent_loop_sends_native_tool_schemas_when_enabled(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {
            "content": "   ",
            "tool_calls": [{"name": "list_files", "arguments": '{"path": "."}'}],
        },
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = run_agent_loop(
        client=client,
        model_id="fake-model",
        task=task,
        backend_name="fake",
        bench_config={"temperature": 0.0, "timeout": 30, "agent_loop_native_tools": True},
    )

    request = client.requests[0]
    tool_names = {tool["function"]["name"] for tool in request["tools"]}
    assert tool_names == {"list_files", "read_file", "write_file", "run_tests", "final"}
    assert request["tool_choice"] == "auto"
    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"


def test_agent_loop_feeds_native_tool_observation_with_tool_role(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {
            "content": "   ",
            "tool_calls": [{"id": "call_list", "name": "list_files", "arguments": '{"path": "."}'}],
        },
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    run_agent_loop(
        client=client,
        model_id="fake-model",
        task=task,
        backend_name="fake",
        bench_config={"temperature": 0.0, "timeout": 30, "agent_loop_native_tools": True},
    )

    second_messages = client.requests[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["id"] == "call_list"
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call_list"
    assert "calc/stats.py" in second_messages[-1]["content"]


def test_agent_loop_counts_nonstream_reasoning_tokens(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {
            "content": '{"tool": "list_files", "args": {"path": "."}}',
            "reasoning_tokens": 7,
        },
        {
            "content": '{"tool": "final", "args": {"summary": "inspected only"}}',
            "reasoning_tokens": 5,
        },
    ])

    result = _run(client, task)

    assert result["reasoning_tokens"] == 12


def test_agent_loop_aggregates_nonstream_usage_metadata(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        {
            "content": '{"tool": "list_files", "args": {"path": "."}}',
            "prompt_tokens": 10,
            "total_tokens": 14,
            "cost": 0.001,
        },
        {
            "content": '{"tool": "final", "args": {"summary": "inspected only"}}',
            "prompt_tokens": 12,
            "total_tokens": 16,
            "cost": 0.002,
        },
    ])

    result = _run(client, task)

    assert result["prompt_tokens"] == 22
    assert result["total_tokens"] == 30
    assert result["api_cost"] == 0.003


def test_agent_loop_accepts_kimi_tool_calls_section_syntax(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.list_files:0<|tool_call_argument_begin|>{"path":"."}<|tool_call_end|><|tool_calls_section_end|>',
        '{"tool": "final", "args": {"summary": "inspected only"}}',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["args"] == {"path": "."}


def test_agent_loop_accepts_triple_quoted_write_file_content(tmp_path):
    task = _task(tmp_path, max_steps=3)
    client = FakeClient([
        '''{"tool": "write_file", "args": {"path": "calc/stats.py", "content": """def mean(values):
    return sum(values) / len(values)
"""}}''',
        '{"tool": "run_tests", "args": {}}',
        '{"tool": "final", "args": {"summary": "fixed mean"}}',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0
    assert result["execution_trace"]["tool_calls"][0]["tool"] == "write_file"
    assert result["execution_trace"]["tool_calls"][0]["ok"] is True


def test_agent_loop_accepts_function_call_tool_syntax(tmp_path):
    task = _task(tmp_path, max_steps=4)
    client = FakeClient([
        'list_files(path=".")',
        '''write_file(path="calc/stats.py", content="""def mean(values):
    return sum(values) / len(values)
""")''',
        "run_tests()",
        'final(summary="fixed mean")',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0
    assert [call["tool"] for call in result["execution_trace"]["tool_calls"]] == [
        "list_files",
        "write_file",
        "run_tests",
        "final",
    ]


def test_agent_loop_accepts_embedded_function_call_action(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        'I will inspect the repo now. list_files(path=".")',
        'final(summary="inspected only")',
    ])

    result = _run(client, task)

    assert result["execution_trace"]["tool_calls"][0]["tool"] == "list_files"
    assert result["execution_trace"]["tool_calls"][0]["ok"] is True


def test_agent_loop_accepts_tool_colon_json_action(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        'I will inspect first. tool:list_files {"path":"."}',
        'final({"summary":"inspected only"})',
    ])

    result = _run(client, task)

    assert [call["tool"] for call in result["execution_trace"]["tool_calls"]] == ["list_files", "final"]
    assert result["execution_trace"]["tool_calls"][0]["args"] == {"path": "."}


def test_agent_loop_accepts_namespaced_function_call_action(tmp_path):
    task = _task(tmp_path, max_steps=2)
    client = FakeClient([
        'files.list_files(path="."){"path":".","depth":2}',
        'final({"summary":"inspected only"})',
    ])

    result = _run(client, task)

    assert [call["tool"] for call in result["execution_trace"]["tool_calls"]] == ["list_files", "final"]
    assert result["execution_trace"]["tool_calls"][0]["args"] == {"path": "."}


def test_agent_loop_accepts_summary_only_json_as_final_action(tmp_path):
    task = _task(tmp_path, max_steps=3)
    client = FakeClient([
        (
            '{"tool": "write_file", "args": {"path": "calc/stats.py", '
            '"content": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'
        ),
        '{"tool": "run_tests", "args": {}}',
        '{"summary": "fixed mean and tests pass"}',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0
    assert result["response"] == "fixed mean and tests pass"
    assert [call["tool"] for call in result["execution_trace"]["tool_calls"]] == [
        "write_file",
        "run_tests",
        "final",
    ]


def test_agent_loop_function_call_decodes_common_string_escapes(tmp_path):
    task = _task(tmp_path, max_steps=3)
    client = FakeClient([
        'write_file(path="calc/stats.py", content="def mean(values):\\n    return sum(values) / len(values)\\n")',
        "run_tests()",
        'final(summary="fixed mean")',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0


def test_agent_loop_accepts_quoted_json_style_function_keyword(tmp_path):
    task = _task(tmp_path, max_steps=3)
    client = FakeClient([
        'write_file(path="calc/stats.py","content":"def mean(values):\\n    return sum(values) / len(values)\\n")',
        "run_tests()",
        'final(summary="fixed mean")',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0


def test_agent_loop_accepts_multiline_single_quoted_function_content(tmp_path):
    task = _task(tmp_path, max_steps=3)
    client = FakeClient([
        '''write_file(path="calc/stats.py", content='def mean(values):
    _ = "same quote marker ' inside content"
    return sum(values) / len(values)
')''',
        "run_tests()",
        'final(summary="fixed mean")',
    ])

    result = _run(client, task)

    assert result["agent_loop_score"] == 1.0


def test_agent_loop_can_use_responses_api_client(tmp_path):
    task = _task(tmp_path)
    client = FakeResponsesClient([
        '{"tool": "read_file", "args": {"path": "calc/stats.py"}}',
        (
            '{"tool": "write_file", "args": {"path": "calc/stats.py", '
            '"content": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'
        ),
        '{"tool": "run_tests", "args": {}}',
        '{"tool": "final", "args": {"summary": "fixed mean"}}',
    ])

    result = run_agent_loop(
        client=client,
        model_id="gpt-5.5",
        task=task,
        backend_name="fake",
        bench_config={"timeout": 30},
        use_responses_api=True,
        responses_params={"reasoning": {"effort": "high"}, "max_output_tokens": 12000},
    )

    assert result["agent_loop_score"] == 1.0
    assert result["completion_tokens"] > 0
    assert client.requests[0]["model"] == "gpt-5.5"
    assert client.requests[0]["reasoning"] == {"effort": "high"}
    assert client.requests[0]["max_output_tokens"] == 12000
    assert client.requests[0]["input"][0]["role"] == "system"
