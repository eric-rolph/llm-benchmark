"""
tests/test_runner_stream.py — _run_once streaming loop via an injected fake client.

ModelRunner accepts `client=` so the TTFT/reasoning/usage handling can be
exercised without a live backend.
"""
from types import SimpleNamespace

from benchmark.runner import ModelRunner


class FakeBackend:
    name = "fake"

    def __init__(self, extra_params=None, responses=False, responses_params=None):
        self._extra = extra_params or {}
        self._responses = responses
        self._responses_params = responses_params or {}

    def get_openai_client(self):
        raise AssertionError("injected client must be used, not backend.get_openai_client()")

    def get_extra_chat_params(self, task):
        return dict(self._extra)

    def use_responses_api(self, model_id, task):
        return self._responses

    def get_responses_params(self, task, bench_config):
        return dict(self._responses_params)


class FakeClient:
    """Mimics openai.OpenAI for streaming chat completions."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self._chunks)


class FakeResponsesClient:
    def __init__(self, output_text="done", output_tokens=5, reasoning_tokens=2, prompt_tokens=3, total_tokens=8, cost=0.0):
        self.kwargs = None
        details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
        usage = SimpleNamespace(
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_tokens_details=details,
            cost=cost,
        )
        self._response = SimpleNamespace(
            output_text=output_text,
            output=[SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="checked")])],
            usage=usage,
            status="completed",
        )
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


class FailingClient:
    def __init__(self, exc):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self._exc = exc

    def _create(self, **kwargs):
        raise self._exc


def content_chunk(text=None, reasoning=None, thinking=None):
    delta = SimpleNamespace(content=text)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    if thinking is not None:
        delta.thinking = thinking
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def usage_chunk(completion_tokens, reasoning_tokens=None, prompt_tokens=None, total_tokens=None, cost=None):
    details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_details=details,
        cost=cost,
    )
    return SimpleNamespace(choices=[], usage=usage)


def _task(**updates):
    task = {"id": "t1", "prompt": "hi", "scoring": {"type": "contains", "value": "x"}}
    task.update(updates)
    return task


def run_once(chunks_or_client, backend=None):
    backend = backend or FakeBackend()
    client = (
        chunks_or_client
        if hasattr(chunks_or_client, "chat")
        else FakeClient(chunks_or_client)
    )
    runner = ModelRunner(backend, "fake-model", {}, client=client)
    return runner._run_once(_task()), client


def test_injected_client_used_and_content_assembled():
    result, _ = run_once([
        content_chunk("Hel"),
        content_chunk("lo"),
        usage_chunk(completion_tokens=2),
    ])
    assert result["error"] is None
    assert result["response"] == "Hello"
    assert result["completion_tokens"] == 2
    assert result["ttft_ms"] is not None
    assert result["total_ms"] is not None
    assert result["tps"] is not None
    assert result["backend"] == "fake"


def test_reasoning_content_captured_separately_from_response():
    result, _ = run_once([
        content_chunk(reasoning="thinking hard..."),
        content_chunk("The answer is 4."),
        usage_chunk(completion_tokens=5, reasoning_tokens=3),
    ])
    assert result["response"] == "The answer is 4."
    assert result["reasoning_preview"].startswith("thinking hard")
    assert result["reasoning_tokens"] == 3


def test_chat_usage_metadata_is_recorded():
    result, _ = run_once([
        content_chunk("ok"),
        usage_chunk(
            completion_tokens=5,
            reasoning_tokens=2,
            prompt_tokens=11,
            total_tokens=16,
            cost=0.00042,
        ),
    ])

    assert result["completion_tokens"] == 5
    assert result["reasoning_tokens"] == 2
    assert result["prompt_tokens"] == 11
    assert result["total_tokens"] == 16
    assert result["api_cost"] == 0.00042


def test_ollama_thinking_field_captured_as_reasoning():
    result, _ = run_once([
        content_chunk(thinking="hmm"),
        content_chunk("42"),
        usage_chunk(completion_tokens=1),
    ])
    assert result["response"] == "42"
    assert result["reasoning_preview"] == "hmm"


def test_think_tags_leaked_into_content_are_stripped():
    result, _ = run_once([
        content_chunk("<think>secret plan</think>"),
        content_chunk("Answer: 42"),
        usage_chunk(completion_tokens=4),
    ])
    assert result["response"] == "Answer: 42"
    assert "<think>" not in result["response"]


def test_reasoning_only_output_falls_back_to_stripped_reasoning():
    # Qwen3-style: everything routed through reasoning_content, content empty
    result, _ = run_once([
        content_chunk(reasoning="the final answer is 7"),
        usage_chunk(completion_tokens=6, reasoning_tokens=6),
    ])
    assert "7" in result["response"]


def test_backend_extra_body_reaches_request_kwargs():
    # Regression for the Ollama think fix: extra_body must ride into create()
    backend = FakeBackend(extra_params={"extra_body": {"think": True}})
    result, client = run_once(
        [content_chunk("ok"), usage_chunk(completion_tokens=1)], backend=backend
    )
    assert result["error"] is None
    assert client.kwargs["model"] == "fake-model"
    assert client.kwargs["stream"] is True
    assert client.kwargs["stream_options"] == {"include_usage": True}
    assert client.kwargs["extra_body"]["think"] is True
    assert "think" not in client.kwargs  # never a top-level kwarg


def test_api_error_returns_error_result_not_exception():
    result, _ = run_once(FailingClient(RuntimeError("connection refused")))
    assert result["error"] == "connection refused"
    assert result["response"] == ""
    assert result["tps"] is None
    assert result["ttft_ms"] is None


def test_responses_api_runner_path_uses_backend_params_and_usage():
    backend = FakeBackend(
        responses=True,
        responses_params={
            "reasoning": {"effort": "high"},
            "text": {"verbosity": "low"},
            "max_output_tokens": 12000,
        },
    )
    client = FakeResponsesClient(
        output_text="Answer: x",
        output_tokens=9,
        reasoning_tokens=4,
        prompt_tokens=7,
        total_tokens=16,
        cost=0.0012,
    )
    runner = ModelRunner(backend, "gpt-5.5", {"timeout": 30}, client=client)

    result = runner._run_once(_task(system="system text"))

    assert result["error"] is None
    assert result["response"] == "Answer: x"
    assert result["completion_tokens"] == 9
    assert result["reasoning_tokens"] == 4
    assert result["prompt_tokens"] == 7
    assert result["total_tokens"] == 16
    assert result["api_cost"] == 0.0012
    assert result["reasoning_preview"] == "checked"
    assert client.kwargs["model"] == "gpt-5.5"
    assert client.kwargs["reasoning"] == {"effort": "high"}
    assert client.kwargs["text"] == {"verbosity": "low"}
    assert client.kwargs["max_output_tokens"] == 12000
    assert client.kwargs["input"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "hi"},
    ]
