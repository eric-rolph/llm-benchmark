import json
from dataclasses import dataclass
from types import SimpleNamespace

from benchmark import arena
from benchmark.arena import ArenaPlayer, arena_results_payload, save_arena_results
from benchmark.session import ApiCostBudget


@dataclass
class DummyModel:
    id: str


class DummyBackend:
    name = "dummy"
    config = {}


def test_arena_results_payload_sorts_leaderboard_and_includes_history():
    players = {
        "model-b": ArenaPlayer(
            model_id="model-b",
            backend_name="ollama",
            elo=1516.25,
            wins=1,
            losses=0,
            ties=0,
            history=[{"task_id": "task_1", "opponent": "model-a", "result": "A"}],
        ),
        "model-a": ArenaPlayer(
            model_id="model-a",
            backend_name="lm_studio",
            elo=1483.75,
            wins=0,
            losses=1,
            ties=0,
            history=[{"task_id": "task_1", "opponent": "model-b", "result": "B"}],
        ),
    }

    payload = arena_results_payload(players)

    assert payload["initial_elo"] == 1500
    assert payload["k_factor"] == 32
    assert [row["model_id"] for row in payload["leaderboard"]] == ["model-b", "model-a"]
    assert payload["leaderboard"][0]["rank"] == 1
    assert payload["leaderboard"][0]["elo"] == 1516.25
    assert payload["leaderboard"][0]["win_rate"] == 1.0
    assert payload["history"]["model-a"] == [{"task_id": "task_1", "opponent": "model-b", "result": "B"}]


def test_save_arena_results_writes_json(tmp_path):
    players = {
        "model-a": ArenaPlayer(
            model_id="model-a",
            backend_name="lm_studio",
            wins=0,
            losses=0,
            ties=1,
        ),
    }

    path = save_arena_results(players, tmp_path)

    assert path is not None
    assert path.name.startswith("arena_")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["leaderboard"][0]["model_id"] == "model-a"
    assert data["leaderboard"][0]["matches"] == 1


def test_save_arena_results_skips_empty_arena(tmp_path):
    assert save_arena_results({}, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_run_arena_stops_before_next_task_when_api_cost_budget_is_exhausted(monkeypatch):
    calls = []

    class FakeRunner:
        def __init__(self, backend, model_id, bench_config):
            self.model_id = model_id

        def run_task(self, task):
            calls.append((self.model_id, task["id"]))
            return {"response": self.model_id, "api_cost": 0.01}

    monkeypatch.setattr(arena, "ModelRunner", FakeRunner)
    monkeypatch.setattr(arena, "_judge_pair", lambda **kwargs: ("A", "judge ok"))

    players = arena.run_arena(
        model_pairs=[(DummyModel("model-a"), DummyBackend()), (DummyModel("model-b"), DummyBackend())],
        tasks=[
            {"id": "task_1", "prompt": "one"},
            {"id": "task_2", "prompt": "two"},
        ],
        bench_config={},
        judge_client=object(),
        judge_model="judge",
        api_cost_budget=ApiCostBudget(limit=0.015),
    )

    assert calls == [("model-a", "task_1"), ("model-b", "task_1")]
    assert players["model-a"].wins == 1


def test_judge_pair_returns_provider_reported_api_cost(monkeypatch):
    class FakeJudgeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            message = SimpleNamespace(content="Reasoning\nWINNER: A")
            choice = SimpleNamespace(message=message)
            usage = SimpleNamespace(cost=0.0042)
            return SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setattr(arena.random, "random", lambda: 0.9)

    result, reasoning, api_cost = arena._judge_pair(
        prompt="question",
        response_a="A",
        response_b="B",
        judge_client=FakeJudgeClient(),
        judge_model="judge",
    )

    assert result == "A"
    assert "WINNER" in reasoning
    assert api_cost == 0.0042


def test_run_arena_counts_judge_api_cost_toward_budget(monkeypatch):
    calls = []
    judge_calls = []

    class FakeRunner:
        def __init__(self, backend, model_id, bench_config):
            self.model_id = model_id

        def run_task(self, task):
            calls.append((self.model_id, task["id"]))
            return {"response": self.model_id, "api_cost": 0.0}

    def fake_judge_pair(**kwargs):
        judge_calls.append((kwargs["response_a"], kwargs["response_b"]))
        return "A", "judge ok", 0.02

    monkeypatch.setattr(arena, "ModelRunner", FakeRunner)
    monkeypatch.setattr(arena, "_judge_pair", fake_judge_pair)
    budget = ApiCostBudget(limit=0.015)

    players = arena.run_arena(
        model_pairs=[
            (DummyModel("model-a"), DummyBackend()),
            (DummyModel("model-b"), DummyBackend()),
            (DummyModel("model-c"), DummyBackend()),
        ],
        tasks=[{"id": "task_1", "prompt": "one"}],
        bench_config={},
        judge_client=object(),
        judge_model="judge",
        api_cost_budget=budget,
    )

    assert calls == [("model-a", "task_1"), ("model-b", "task_1"), ("model-c", "task_1")]
    assert len(judge_calls) == 1
    assert budget.spent == 0.02
    assert players["model-a"].wins == 1
    assert players["model-c"].wins == 0
