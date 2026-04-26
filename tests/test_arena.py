import json

from benchmark.arena import ArenaPlayer, arena_results_payload, save_arena_results


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
