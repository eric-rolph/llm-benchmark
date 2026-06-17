from benchmark.evaluation import is_leaderboard_task, task_tier


def test_task_tier_defaults_to_leaderboard():
    assert task_tier({}) == "leaderboard"
    assert task_tier({"tier": "fresh"}) == "fresh"
    assert task_tier({"benchmark_tier": "repo_agent"}) == "repo_agent"


def test_leaderboard_task_excludes_smoke_diagnostic_and_high_contamination():
    assert is_leaderboard_task({"id": "core"})
    assert is_leaderboard_task({"benchmark_tier": "fresh"})
    assert is_leaderboard_task({"benchmark_tier": "repo_agent"})

    assert not is_leaderboard_task({"benchmark_tier": "smoke"})
    assert not is_leaderboard_task({"benchmark_tier": "diagnostic"})
    assert not is_leaderboard_task({"contamination_risk": "high"})
