from pathlib import Path

from benchmark.loader import load_tasks


def test_harder_agent_loop_tasks_are_registered_with_hidden_tests():
    tasks = {task["id"]: task for task in load_tasks("agent_loop")}

    for task_id in {
        "agent_loop_004_csv_import_reconciliation",
        "agent_loop_005_ttl_cache_invalidation",
    }:
        task = tasks[task_id]
        scoring = task["scoring"]
        fixture = Path(scoring["repo_fixture"])

        assert task["benchmark_tier"] == "repo_agent"
        assert task["execution_surface"] == "observed_agent_loop"
        assert "misleading_or_incomplete_visible_tests" in task["criticisms_addressed"]
        assert "multi_file_patch_required" in task["criticisms_addressed"]
        assert scoring["type"] == "agent_loop"
        assert fixture.exists()
        assert scoring["hidden_tests"]
        assert scoring["max_steps"] >= 12
