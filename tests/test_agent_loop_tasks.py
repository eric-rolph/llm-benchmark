import shutil
from pathlib import Path

from benchmark.loader import load_tasks
from benchmark.repo_patch import _run_test_command, _write_hidden_tests


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


def test_dependency_planner_agent_loop_task_is_calibrated(tmp_path):
    tasks = {task["id"]: task for task in load_tasks("agent_loop")}
    task = tasks["agent_loop_006_deploy_dependency_planner"]
    scoring = task["scoring"]
    fixture = Path(scoring["repo_fixture"])

    assert task["benchmark_tier"] == "repo_agent"
    assert task["difficulty"] == "hard"
    assert "misleading_or_incomplete_visible_tests" in task["criticisms_addressed"]
    assert scoring["max_steps"] == 12
    assert len(scoring["hidden_tests"]) == 1

    workspace = tmp_path / "workspace"
    shutil.copytree(fixture, workspace)

    visible = _run_test_command(scoring["test_command"], workspace, 30)
    assert visible.returncode == 0, visible.stdout + visible.stderr

    ok, detail = _write_hidden_tests(workspace, scoring["hidden_tests"])
    assert ok, detail
    hidden = _run_test_command(scoring["test_command"], workspace, 30)
    assert hidden.returncode != 0
    assert "test_hidden_dependency_planner" in hidden.stdout
