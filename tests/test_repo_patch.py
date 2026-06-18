from benchmark import repo_patch
from benchmark.scorer import score_response


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
        "type": "repo_patch",
        "repo_fixture": str(fixture),
        "test_command": ["{python}", "-m", "pytest", "-q"],
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
        "id": "repo_patch_mean",
        "prompt": "Fix calc.stats.mean so it returns the arithmetic mean.",
        "category": "agentic",
        "scoring": scoring,
    }


def _result(response: str) -> dict:
    return {
        "task_id": "repo_patch_mean",
        "response": response,
        "error": None,
        "ttft_ms": None,
        "total_ms": 100.0,
        "tps": 10.0,
        "completion_tokens": 5,
        "reasoning_tokens": 0,
        "backend": "test",
    }


def test_repo_patch_is_gated_by_allow_code_exec(tmp_path):
    task = _task(tmp_path)
    response = '{"files": {"calc/stats.py": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=False)

    assert scored["score"] == 0.0
    assert "allow-code-exec" in scored["score_detail"] or "disabled" in scored["score_detail"]


def test_repo_patch_applies_json_file_replacements_and_runs_hidden_tests(tmp_path):
    task = _task(tmp_path)
    response = '{"files": {"calc/stats.py": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 1.0
    assert scored["passed"] is True
    assert "tests passed" in scored["score_detail"]


def test_repo_patch_hidden_tests_reject_incomplete_fix(tmp_path):
    task = _task(tmp_path)
    response = '{"files": {"calc/stats.py": "def mean(values):\\n    return round(sum(values) / len(values))\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert scored["passed"] is False
    assert "tests failed" in scored["score_detail"]


def test_repo_patch_rejects_unsafe_paths(tmp_path):
    task = _task(tmp_path)
    response = '{"files": {"../outside.py": "print(\\"bad\\")\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert "unsafe path" in scored["score_detail"].lower()


def test_repo_patch_rejects_hidden_test_collection_sabotage(tmp_path):
    task = _task(tmp_path)
    response = (
        '{"files": {'
        '"conftest.py": "def pytest_ignore_collect(collection_path, config):\\n'
        '    return collection_path.name.startswith(\\"test_hidden\\")\\n"'
        '}}'
    )

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert "protected path" in scored["score_detail"].lower()


def test_repo_patch_rejects_diff_that_adds_pytest_control_file(tmp_path):
    task = _task(tmp_path)
    response = """
```diff
diff --git a/conftest.py b/conftest.py
new file mode 100644
--- /dev/null
+++ b/conftest.py
@@ -0,0 +1,2 @@
+def pytest_ignore_collect(collection_path, config):
+    return collection_path.name.startswith("test_hidden")
```
"""

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert "protected path" in scored["score_detail"].lower()


def test_repo_patch_applies_unified_diff_blocks(tmp_path):
    task = _task(tmp_path)
    response = """
```diff
diff --git a/calc/stats.py b/calc/stats.py
--- a/calc/stats.py
+++ b/calc/stats.py
@@ -1,2 +1,2 @@
 def mean(values):
-    return sum(values) // len(values)
+    return sum(values) / len(values)
```
"""

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 1.0
    assert scored["passed"] is True


def test_repo_patch_rejects_zero_exit_when_tests_do_not_run(tmp_path):
    task = _task(tmp_path)
    response = '{"files": {"calc/stats.py": "import os\\nos._exit(0)\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert scored["passed"] is False
    assert "sentinel" in scored["score_detail"].lower()


def test_repo_patch_does_not_expose_parent_environment_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BENCH_SECRET_TOKEN", "do-not-leak")
    task = _task(tmp_path)
    response = (
        '{"files": {"calc/stats.py": "import os\\n\\n'
        'def mean(values):\\n'
        '    if os.environ.get(\\"LLM_BENCH_SECRET_TOKEN\\"):\\n'
        '        raise RuntimeError(\\"secret leaked\\")\\n'
        '    return sum(values) / len(values)\\n"}}'
    )

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 1.0
    assert scored["passed"] is True


def test_repo_patch_timeout_is_reported_as_score_failure(tmp_path):
    task = _task(tmp_path, timeout=0.5)
    response = (
        '{"files": {"calc/stats.py": "def mean(values):\\n'
        '    while True:\\n'
        '        pass\\n"}}'
    )

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert scored["passed"] is False
    assert "timed out" in scored["score_detail"].lower()


def test_repo_patch_rejects_string_test_commands(tmp_path):
    task = _task(tmp_path, test_command="{python} -m pytest -q")
    response = '{"files": {"calc/stats.py": "def mean(values):\\n    return sum(values) / len(values)\\n"}}'

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 0.0
    assert "list-form" in scored["score_detail"].lower()


def test_repo_patch_finds_valid_json_patch_after_earlier_json(tmp_path):
    task = _task(tmp_path)
    response = """
{"note": "analysis metadata, not a patch"}
```json
{"files": {"calc/stats.py": "def mean(values):\\n    return sum(values) / len(values)\\n"}}
```
"""

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 1.0
    assert scored["passed"] is True


def test_repo_patch_accepts_patch_fenced_unified_diff(tmp_path):
    task = _task(tmp_path)
    response = """
```patch
--- a/calc/stats.py
+++ b/calc/stats.py
@@ -1,2 +1,2 @@
 def mean(values):
-    return sum(values) // len(values)
+    return sum(values) / len(values)
```
"""

    scored = score_response(task, _result(response), allow_code_exec=True)

    assert scored["score"] == 1.0
    assert scored["passed"] is True


def test_repo_patch_resolves_installed_package_fixture_layout(tmp_path, monkeypatch):
    package_root = tmp_path / "site-packages" / "benchmark"
    fixture = package_root / "tasks" / "fixtures" / "repo_patch" / "example"
    fixture.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(repo_patch, "__file__", str(package_root / "repo_patch.py"))

    resolved = repo_patch._resolve_fixture_path("tasks/fixtures/repo_patch/example")

    assert resolved == fixture
