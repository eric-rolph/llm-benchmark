"""Repo-level patch scoring for small, local software-engineering tasks."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


_FENCED_RE = re.compile(r"```(?P<header>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_LAUNCH_ERROR = 127
_TIMEOUT_ERROR = 124
_PROTECTED_FILENAMES = {
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "sitecustomize.py",
}
_PROTECTED_ROOT_FILES = {
    "pyproject.toml",
    "setup.cfg",
}
_PROTECTED_DIRS = {
    ".llm_bench_sentinel",
    "tests",
}
_ENV_KEEP = {
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
}


def score_repo_patch(response: str, scoring: dict) -> tuple[float, str]:
    """Apply a model-produced repo edit in a temp workspace and run tests."""
    fixture_raw = scoring.get("repo_fixture")
    if not fixture_raw:
        return 0.0, "repo_patch: no repo_fixture configured"

    fixture = _resolve_fixture_path(str(fixture_raw))
    if fixture is None or not fixture.exists() or not fixture.is_dir():
        return 0.0, f"repo_patch: fixture not found: {fixture_raw}"

    command = scoring.get("test_command")
    if not command:
        return 0.0, "repo_patch: no test_command configured"
    ok, detail = _validate_test_command(command)
    if not ok:
        return 0.0, detail

    with tempfile.TemporaryDirectory(prefix="llm-bench-repo-") as tmp:
        workspace = Path(tmp) / "workspace"
        shutil.copytree(
            fixture,
            workspace,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"),
        )

        ok, detail, changed_count = _apply_response(response, workspace)
        if not ok:
            return 0.0, detail

        sentinel = _make_sentinel_hidden_test()
        hidden_tests = list(scoring.get("hidden_tests", [])) + [sentinel["test"]]
        ok, detail = _write_hidden_tests(workspace, hidden_tests)
        if not ok:
            return 0.0, detail

        timeout_s = float(scoring.get("timeout", 60))
        proc = _run_test_command(command, workspace, timeout_s)
        if proc.returncode == _TIMEOUT_ERROR:
            return 0.0, f"repo_patch: tests timed out after {timeout_s:g}s"
        if proc.returncode == _LAUNCH_ERROR:
            return 0.0, f"repo_patch: test command failed to launch: {_truncate(proc.stderr or proc.stdout, 500)}"
        if proc.returncode == 0:
            if not _sentinel_completed(workspace, sentinel):
                return 0.0, "repo_patch: test sentinel did not run; rejecting zero-exit result"
            return 1.0, f"repo_patch: tests passed ({changed_count} file(s) changed)"
        return 0.0, (
            f"repo_patch: tests failed (exit {proc.returncode}): "
            f"{_truncate((proc.stdout or '') + (proc.stderr or ''), 500)}"
        )


def _resolve_fixture_path(raw: str) -> Path | None:
    path = Path(raw)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parent.parent / path,
        Path(__file__).resolve().parent / path,
    ]
    parts = path.parts
    if parts and parts[0] == "tasks":
        candidates.append(Path(__file__).resolve().parent / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _apply_response(response: str, workspace: Path) -> tuple[bool, str, int]:
    files = _extract_file_replacements(response)
    if files:
        ok, detail = _write_files(workspace, files)
        return ok, detail, len(files)

    diff = _extract_unified_diff(response)
    if diff:
        protected_path = _first_protected_diff_path(diff)
        if protected_path:
            return False, f"repo_patch: protected path rejected: {protected_path}", 0

        ok, detail, changed = _apply_simple_unified_diff(diff, workspace)
        if ok:
            return True, detail, changed

        try:
            proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=workspace,
                input=diff,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            git_detail = _truncate(proc.stderr or proc.stdout, 500)
            if proc.returncode == 0 and _count_changed_files_in_diff(diff) > 0:
                return True, "repo_patch: diff applied", _count_changed_files_in_diff(diff)
        except subprocess.TimeoutExpired:
            git_detail = "git apply timed out"
        except OSError as exc:
            git_detail = str(exc)
        return False, f"repo_patch: patch apply failed: {detail}; {git_detail}", 0

    return False, "repo_patch: no JSON files object or unified diff found", 0


def _extract_file_replacements(response: str) -> dict[str, str] | None:
    for data in _iter_json_values(response):
        files = _files_from_json_value(data)
        if files is not None:
            return files

    fenced_files: dict[str, str] = {}
    for match in _FENCED_RE.finditer(response):
        header = match.group("header").strip()
        path_match = re.search(r"(?:^|\s)path=(?P<quote>['\"]?)(?P<path>[^'\"\s]+)(?P=quote)", header)
        if path_match:
            fenced_files[path_match.group("path")] = match.group("body")
    return fenced_files or None


def _files_from_json_value(data) -> dict[str, str] | None:
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return None
    files: dict[str, str] = {}
    for path, value in data["files"].items():
        if isinstance(value, str):
            files[str(path)] = value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            files[str(path)] = value["content"]
        else:
            return None
    return files


def _iter_json_values(response: str):
    text = response.strip()
    candidates = [text]
    candidates.extend(block.strip() for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE))
    for candidate in candidates:
        try:
            yield json.loads(candidate)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[idx:])
            yield data
        except json.JSONDecodeError:
            continue


def _extract_unified_diff(response: str) -> str | None:
    for match in _FENCED_RE.finditer(response):
        header = match.group("header").strip().lower()
        body = match.group("body").strip()
        if header.startswith(("diff", "patch")) or body.startswith(("diff --git ", "--- ")) or "\n--- " in body:
            return body + "\n"
    text = response.strip()
    if text.startswith("diff --git ") or text.startswith("--- "):
        return text + "\n"
    return None


def _write_files(
    workspace: Path,
    files: dict[str, str],
    *,
    allow_protected_paths: bool = False,
) -> tuple[bool, str]:
    for rel, content in files.items():
        safe_target = _safe_target(workspace, rel)
        if safe_target is None:
            return False, f"repo_patch: unsafe path rejected: {rel}"
        if not allow_protected_paths and _is_protected_harness_path(rel):
            return False, f"repo_patch: protected path rejected: {rel}"
        safe_target.parent.mkdir(parents=True, exist_ok=True)
        safe_target.write_text(content, encoding="utf-8")
    return True, f"repo_patch: wrote {len(files)} file(s)"


def _write_hidden_tests(workspace: Path, hidden_tests: list) -> tuple[bool, str]:
    if not hidden_tests:
        return True, "repo_patch: no hidden tests"
    files = {}
    for item in hidden_tests:
        if not isinstance(item, dict) or not item.get("path") or not isinstance(item.get("content"), str):
            return False, "repo_patch: hidden_tests entries require path and content"
        files[str(item["path"])] = item["content"]
    ok, detail = _write_files(workspace, files, allow_protected_paths=True)
    if not ok:
        return False, detail
    return True, f"repo_patch: wrote {len(files)} hidden test file(s)"


def _safe_target(workspace: Path, rel: str) -> Path | None:
    rel_path = Path(rel)
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        return None
    target = (workspace / rel_path).resolve()
    root = workspace.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _is_protected_harness_path(rel: str) -> bool:
    parts = _normal_rel_parts(rel)
    if not parts:
        return False
    if any(part in _PROTECTED_DIRS for part in parts):
        return True
    name = parts[-1]
    if name in _PROTECTED_FILENAMES:
        return True
    return len(parts) == 1 and name in _PROTECTED_ROOT_FILES


def _normal_rel_parts(rel: str) -> tuple[str, ...]:
    rel_path = Path(str(rel).replace("\\", "/"))
    if rel_path.is_absolute():
        return ()
    return tuple(part.lower() for part in rel_path.parts if part not in {"", "."})


def _first_protected_diff_path(diff: str) -> str | None:
    for rel in _iter_diff_paths(diff):
        if _is_protected_harness_path(rel):
            return rel
    return None


def _iter_diff_paths(diff: str):
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git\s+a/(?P<old>\S+)\s+b/(?P<new>\S+)", line)
            if match:
                yield match.group("old")
                yield match.group("new")
            continue
        if line.startswith(("--- ", "+++ ")):
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            yield raw[2:] if raw.startswith(("a/", "b/")) else raw


def _run_test_command(command, workspace: Path, timeout_s: float) -> subprocess.CompletedProcess:
    env = _minimal_test_env(workspace)
    command = _resolve_command_tokens(command)
    args = [str(part) for part in command]
    try:
        return subprocess.run(
            args,
            cwd=workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(args, _LAUNCH_ERROR, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = _stream_text(exc.stdout)
        stderr = _stream_text(exc.stderr)
        return subprocess.CompletedProcess(args, _TIMEOUT_ERROR, stdout, stderr)


def _validate_test_command(command) -> tuple[bool, str]:
    if not isinstance(command, list):
        return False, "repo_patch: test_command must be list-form to avoid shell execution"
    if not _command_invokes_pytest(command):
        return False, "repo_patch: test_command must invoke pytest so hidden sentinel tests are verified"
    return True, "repo_patch: test command ok"


def _command_invokes_pytest(command: list) -> bool:
    for part in command:
        text = str(part).lower()
        name = Path(text).name
        if text == "pytest" or name in {"pytest", "pytest.exe"}:
            return True
    return any(str(part).lower() == "pytest" for part in command)


def _minimal_test_env(workspace: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP}
    env["PYTHONPATH"] = str(workspace)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _stream_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_command_tokens(command):
    return [sys.executable if str(part) == "{python}" else part for part in command]


def _make_sentinel_hidden_test() -> dict:
    token = uuid.uuid4().hex
    marker_rel = f".llm_bench_sentinel/{token}.txt"
    test_name = f"tests/test_llm_bench_sentinel_{token}.py"
    content = f"""
from pathlib import Path


def test_llm_bench_sentinel_{token}():
    marker = Path(__file__).resolve().parents[1] / ".llm_bench_sentinel" / "{token}.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{token}", encoding="utf-8")
    assert marker.read_text(encoding="utf-8") == "{token}"
"""
    return {
        "token": token,
        "marker_rel": marker_rel,
        "test": {"path": test_name, "content": content.lstrip()},
    }


def _sentinel_completed(workspace: Path, sentinel: dict) -> bool:
    marker = workspace / sentinel["marker_rel"]
    try:
        return marker.read_text(encoding="utf-8") == sentinel["token"]
    except OSError:
        return False


def _apply_simple_unified_diff(diff: str, workspace: Path) -> tuple[bool, str, int]:
    patches = _split_file_patches(diff)
    if not patches:
        return False, "no file patches found", 0

    changed = 0
    for rel, lines in patches:
        target = _safe_target(workspace, rel)
        if target is None:
            return False, f"unsafe path rejected: {rel}", changed
        if not target.exists():
            return False, f"target file does not exist: {rel}", changed
        old_text = target.read_text(encoding="utf-8")
        ok, new_text_or_error = _apply_hunks(old_text, lines)
        if not ok:
            return False, new_text_or_error, changed
        if new_text_or_error != old_text:
            target.write_text(new_text_or_error, encoding="utf-8")
            changed += 1

    if changed == 0:
        return False, "diff produced no file changes", 0
    return True, f"repo_patch: diff applied ({changed} file(s) changed)", changed


def _split_file_patches(diff: str) -> list[tuple[str, list[str]]]:
    lines = diff.splitlines()
    patches: list[tuple[str, list[str]]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("+++ "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                current_path = None
                current_lines = []
                continue
            current_path = raw[2:] if raw.startswith("b/") else raw
            current_lines = []
            patches.append((current_path, current_lines))
            continue
        if current_path is not None:
            current_lines.append(line)
    return patches


def _apply_hunks(old_text: str, patch_lines: list[str]) -> tuple[bool, str]:
    old_lines = old_text.splitlines()
    old_had_trailing_newline = old_text.endswith("\n")
    new_lines: list[str] = []
    old_index = 0
    line_index = 0

    while line_index < len(patch_lines):
        line = patch_lines[line_index]
        if not line.startswith("@@"):
            line_index += 1
            continue

        match = re.match(r"@@ -(?P<start>\d+)(?:,\d+)? \+(?:\d+)(?:,\d+)? @@", line)
        if not match:
            return False, f"invalid hunk header: {line}"
        old_start = int(match.group("start")) - 1
        if old_start < old_index or old_start > len(old_lines):
            return False, f"hunk start out of range: {line}"
        new_lines.extend(old_lines[old_index:old_start])
        old_index = old_start
        line_index += 1

        while line_index < len(patch_lines) and not patch_lines[line_index].startswith("@@"):
            hunk_line = patch_lines[line_index]
            if hunk_line.startswith(" "):
                expected = hunk_line[1:]
                if old_index >= len(old_lines) or old_lines[old_index] != expected:
                    return False, f"hunk context mismatch near: {expected}"
                new_lines.append(expected)
                old_index += 1
            elif hunk_line.startswith("-"):
                expected = hunk_line[1:]
                if old_index >= len(old_lines) or old_lines[old_index] != expected:
                    return False, f"hunk removal mismatch near: {expected}"
                old_index += 1
            elif hunk_line.startswith("+"):
                new_lines.append(hunk_line[1:])
            elif hunk_line.startswith("\\"):
                pass
            else:
                return False, f"invalid hunk line: {hunk_line}"
            line_index += 1

    new_lines.extend(old_lines[old_index:])
    new_text = "\n".join(new_lines)
    if old_had_trailing_newline:
        new_text += "\n"
    return True, new_text


def _count_changed_files_in_diff(diff: str) -> int:
    paths = set(re.findall(r"^\+\+\+\s+b/(.+)$", diff, re.MULTILINE))
    return len(paths) or 1


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text[:limit]
