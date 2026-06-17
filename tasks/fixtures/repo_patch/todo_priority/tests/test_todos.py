from todo_app.filters import open_titles
from todo_app.models import Todo


def test_open_titles_returns_only_incomplete_items():
    todos = [Todo("write tests"), Todo("ship", done=True)]
    assert open_titles(todos) == ["write tests"]
