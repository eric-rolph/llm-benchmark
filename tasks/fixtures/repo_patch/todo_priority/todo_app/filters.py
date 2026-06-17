def open_titles(todos):
    return [todo.title for todo in todos if not todo.done]
