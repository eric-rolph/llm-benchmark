from io import BytesIO, TextIOWrapper

from benchmark.console import make_console


def test_console_replaces_unencodable_characters_on_legacy_stream():
    raw = BytesIO()
    stream = TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    console = make_console(file=stream)

    console.print("status ✓ -> — 🏆")

    output = raw.getvalue().decode("cp1252")
    assert "status" in output
    assert "?" in output
