from settings_app.config import load_flags, parse_bool


def test_parse_bool_accepts_basic_values():
    assert parse_bool("true") is True
    assert parse_bool("false") is False


def test_load_flags_parses_key_value_lines():
    assert load_flags(["debug=true", "cache=false"]) == {
        "debug": True,
        "cache": False,
    }
