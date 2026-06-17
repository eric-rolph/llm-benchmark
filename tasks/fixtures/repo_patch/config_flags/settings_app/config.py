def parse_bool(value):
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"invalid boolean: {value}")


def load_flags(lines):
    flags = {}
    for line in lines:
        if not line.strip():
            continue
        key, value = line.split("=")
        flags[key] = parse_bool(value)
    return flags
