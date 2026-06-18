from .parser import parse_services


def deployment_order(lines):
    return list(parse_services(lines))


def impacted_services(lines, changed):
    services = parse_services(lines)
    return sorted(name for name in changed if name in services)
