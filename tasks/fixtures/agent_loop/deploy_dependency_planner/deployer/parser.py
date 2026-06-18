def parse_services(lines):
    services = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        name, _, deps_raw = line.partition(":")
        deps = [dep.strip() for dep in deps_raw.split(",") if dep.strip()]
        services[name.strip()] = deps
    return services
