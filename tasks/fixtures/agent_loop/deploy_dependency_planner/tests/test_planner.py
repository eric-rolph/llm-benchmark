from deployer.planner import deployment_order, impacted_services


def test_deployment_order_accepts_simple_ordered_chain():
    lines = [
        "db:",
        "api: db",
    ]

    assert deployment_order(lines) == ["db", "api"]


def test_impacted_services_returns_known_changed_service():
    lines = [
        "api:",
        "worker:",
    ]

    assert impacted_services(lines, {"api"}) == ["api"]
