from catalog.cache import TTLCache
from catalog.clock import ManualClock
from catalog.service import ProductService


def test_cache_reuses_value_before_ttl():
    clock = ManualClock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    calls = []

    def factory():
        calls.append("called")
        return f"value-{len(calls)}"

    assert cache.get_or_set("a", factory) == "value-1"
    assert cache.get_or_set("a", factory) == "value-1"
    assert calls == ["called"]


def test_product_service_caches_fetches():
    clock = ManualClock()
    calls = []

    class Client:
        def fetch_product(self, product_id, include_archived=False):
            calls.append((product_id, include_archived))
            return {"id": product_id, "archived": include_archived}

    service = ProductService(Client(), clock, ttl_seconds=10)

    assert service.get_product("sku-1") == {"id": "sku-1", "archived": False}
    assert service.get_product("sku-1") == {"id": "sku-1", "archived": False}
    assert calls == [("sku-1", False)]
