from .cache import TTLCache


class ProductService:
    def __init__(self, client, clock, ttl_seconds=60):
        self.client = client
        self.cache = TTLCache(ttl_seconds=ttl_seconds, clock=clock)

    def get_product(self, product_id, include_archived=False):
        return self.cache.get_or_set(
            product_id,
            lambda: self.client.fetch_product(
                product_id,
                include_archived=include_archived,
            ),
        )

    def refresh_product(self, product_id):
        self.cache.invalidate(product_id)
        return self.get_product(product_id)
