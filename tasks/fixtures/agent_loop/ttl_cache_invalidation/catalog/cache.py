class TTLCache:
    def __init__(self, ttl_seconds, clock):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._values = {}

    def get_or_set(self, key, factory):
        if key not in self._values:
            self._values[key] = factory()
        return self._values[key]

    def invalidate(self, key=None):
        if key is None:
            self._values.clear()
            return
        self._values.pop(key, None)
