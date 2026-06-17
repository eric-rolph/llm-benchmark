class ManualClock:
    def __init__(self):
        self._now = 0.0

    def now(self):
        return self._now

    def advance(self, seconds):
        self._now += seconds
