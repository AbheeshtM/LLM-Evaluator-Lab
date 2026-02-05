import time
from threading import Lock

class RateLimiter:
    """
    Simple global rate limiter.
    Ensures a minimum delay between API calls.
    """
    def __init__(self, min_interval_sec=10):
        self.min_interval = min_interval_sec
        self.last_call = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()
