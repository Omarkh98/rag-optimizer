import asyncio
import logging

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        while True:
            async with self.lock:
                now = asyncio.get_event_loop().time()
                self.calls = [t for t in self.calls if now - t < self.period]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                else:
                    sleep_time = self.period - (now - min(self.calls))
            await asyncio.sleep(sleep_time)