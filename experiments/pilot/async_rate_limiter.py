"""
Async Rate Limiter for LLM API Calls

Provides async-compatible token bucket rate limiting for parallel API calls.
Uses asyncio.Lock and asyncio.sleep for non-blocking rate control.
"""

import asyncio
import time


class AsyncRateLimiter:
    """Async token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        """
        Initialize async rate limiter.

        Args:
            requests_per_minute: Maximum sustained request rate
            burst_size: Maximum burst capacity (tokens)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.min_interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire a token, waiting asynchronously if necessary.

        This method is coroutine-safe and can be called concurrently
        from multiple tasks.
        """
        async with self._lock:
            now = time.time()

            # Refill tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * (self.requests_per_minute / 60.0)
            )
            self.last_update = now

            if self.tokens < 1:
                # Calculate wait time and sleep asynchronously
                wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)
                await asyncio.sleep(wait_time)
                self.tokens = 1.0
                self.last_update = time.time()

            self.tokens -= 1

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.time()

    @property
    def available_tokens(self) -> float:
        """Return current available tokens (approximate)."""
        now = time.time()
        elapsed = now - self.last_update
        return min(
            self.burst_size,
            self.tokens + elapsed * (self.requests_per_minute / 60.0)
        )
