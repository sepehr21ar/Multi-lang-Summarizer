import asyncio
import time
from collections import deque
from typing import Deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/health", "/ready", "/docs", "/openapi.json", "/"}:
            return await call_next(request)

        now = time.time()
        key = request.client.host if request.client else "unknown"

        async with self._lock:
            queue = self._requests.setdefault(key, deque())
            while queue and queue[0] <= now - 60:
                queue.popleft()

            if len(queue) >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again in a minute."},
                )

            queue.append(now)

        return await call_next(request)
