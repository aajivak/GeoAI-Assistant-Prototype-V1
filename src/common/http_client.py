
"""Asynchronous HTTP transport for public GeoAI data services.

Supports:
- Nominatim geocoding
- OpenStreetMap Overpass API
- Open-Meteo weather and air-quality APIs

Features:
- Configurable timeouts
- Bounded retries with exponential backoff
- Retry-After handling
- TTL-based in-memory caching
- Explicit HTTP and JSON error handling
- Connection pooling within each request operation
- Fast-failing Overpass requests for endpoint failover

This layer transports and validates JSON structure.
Domain-specific validation belongs to the data clients.
"""

import asyncio
import copy
import json
import logging
import random
import threading
import time

from typing import Any, Dict, Optional

import httpx

from config.settings import settings


logger = logging.getLogger("geoai.http_client")


class APIError(Exception):
    """Base exception for external API communication errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class RateLimitError(APIError):
    """Raised when an external API rate limit is exhausted."""


class NotFoundError(APIError):
    """Raised when an external resource cannot be found."""


class GeocodingError(APIError):
    """Raised when geographic resolution fails."""


class SimpleMemoryCache:
    """Thread-safe TTL cache for JSON-compatible responses."""

    MAX_ENTRIES = 512

    def __init__(self):
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            expires_at, value = entry

            if time.monotonic() >= expires_at:
                self._cache.pop(key, None)
                return None

            return copy.deepcopy(value)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()

            expired_keys = [
                existing_key
                for existing_key, (expiry, _) in self._cache.items()
                if expiry <= now
            ]

            for expired_key in expired_keys:
                self._cache.pop(expired_key, None)

            if len(self._cache) >= self.MAX_ENTRIES:
                oldest_key = min(
                    self._cache,
                    key=lambda item: self._cache[item][0],
                )
                self._cache.pop(oldest_key, None)

            self._cache[key] = (
                now + ttl_seconds,
                copy.deepcopy(value),
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_memory_cache = SimpleMemoryCache()


class AsyncHttpClient:
    """Async JSON HTTP client with retries and bounded caching."""

    RETRYABLE_STATUS_CODES = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    OVERPASS_TIMEOUT_SECONDS = 12.0

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
    ):
        self.timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(settings.HTTP_TIMEOUT_SECONDS)
        )

        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(settings.HTTP_MAX_RETRIES)
        )

        self.backoff_factor = (
            float(backoff_factor)
            if backoff_factor is not None
            else float(settings.HTTP_BACKOFF_FACTOR)
        )

        if self.timeout <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if self.backoff_factor <= 0:
            raise ValueError(
                "backoff_factor must be positive."
            )

        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "application/json",
        }

    @staticmethod
    def _cache_key(
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> str:
        """Build a deterministic cache key."""

        payload = {
            "method": method.upper(),
            "url": url,
            "params": params or {},
            "headers": headers or {},
            "body": body,
        }

        return json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    def _timeout_config(
        self,
        request_timeout: Optional[float] = None,
    ) -> httpx.Timeout:
        """Construct a bounded timeout configuration."""

        timeout = (
            request_timeout
            if request_timeout is not None
            else self.timeout
        )

        connect_timeout = min(
            timeout,
            float(settings.HTTP_CONNECT_TIMEOUT_SECONDS),
        )

        return httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
        )

    def _retry_delay(
        self,
        attempt_index: int,
        response: Optional[httpx.Response] = None,
    ) -> float:
        """Calculate bounded retry delay."""

        if response is not None:
            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after:
                try:
                    seconds = float(retry_after)

                    if seconds >= 0:
                        return min(seconds, 30.0)

                except ValueError:
                    pass

        base = self.backoff_factor ** (
            attempt_index + 1
        )

        jitter = random.uniform(0.0, 0.25)

        return min(
            base + jitter,
            15.0,
        )

    @staticmethod
    def _decode_json(
        response: httpx.Response,
        url: str,
    ) -> Any:
        """Decode JSON and reject malformed API responses."""

        try:
            data = response.json()

        except ValueError as exc:
            raise APIError(
                f"Invalid JSON returned by {url}.",
                status_code=response.status_code,
                response_text=response.text[:500],
            ) from exc

        if data is None:
            raise APIError(
                f"Empty JSON response returned by {url}.",
                status_code=response.status_code,
                response_text=response.text[:500],
            )

        return data

    async def _request_json(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        form_data: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: Optional[int] = None,
        request_timeout: Optional[float] = None,
        max_retries_override: Optional[int] = None,
    ) -> Any:
        """Execute a JSON request with retry and cache support."""

        merged_headers = {
            **self.headers,
            **(headers or {}),
        }

        cache_key = None

        if (
            cache_ttl_seconds is not None
            and cache_ttl_seconds > 0
        ):
            cache_key = self._cache_key(
                method=method,
                url=url,
                params=params,
                headers=merged_headers,
                body=(
                    json.dumps(
                        form_data,
                        sort_keys=True,
                        default=str,
                    )
                    if form_data is not None
                    else None
                ),
            )

            cached = _memory_cache.get(cache_key)

            if cached is not None:
                logger.debug(
                    "HTTP cache hit: %s %s",
                    method,
                    url,
                )
                return cached

        timeout_config = self._timeout_config(
            request_timeout
        )

        retry_count = (
            self.max_retries
            if max_retries_override is None
            else int(max_retries_override)
        )

        if retry_count < 0:
            raise ValueError(
                "max_retries_override cannot be negative."
            )

        total_attempts = retry_count + 1

        async with httpx.AsyncClient(
            timeout=timeout_config,
            follow_redirects=True,
        ) as client:

            for attempt_index in range(total_attempts):
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        data=form_data,
                        headers=merged_headers,
                    )

                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:

                    if attempt_index == total_attempts - 1:
                        raise APIError(
                            f"{method} request to {url} failed "
                            f"after {total_attempts} attempt(s): "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    delay = self._retry_delay(
                        attempt_index
                    )

                    logger.warning(
                        "HTTP transport error for %s: %s. "
                        "Retrying in %.2f seconds.",
                        url,
                        exc,
                        delay,
                    )

                    await asyncio.sleep(delay)
                    continue

                if response.is_success:
                    data = self._decode_json(
                        response,
                        url,
                    )

                    if (
                        cache_key is not None
                        and cache_ttl_seconds is not None
                    ):
                        _memory_cache.set(
                            cache_key,
                            data,
                            cache_ttl_seconds,
                        )

                    return data

                status = response.status_code

                if status == 404:
                    raise NotFoundError(
                        f"Resource not found: {url}",
                        status_code=404,
                        response_text=response.text[:500],
                    )

                if status in self.RETRYABLE_STATUS_CODES:
                    if attempt_index < total_attempts - 1:
                        delay = self._retry_delay(
                            attempt_index,
                            response,
                        )

                        logger.warning(
                            "HTTP %s from %s. "
                            "Retrying in %.2f seconds "
                            "(attempt %s/%s).",
                            status,
                            url,
                            delay,
                            attempt_index + 1,
                            total_attempts,
                        )

                        await asyncio.sleep(delay)
                        continue

                    error_class = (
                        RateLimitError
                        if status == 429
                        else APIError
                    )

                    raise error_class(
                        f"HTTP {status} from {url} "
                        f"after {total_attempts} attempt(s).",
                        status_code=status,
                        response_text=response.text[:500],
                    )

                raise APIError(
                    f"HTTP {status} from {url}: "
                    f"{response.text[:200]}",
                    status_code=status,
                    response_text=response.text[:500],
                )

        raise APIError(
            f"Request to {url} ended unexpectedly."
        )

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Any:
        """GET JSON with retry, timeout, and optional caching."""

        return await self._request_json(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def post_overpass(
        self,
        url: str,
        query: str,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Any:
        """POST Overpass QL with a fast-failing request policy.

        One attempt per endpoint, without exponential backoff.

        OverpassClient.execute_query() is responsible for
        switching to the fallback endpoint when needed.

        This prevents a failed Overpass request from blocking
        the recommendation pipeline for several minutes.
        """

        return await self._request_json(
            method="POST",
            url=url,
            form_data={
                "data": query,
            },
            cache_ttl_seconds=cache_ttl_seconds,
            request_timeout=self.OVERPASS_TIMEOUT_SECONDS,
            max_retries_override=0,
        )
