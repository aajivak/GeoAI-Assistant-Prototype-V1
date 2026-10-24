"""Asynchronous HTTP client with retries, timeouts, caching, and custom error handling.

Engineered to handle public rate-limited endpoints (Nominatim, Overpass, Open-Meteo) gracefully.
"""
import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional
import httpx

from config.settings import settings

logger = logging.getLogger("geoai.http_client")


class APIError(Exception):
    """Base exception for external API communication errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class RateLimitError(APIError):
    """Raised when an external API returns HTTP 429 (Too Many Requests)."""
    pass


class NotFoundError(APIError):
    """Raised when a requested resource is not found (HTTP 404 or empty geocode)."""
    pass


class GeocodingError(APIError):
    """Raised when geocoding fails or location is ambiguous/unresolvable."""
    pass


class SimpleMemoryCache:
    """Thread-safe in-memory cache with TTL."""
    def __init__(self):
        self._cache: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            expires_at, val = self._cache[key]
            if time.time() < expires_at:
                return val
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._cache[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        self._cache.clear()


# Shared global memory cache for API responses
_memory_cache = SimpleMemoryCache()


class AsyncHttpClient:
    """Robust async HTTP client wrapper over httpx.AsyncClient."""

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
    ):
        self.timeout = timeout_seconds or settings.HTTP_TIMEOUT_SECONDS
        self.max_retries = max_retries or settings.HTTP_MAX_RETRIES
        self.backoff_factor = backoff_factor or settings.HTTP_BACKOFF_FACTOR
        self.headers = {"User-Agent": settings.USER_AGENT}

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Execute an asynchronous GET request with caching, timeouts, and exponential backoff."""
        cache_key = None
        if cache_ttl_seconds and cache_ttl_seconds > 0:
            params_str = json.dumps(params or {}, sort_keys=True)
            cache_key = f"GET:{url}:{params_str}"
            cached_data = _memory_cache.get(cache_key)
            if cached_data is not None:
                logger.debug("Cache hit for URL: %s", url)
                return cached_data

        merged_headers = {**self.headers, **(headers or {})}
        timeout_config = httpx.Timeout(
            timeout=self.timeout,
            connect=settings.HTTP_CONNECT_TIMEOUT_SECONDS,
        )

        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.get(url, params=params, headers=merged_headers)

                    if response.status_code == 200:
                        data = response.json()
                        if cache_key and cache_ttl_seconds:
                            _memory_cache.set(cache_key, data, cache_ttl_seconds)
                        return data

                    if response.status_code == 429:
                        wait_time = self.backoff_factor ** attempt
                        logger.warning(
                            "Rate limited (429) on %s. Retrying in %.2fs (attempt %d/%d)...",
                            url, wait_time, attempt, self.max_retries
                        )
                        if attempt == self.max_retries:
                            raise RateLimitError(
                                f"Rate limit exceeded on {url} after {self.max_retries} retries.",
                                status_code=429,
                                response_text=response.text,
                            )
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code in (500, 502, 503, 504):
                        wait_time = self.backoff_factor ** attempt
                        logger.warning(
                            "Server error (%d) on %s. Retrying in %.2fs (attempt %d/%d)...",
                            response.status_code, url, wait_time, attempt, self.max_retries
                        )
                        if attempt == self.max_retries:
                            raise APIError(
                                f"Server error {response.status_code} on {url}",
                                status_code=response.status_code,
                                response_text=response.text,
                            )
                        await asyncio.sleep(wait_time)
                        continue

                    raise APIError(
                        f"HTTP {response.status_code} error from {url}: {response.text[:200]}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt
                logger.warning(
                    "Network/Timeout exception '%s' on %s. Retrying in %.2fs (attempt %d/%d)...",
                    str(e), url, wait_time, attempt, self.max_retries
                )
                if attempt == self.max_retries:
                    raise APIError(f"Request failed after {self.max_retries} attempts: {str(e)}") from e
                await asyncio.sleep(wait_time)

        if last_exception:
            raise APIError(f"Request failed: {str(last_exception)}") from last_exception
        raise APIError("Unknown HTTP failure")

    async def post_overpass(
        self,
        url: str,
        query: str,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Execute an asynchronous POST request to Overpass API with snappy 6.0s timeout."""
        cache_key = None
        if cache_ttl_seconds and cache_ttl_seconds > 0:
            cache_key = f"OVERPASS:{query}"
            cached_data = _memory_cache.get(cache_key)
            if cached_data is not None:
                logger.debug("Cache hit for Overpass query")
                return cached_data

        # 6s timeout to avoid hanging when public server is overloaded
        timeout_config = httpx.Timeout(timeout=6.0, connect=3.0)

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    url,
                    data={"data": query},
                    headers=self.headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    if cache_key and cache_ttl_seconds:
                        _memory_cache.set(cache_key, data, cache_ttl_seconds)
                    return data

                if response.status_code in (429, 502, 503, 504):
                    raise RateLimitError(
                        f"Overpass API busy/throttled ({response.status_code}) on {url}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )

                raise APIError(
                    f"Overpass returned HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    response_text=response.text,
                )

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise APIError(f"Overpass connection timed out on {url}") from e
