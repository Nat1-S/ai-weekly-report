"""HTTP client with throttling, per-domain serialization, cache, and retries."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import requests

import config

log = logging.getLogger(__name__)


@dataclass
class HttpResult:
    url: str
    status_code: int
    content: bytes
    headers: dict[str, str]

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} for {self.url}",
                response=requests.Response(),
            )


@dataclass
class FetchStats:
    requests_per_domain: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    rate_limit_events: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def log_summary(self) -> None:
        log.info(
            "HTTP stats: requests=%s retries=%d rate_limits=%d cache_hits=%d cache_misses=%d",
            dict(self.requests_per_domain),
            self.retries,
            self.rate_limit_events,
            self.cache_hits,
            self.cache_misses,
        )


def _resolved_url(url: str, params: dict[str, Any] | None) -> str:
    req = requests.Request("GET", url, params=params).prepare()
    return req.url or url


def _normalize_domain(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_throttle_seconds(domain: str) -> float:
    for pattern, seconds in config.DOMAIN_THROTTLE_SECONDS.items():
        if domain == pattern or domain.endswith(f".{pattern}") or pattern in domain:
            return float(seconds)
    return random.uniform(config.THROTTLE_DELAY_MIN, config.THROTTLE_DELAY_MAX)


def _parse_retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _is_rate_limited(status_code: int, body_sample: str) -> bool:
    if status_code == 429:
        return True
    lower = body_sample[:500].lower()
    return "rate limit" in lower or "too many requests" in lower


class HttpFetchClient:
    """Per-domain serialized HTTP GET with daily cache and rate-limit retries."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": config.USER_AGENT,
                "Accept": config.HTTP_ACCEPT,
                "Accept-Language": config.HTTP_ACCEPT_LANGUAGE,
            }
        )
        self._domain_locks: dict[str, Lock] = {}
        self._last_request_at: dict[str, float] = {}
        self._stats = FetchStats()
        if config.HTTP_CACHE_ENABLED:
            config.HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _lock_for(self, domain: str) -> Lock:
        if domain not in self._domain_locks:
            self._domain_locks[domain] = Lock()
        return self._domain_locks[domain]

    def _throttle(self, domain: str) -> None:
        delay = _domain_throttle_seconds(domain)
        last = self._last_request_at.get(domain, 0.0)
        wait = delay - (time.monotonic() - last)
        if wait > 0:
            log.debug("Throttling %s for %.1fs", domain, wait)
            time.sleep(wait)
        self._last_request_at[domain] = time.monotonic()

    def _cache_path(self, source_name: str, url: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        digest = hashlib.sha256(f"{source_name}|{url}|{day}".encode()).hexdigest()
        return config.HTTP_CACHE_DIR / f"{digest}.json"

    def _read_cache(self, source_name: str, url: str) -> HttpResult | None:
        if not config.HTTP_CACHE_ENABLED:
            return None
        path = self._cache_path(source_name, url)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            content = base64.b64decode(payload["content_b64"])
            return HttpResult(
                url=url,
                status_code=int(payload["status_code"]),
                content=content,
                headers=payload.get("headers") or {},
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            log.debug("Cache read failed for %s: %s", url, exc)
            return None

    def _write_cache(self, source_name: str, url: str, result: HttpResult) -> None:
        if not config.HTTP_CACHE_ENABLED or result.status_code != 200:
            return
        path = self._cache_path(source_name, url)
        payload = {
            "source_name": source_name,
            "url": url,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "status_code": result.status_code,
            "content_b64": base64.b64encode(result.content).decode("ascii"),
            "headers": result.headers,
        }
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            log.debug("Cache write failed for %s: %s", url, exc)

    def _record_request(self, domain: str) -> None:
        self._stats.requests_per_domain[domain] = (
            self._stats.requests_per_domain.get(domain, 0) + 1
        )

    def fetch(
        self,
        url: str,
        source_name: str,
        *,
        timeout: int | None = None,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        allow_cache: bool = True,
    ) -> HttpResult:
        """GET with per-domain lock, throttle, cache, and single retry on rate limits."""
        resolved = _resolved_url(url, params)
        domain = _normalize_domain(resolved)
        timeout = timeout or config.REQUEST_TIMEOUT

        with self._lock_for(domain):
            if allow_cache:
                cached = self._read_cache(source_name, resolved)
                if cached is not None:
                    self._stats.cache_hits += 1
                    log.debug("Cache hit: %s (%s)", source_name, resolved)
                    return cached
                self._stats.cache_misses += 1

            self._throttle(domain)
            result = self._request_once(
                resolved, url, source_name, domain, timeout, params, extra_headers
            )

            if _is_rate_limited(result.status_code, result.text):
                self._stats.rate_limit_events += 1
                wait = _parse_retry_after(result.headers) or float(
                    config.RATE_LIMIT_RETRY_SECONDS
                )
                log.warning(
                    "Rate limit on %s (%s); waiting %.0fs then retrying once",
                    source_name,
                    domain,
                    wait,
                )
                time.sleep(wait)
                self._throttle(domain)
                self._stats.retries += 1
                self._record_request(domain)
                result = self._request_once(
                    resolved, url, source_name, domain, timeout, params, extra_headers
                )

            if result.status_code == 200 and allow_cache:
                self._write_cache(source_name, resolved, result)

            return result

    def _request_once(
        self,
        resolved_url: str,
        url: str,
        source_name: str,
        domain: str,
        timeout: int,
        params: dict[str, Any] | None,
        extra_headers: dict[str, str] | None,
    ) -> HttpResult:
        self._record_request(domain)
        headers = dict(extra_headers or {})
        log.debug("Fetching %s (%s)", source_name, resolved_url)
        resp = self._session.get(
            url,
            params=params,
            timeout=timeout,
            allow_redirects=True,
            headers=headers or None,
        )
        return HttpResult(
            url=resolved_url,
            status_code=resp.status_code,
            content=resp.content,
            headers={k: v for k, v in resp.headers.items()},
        )

    def record_retry(self) -> None:
        self._stats.retries += 1

    def log_stats(self) -> None:
        self._stats.log_summary()


_client: HttpFetchClient | None = None


def get_http_client() -> HttpFetchClient:
    global _client
    if _client is None:
        _client = HttpFetchClient()
    return _client


def reset_http_client() -> None:
    """Reset singleton (useful for tests)."""
    global _client
    _client = None
