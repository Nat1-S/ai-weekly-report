"""Collect AI news items from RSS feeds, APIs, and public Twitter mirrors."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

import config
from fetch_client import get_http_client

log = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: str | None = None
    summary: str | None = None
    category: str = "general"
    score: int | None = None

    def key(self) -> str:
        return self.url.strip().lower() or f"{self.source}:{self.title}".lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceResult:
    source_name: str
    source_type: str
    status: str  # "success" | "failed"
    articles_collected: int = 0
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def items_count(self) -> int:
        return self.articles_collected

    @property
    def error(self) -> str | None:
        return self.failure_reason


@dataclass
class ScrapeResult:
    items: list[NewsItem] = field(default_factory=list)
    sources: list[SourceResult] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def sources_scanned(self) -> int:
        return len(self.sources)

    @property
    def sources_succeeded(self) -> int:
        return sum(1 for s in self.sources if s.succeeded)

    @property
    def sources_failed(self) -> int:
        return sum(1 for s in self.sources if not s.succeeded)

    @property
    def coverage_pct(self) -> int:
        if not self.sources_scanned:
            return 0
        return round(100 * self.sources_succeeded / self.sources_scanned)

    @property
    def total_articles_collected(self) -> int:
        return len(self.items)

    @property
    def failed_sources(self) -> list[dict[str, str]]:
        return [
            {
                "name": _display_source_name(s.source_name),
                "error": s.failure_reason or "Unknown error",
            }
            for s in self.sources
            if s.status == "failed"
        ]

    # Backward-compatible alias for logging
    @property
    def errors(self) -> list[str]:
        return [f"{s.name}: {s.error}" for s in self.sources if not s.succeeded and s.error]


def _display_source_name(name: str) -> str:
    cleaned = re.sub(r"^(RSS:\s*|API:\s*|X:\s*)", "", name).strip()
    cleaned = cleaned.replace("r/", "")
    return cleaned or name


def _humanize_error(msg: str | None) -> str:
    if not msg:
        return "Unknown error"
    text = re.sub(r"<[^>]+>", "", msg)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    lower = text.lower()
    if "403" in text or "forbidden" in lower or "blocked" in lower:
        return "HTTP 403"
    if "429" in text or "rate limit" in lower:
        return "Rate Limit"
    if "timeout" in lower or "timed out" in lower:
        return "Timeout"
    if "parse" in lower:
        return "Parse Error"
    if "no rss mirror" in lower or "unavailable" in lower:
        return "Unavailable"
    if "connection" in lower:
        return "Connection Error"
    if len(text) > 60:
        return text[:57] + "..."
    return text or "Unknown error"


def _record_source(
    result: ScrapeResult,
    source_name: str,
    source_type: str,
    succeeded: bool,
    articles_collected: int = 0,
    error: str | None = None,
    stat_key: str | None = None,
) -> None:
    result.sources.append(
        SourceResult(
            source_name=source_name,
            source_type=source_type,
            status="success" if succeeded else "failed",
            articles_collected=articles_collected,
            failure_reason=_humanize_error(error) if error else None,
        )
    )
    if stat_key and articles_collected:
        result.stats[stat_key] = result.stats.get(stat_key, 0) + articles_collected


def _snippet_from_html(text: str | None) -> str | None:
    if not text:
        return None
    plain = BeautifulSoup(str(text), "html.parser").get_text(" ", strip=True)
    return _cut(plain)


def _cut(text: str | None, limit: int = config.MAX_ITEM_SUMMARY_CHARS) -> str | None:
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if hasattr(value, "tm_year"):
            return datetime(*value[:6], tzinfo=timezone.utc)
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        dt = date_parser.parse(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _within_lookback(dt: datetime | None, since: datetime) -> bool:
    if dt is None:
        return True
    return dt >= since


def _since_cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)


def _add_items(
    result: ScrapeResult,
    items: list[NewsItem],
    seen: set[str],
) -> list[NewsItem]:
    added: list[NewsItem] = []
    for item in items[: config.MAX_ITEMS_PER_SOURCE]:
        k = item.key()
        if k in seen:
            continue
        seen.add(k)
        result.items.append(item)
        added.append(item)
    return added


def _entries_from_feed(content: bytes) -> list[Any]:
    parsed = feedparser.parse(content)
    if not parsed.entries:
        return []
    return list(parsed.entries)


def _items_from_feed_entries(
    entries: list[Any],
    source_name: str,
    since: datetime,
) -> list[NewsItem]:
    batch: list[NewsItem] = []
    for entry in entries:
        published = _parse_date(
            getattr(entry, "published", None) or getattr(entry, "updated", None)
        )
        if not _within_lookback(published, since):
            continue
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "").strip() or "(no title)"
        raw_summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
        summary = _snippet_from_html(raw_summary)
        batch.append(
            NewsItem(
                title=title,
                url=link,
                source=source_name,
                published=published.isoformat() if published else None,
                summary=summary,
                category=_guess_category(title, summary or ""),
            )
        )
    return batch


def _scrape_html_listing_only(
    page_url: str,
    source_name: str,
    since: datetime,
) -> list[NewsItem]:
    """Parse a single listing page for title/url only (no per-article fetches)."""
    client = get_http_client()
    resp = client.fetch(page_url, source_name, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    seen_urls: set[str] = set()
    batch: list[NewsItem] = []

    selectors = [
        "article h2 a",
        "article h3 a",
        "article a",
        ".post-title a",
        "h2 a",
        "h3 a",
    ]
    for selector in selectors:
        for link in soup.select(selector):
            href = link.get("href") or ""
            title = link.get_text(" ", strip=True)
            if not href or not title or len(title) < 12:
                continue
            url = urljoin(page_url, href)
            if url in seen_urls or not url.startswith("http"):
                continue
            if any(skip in url for skip in ("/tag/", "/author/", "/category/", "#", "javascript:")):
                continue
            seen_urls.add(url)
            batch.append(
                NewsItem(
                    title=title,
                    url=url,
                    source=f"{source_name} (HTML)",
                    published=None,
                    summary=None,
                    category=_guess_category(title, ""),
                )
            )
            if len(batch) >= config.MAX_ITEMS_PER_SOURCE:
                return batch
        if batch:
            return batch
    return batch


def fetch_rss_feeds(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    client = get_http_client()
    for name, url in config.RSS_FEEDS.items():
        source_label = f"RSS: {name}"
        batch: list[NewsItem] = []
        error: str | None = None
        try:
            resp = client.fetch(url, source_label, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                resp.raise_for_status()
            entries = _entries_from_feed(resp.content)
            if entries:
                batch = _items_from_feed_entries(entries, source_label, since)
            else:
                error = "RSS parse error"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        if not batch and name in config.RSS_HTML_FALLBACKS:
            try:
                batch = _scrape_html_listing_only(
                    config.RSS_HTML_FALLBACKS[name], source_label, since
                )
                if batch:
                    error = None
            except Exception as exc:  # noqa: BLE001
                error = error or str(exc)

        added = _add_items(result, batch, seen)
        _record_source(
            result,
            source_name=name,
            source_type="RSS",
            succeeded=bool(added),
            articles_collected=len(added),
            error=error if not added else None,
            stat_key=f"rss:{name}",
        )


def fetch_hf_trending_papers(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    name = "Hugging Face Papers Trending"
    try:
        client = get_http_client()
        url = (
            f"{config.HF_DAILY_PAPERS_URL}?limit=25&sort=trending"
        )
        resp = client.fetch(url, name, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        batch: list[NewsItem] = []
        for row in data if isinstance(data, list) else []:
            paper = row.get("paper") or row
            title = (row.get("title") or paper.get("title") or "").strip()
            if not title:
                continue
            arxiv_id = paper.get("id") or paper.get("arxivId") or ""
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "https://huggingface.co/papers"
            published = _parse_date(
                row.get("publishedAt")
                or paper.get("submittedOnDailyAt")
                or paper.get("publishedAt")
            )
            if not _within_lookback(published, since):
                continue
            upvotes = paper.get("upvotes") or paper.get("numComments")
            batch.append(
                NewsItem(
                    title=title,
                    url=url,
                    source=f"API: {name}",
                    published=published.isoformat() if published else None,
                    summary=_cut(paper.get("summary") or paper.get("abstract")),
                    category="research",
                    score=int(upvotes) if upvotes is not None else None,
                )
            )
        added = _add_items(result, batch, seen)
        _record_source(
            result, name, "API", bool(added), len(added), stat_key="hf_papers"
        )
    except Exception as exc:  # noqa: BLE001
        _record_source(result, name, "API", False, 0, str(exc))


def fetch_hacker_news(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    name = "Hacker News"
    since_ts = int(since.timestamp())
    queries = ["AI", "LLM", "machine learning", "OpenAI", "Anthropic"]
    batch: list[NewsItem] = []
    client = get_http_client()
    errors: list[str] = []

    for q in queries:
        try:
            resp = client.fetch(
                config.HN_ALGOLIA_URL,
                f"{name} ({q})",
                timeout=config.REQUEST_TIMEOUT,
                params={
                    "query": q,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{since_ts}",
                    "hitsPerPage": 10,
                },
            )
            resp.raise_for_status()
            for hit in resp.json().get("hits", []):
                title = hit.get("title", "").strip()
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                created = hit.get("created_at_i")
                published = (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created
                    else None
                )
                batch.append(
                    NewsItem(
                        title=title,
                        url=url,
                        source=f"API: {name}",
                        published=published,
                        summary=_cut(hit.get("story_text")),
                        category=_guess_category(title, ""),
                        score=hit.get("points"),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    added = _add_items(result, batch, seen)
    _record_source(
        result,
        source_name=name,
        source_type="API",
        succeeded=bool(added),
        articles_collected=len(added),
        error="; ".join(errors) if not added and errors else None,
        stat_key="hacker_news",
    )


def fetch_arxiv(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    name = "ArXiv"
    cat_query = "+OR+".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    params = {
        "search_query": cat_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 20,
    }
    client = get_http_client()
    last_exc: Exception | None = None

    for attempt in range(2):
        try:
            resp = client.fetch(
                config.ARXIV_API_URL,
                name,
                timeout=config.ARXIV_TIMEOUT,
                params=params,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            batch: list[NewsItem] = []
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                link_el = entry.find("atom:id", ns)
                url = (link_el.text or "").strip() if link_el is not None else ""
                published_el = entry.find("atom:published", ns)
                published_dt = _parse_date(
                    published_el.text if published_el is not None else None
                )
                if not _within_lookback(published_dt, since):
                    continue
                summary_el = entry.find("atom:summary", ns)
                summary = _cut(summary_el.text if summary_el is not None else None)
                batch.append(
                    NewsItem(
                        title=title,
                        url=url,
                        source=f"API: {name}",
                        published=published_dt.isoformat() if published_dt else None,
                        summary=summary,
                        category="research",
                    )
                )
            added = _add_items(result, batch, seen)
            _record_source(result, name, "API", bool(added), len(added), stat_key="arxiv")
            return
        except requests.Timeout as exc:
            last_exc = exc
            client.record_retry()
            log.warning("ArXiv timeout (attempt %d/2): %s", attempt + 1, exc)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            break

    _record_source(result, name, "API", False, 0, str(last_exc))


def fetch_reddit(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    client = get_http_client()

    for sub in config.REDDIT_SUBREDDITS:
        name = f"Reddit r/{sub}"
        url = f"{config.REDDIT_BASE}/r/{sub}/top/.rss"
        batch: list[NewsItem] = []
        error: str | None = None
        try:
            resp = client.fetch(
                url,
                name,
                timeout=config.REQUEST_TIMEOUT,
                params={"t": "week", "limit": 15},
                extra_headers={"User-Agent": config.REDDIT_USER_AGENT},
            )
            if resp.status_code == 403:
                raise requests.HTTPError("Blocked by Reddit (403)")
            resp.raise_for_status()
            entries = _entries_from_feed(resp.content)
            batch = _items_from_feed_entries(entries, name, since)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        added = _add_items(result, batch, seen)
        _record_source(
            result,
            source_name=f"Reddit {sub}",
            source_type="RSS",
            succeeded=bool(added),
            articles_collected=len(added),
            error=error if not added else None,
            stat_key=f"reddit:{sub}",
        )


def fetch_twitter_profiles(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    for display_name, username in config.TWITTER_PROFILES.items():
        name = f"X: {display_name}"
        batch: list[NewsItem] = []
        error: str | None = None

        for instance in config.NITTER_RSS_INSTANCES:
            feed_url = f"{instance.rstrip('/')}/{username}/rss"
            try:
                resp = get_http_client().fetch(
                    feed_url,
                    name,
                    timeout=config.REQUEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    continue
                entries = _entries_from_feed(resp.content)
                for entry in entries[:10]:
                    published = _parse_date(
                        getattr(entry, "published", None) or getattr(entry, "updated", None)
                    )
                    if not _within_lookback(published, since):
                        continue
                    title = getattr(entry, "title", "").strip() or "(tweet)"
                    link = getattr(entry, "link", "") or ""
                    summary = _snippet_from_html(
                        getattr(entry, "summary", None) or getattr(entry, "description", None)
                    )
                    batch.append(
                        NewsItem(
                            title=f"@{username}: {title}",
                            url=link,
                            source=name,
                            published=published.isoformat() if published else None,
                            summary=summary,
                            category="social",
                        )
                    )
                if batch:
                    break
            except Exception:
                continue

        if not batch:
            error = "No RSS mirror responded"

        added = _add_items(result, batch, seen)
        _record_source(
            result,
            source_name=display_name,
            source_type="Social",
            succeeded=bool(added),
            articles_collected=len(added),
            error=error if not added else None,
            stat_key=f"twitter:{username}",
        )


def _guess_category(title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    research_kw = ("paper", "arxiv", "model", "benchmark", "dataset", "training", "research")
    product_kw = ("launch", "release", "tool", "api", "open-source", "github", "product")
    business_kw = ("funding", "acquisition", "ipo", "market", "billion", "investment", "regulation")
    if any(k in text for k in research_kw):
        return "research"
    if any(k in text for k in product_kw):
        return "products"
    if any(k in text for k in business_kw):
        return "business"
    return "general"


def _safe_fetch(name: str, fn, result: ScrapeResult, seen: set[str], since: datetime) -> None:
    try:
        fn(result, seen, since)
    except Exception as exc:  # noqa: BLE001
        log.exception("Source collector crashed (%s): %s", name, exc)


def scrape_all() -> ScrapeResult:
    since = _since_cutoff()
    result = ScrapeResult()
    seen: set[str] = set()
    client = get_http_client()

    collectors = (
        ("rss", fetch_rss_feeds),
        ("hf_papers", fetch_hf_trending_papers),
        ("hacker_news", fetch_hacker_news),
        ("arxiv", fetch_arxiv),
        ("reddit", fetch_reddit),
        ("twitter", fetch_twitter_profiles),
    )
    for label, fn in collectors:
        _safe_fetch(label, fn, result, seen, since)

    if len(result.items) > config.MAX_TOTAL_ITEMS:
        result.items.sort(key=lambda i: i.published or "", reverse=True)
        result.items = result.items[: config.MAX_TOTAL_ITEMS]

    client.log_stats()
    return result


def items_to_prompt_blob(items: list[NewsItem]) -> str:
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        lines.append(f"[{i}] ({item.source}) {item.title}")
        if item.url:
            lines.append(f"    URL: {item.url}")
        if item.published:
            lines.append(f"    Date: {item.published}")
        if item.score is not None:
            lines.append(f"    Score: {item.score}")
        if item.summary:
            lines.append(f"    Snippet: {item.summary}")
        lines.append(f"    Suggested bucket: {item.category}")
        lines.append("")
    return "\n".join(lines)
