"""Collect AI news items from RSS feeds, APIs, and public Twitter mirrors."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
import feedparser
import requests
from dateutil import parser as date_parser

import config

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


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
class ScrapeResult:
    items: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


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
    source_key: str,
) -> None:
    added = 0
    for item in items[: config.MAX_ITEMS_PER_SOURCE]:
        k = item.key()
        if k in seen:
            continue
        seen.add(k)
        result.items.append(item)
        added += 1
    result.stats[source_key] = result.stats.get(source_key, 0) + added


def fetch_rss_feeds(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    session = _session()
    for name, url in config.RSS_FEEDS.items():
        try:
            resp = session.get(url, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
            if getattr(parsed, "bozo", False) and not parsed.entries:
                result.errors.append(f"RSS {name}: parse error")
                continue

            batch: list[NewsItem] = []
            for entry in parsed.entries:
                published = _parse_date(
                    getattr(entry, "published", None)
                    or getattr(entry, "updated", None)
                )
                if not _within_lookback(published, since):
                    continue
                link = getattr(entry, "link", "") or ""
                title = getattr(entry, "title", "").strip() or "(no title)"
                summary = _cut(
                    getattr(entry, "summary", None) or getattr(entry, "description", None)
                )
                batch.append(
                    NewsItem(
                        title=title,
                        url=link,
                        source=f"RSS: {name}",
                        published=published.isoformat() if published else None,
                        summary=summary,
                        category=_guess_category(title, summary or ""),
                    )
                )
            _add_items(result, batch, seen, f"rss:{name}")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"RSS {name}: {exc}")


def fetch_hf_trending_papers(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    """Hugging Face daily papers sorted by trending (free API)."""
    try:
        resp = _session().get(
            config.HF_DAILY_PAPERS_URL,
            params={"limit": 25, "sort": "trending"},
            timeout=config.REQUEST_TIMEOUT,
        )
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
                or paper.get("published_at")
            )
            if not _within_lookback(published, since):
                continue
            upvotes = paper.get("upvotes") or paper.get("numComments")
            batch.append(
                NewsItem(
                    title=title,
                    url=url,
                    source="API: Hugging Face Papers Trending",
                    published=published.isoformat() if published else None,
                    summary=_cut(paper.get("summary") or paper.get("abstract")),
                    category="research",
                    score=int(upvotes) if upvotes is not None else None,
                )
            )
        _add_items(result, batch, seen, "hf_papers")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Hugging Face Papers API: {exc}")


def fetch_hacker_news(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    """HN stories from the last week via Algolia (free)."""
    since_ts = int(since.timestamp())
    queries = ["AI", "LLM", "machine learning", "OpenAI", "Anthropic", "Claude"]
    batch: list[NewsItem] = []
    session = _session()

    for q in queries:
        try:
            resp = session.get(
                config.HN_ALGOLIA_URL,
                params={
                    "query": q,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{since_ts}",
                    "hitsPerPage": 15,
                },
                timeout=config.REQUEST_TIMEOUT,
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
                        source="API: Hacker News",
                        published=published,
                        summary=_cut(hit.get("story_text")),
                        category=_guess_category(title, ""),
                        score=hit.get("points"),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Hacker News ({q}): {exc}")

    _add_items(result, batch, seen, "hacker_news")


def fetch_arxiv(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    cat_query = "+OR+".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    params = {
        "search_query": cat_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 20,
    }
    session = _session()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            if attempt:
                time.sleep(4 * attempt)
            resp = session.get(
                config.ARXIV_API_URL,
                params=params,
                timeout=config.REQUEST_TIMEOUT,
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
                        source="API: ArXiv",
                        published=published_dt.isoformat() if published_dt else None,
                        summary=summary,
                        category="research",
                    )
                )
            _add_items(result, batch, seen, "arxiv")
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    result.errors.append(f"ArXiv: {last_exc}")


def fetch_reddit(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    """Reddit via public RSS (more reliable than JSON from CI runners)."""
    session = _session()
    batch: list[NewsItem] = []
    for sub in config.REDDIT_SUBREDDITS:
        url = f"{config.REDDIT_BASE}/r/{sub}/top/.rss"
        try:
            resp = session.get(
                url,
                params={"t": "week", "limit": 15},
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": config.USER_AGENT},
            )
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
            for entry in parsed.entries:
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "") or ""
                published = _parse_date(
                    getattr(entry, "published", None) or getattr(entry, "updated", None)
                )
                if not _within_lookback(published, since):
                    continue
                summary = _cut(
                    getattr(entry, "summary", None) or getattr(entry, "description", None)
                )
                batch.append(
                    NewsItem(
                        title=title,
                        url=link,
                        source=f"Reddit r/{sub}",
                        published=published.isoformat() if published else None,
                        summary=summary,
                        category=_guess_category(title, summary or ""),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Reddit r/{sub}: {exc}")

    _add_items(result, batch, seen, "reddit")


def fetch_twitter_profiles(result: ScrapeResult, seen: set[str], since: datetime) -> None:
    """Public X profiles via Nitter-style RSS mirrors (best-effort, free)."""
    session = _session()
    batch: list[NewsItem] = []

    for display_name, username in config.TWITTER_PROFILES.items():
        fetched = False
        for instance in config.NITTER_RSS_INSTANCES:
            feed_url = f"{instance.rstrip('/')}/{username}/rss"
            try:
                parsed = feedparser.parse(
                    feed_url,
                    agent=config.USER_AGENT,
                    request_headers={
                        "User-Agent": config.USER_AGENT,
                        "Accept": "application/rss+xml, application/xml, text/xml",
                    },
                )
                if not parsed.entries:
                    continue
                for entry in parsed.entries[:10]:
                    published = _parse_date(
                        getattr(entry, "published", None)
                        or getattr(entry, "updated", None)
                    )
                    if not _within_lookback(published, since):
                        continue
                    title = getattr(entry, "title", "").strip() or "(tweet)"
                    link = getattr(entry, "link", "") or ""
                    if link and "twitter.com" not in link and "x.com" not in link:
                        link = link.replace(instance, "https://twitter.com")
                    summary = _cut(
                        getattr(entry, "summary", None) or getattr(entry, "description", None)
                    )
                    batch.append(
                        NewsItem(
                            title=f"@{username}: {title}",
                            url=link,
                            source=f"X: {display_name}",
                            published=published.isoformat() if published else None,
                            summary=summary,
                            category="social",
                        )
                    )
                fetched = True
                break
            except Exception:
                continue
        if not fetched:
            result.errors.append(
                f"X/@{username}: no RSS mirror responded (Nitter instances may be down)"
            )

    _add_items(result, batch, seen, "twitter")


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


def scrape_all() -> ScrapeResult:
    since = _since_cutoff()
    result = ScrapeResult()
    seen: set[str] = set()

    fetch_rss_feeds(result, seen, since)
    fetch_hf_trending_papers(result, seen, since)
    fetch_hacker_news(result, seen, since)
    fetch_arxiv(result, seen, since)
    fetch_reddit(result, seen, since)
    fetch_twitter_profiles(result, seen, since)

    if len(result.items) > config.MAX_TOTAL_ITEMS:
        result.items.sort(key=lambda i: i.published or "", reverse=True)
        result.items = result.items[: config.MAX_TOTAL_ITEMS]

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
