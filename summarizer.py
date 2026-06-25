"""Generate structured weekly report using Claude API."""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from anthropic import Anthropic

import config
from scraper import NewsItem, ScrapeResult, items_to_prompt_blob

log = logging.getLogger(__name__)


def sanitize_plain_text(text: str) -> str:
    """Strip HTML/escaped markup from LLM or scrape text before rendering."""
    if not text:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"&lt;[^&]*?&gt;", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?span[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*dir="[^"]*"', "", cleaned)
    cleaned = re.sub(r'\s*style="[^"]*"', "", cleaned)
    cleaned = re.sub(r"unicode-bidi:\s*\w+;?", "", cleaned)
    cleaned = re.sub(r"display:\s*inline-block;?", "", cleaned)
    cleaned = re.sub(r"text-align:\s*\w+;?", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class ResearchItem:
    title: str
    summary: str
    why_it_matters: str


@dataclass
class ProductItem:
    title: str
    summary: str
    relevance: str


@dataclass
class BusinessItem:
    title: str
    summary: str
    why_it_matters: str


@dataclass
class TechnicalCorner:
    title: str
    explanation: str


@dataclass
class SourceRef:
    name: str
    url: str


@dataclass
class SourceCoverageDetail:
    source_name: str
    source_type: str
    status: str
    articles_collected: int
    failure_reason: str | None = None


@dataclass
class ScrapeStatusSummary:
    total_sources: int
    successful_sources: int
    failed_source_count: int
    coverage_percentage: int
    total_articles_collected: int
    source_details: list[SourceCoverageDetail] = field(default_factory=list)
    failed_source_list: list[dict[str, str]] = field(default_factory=list)

    @property
    def sources_scanned(self) -> int:
        return self.total_sources

    @property
    def sources_succeeded(self) -> int:
        return self.successful_sources

    @property
    def sources_failed(self) -> int:
        return self.failed_source_count

    @property
    def coverage_pct(self) -> int:
        return self.coverage_percentage

    def reliability_label(self, hebrew: bool = True) -> str:
        pct = self.coverage_percentage
        if hebrew:
            if pct >= 90:
                return "🟢 כיסוי גבוה"
            if pct >= 75:
                return "🟡 כיסוי בינוני"
            return "🔴 כיסוי נמוך"
        if pct >= 90:
            return "🟢 High coverage"
        if pct >= 75:
            return "🟡 Medium coverage"
        return "🔴 Low coverage"

    def completeness_text(self, hebrew: bool = True) -> str:
        pct = self.coverage_percentage
        if hebrew:
            if pct >= 90:
                return (
                    "הדוח מבוסס על רוב המקורות המתוכננים ולכן צפוי לשקף באופן מלא כמעט "
                    "את ההתפתחויות המרכזיות בשבוע האחרון."
                )
            if pct >= 75:
                return (
                    "מרבית המקורות נסרקו בהצלחה ולכן הדוח צפוי לשקף את רוב ההתפתחויות המרכזיות, "
                    "אך ייתכן שחלק מהעדכונים לא נכללו."
                )
            return (
                "מספר מקורות מרכזיים לא היו זמינים בזמן יצירת הדוח ולכן ייתכן "
                "שחלק מהעדכונים המשמעותיים לא נכללו."
            )
        if pct >= 90:
            return "The report reflects nearly all major developments from the past week."
        if pct >= 75:
            return "Most sources succeeded; some updates may be missing."
        return "Several key sources were unavailable; significant updates may be missing."

    def transparency_text(self, hebrew: bool = True) -> str:
        if hebrew:
            return (
                f"הדוח נוצר מתוך {self.total_articles_collected} פריטים שנאספו "
                f"מ-{self.successful_sources} מקורות פעילים לאחר סינון כפילויות, "
                "איחוד ידיעות דומות ותיעדוף לפי השפעה עסקית, מוצרית וטכנולוגית."
            )
        return (
            f"This report was built from {self.total_articles_collected} items collected "
            f"across {self.successful_sources} active sources after deduplication and prioritization."
        )


@dataclass
class ReportContent:
    report_date: str
    period_display: str
    period_start: str
    period_end: str
    executive_summary: list[str]
    models_research: list[ResearchItem]
    products_tools: list[ProductItem]
    business_market: list[BusinessItem]
    technical_corner: TechnicalCorner | None
    pm_takeaways: list[str]
    sources: list[SourceRef]
    scrape_status: ScrapeStatusSummary
    items_collected: int


SYSTEM_PROMPT = """You are an expert AI industry analyst preparing a concise weekly intelligence brief for a product manager.
Use ONLY the provided source items. Do not invent news.
Write in clear Hebrew when requested, but preserve English product/model names as-is (e.g. GPT-4, Claude, OpenAI).
Keep summaries short. Target reading time: 5 minutes.
Return valid JSON only. No markdown. No code fences. No HTML tags. Plain text strings only.
Limits: models_research max 3 items, products_tools max 5, business_market max 5, executive_summary 3-4 bullets, pm_takeaways 2-3 bullets.
Put English product/company names in the title field; keep summary and impact fields in Hebrew when Hebrew is requested."""


def _language_instruction() -> str:
    if config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית"):
        return (
            "Write all text fields in Hebrew (עברית). "
            "Keep English names for products, companies, models, and APIs."
        )
    return "Write all text fields in English."


def _build_user_prompt(scrape: ScrapeResult) -> str:
    now = datetime.now(config.LOCAL_TZ)
    period_end = now.strftime("%Y-%m-%d")
    blob = items_to_prompt_blob(scrape.items)

    return f"""{_language_instruction()}

Report period: last {config.LOOKBACK_DAYS} days (ending {period_end}).
Collected items: {len(scrape.items)}

SOURCE ITEMS:
{blob}

Return JSON with exactly this schema:
{{
  "executive_summary": ["bullet 1", "bullet 2"],
  "models_research": [
    {{"title": "...", "summary": "...", "why_it_matters": "..."}}
  ],
  "products_tools": [
    {{"title": "...", "summary": "...", "relevance": "..."}}
  ],
  "business_market": [
    {{"title": "...", "summary": "...", "why_it_matters": "..."}}
  ],
  "technical_corner": {{"title": "...", "explanation": "..."}},
  "pm_takeaways": ["actionable takeaway 1", "actionable takeaway 2"],
  "sources": [{{"name": "OpenAI News - Codex plugins", "url": ""}}]
}}

Rules:
- executive_summary: 3-4 short bullets
- models_research: maximum 3 items
- products_tools: maximum 5 items
- business_market: maximum 5 items
- pm_takeaways: 2-3 actionable bullets for a PM
- title fields: use English product/company names where appropriate
- summary/why_it_matters/relevance: Hebrew when Hebrew is requested
- sources: up to 8 key citations as plain text lines "Source Name - Article title" in the name field (no HTML, no URLs in name)
- Do NOT include scrape_status in your response
"""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _find_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _safe_response_preview(text: str, limit: int = 200) -> str:
    preview = re.sub(r"\s+", " ", text.strip())[:limit]
    return preview + ("..." if len(text.strip()) > limit else "")


def _extract_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Claude returned empty response")

    original = text.strip()
    candidates: list[str] = [original]
    unfenced = _strip_markdown_fences(original)
    if unfenced != original:
        candidates.append(unfenced)
    for source in (original, unfenced):
        span = _find_json_object(source)
        if span and span not in candidates:
            candidates.append(span)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc

    preview = _safe_response_preview(original)
    log.warning("Failed to parse Claude JSON response (preview): %s", preview)
    message = "Claude returned invalid JSON after extraction attempts"
    if last_error:
        message += f": {last_error.msg} at position {last_error.pos}"
    raise ValueError(message)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [sanitize_plain_text(str(v)) for v in value if sanitize_plain_text(str(v))]
    if isinstance(value, str) and value.strip():
        return [sanitize_plain_text(value)]
    return []


def _parse_research(items: Any) -> list[ResearchItem]:
    out: list[ResearchItem] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        title = sanitize_plain_text(str(row.get("title", "")))
        if not title:
            continue
        out.append(
            ResearchItem(
                title=title,
                summary=sanitize_plain_text(str(row.get("summary", ""))),
                why_it_matters=sanitize_plain_text(str(row.get("why_it_matters", ""))),
            )
        )
    return out


def _parse_products(items: Any) -> list[ProductItem]:
    out: list[ProductItem] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        title = sanitize_plain_text(str(row.get("title", "")))
        if not title:
            continue
        out.append(
            ProductItem(
                title=title,
                summary=sanitize_plain_text(str(row.get("summary", ""))),
                relevance=sanitize_plain_text(str(row.get("relevance", ""))),
            )
        )
    return out


def _parse_business(items: Any) -> list[BusinessItem]:
    out: list[BusinessItem] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        title = sanitize_plain_text(str(row.get("title", "")))
        if not title:
            continue
        out.append(
            BusinessItem(
                title=title,
                summary=sanitize_plain_text(str(row.get("summary", ""))),
                why_it_matters=sanitize_plain_text(str(row.get("why_it_matters", ""))),
            )
        )
    return out


def _parse_technical(value: Any) -> TechnicalCorner | None:
    if not isinstance(value, dict):
        return None
    title = sanitize_plain_text(str(value.get("title", "")))
    explanation = sanitize_plain_text(str(value.get("explanation", "")))
    if not title and not explanation:
        return None
    return TechnicalCorner(title=title or "Technical Corner", explanation=explanation)


def _format_period(now: datetime) -> tuple[str, str, str, str]:
    """Return (report_date, period_display, period_start, period_end)."""
    end = now.date()
    start = end - timedelta(days=config.LOOKBACK_DAYS - 1)
    he = config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית")
    if he:
        display = f"{start.strftime('%d.%m.%Y')} - {end.strftime('%d.%m.%Y')}"
    else:
        display = f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"
    report_date = now.strftime("%d %B %Y")
    return report_date, display, start.isoformat(), end.isoformat()


def _parse_sources(items: Any) -> list[SourceRef]:
    out: list[SourceRef] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        name = sanitize_plain_text(str(row.get("name", "")))
        url = sanitize_plain_text(str(row.get("url", "")))
        if name:
            out.append(SourceRef(name=name, url=url if url.startswith("http") else ""))
    return out


def _build_scrape_status(scrape: ScrapeResult) -> ScrapeStatusSummary:
    details = [
        SourceCoverageDetail(
            source_name=s.source_name,
            source_type=s.source_type,
            status=s.status,
            articles_collected=s.articles_collected,
            failure_reason=s.failure_reason,
        )
        for s in scrape.sources
    ]
    failed_list = [
        {
            "name": sanitize_plain_text(f["name"]),
            "error": sanitize_plain_text(f["error"]),
        }
        for f in scrape.failed_sources
    ]
    return ScrapeStatusSummary(
        total_sources=scrape.sources_scanned,
        successful_sources=scrape.sources_succeeded,
        failed_source_count=scrape.sources_failed,
        coverage_percentage=scrape.coverage_pct,
        total_articles_collected=scrape.total_articles_collected,
        source_details=details,
        failed_source_list=failed_list,
    )


def _is_hebrew_report() -> bool:
    return config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית")


def _source_label(item: NewsItem) -> str:
    return re.sub(r"^(RSS:\s*|API:\s*|X:\s*)", "", item.source).strip() or item.source


def _report_content_from_data(data: dict[str, Any], scrape: ScrapeResult) -> ReportContent:
    report_date, period_display, period_start, period_end = _format_period(
        datetime.now(config.LOCAL_TZ)
    )
    return ReportContent(
        report_date=report_date,
        period_display=period_display,
        period_start=period_start,
        period_end=period_end,
        executive_summary=_as_list(data.get("executive_summary")),
        models_research=_parse_research(data.get("models_research"))[:3],
        products_tools=_parse_products(data.get("products_tools"))[:5],
        business_market=_parse_business(data.get("business_market"))[:5],
        technical_corner=_parse_technical(data.get("technical_corner")),
        pm_takeaways=_as_list(data.get("pm_takeaways"))[:3],
        sources=_parse_sources(data.get("sources")),
        scrape_status=_build_scrape_status(scrape),
        items_collected=len(scrape.items),
    )


def _build_fallback_report(scrape: ScrapeResult) -> ReportContent:
    """Build a sendable report from scraped items when Claude JSON parsing fails."""
    log.warning("Using fallback report from scraped items because Claude JSON parsing failed")

    he = _is_hebrew_report()
    items = sorted(
        scrape.items,
        key=lambda i: (i.published or "", i.score or 0),
        reverse=True,
    )
    by_cat: dict[str, list[NewsItem]] = {
        "research": [],
        "products": [],
        "business": [],
        "general": [],
        "social": [],
    }
    for item in items:
        bucket = item.category if item.category in by_cat else "general"
        by_cat[bucket].append(item)

    if he:
        summary_default = "פריט שנאסף מהמקורות השבוע."
        why_research = "עדכון רלוונטי מתחום המחקר והמודלים."
        why_products = "עדכון רלוונטי לכלי מוצר ופיתוח."
        why_business = "עדכון רלוונטי לשוק ולעסקים."
        executive_summary = [
            f"נאספו {len(scrape.items)} פריטים מ-{scrape.sources_succeeded} מקורות פעילים בשבוע האחרון.",
            "להלן עיקרי העדכונים שנאספו אוטומטית מהמקורות.",
        ]
        pm_takeaways = [
            "עיין בפריטים המפורטים למטה לפי קטגוריה.",
            "כדאי לעקוב אחר מקורות המחקר והמוצרים לעדכונים נוספים.",
        ]
    else:
        summary_default = "Item collected from sources this week."
        why_research = "Relevant research and models update."
        why_products = "Relevant product and tooling update."
        why_business = "Relevant business and market update."
        executive_summary = [
            f"Collected {len(scrape.items)} items from {scrape.sources_succeeded} active sources this week.",
            "Below are key updates gathered automatically from sources.",
        ]
        pm_takeaways = [
            "Review the categorized items below for this week's highlights.",
            "Follow research and product sources for additional updates.",
        ]

    if items:
        executive_summary.append(items[0].title)

    models_research = [
        ResearchItem(
            title=item.title,
            summary=item.summary or summary_default,
            why_it_matters=why_research,
        )
        for item in by_cat["research"][:3]
    ]
    products_tools = [
        ProductItem(
            title=item.title,
            summary=item.summary or summary_default,
            relevance=why_products,
        )
        for item in (by_cat["products"] + by_cat["general"] + by_cat["social"])[:5]
    ]
    business_market = [
        BusinessItem(
            title=item.title,
            summary=item.summary or summary_default,
            why_it_matters=why_business,
        )
        for item in by_cat["business"][:5]
    ]

    used_titles = {
        *(i.title for i in models_research),
        *(i.title for i in products_tools),
        *(i.title for i in business_market),
    }
    for item in items:
        if len(models_research) >= 3 and len(products_tools) >= 5 and len(business_market) >= 5:
            break
        if item.title in used_titles:
            continue
        used_titles.add(item.title)
        if len(models_research) < 3:
            models_research.append(
                ResearchItem(
                    title=item.title,
                    summary=item.summary or summary_default,
                    why_it_matters=why_research,
                )
            )
        elif len(products_tools) < 5:
            products_tools.append(
                ProductItem(
                    title=item.title,
                    summary=item.summary or summary_default,
                    relevance=why_products,
                )
            )
        elif len(business_market) < 5:
            business_market.append(
                BusinessItem(
                    title=item.title,
                    summary=item.summary or summary_default,
                    why_it_matters=why_business,
                )
            )

    sources = [
        SourceRef(
            name=f"{_source_label(item)} - {item.title}",
            url=item.url if item.url.startswith("http") else "",
        )
        for item in items[:8]
    ]

    report_date, period_display, period_start, period_end = _format_period(
        datetime.now(config.LOCAL_TZ)
    )
    return ReportContent(
        report_date=report_date,
        period_display=period_display,
        period_start=period_start,
        period_end=period_end,
        executive_summary=executive_summary[:4],
        models_research=models_research[:3],
        products_tools=products_tools[:5],
        business_market=business_market[:5],
        technical_corner=None,
        pm_takeaways=pm_takeaways[:3],
        sources=sources,
        scrape_status=_build_scrape_status(scrape),
        items_collected=len(scrape.items),
    )


def summarize(scrape: ScrapeResult) -> ReportContent:
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = config.ANTHROPIC_MODEL
    print(f"Using Anthropic model: {model}")
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(scrape)}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Claude API call failed (model={model}): {exc}"
        ) from exc

    raw = ""
    for block in message.content:
        if hasattr(block, "text"):
            raw += block.text

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Claude JSON parsing failed, using fallback report: %s", exc)
        return _build_fallback_report(scrape)

    return _report_content_from_data(data, scrape)
