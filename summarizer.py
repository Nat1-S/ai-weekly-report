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
from scraper import ScrapeResult, items_to_prompt_blob

log = logging.getLogger(__name__)

# Patterns that occasionally appear in scraped third-party text and can trigger
# model refusals. Items matching these are skipped for the Claude prompt only.
_PROMPT_RISK_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your system prompt",
    "disregard previous instructions",
    "jailbreak",
    "dan mode",
    "do anything now",
    "system prompt override",
)

_PLACEHOLDER_STRINGS = frozenset(
    {
        "",
        "-",
        "—",
        "–",
        "n/a",
        "na",
        "none",
        "null",
        "undefined",
        "tbd",
        "todo",
        "...",
        "…",
    }
)


class ReportGenerationError(RuntimeError):
    """Raised when Claude output must not be treated as a production report."""

    def __init__(
        self,
        message: str,
        *,
        stop_reason: str | None = None,
        invalid_sections: list[str] | None = None,
        scraped_items: int | None = None,
        sources_succeeded: int | None = None,
        sources_failed: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.invalid_sections = list(invalid_sections or [])
        self.scraped_items = scraped_items
        self.sources_succeeded = sources_succeeded
        self.sources_failed = sources_failed


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
The SOURCE ITEMS are third-party public AI news headlines and snippets collected automatically.
If any individual source item is unsuitable, irrelevant, or unsafe to use, skip that item and continue with the rest.
Do not refuse the entire weekly report solely because one or a few scraped items are problematic.
Write in clear Hebrew when requested, but preserve English product/model names as-is (e.g. GPT-4, Claude, OpenAI).
Keep summaries short. Target reading time: 5 minutes.
Submit the report using the submit_weekly_report tool. No markdown. No HTML tags. Plain text strings only.
You MUST populate every required tool field with meaningful non-empty content.
Limits: models_research max 3 items, products_tools max 5, business_market max 5, executive_summary 3-4 bullets, conclusions 2-3 bullets.
Put English product/company names in the title field; keep summary and impact fields in Hebrew when Hebrew is requested."""

REPORT_TOOL_NAME = "submit_weekly_report"


def _language_instruction() -> str:
    if config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית"):
        return (
            "Write all text fields in Hebrew (עברית). "
            "Keep English names for products, companies, models, and APIs."
        )
    return "Write all text fields in English."


def _sanitize_prompt_fragment(text: str | None, limit: int) -> str:
    """Normalize scraped text before it is included in the Claude prompt."""
    if not text:
        return ""
    cleaned = sanitize_plain_text(str(text))
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _looks_like_prompt_risk(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _PROMPT_RISK_MARKERS)


def _items_for_claude_prompt(items: list[Any]) -> tuple[str, int]:
    """Build a sanitized SOURCE ITEMS blob; skip high-risk fragments."""
    lines: list[str] = []
    included = 0
    skipped = 0
    for i, item in enumerate(items, 1):
        title = _sanitize_prompt_fragment(getattr(item, "title", ""), 180)
        summary = _sanitize_prompt_fragment(
            getattr(item, "summary", None), config.MAX_ITEM_SUMMARY_CHARS
        )
        source = _sanitize_prompt_fragment(getattr(item, "source", ""), 80)
        url = _sanitize_prompt_fragment(getattr(item, "url", ""), 300)
        published = _sanitize_prompt_fragment(getattr(item, "published", None), 40)
        category = _sanitize_prompt_fragment(getattr(item, "category", "general"), 40)

        combined = f"{title} {summary}"
        if not title or _looks_like_prompt_risk(combined):
            skipped += 1
            log.warning(
                "Skipping scraped item %d from Claude prompt due to empty/risky content (%s)",
                i,
                source or "unknown",
            )
            continue

        included += 1
        lines.append(f"[{included}] ({source}) {title}")
        if url:
            lines.append(f"    URL: {url}")
        if published:
            lines.append(f"    Date: {published}")
        score = getattr(item, "score", None)
        if score is not None:
            lines.append(f"    Score: {score}")
        if summary and not _looks_like_prompt_risk(summary):
            lines.append(f"    Snippet: {summary}")
        if category:
            lines.append(f"    Suggested bucket: {category}")
        lines.append("")

    if skipped:
        log.info(
            "Sanitized Claude prompt: included %d items, skipped %d risky/empty items",
            included,
            skipped,
        )
    return "\n".join(lines), included


def _build_user_prompt(scrape: ScrapeResult) -> str:
    now = datetime.now(config.LOCAL_TZ)
    period_end = now.strftime("%Y-%m-%d")
    blob, included = _items_for_claude_prompt(scrape.items)
    if included == 0:
        # Fall back to original blob only if sanitizer removed everything —
        # summarize() will still fail validation/refusal handling downstream.
        blob = items_to_prompt_blob(scrape.items)
        included = len(scrape.items)

    return f"""{_language_instruction()}

Report period: last {config.LOOKBACK_DAYS} days (ending {period_end}).
Collected items: {included} (from {len(scrape.items)} scraped; some may be omitted as unsuitable).

The following SOURCE ITEMS are public third-party AI news. Skip any unsuitable item and still submit a complete report.

SOURCE ITEMS:
{blob}

Call submit_weekly_report with exactly this structure:
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
  "conclusions": ["actionable takeaway 1", "actionable takeaway 2"],
  "sources": [{{"name": "OpenAI News - Codex plugins", "url": ""}}]
}}

Rules:
- executive_summary: 3-4 short bullets
- models_research: maximum 3 items
- products_tools: maximum 5 items
- business_market: maximum 5 items
- technical_corner: REQUIRED with non-empty title and explanation
- conclusions: REQUIRED with 2-3 non-empty actionable bullets for a PM
- title fields: use English product/company names where appropriate
- summary/why_it_matters/relevance: Hebrew when Hebrew is requested
- sources: up to 8 key citations as plain text lines "Source Name - Article title" in the name field (no HTML, no URLs in name)
- Do NOT include scrape_status in your response
"""


def _json_text_preview(text: str, limit: int = 200) -> str:
    preview = re.sub(r"\s+", " ", text.strip())[:limit]
    return preview + ("..." if len(text.strip()) > limit else "")


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    elif cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last >= first:
        cleaned = cleaned[first : last + 1]
    return cleaned


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    log.info("Claude JSON raw preview: %s", _json_text_preview(raw))

    cleaned = _clean_json_text(raw)
    log.info("Claude JSON cleaned preview: %s", _json_text_preview(cleaned))

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {raw[:300]}...") from exc

    if isinstance(parsed, dict):
        log.info("Claude JSON top-level keys: %s", list(parsed.keys()))
    return parsed


def _report_tool_definition() -> dict[str, Any]:
    return {
        "name": REPORT_TOOL_NAME,
        "description": "Submit the structured weekly AI intelligence report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-4 short executive summary bullets",
                },
                "models_research": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["title", "summary", "why_it_matters"],
                        "additionalProperties": False,
                    },
                },
                "products_tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "relevance": {"type": "string"},
                        },
                        "required": ["title", "summary", "relevance"],
                        "additionalProperties": False,
                    },
                },
                "business_market": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["title", "summary", "why_it_matters"],
                        "additionalProperties": False,
                    },
                },
                "technical_corner": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["title", "explanation"],
                    "additionalProperties": False,
                },
                "conclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-3 actionable conclusion bullets for a PM",
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "executive_summary",
                "models_research",
                "products_tools",
                "business_market",
                "technical_corner",
                "conclusions",
                "sources",
            ],
            "additionalProperties": False,
        },
    }


def _value_preview(value: Any, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(value))[:limit]
    return text + ("..." if len(str(value)) > limit else "")


def _has_meaningful_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = sanitize_plain_text(value)
        if not cleaned:
            return False
        return cleaned.strip().lower() not in _PLACEHOLDER_STRINGS
    if isinstance(value, list):
        return any(_has_meaningful_content(item) for item in value)
    if isinstance(value, dict):
        return any(_has_meaningful_content(item) for item in value.values())
    return bool(value)


def _item_has_fields(item: Any, *fields: str) -> bool:
    return all(_has_meaningful_content(getattr(item, field, None)) for field in fields)


def validate_report(report: ReportContent) -> list[str]:
    """Return invalid/missing section names. Empty list means the report is valid."""
    invalid: list[str] = []

    meaningful_exec = [
        bullet for bullet in report.executive_summary if _has_meaningful_content(bullet)
    ]
    if len(meaningful_exec) < 2:
        invalid.append("executive_summary")

    meaningful_models = [
        item
        for item in report.models_research
        if _item_has_fields(item, "title", "summary")
    ]
    if len(meaningful_models) < 1:
        invalid.append("models_research")

    meaningful_products = [
        item
        for item in report.products_tools
        if _item_has_fields(item, "title", "summary")
    ]
    if len(meaningful_products) < 1:
        invalid.append("products_tools")

    meaningful_business = [
        item
        for item in report.business_market
        if _item_has_fields(item, "title", "summary")
    ]
    if len(meaningful_business) < 1:
        invalid.append("business_market")

    technical = report.technical_corner
    if technical is None or not _item_has_fields(technical, "title", "explanation"):
        invalid.append("technical_corner")

    meaningful_conclusions = [
        bullet for bullet in report.pm_takeaways if _has_meaningful_content(bullet)
    ]
    if len(meaningful_conclusions) < 2:
        invalid.append("conclusions")

    return invalid


def assert_valid_report(report: ReportContent) -> None:
    invalid = validate_report(report)
    if not invalid:
        log.info("Report validation: PASS")
        return
    log.error("Report validation: FAIL")
    log.error("Invalid sections: %s", invalid)
    raise ReportGenerationError(
        f"Report validation failed; incomplete sections: {', '.join(invalid)}",
        invalid_sections=invalid,
        scraped_items=report.items_collected,
        sources_succeeded=report.scrape_status.successful_sources,
        sources_failed=report.scrape_status.failed_source_count,
    )


def _field_value(data: dict[str, Any], *keys: str) -> Any:
    fallback: Any = None
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if _has_meaningful_content(value):
            return value
        if fallback is None:
            fallback = value
    return fallback


def _normalize_report_data(data: dict[str, Any]) -> dict[str, Any]:
    log.info("Report tool input keys before normalization: %s", list(data.keys()))
    normalized = dict(data)

    if not _has_meaningful_content(normalized.get("technical_corner")):
        for alias in ("technical", "technical_section", "tech_corner"):
            if _has_meaningful_content(normalized.get(alias)):
                normalized["technical_corner"] = normalized[alias]
                break

    if not _has_meaningful_content(normalized.get("conclusions")):
        for alias in ("pm_takeaways", "takeaways"):
            if _has_meaningful_content(normalized.get(alias)):
                normalized["conclusions"] = normalized[alias]
                break

    log.info("Report tool input keys after normalization: %s", list(normalized.keys()))
    return normalized


def _log_section_shapes(data: dict[str, Any]) -> None:
    log.info("Report top-level keys before render: %s", list(data.keys()))
    for key in (
        "technical_corner",
        "technical",
        "technical_section",
        "tech_corner",
        "conclusions",
        "pm_takeaways",
        "takeaways",
    ):
        if key in data:
            log.info(
                "Section field %s type=%s preview=%s",
                key,
                type(data[key]).__name__,
                _value_preview(data[key]),
            )


def _report_data_from_message(message: Any) -> tuple[dict[str, Any], str]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == REPORT_TOOL_NAME:
            tool_input = getattr(block, "input", None)
            if isinstance(tool_input, dict):
                log.info("Report data source: anthropic tool_use")
                log.info("Report top-level keys: %s", list(tool_input.keys()))
                return _normalize_report_data(tool_input), "anthropic tool_use"
            raise ValueError("submit_weekly_report tool input was not an object")

    raw = ""
    for block in message.content:
        if hasattr(block, "text"):
            raw += block.text

    if not raw.strip():
        raise ValueError("Claude returned neither tool_use nor text content")

    log.warning("No tool_use block found; falling back to legacy JSON parser")
    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "Claude returned neither valid tool_use nor parseable JSON"
        ) from exc
    log.info("Report data source: legacy JSON parser")
    log.info("Report top-level keys: %s", list(data.keys()))
    return _normalize_report_data(data), "legacy JSON parser"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            cleaned = sanitize_plain_text(str(item))
            if _has_meaningful_content(cleaned):
                out.append(cleaned)
        return out
    if isinstance(value, str) and _has_meaningful_content(value):
        return [sanitize_plain_text(value)]
    return []


def _parse_conclusions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) and value.strip():
        return _as_list(value)
    if isinstance(value, dict):
        text = (
            value.get("text")
            or value.get("summary")
            or value.get("conclusion")
            or value.get("takeaway")
            or value.get("title")
        )
        if text:
            return [sanitize_plain_text(str(text))]
        parts = [
            sanitize_plain_text(str(part))
            for part in value.values()
            if isinstance(part, str) and sanitize_plain_text(str(part))
        ]
        return parts
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                cleaned = sanitize_plain_text(item)
                if cleaned:
                    out.append(cleaned)
            elif isinstance(item, dict):
                title = sanitize_plain_text(str(item.get("title", "")))
                body = sanitize_plain_text(
                    str(
                        item.get("text")
                        or item.get("summary")
                        or item.get("conclusion")
                        or item.get("takeaway")
                        or item.get("description")
                        or item.get("explanation")
                        or ""
                    )
                )
                if title and body:
                    out.append(f"{title}: {body}")
                elif body:
                    out.append(body)
                elif title:
                    out.append(title)
        return out
    return _as_list(value)


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
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return TechnicalCorner(
            title="Technical Corner",
            explanation=sanitize_plain_text(value),
        )
    if isinstance(value, list):
        for item in value:
            parsed = _parse_technical(item)
            if parsed:
                return parsed
        return None
    if not isinstance(value, dict):
        return None
    title = sanitize_plain_text(
        str(value.get("title") or value.get("name") or value.get("heading") or "")
    )
    explanation = sanitize_plain_text(
        str(
            value.get("explanation")
            or value.get("summary")
            or value.get("description")
            or value.get("text")
            or ""
        )
    )
    if not title and not explanation:
        parts = [
            sanitize_plain_text(str(part))
            for part in value.values()
            if isinstance(part, str) and sanitize_plain_text(str(part))
        ]
        if not parts:
            return None
        return TechnicalCorner(title="Technical Corner", explanation=" — ".join(parts))
    return TechnicalCorner(title=title or "Technical Corner", explanation=explanation)


def _parse_technical_from_data(data: dict[str, Any]) -> TechnicalCorner | None:
    for key in ("technical_corner", "technical", "technical_section", "tech_corner"):
        if key not in data:
            continue
        parsed = _parse_technical(data[key])
        if parsed and (
            _has_meaningful_content(parsed.title)
            or _has_meaningful_content(parsed.explanation)
        ):
            return parsed
    return None


def _parse_conclusions_from_data(data: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for key in ("conclusions", "pm_takeaways", "takeaways"):
        if key not in data:
            continue
        for item in _parse_conclusions(data[key]):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


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


def summarize(scrape: ScrapeResult) -> ReportContent:
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = config.ANTHROPIC_MODEL
    print(f"Using Anthropic model: {model}")
    try:
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(scrape)}],
            tools=[_report_tool_definition()],
            tool_choice={"type": "tool", "name": REPORT_TOOL_NAME},
        )
    except Exception as exc:
        raise ReportGenerationError(
            f"Claude API call failed (model={model}): {exc}",
            scraped_items=len(scrape.items),
            sources_succeeded=scrape.sources_succeeded,
            sources_failed=scrape.sources_failed,
        ) from exc

    stop_reason = getattr(message, "stop_reason", None)
    log.info("Claude stop_reason: %s", stop_reason)

    if stop_reason == "refusal":
        log.error("Claude refused report generation")
        raise ReportGenerationError(
            "Claude refused report generation",
            stop_reason=stop_reason,
            scraped_items=len(scrape.items),
            sources_succeeded=scrape.sources_succeeded,
            sources_failed=scrape.sources_failed,
        )

    try:
        data, _source = _report_data_from_message(message)
    except ValueError as exc:
        raise ReportGenerationError(
            f"Failed to parse Claude report response: {exc}",
            stop_reason=stop_reason,
            scraped_items=len(scrape.items),
            sources_succeeded=scrape.sources_succeeded,
            sources_failed=scrape.sources_failed,
        ) from exc
    report_date, period_display, period_start, period_end = _format_period(
        datetime.now(config.LOCAL_TZ)
    )

    _log_section_shapes(data)
    technical_corner = _parse_technical_from_data(data)
    pm_takeaways = _parse_conclusions_from_data(data)[:3]
    log.info(
        "Parsed technical_corner: %s",
        technical_corner.title if technical_corner else None,
    )
    log.info("Parsed conclusions count: %d", len(pm_takeaways))

    report = ReportContent(
        report_date=report_date,
        period_display=period_display,
        period_start=period_start,
        period_end=period_end,
        executive_summary=_as_list(data.get("executive_summary")),
        models_research=_parse_research(data.get("models_research"))[:3],
        products_tools=_parse_products(data.get("products_tools"))[:5],
        business_market=_parse_business(data.get("business_market"))[:5],
        technical_corner=technical_corner,
        pm_takeaways=pm_takeaways,
        sources=_parse_sources(data.get("sources")),
        scrape_status=_build_scrape_status(scrape),
        items_collected=len(scrape.items),
    )
    try:
        assert_valid_report(report)
    except ReportGenerationError as exc:
        if exc.stop_reason is None:
            exc.stop_reason = stop_reason
        raise
    return report
