"""Generate structured weekly report using Claude API."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from anthropic import Anthropic

import config
from scraper import ScrapeResult, items_to_prompt_blob


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
class ScrapeStatusSummary:
    sources_scanned: int
    sources_succeeded: int
    sources_failed: int
    coverage_pct: int
    failed_sources: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ReportContent:
    report_date: str
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
Keep the full report concise — roughly 2 printed pages when rendered.
Return valid JSON only. No markdown. No code fences. No prose outside JSON.
Each list section should have 3-5 focused items maximum."""


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
  "sources": [{{"name": "...", "url": "..."}}]
}}

Rules:
- executive_summary: 3-5 short bullets
- pm_takeaways: 2-4 actionable bullets for a PM
- sources: up to 8 most important cited sources with real URLs from the input
- Do NOT include scrape_status in your response
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_research(items: Any) -> list[ResearchItem]:
    out: list[ResearchItem] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        out.append(
            ResearchItem(
                title=title,
                summary=str(row.get("summary", "")).strip(),
                why_it_matters=str(row.get("why_it_matters", "")).strip(),
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
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        out.append(
            ProductItem(
                title=title,
                summary=str(row.get("summary", "")).strip(),
                relevance=str(row.get("relevance", "")).strip(),
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
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        out.append(
            BusinessItem(
                title=title,
                summary=str(row.get("summary", "")).strip(),
                why_it_matters=str(row.get("why_it_matters", "")).strip(),
            )
        )
    return out


def _parse_technical(value: Any) -> TechnicalCorner | None:
    if not isinstance(value, dict):
        return None
    title = str(value.get("title", "")).strip()
    explanation = str(value.get("explanation", "")).strip()
    if not title and not explanation:
        return None
    return TechnicalCorner(title=title or "Technical Corner", explanation=explanation)


def _parse_sources(items: Any) -> list[SourceRef]:
    out: list[SourceRef] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        if name and url:
            out.append(SourceRef(name=name, url=url))
    return out


def _build_scrape_status(scrape: ScrapeResult) -> ScrapeStatusSummary:
    return ScrapeStatusSummary(
        sources_scanned=scrape.sources_scanned,
        sources_succeeded=scrape.sources_succeeded,
        sources_failed=scrape.sources_failed,
        coverage_pct=scrape.coverage_pct,
        failed_sources=scrape.failed_sources,
    )


def summarize(scrape: ScrapeResult) -> ReportContent:
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(scrape)}],
    )

    raw = ""
    for block in message.content:
        if hasattr(block, "text"):
            raw += block.text

    data = _extract_json(raw)
    report_date = datetime.now(config.LOCAL_TZ).strftime("%d %B %Y")

    return ReportContent(
        report_date=report_date,
        executive_summary=_as_list(data.get("executive_summary")),
        models_research=_parse_research(data.get("models_research")),
        products_tools=_parse_products(data.get("products_tools")),
        business_market=_parse_business(data.get("business_market")),
        technical_corner=_parse_technical(data.get("technical_corner")),
        pm_takeaways=_as_list(data.get("pm_takeaways")),
        sources=_parse_sources(data.get("sources")),
        scrape_status=_build_scrape_status(scrape),
        items_collected=len(scrape.items),
    )
