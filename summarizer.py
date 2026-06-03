"""Generate structured weekly report using Claude API."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from anthropic import Anthropic

import config
from scraper import ScrapeResult, items_to_prompt_blob


@dataclass
class ReportContent:
    report_date: str
    executive_summary: str
    models_research: str
    products_tools: str
    business_market: str
    technical_corner: str
    key_takeaway: str
    sources_used: int
    scrape_errors: list[str]

    def sections(self) -> list[tuple[str, str, str]]:
        """Return (id, title, html_content) for email template."""
        return [
            ("executive", "Executive Summary", self.executive_summary),
            ("research", "Models & Research", self.models_research),
            ("products", "Products & Tools", self.products_tools),
            ("business", "Business & Market", self.business_market),
            ("technical", "Technical Corner", self.technical_corner),
            ("takeaway", "Key Takeaway", self.key_takeaway),
        ]


SECTION_WEIGHTS = {
    "executive_summary": "20%",
    "models_research": "20%",
    "products_tools": "25%",
    "business_market": "25%",
    "technical_corner": "5%",
    "key_takeaway": "5%",
}

SYSTEM_PROMPT = """You are an expert AI industry analyst preparing a concise weekly intelligence brief.
Use ONLY the provided source items. Do not invent news. If a section lacks evidence, say so briefly.
Write clearly for a technical executive audience. Prefer bullets where appropriate.
The full report must fit roughly 2 printed pages when rendered as HTML (about 900-1100 words total).
Allocate content roughly by section weight: Executive 20%, Models & Research 20%, Products & Tools 25%, Business & Market 25%, Technical Corner 5%, Key Takeaway 5%.
Respond with valid JSON only — no markdown fences."""


def _language_instruction() -> str:
    if config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית"):
        return "Write the entire report in Hebrew (עברית), including section titles in the JSON values."
    return "Write the report in English."


def _build_user_prompt(scrape: ScrapeResult) -> str:
    now = datetime.now(config.LOCAL_TZ)
    period_end = now.strftime("%Y-%m-%d")
    blob = items_to_prompt_blob(scrape.items)
    errors = "\n".join(scrape.errors) if scrape.errors else "None"

    return f"""{_language_instruction()}

Report period: last {config.LOOKBACK_DAYS} days (ending {period_end}).
Collected items: {len(scrape.items)}
Scrape warnings:
{errors}

SOURCE ITEMS:
{blob}

Return JSON with exactly these keys:
{{
  "executive_summary": "string — high-level week in AI",
  "models_research": "string — papers, models, benchmarks, research breakthroughs",
  "products_tools": "string — launches, APIs, open-source tools, products",
  "business_market": "string — funding, policy, market moves, partnerships",
  "technical_corner": "string — one sharp technical insight or tip",
  "key_takeaway": "string — single most important action-oriented takeaway"
}}
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
        executive_summary=data.get("executive_summary", "").strip(),
        models_research=data.get("models_research", "").strip(),
        products_tools=data.get("products_tools", "").strip(),
        business_market=data.get("business_market", "").strip(),
        technical_corner=data.get("technical_corner", "").strip(),
        key_takeaway=data.get("key_takeaway", "").strip(),
        sources_used=len(scrape.items),
        scrape_errors=scrape.errors,
    )
