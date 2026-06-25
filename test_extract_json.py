"""Tests for structured report extraction from Anthropic responses."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from summarizer import (
    REPORT_TOOL_NAME,
    ReportContent,
    ScrapeStatusSummary,
    _as_list,
    _build_scrape_status,
    _extract_json,
    _field_value,
    _normalize_report_data,
    _parse_technical,
    _report_data_from_message,
)
from scraper import ScrapeResult
from sender import build_html_email


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self) -> None:
        data = _extract_json('{"executive_summary": ["a"], "models_research": []}')
        self.assertEqual(data["executive_summary"], ["a"])

    def test_fenced_json(self) -> None:
        text = '```json\n{"executive_summary": ["b"]}\n```'
        data = _extract_json(text)
        self.assertEqual(data["executive_summary"], ["b"])

    def test_text_before_and_after_json(self) -> None:
        text = 'Here is the report:\n{"executive_summary": ["c"]}\nThanks!'
        data = _extract_json(text)
        self.assertEqual(data["executive_summary"], ["c"])

    def test_invalid_json_still_fails(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("This is not JSON at all.")
        self.assertIn("invalid JSON", str(ctx.exception))


class TestReportDataFromMessage(unittest.TestCase):
    def test_uses_tool_input_when_present(self) -> None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = REPORT_TOOL_NAME
        tool_block.input = {
            "executive_summary": ["bullet"],
            "models_research": [],
            "products_tools": [],
            "business_market": [],
            "conclusions": ["takeaway"],
            "sources": [],
        }
        message = MagicMock(content=[tool_block])

        data, source = _report_data_from_message(message)

        self.assertEqual(source, "anthropic tool_use")
        self.assertEqual(data["executive_summary"], ["bullet"])

    def test_falls_back_to_legacy_json_parser(self) -> None:
        text_block = MagicMock()
        text_block.text = '{"executive_summary": ["legacy"], "models_research": []}'
        message = MagicMock(content=[text_block])

        data, source = _report_data_from_message(message)

        self.assertEqual(source, "legacy JSON parser")
        self.assertEqual(data["executive_summary"], ["legacy"])

    def test_fails_when_both_paths_unavailable(self) -> None:
        message = MagicMock(content=[])

        with self.assertRaises(ValueError) as ctx:
            _report_data_from_message(message)
        self.assertIn("neither tool_use nor text", str(ctx.exception))


class TestReportSectionMapping(unittest.TestCase):
    def test_conclusions_and_technical_corner_render_non_empty(self) -> None:
        data = _normalize_report_data(
            {
                "executive_summary": ["summary"],
                "models_research": [],
                "products_tools": [],
                "business_market": [],
                "conclusions": ["משקיעים בכלי agents", "לעקוב אחר מודלים חדשים"],
                "technical_corner": {"title": "RAG", "explanation": "הסבר טכני"},
                "sources": [],
            }
        )
        technical = _parse_technical(
            _field_value(data, "technical_corner", "technical", "technical_section", "tech_corner")
        )
        conclusions = _as_list(_field_value(data, "conclusions", "pm_takeaways", "takeaways"))

        report = ReportContent(
            report_date="1 January 2026",
            period_display="01.01.2026 - 07.01.2026",
            period_start="2026-01-01",
            period_end="2026-01-07",
            executive_summary=["summary"],
            models_research=[],
            products_tools=[],
            business_market=[],
            technical_corner=technical,
            pm_takeaways=conclusions,
            sources=[],
            scrape_status=_build_scrape_status(ScrapeResult()),
            items_collected=0,
        )

        html = build_html_email(report)

        self.assertIsNotNone(technical)
        self.assertEqual(len(conclusions), 2)
        self.assertIn("משקיעים בכלי", html)
        self.assertIn("agents", html)
        self.assertIn("לעקוב אחר מודלים חדשים", html)
        self.assertIn("RAG", html)
        self.assertIn("הסבר טכני", html)

    def test_aliases_normalize_to_expected_keys(self) -> None:
        data = _normalize_report_data(
            {
                "takeaways": ["one"],
                "tech_corner": {"title": "LoRA", "explanation": "fine-tuning"},
            }
        )
        self.assertIn("conclusions", data)
        self.assertIn("technical_corner", data)
        self.assertEqual(data["conclusions"], ["one"])
        self.assertEqual(data["technical_corner"]["title"], "LoRA")


if __name__ == "__main__":
    unittest.main()
