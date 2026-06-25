"""Tests for summarizer JSON parsing and fallback report generation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from scraper import NewsItem, ScrapeResult, SourceResult
from summarizer import _build_fallback_report, _extract_json, summarize


def _sample_scrape() -> ScrapeResult:
    items = [
        NewsItem(
            title="New GPT model paper",
            url="https://example.com/1",
            source="API: ArXiv",
            published="2026-06-01T00:00:00+00:00",
            summary="Abstract here",
            category="research",
        ),
        NewsItem(
            title="Tool launch",
            url="https://example.com/2",
            source="RSS: TechCrunch AI",
            published="2026-06-02T00:00:00+00:00",
            category="products",
        ),
    ]
    sources = [
        SourceResult("ArXiv", "API", "success", 1),
        SourceResult("TechCrunch AI", "RSS", "success", 1),
    ]
    return ScrapeResult(items=items, sources=sources)


class TestExtractJson(unittest.TestCase):
    def test_plain_json_parses(self) -> None:
        data = _extract_json('{"executive_summary": ["a"], "models_research": []}')
        self.assertEqual(data["executive_summary"], ["a"])

    def test_json_inside_fences_parses(self) -> None:
        text = '```json\n{"executive_summary": ["b"]}\n```'
        data = _extract_json(text)
        self.assertEqual(data["executive_summary"], ["b"])

    def test_json_with_text_before_and_after_parses(self) -> None:
        text = 'Here is the report:\n{"executive_summary": ["c"]}\nThanks!'
        data = _extract_json(text)
        self.assertEqual(data["executive_summary"], ["c"])

    def test_invalid_json_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("This is not JSON at all.")
        self.assertIn("invalid JSON", str(ctx.exception))


class TestFallbackReport(unittest.TestCase):
    def test_fallback_produces_sendable_report(self) -> None:
        report = _build_fallback_report(_sample_scrape())
        self.assertTrue(report.executive_summary)
        self.assertTrue(report.models_research)
        self.assertTrue(report.products_tools)
        self.assertTrue(report.sources)
        self.assertEqual(report.items_collected, 2)


class TestSummarizePipelineFallback(unittest.TestCase):
    @patch("summarizer.Anthropic")
    def test_summarize_uses_fallback_on_bad_json(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        block = MagicMock()
        block.text = "not valid json {{{"
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("summarizer.config.ANTHROPIC_API_KEY", "test-key"):
            report = summarize(_sample_scrape())

        self.assertTrue(report.executive_summary)
        self.assertEqual(report.items_collected, 2)


if __name__ == "__main__":
    unittest.main()
