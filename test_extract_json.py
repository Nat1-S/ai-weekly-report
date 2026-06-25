"""Tests for structured report extraction from Anthropic responses."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from summarizer import REPORT_TOOL_NAME, _extract_json, _report_data_from_message


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
            "pm_takeaways": [],
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


if __name__ == "__main__":
    unittest.main()
