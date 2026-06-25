"""Tests for summarizer._extract_json only."""

from __future__ import annotations

import unittest

from summarizer import _extract_json


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

    def test_executive_summary_list_passes(self) -> None:
        text = '{ "executive_summary": ["bullet one", "bullet two"], "models_research": [] }'
        data = _extract_json(text)
        self.assertIsInstance(data, dict)
        self.assertEqual(data["executive_summary"], ["bullet one", "bullet two"])

    def test_object_not_replaced_by_inner_array(self) -> None:
        text = (
            'Here is the report:\n'
            '{ "executive_summary": ["a", "b"], "models_research": [], "pm_takeaways": ["x"] }\n'
            "Thanks!"
        )
        data = _extract_json(text)
        self.assertIsInstance(data, dict)
        self.assertIn("models_research", data)
        self.assertIn("pm_takeaways", data)

    def test_invalid_json_still_fails(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("This is not JSON at all.")
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
