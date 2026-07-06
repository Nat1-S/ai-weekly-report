#!/usr/bin/env python3
"""AI Weekly Report — scrape, summarize, deliver."""

from __future__ import annotations

import logging
import sys
import traceback

from scraper import scrape_all
from sender import _email_recipients, _error_recipients, send_email, send_error_email
from summarizer import summarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _notify_failure(message: str, details: str = "") -> None:
    try:
        send_error_email(message, details)
        log.info("Failure notification sent to %s", ", ".join(_error_recipients()))
    except Exception:
        log.error("Failed to send failure notification:\n%s", traceback.format_exc())


def main() -> int:
    try:
        log.info("Starting AI weekly report pipeline")

        scrape = scrape_all()
        log.info("Scraped %d items (%d source buckets)", len(scrape.items), len(scrape.stats))
        for source, count in sorted(scrape.stats.items()):
            log.info("  %s: %d", source, count)
        if scrape.errors:
            log.warning("%d sources failed", scrape.sources_failed)
            for err in scrape.errors:
                log.warning("  %s", err)

        if not scrape.items:
            log.error("No items collected — aborting to avoid empty report")
            _notify_failure("No items collected — aborting to avoid empty report")
            return 1

        report = summarize(scrape)
        log.info(
            "Report generated for %s (coverage %d%%)",
            report.report_date,
            report.scrape_status.coverage_percentage,
        )

        send_email(report)
        log.info("HTML email sent to %s", ", ".join(_email_recipients()))

        log.info("Pipeline completed successfully")
        return 0
    except Exception:
        tb = traceback.format_exc()
        log.error("Pipeline failed:\n%s", tb)
        _notify_failure("Pipeline failed", tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
