#!/usr/bin/env python3
"""AI Weekly Report — scrape, summarize, deliver."""

from __future__ import annotations

import logging
import sys

from scraper import scrape_all
from sender import send_email
from summarizer import summarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> int:
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
        return 1

    report = summarize(scrape)
    log.info(
        "Report generated for %s (coverage %d%%)",
        report.report_date,
        report.scrape_status.coverage_pct,
    )

    send_email(report)
    log.info("HTML email sent to %s", __import__("config").EMAIL_RECIPIENT)

    log.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
