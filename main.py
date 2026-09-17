#!/usr/bin/env python3
"""AI Weekly Report — scrape, summarize, deliver."""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime

import config
from scraper import ScrapeResult, scrape_all
from sender import _email_recipients, _error_recipients, send_email, send_error_email
from summarizer import ReportGenerationError, summarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _notify_failure(
    message: str,
    details: str = "",
    *,
    stop_reason: str | None = None,
    scraped_items: int | None = None,
    sources_succeeded: int | None = None,
    sources_failed: int | None = None,
    invalid_sections: list[str] | None = None,
    stop_details: str | None = None,
    isolation_attempted: bool | None = None,
    problematic_items: list[str] | None = None,
    items_excluded: int | None = None,
    final_item_count: int | None = None,
    recovery_succeeded: bool | None = None,
    initial_item_count: int | None = None,
) -> None:
    log.error("Production email blocked")
    log.info("Sending failure report to ADMIN_EMAIL only")
    try:
        send_error_email(
            message,
            details,
            stop_reason=stop_reason,
            scraped_items=scraped_items,
            sources_succeeded=sources_succeeded,
            sources_failed=sources_failed,
            invalid_sections=invalid_sections,
            report_date=datetime.now(config.LOCAL_TZ).strftime("%Y-%m-%d"),
            stop_details=stop_details,
            isolation_attempted=isolation_attempted,
            problematic_items=problematic_items,
            items_excluded=items_excluded,
            final_item_count=final_item_count,
            recovery_succeeded=recovery_succeeded,
            initial_item_count=initial_item_count,
        )
        log.info("Failure notification sent to %s", ", ".join(_error_recipients()))
    except Exception:
        log.error("Failed to send failure notification:\n%s", traceback.format_exc())


def _scrape_counts(scrape: ScrapeResult | None) -> tuple[int | None, int | None, int | None]:
    if scrape is None:
        return None, None, None
    return len(scrape.items), scrape.sources_succeeded, scrape.sources_failed


def main() -> int:
    scrape: ScrapeResult | None = None
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
            raise ReportGenerationError(
                "No items collected — aborting to avoid empty report",
                scraped_items=0,
                sources_succeeded=scrape.sources_succeeded,
                sources_failed=scrape.sources_failed,
            )

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
    except ReportGenerationError as exc:
        log.error("%s", exc)
        if exc.invalid_sections:
            log.error("Invalid sections: %s", exc.invalid_sections)
        items, ok, failed = _scrape_counts(scrape)
        _notify_failure(
            str(exc),
            traceback.format_exc(),
            stop_reason=exc.stop_reason,
            scraped_items=exc.scraped_items if exc.scraped_items is not None else items,
            sources_succeeded=(
                exc.sources_succeeded if exc.sources_succeeded is not None else ok
            ),
            sources_failed=exc.sources_failed if exc.sources_failed is not None else failed,
            invalid_sections=exc.invalid_sections,
            stop_details=exc.stop_details,
            isolation_attempted=exc.isolation_attempted,
            problematic_items=exc.problematic_items,
            items_excluded=exc.items_excluded,
            final_item_count=exc.final_item_count,
            recovery_succeeded=exc.recovery_succeeded,
            initial_item_count=exc.initial_item_count,
        )
        return 1
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Pipeline failed:\n%s", tb)
        items, ok, failed = _scrape_counts(scrape)
        _notify_failure(
            f"Pipeline failed: {exc}",
            tb,
            scraped_items=items,
            sources_succeeded=ok,
            sources_failed=failed,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
