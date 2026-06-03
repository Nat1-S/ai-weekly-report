"""Deliver report via Gmail SMTP with email-safe RTL HTML."""

from __future__ import annotations

import html
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from summarizer import (
    BusinessItem,
    ProductItem,
    ReportContent,
    ResearchItem,
    TechnicalCorner,
    sanitize_plain_text,
)

LTR_SPAN_OPEN = '<span dir="ltr" style="unicode-bidi:isolate;display:inline-block;text-align:left;">'
LTR_SPAN_CLOSE = "</span>"


def _is_hebrew_report() -> bool:
    return config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית")


def _esc(text: str) -> str:
    return html.escape(text.strip())


def _wrap_ltr(text: str) -> str:
    """Wrap Latin runs in LTR spans; input must be plain text only."""
    text = sanitize_plain_text(text)
    if not text:
        return ""
    parts: list[str] = []
    last = 0
    latin_pattern = re.compile(
        r"[A-Za-z][A-Za-z0-9+.#\-_/]*(?:\s+[A-Za-z][A-Za-z0-9+.#\-_/]*)*"
    )
    for match in latin_pattern.finditer(text):
        if match.start() > last:
            parts.append(html.escape(text[last : match.start()]))
        parts.append(f"{LTR_SPAN_OPEN}{html.escape(match.group(0))}{LTR_SPAN_CLOSE}")
        last = match.end()
    if last < len(text):
        parts.append(html.escape(text[last:]))
    return "".join(parts) if parts else html.escape(text)


def _validate_report_html(html_body: str, labels: dict[str, str]) -> None:
    """Block emails where escaped HTML would show as visible text."""
    body = re.sub(r"<style[^>]*>.*?</style>", "", html_body, flags=re.DOTALL | re.IGNORECASE)
    for fragment in ("&lt;span", "&lt;/span", "&lt;style"):
        if fragment in body:
            raise ValueError(f"Report validation failed: visible fragment '{fragment}'")
    if labels["title"] not in html_body:
        raise ValueError("Report validation failed: title missing")
    if labels["pm_takeaways"] not in html_body:
        raise ValueError("Report validation failed: PM section title missing")


def _labels() -> dict[str, str]:
    if _is_hebrew_report():
        return {
            "title": "דוח AI שבועי",
            "period": "תקופת הדוח",
            "items_collected": "פריטים שנאספו",
            "sources_scanned": "מקורות שנסרקו",
            "executive": "סיכום מנהלים",
            "research": "מודלים ומחקר",
            "products": "מוצרים וכלים",
            "business": "עסקים ושוק",
            "technical": "פינה טכנית",
            "pm_takeaways": "מסקנות",
            "sources": "מקורות מרכזיים",
            "coverage_quality": "מקורות ואיכות הכיסוי",
            "sources_succeeded": "מקורות שהצליחו",
            "sources_failed": "מקורות שנכשלו",
            "coverage_pct_label": "אחוז כיסוי",
            "total_items_week": "סה\"כ פריטים שנאספו השבוע",
            "failed_sources_title": "מקורות שנכשלו",
            "completeness_title": "הערכת שלמות המידע",
            "footer": "דוח אוטומטי · GitHub Actions",
            "summary_label": "סיכום",
            "why_matters": "למה זה חשוב",
            "relevance": "רלוונטיות",
        }
    return {
        "title": "AI Weekly Intelligence Report",
        "period": "Report period",
        "items_collected": "Items collected",
        "sources_scanned": "Sources scanned",
        "executive": "Executive Summary",
        "research": "Models & Research",
        "products": "Products & Tools",
        "business": "Business & Market",
        "technical": "Technical Corner",
        "pm_takeaways": "PM Takeaways",
        "sources": "Key Sources",
        "coverage_quality": "Source Coverage",
        "sources_succeeded": "Sources succeeded",
        "sources_failed": "Sources failed",
        "coverage_pct_label": "Coverage",
        "total_items_week": "Total items collected",
        "failed_sources_title": "Failed sources",
        "completeness_title": "Information completeness",
        "footer": "Automated report · GitHub Actions",
        "summary_label": "Summary",
        "why_matters": "Why it matters",
        "relevance": "Relevance",
    }


def _report_card(
    title: str,
    summary: str,
    impact: str | None,
    impact_label: str,
    summary_label: str,
) -> str:
    impact_block = ""
    if impact:
        impact_block = (
            f'<div class="item-impact" style="margin-top:8px;font-size:14px;color:#475569;">'
            f'<strong>{_esc(impact_label)}:</strong> {_wrap_ltr(impact)}'
            f"</div>"
        )
    return f"""
    <div class="report-card" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;margin-bottom:12px;">
      <div class="item-title" style="font-size:15px;font-weight:700;color:#0f3460;margin-bottom:8px;">{_wrap_ltr(title)}</div>
      <div class="item-summary" style="font-size:14px;line-height:1.8;color:#1e293b;">
        <strong>{_esc(summary_label)}:</strong> {_wrap_ltr(summary)}
      </div>
      {impact_block}
    </div>
    """


def _render_card_list(cards: list[str]) -> str:
    if not cards:
        return '<p style="color:#94a3b8;margin:0;">—</p>'
    items = "".join(f"<li style=\"margin-bottom:0;list-style:none;\">{c}</li>" for c in cards)
    return (
        f'<ul class="report-list" style="direction:rtl;text-align:right;'
        f'list-style:none;padding:0;margin:0;">{items}</ul>'
    )


def _render_bullets(items: list[str]) -> str:
    if not items:
        return '<p style="color:#94a3b8;margin:0;">—</p>'
    lis = "".join(
        f'<li style="margin-bottom:16px;line-height:1.8;">{_wrap_ltr(item)}</li>'
        for item in items
    )
    return (
        f'<ul class="report-list" style="direction:rtl;text-align:right;'
        f'list-style-position:outside;padding-right:24px;padding-left:0;margin:0;">{lis}</ul>'
    )


def _render_research_items(items: list[ResearchItem], labels: dict[str, str]) -> str:
    cards = [
        _report_card(
            item.title, item.summary, item.why_it_matters or None,
            labels["why_matters"], labels["summary_label"],
        )
        for item in items
    ]
    return _render_card_list(cards)


def _render_product_items(items: list[ProductItem], labels: dict[str, str]) -> str:
    cards = [
        _report_card(
            item.title, item.summary, item.relevance or None,
            labels["relevance"], labels["summary_label"],
        )
        for item in items
    ]
    return _render_card_list(cards)


def _render_business_items(items: list[BusinessItem], labels: dict[str, str]) -> str:
    cards = [
        _report_card(
            item.title, item.summary, item.why_it_matters or None,
            labels["why_matters"], labels["summary_label"],
        )
        for item in items
    ]
    return _render_card_list(cards)


def _render_technical(item: TechnicalCorner | None, labels: dict[str, str]) -> str:
    if not item:
        return '<p style="color:#94a3b8;margin:0;">—</p>'
    return _render_card_list([
        _report_card(item.title, item.explanation, None, "", labels["summary_label"])
    ])


def _render_sources(report: ReportContent, labels: dict[str, str]) -> str:
    if not report.sources:
        return ""
    items = "".join(
        f'<li style="margin-bottom:10px;line-height:1.6;">'
        f"{_wrap_ltr(s.name)}</li>"
        for s in report.sources[:8]
        if s.name
    )
    return f"""
    <section style="margin-top:28px;">
      <h2 style="font-size:17px;color:#16213e;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin:0 0 12px;text-align:right;">{_esc(labels["sources"])}</h2>
      <ul class="report-list" style="direction:rtl;text-align:right;list-style-position:outside;padding-right:24px;padding-left:0;margin:0;">{items}</ul>
    </section>
    """


def _render_coverage_quality_section(report: ReportContent, labels: dict[str, str]) -> str:
    he = _is_hebrew_report()
    s = report.scrape_status
    completeness = s.completeness_text(hebrew=he)
    transparency = s.transparency_text(hebrew=he)

    failed_block = ""
    if s.failed_source_list:
        failed_items = "".join(
            f"<li style=\"margin-bottom:6px;\">"
            f"{_wrap_ltr(sanitize_plain_text(f['name']))} — "
            f"{_esc(sanitize_plain_text(f['error']))}</li>"
            for f in s.failed_source_list[:12]
        )
        failed_block = f"""
        <div style="margin-top:10px;font-size:13px;color:#64748b;">
          <strong>{_esc(labels["failed_sources_title"])}:</strong>
          <ul style="direction:rtl;text-align:right;padding-right:20px;padding-left:0;margin:6px 0 0;">{failed_items}</ul>
        </div>
        """

    return f"""
    <section style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;margin-top:28px;font-size:14px;color:#475569;direction:rtl;text-align:right;">
      <h2 style="font-size:16px;color:#16213e;border-bottom:1px solid #e2e8f0;padding-bottom:6px;margin:0 0 12px;text-align:right;">{_esc(labels["coverage_quality"])}</h2>
      <table width="100%" cellpadding="0" cellspacing="0" dir="rtl" style="font-size:14px;color:#475569;">
        <tr><td style="padding:3px 0;">{_esc(labels["sources_scanned"])}: <strong style="color:#0f3460;">{s.total_sources}</strong></td></tr>
        <tr><td style="padding:3px 0;">{_esc(labels["sources_succeeded"])}: <strong style="color:#0f3460;">{s.successful_sources}</strong></td></tr>
        <tr><td style="padding:3px 0;">{_esc(labels["sources_failed"])}: <strong style="color:#0f3460;">{s.failed_source_count}</strong></td></tr>
        <tr><td style="padding:3px 0;">{_esc(labels["coverage_pct_label"])}: <strong style="color:#0f3460;">{s.coverage_percentage}%</strong></td></tr>
        <tr><td style="padding:3px 0;">{_esc(labels["total_items_week"])}: <strong style="color:#0f3460;">{s.total_articles_collected}</strong></td></tr>
      </table>
      {failed_block}
      <p style="margin:10px 0 0;font-size:13px;line-height:1.6;color:#64748b;"><strong>{_esc(labels["completeness_title"])}:</strong> {_esc(completeness)}</p>
      <p style="margin:8px 0 0;font-size:13px;line-height:1.6;color:#64748b;">{_esc(transparency)}</p>
    </section>
    """


def _section(title: str, body_html: str, extra_style: str = "") -> str:
    return f"""
    <section style="margin-top:28px;{extra_style}">
      <h2 style="font-size:17px;color:#16213e;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin:0 0 14px;text-align:right;">{_esc(title)}</h2>
      <div>{body_html}</div>
    </section>
    """


def _render_header(report: ReportContent, labels: dict[str, str]) -> str:
    s = report.scrape_status
    return f"""
    <div style="margin-bottom:20px;direction:rtl;text-align:right;">
      <h1 style="font-size:22px;color:#0f3460;margin:0 0 14px;font-weight:700;text-align:right;">{_esc(labels["title"])}</h1>
      <table width="100%" cellpadding="0" cellspacing="0" dir="rtl" style="font-size:14px;color:#64748b;line-height:1.9;">
        <tr><td style="padding:2px 0;">🗓️ {_esc(labels["period"])}: <strong style="color:#334155;">{_esc(report.period_display)}</strong></td></tr>
        <tr><td style="padding:2px 0;">📰 {_esc(labels["items_collected"])}: <strong style="color:#334155;">{report.items_collected}</strong></td></tr>
        <tr><td style="padding:2px 0;">🔎 {_esc(labels["sources_scanned"])}: <strong style="color:#334155;">{s.total_sources}</strong></td></tr>
      </table>
    </div>
    """


def build_plain_text_email(report: ReportContent) -> str:
    labels = _labels()
    lines = [
        labels["title"],
        f"{labels['period']}: {report.period_display}",
        f"{labels['items_collected']}: {report.items_collected}",
        f"{labels['sources_scanned']}: {report.scrape_status.total_sources}",
        "",
    ]
    lines.append(labels["executive"].upper())
    lines.extend(f"• {b}" for b in report.executive_summary)
    lines.append("")

    for title, items, impact_key in [
        (labels["research"], report.models_research, "why_it_matters"),
        (labels["products"], report.products_tools, "relevance"),
        (labels["business"], report.business_market, "why_it_matters"),
    ]:
        lines.append(title.upper())
        for item in items:
            lines.append(f"\n{item.title}")
            lines.append(f"{labels['summary_label']}: {item.summary}")
            extra = getattr(item, impact_key, "")
            if extra:
                label = labels["why_matters"] if impact_key == "why_it_matters" else labels["relevance"]
                lines.append(f"{label}: {extra}")
        lines.append("")

    if report.technical_corner:
        lines.append(labels["technical"].upper())
        lines.append(f"\n{report.technical_corner.title}")
        lines.append(f"{labels['summary_label']}: {report.technical_corner.explanation}")
        lines.append("")

    lines.append(labels["pm_takeaways"].upper())
    lines.extend(f"• {t}" for t in report.pm_takeaways)

    s = report.scrape_status
    lines.extend(["", labels["coverage_quality"].upper()])
    lines.extend([
        f"{labels['sources_scanned']}: {s.total_sources}",
        f"{labels['sources_succeeded']}: {s.successful_sources}",
        f"{labels['sources_failed']}: {s.failed_source_count}",
        f"{labels['coverage_pct_label']}: {s.coverage_percentage}%",
        f"{labels['total_items_week']}: {s.total_articles_collected}",
    ])
    return "\n".join(lines).strip()


def build_html_email(report: ReportContent) -> str:
    labels = _labels()
    he = _is_hebrew_report()
    html_lang = "he" if he else "en"
    html_dir = "rtl" if he else "ltr"

    sections = [
        _section(labels["executive"], _render_bullets(report.executive_summary)),
        _section(labels["research"], _render_research_items(report.models_research, labels)),
        _section(labels["products"], _render_product_items(report.products_tools, labels)),
        _section(labels["business"], _render_business_items(report.business_market, labels)),
        _section(labels["technical"], _render_technical(report.technical_corner, labels)),
        _section(
            labels["pm_takeaways"],
            _render_bullets(report.pm_takeaways),
            extra_style="background:#eef2ff;border-radius:8px;padding:14px 16px;",
        ),
    ]

    return f"""<!DOCTYPE html>
<html lang="{html_lang}" dir="{html_dir}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>{_esc(labels["title"])} — {_esc(report.period_display)}</title>
  <style>
    body {{
      margin: 0; padding: 0;
      font-family: Arial, Helvetica, sans-serif;
      direction: rtl; text-align: right;
      unicode-bidi: embed;
      background: #f8f9fc; color: #1a1a2e;
    }}
    .report-list {{
      direction: rtl; text-align: right;
      list-style-position: outside;
      padding-right: 24px; padding-left: 0;
    }}
    .report-list li {{ margin-bottom: 16px; line-height: 1.8; }}
    .ltr {{ unicode-bidi: isolate; }}
    @media only screen and (max-width: 620px) {{
      .email-container {{ width: 100% !important; padding: 12px !important; }}
    }}
  </style>
</head>
<body dir="{html_dir}" style="direction:{html_dir};text-align:{'right' if he else 'left'};unicode-bidi:embed;margin:0;padding:0;background:#f8f9fc;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" dir="{html_dir}" style="background:#f8f9fc;">
    <tr>
      <td align="center" style="padding:20px 12px;">
        <table role="presentation" class="email-container" width="680" cellpadding="0" cellspacing="0" dir="{html_dir}"
               style="max-width:680px;background:#ffffff;border-radius:12px;padding:28px 32px;direction:{html_dir};text-align:{'right' if he else 'left'};">
          <tr><td dir="{html_dir}" style="direction:{html_dir};text-align:{'right' if he else 'left'};unicode-bidi:embed;">
            {_render_header(report, labels)}
            {''.join(sections)}
            {_render_sources(report, labels)}
            {_render_coverage_quality_section(report, labels)}
          </td></tr>
        </table>
        <p style="text-align:center;font-size:12px;color:#94a3b8;margin-top:16px;">{_esc(labels["footer"])}</p>
      </td>
    </tr>
  </table>
</body>
</html>"""


def send_email(report: ReportContent) -> None:
    if not config.GMAIL_USER or not config.GMAIL_APP_PASSWORD:
        raise ValueError("GMAIL_USER and GMAIL_APP_PASSWORD must be set")

    html_body = build_html_email(report)
    _validate_report_html(html_body, _labels())
    plain_body = build_plain_text_email(report)
    subject = f"{config.EMAIL_SUBJECT_PREFIX} — {report.period_display}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.GMAIL_USER
    msg["To"] = config.EMAIL_RECIPIENT
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
        server.sendmail(config.GMAIL_USER, [config.EMAIL_RECIPIENT], msg.as_string())
