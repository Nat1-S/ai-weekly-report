"""Deliver report via Gmail SMTP."""

from __future__ import annotations

import html
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
)


def _is_hebrew_report() -> bool:
    return config.REPORT_LANGUAGE.lower() in ("he", "hebrew", "עברית")


def _esc(text: str) -> str:
    return html.escape(text.strip())


def _labels() -> dict[str, str]:
    if _is_hebrew_report():
        return {
            "title": "דוח מודיעין AI שבועי",
            "executive": "סיכום מנהלים",
            "research": "מודלים ומחקר",
            "products": "מוצרים וכלים",
            "business": "עסקים ושוק",
            "technical": "פינה טכנית",
            "pm_takeaways": "מסקנות ל-PM",
            "sources": "מקורות מרכזיים",
            "items_collected": "פריטים שנאספו",
            "coverage_quality": "מקורות ואיכות הכיסוי",
            "sources_scanned": "מקורות שנסרקו",
            "sources_succeeded": "מקורות שהצליחו",
            "sources_failed": "מקורות שנכשלו",
            "coverage_pct_label": "אחוז כיסוי",
            "total_items_week": "סה\"כ פריטים שנאספו השבוע",
            "failed_sources_title": "מקורות שנכשלו",
            "completeness_title": "הערכת שלמות המידע",
            "footer": "דוח אוטומטי · GitHub Actions",
            "why_matters": "למה זה חשוב",
            "relevance": "רלוונטיות",
        }
    return {
        "title": "AI Weekly Intelligence Report",
        "executive": "Executive Summary",
        "research": "Models & Research",
        "products": "Products & Tools",
        "business": "Business & Market",
        "technical": "Technical Corner",
        "pm_takeaways": "PM Takeaways",
        "sources": "Key Sources",
        "items_collected": "Items collected",
        "coverage_quality": "Sources & Coverage Quality",
        "sources_scanned": "Sources scanned",
        "sources_succeeded": "Sources succeeded",
        "sources_failed": "Sources failed",
        "coverage_pct_label": "Coverage",
        "total_items_week": "Total items collected this week",
        "failed_sources_title": "Failed sources",
        "completeness_title": "Information completeness",
        "footer": "Automated report · GitHub Actions",
        "why_matters": "Why it matters",
        "relevance": "Relevance",
    }


def _render_bullets(items: list[str]) -> str:
    if not items:
        return "<p class=\"empty\">—</p>"
    lis = "".join(f'<li dir="auto">{_esc(item)}</li>' for item in items)
    return f"<ul>{lis}</ul>"


def _render_research_items(items: list[ResearchItem], labels: dict[str, str]) -> str:
    if not items:
        return "<p class=\"empty\">—</p>"
    parts: list[str] = []
    for item in items:
        body = _esc(item.summary)
        if item.why_it_matters:
            body = f'{_esc(item.summary)} <span class="meta-inline">· {_esc(labels["why_matters"])}: {_esc(item.why_it_matters)}</span>'
        parts.append(
            f'<li dir="auto"><strong>{_esc(item.title)}</strong> — {body}</li>'
        )
    return f"<ul>{''.join(parts)}</ul>"


def _render_product_items(items: list[ProductItem], labels: dict[str, str]) -> str:
    if not items:
        return "<p class=\"empty\">—</p>"
    parts: list[str] = []
    for item in items:
        body = _esc(item.summary)
        if item.relevance:
            body = f'{_esc(item.summary)} <span class="meta-inline">· {_esc(labels["relevance"])}: {_esc(item.relevance)}</span>'
        parts.append(
            f'<li dir="auto"><strong>{_esc(item.title)}</strong> — {body}</li>'
        )
    return f"<ul>{''.join(parts)}</ul>"


def _render_business_items(items: list[BusinessItem], labels: dict[str, str]) -> str:
    if not items:
        return "<p class=\"empty\">—</p>"
    parts: list[str] = []
    for item in items:
        body = _esc(item.summary)
        if item.why_it_matters:
            body = f'{_esc(item.summary)} <span class="meta-inline">· {_esc(labels["why_matters"])}: {_esc(item.why_it_matters)}</span>'
        parts.append(
            f'<li dir="auto"><strong>{_esc(item.title)}</strong> — {body}</li>'
        )
    return f"<ul>{''.join(parts)}</ul>"


def _render_technical(item: TechnicalCorner | None) -> str:
    if not item:
        return "<p class=\"empty\">—</p>"
    return (
        f'<ul><li dir="auto"><strong>{_esc(item.title)}</strong> — '
        f'{_esc(item.explanation)}</li></ul>'
    )


def _render_sources(report: ReportContent, labels: dict[str, str]) -> str:
    if not report.sources:
        return ""
    items = "".join(
        f'<li dir="auto"><a href="{_esc(s.url)}">{_esc(s.name)}</a></li>'
        for s in report.sources[:8]
    )
    return f"""
    <section class="block sources">
      <h2>{_esc(labels["sources"])}</h2>
      <ul>{items}</ul>
    </section>
    """


def _render_coverage_quality_section(report: ReportContent, labels: dict[str, str]) -> str:
    he = _is_hebrew_report()
    s = report.scrape_status
    reliability = s.reliability_label(hebrew=he)
    completeness = s.completeness_text(hebrew=he)
    transparency = s.transparency_text(hebrew=he)

    failed_block = ""
    if s.failed_source_list:
        failed_items = "".join(
            f"<li>{_esc(f['name'])} — {_esc(f['error'])}</li>"
            for f in s.failed_source_list[:12]
        )
        failed_block = f"""
        <div class="cq-failed">
          <strong>{_esc(labels["failed_sources_title"])}:</strong>
          <ul>{failed_items}</ul>
        </div>
        """

    return f"""
    <section class="coverage-quality">
      <h2>{_esc(labels["coverage_quality"])}</h2>
      <div class="cq-stats">
        <div>{_esc(labels["sources_scanned"])}: <strong>{s.total_sources}</strong></div>
        <div>{_esc(labels["sources_succeeded"])}: <strong>{s.successful_sources}</strong></div>
        <div>{_esc(labels["sources_failed"])}: <strong>{s.failed_source_count}</strong></div>
        <div>{_esc(labels["coverage_pct_label"])}: <strong>{s.coverage_percentage}%</strong></div>
        <div>{_esc(labels["total_items_week"])}: <strong>{s.total_articles_collected}</strong></div>
      </div>
      <p class="cq-reliability">{_esc(reliability)}</p>
      {failed_block}
      <p class="cq-completeness"><strong>{_esc(labels["completeness_title"])}:</strong> {_esc(completeness)}</p>
      <p class="cq-transparency">{_esc(transparency)}</p>
    </section>
    """


def _section(title: str, body_html: str, section_class: str = "block") -> str:
    return f"""
    <section class="{section_class}">
      <h2>{_esc(title)}</h2>
      <div class="body">{body_html}</div>
    </section>
    """


def build_plain_text_email(report: ReportContent) -> str:
    labels = _labels()
    lines = [
        f"{labels['title']} — {report.report_date}",
        f"{labels['items_collected']}: {report.items_collected}",
        "",
    ]
    lines.append(labels["executive"].upper())
    lines.extend(f"• {b}" for b in report.executive_summary)
    lines.append("")

    for title, items in [
        (labels["research"], report.models_research),
        (labels["products"], report.products_tools),
        (labels["business"], report.business_market),
    ]:
        lines.append(title.upper())
        for item in items:
            lines.append(f"• {item.title} — {item.summary}")
        lines.append("")

    if report.technical_corner:
        lines.append(labels["technical"].upper())
        lines.append(
            f"• {report.technical_corner.title} — {report.technical_corner.explanation}"
        )
        lines.append("")

    lines.append(labels["pm_takeaways"].upper())
    lines.extend(f"• {t}" for t in report.pm_takeaways)

    s = report.scrape_status
    lines.extend(
        [
            "",
            labels["coverage_quality"].upper(),
            f"{labels['sources_scanned']}: {s.total_sources}",
            f"{labels['sources_succeeded']}: {s.successful_sources}",
            f"{labels['sources_failed']}: {s.failed_source_count}",
            f"{labels['coverage_pct_label']}: {s.coverage_percentage}%",
            f"{labels['total_items_week']}: {s.total_articles_collected}",
            s.reliability_label(hebrew=_is_hebrew_report()),
        ]
    )
    if s.failed_source_list:
        lines.append("")
        lines.append(f"{labels['failed_sources_title']}:")
        for f in s.failed_source_list:
            lines.append(f"• {f['name']} — {f['error']}")
    lines.extend(
        [
            "",
            f"{labels['completeness_title']}: {s.completeness_text(hebrew=_is_hebrew_report())}",
            s.transparency_text(hebrew=_is_hebrew_report()),
        ]
    )
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
        _section(labels["technical"], _render_technical(report.technical_corner)),
        _section(
            labels["pm_takeaways"],
            _render_bullets(report.pm_takeaways),
            section_class="block takeaway",
        ),
    ]

    return f"""<!DOCTYPE html>
<html lang="{html_lang}" dir="{html_dir}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(labels["title"])} — {_esc(report.report_date)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1a1a2e;
      max-width: 720px;
      margin: 0 auto;
      padding: 24px 20px;
      background: #f8f9fc;
      direction: {html_dir};
      text-align: {"right" if he else "left"};
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,.06);
      padding: 28px 32px;
    }}
    h1, h2, h3 {{
      text-align: {"right" if he else "left"};
    }}
    h1 {{
      font-size: 1.45rem;
      margin: 0 0 8px;
      color: #0f3460;
    }}
    h2 {{
      font-size: 1.05rem;
      margin: 22px 0 10px;
      padding-bottom: 6px;
      border-bottom: 2px solid #e2e8f0;
      color: #16213e;
    }}
    .meta {{
      color: #64748b;
      font-size: 0.875rem;
      margin-bottom: 16px;
    }}
    ul, ol {{
      direction: {html_dir};
      text-align: {"right" if he else "left"};
      padding-right: {"24px" if he else "0"};
      padding-left: {"0" if he else "24px"};
      margin: 0 0 12px;
    }}
    li {{
      margin-bottom: 10px;
      line-height: 1.6;
    }}
    .meta-inline {{
      color: #64748b;
      font-size: 0.92em;
    }}
    .coverage-quality {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 16px 18px;
      margin-top: 24px;
      font-size: 0.875rem;
      color: #475569;
    }}
    .coverage-quality h2 {{
      font-size: 0.95rem;
      margin-top: 0;
      border-bottom: 1px solid #e2e8f0;
    }}
    .cq-stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 16px;
      margin: 12px 0;
    }}
    .cq-stats strong {{
      color: #0f3460;
    }}
    .cq-reliability {{
      font-size: 0.95rem;
      font-weight: 600;
      margin: 10px 0;
      color: #1e293b;
    }}
    .cq-failed {{
      margin: 10px 0;
      font-size: 0.82rem;
    }}
    .cq-failed ul {{
      margin-top: 6px;
      font-size: 0.82rem;
      color: #64748b;
    }}
    .cq-completeness, .cq-transparency {{
      margin: 8px 0 0;
      font-size: 0.82rem;
      line-height: 1.55;
      color: #64748b;
    }}
    .takeaway {{
      background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .sources a {{
      color: #2563eb;
      text-decoration: none;
    }}
    .empty {{
      color: #94a3b8;
    }}
    footer {{
      text-align: center;
      font-size: 0.75rem;
      color: #94a3b8;
      margin-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🤖 {_esc(labels["title"])}</h1>
    <p class="meta">{_esc(report.report_date)} · {_esc(labels["items_collected"])}: {report.items_collected}</p>
    {''.join(sections)}
    {_render_sources(report, labels)}
    {_render_coverage_quality_section(report, labels)}
  </div>
  <footer>{_esc(labels["footer"])}</footer>
</body>
</html>"""


def send_email(report: ReportContent) -> None:
    if not config.GMAIL_USER or not config.GMAIL_APP_PASSWORD:
        raise ValueError("GMAIL_USER and GMAIL_APP_PASSWORD must be set")

    html_body = build_html_email(report)
    plain_body = build_plain_text_email(report)
    subject = f"{config.EMAIL_SUBJECT_PREFIX} — {report.report_date}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.GMAIL_USER
    msg["To"] = config.EMAIL_RECIPIENT
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
        server.sendmail(config.GMAIL_USER, [config.EMAIL_RECIPIENT], msg.as_string())
