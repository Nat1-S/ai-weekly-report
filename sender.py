"""Deliver report via Gmail SMTP."""

from __future__ import annotations

import html
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from summarizer import ReportContent


def _text_to_html(text: str) -> str:
    escaped = html.escape(text.strip())
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", escaped) if p.strip()]
    if not paragraphs:
        return "<p>—</p>"
    parts: list[str] = []
    for p in paragraphs:
        if p.startswith(("- ", "• ", "* ")):
            items = re.split(r"\n(?=[-*•]\s)", p)
            lis = "".join(f"<li>{item.lstrip('-•* ')}</li>" for item in items if item.strip())
            parts.append(f"<ul>{lis}</ul>")
        else:
            lines = p.replace("\n", "<br>")
            parts.append(f"<p>{lines}</p>")
    return "\n".join(parts)


def build_plain_text_email(report: ReportContent) -> str:
    lines = [
        f"AI Weekly Intelligence Report — {report.report_date}",
        f"Sources scanned: {report.sources_used}",
        "",
    ]
    for _id, title, body in report.sections():
        lines.append(title.upper())
        lines.append("-" * len(title))
        lines.append(body.strip())
        lines.append("")
    if report.scrape_errors:
        lines.append("SCRAPE NOTES")
        lines.extend(f"- {e}" for e in report.scrape_errors[:8])
    return "\n".join(lines).strip()


def build_html_email(report: ReportContent) -> str:
    sections_html = []
    for _id, title, body in report.sections():
        sections_html.append(
            f"""
            <section class="block" id="{_id}">
              <h2>{html.escape(title)}</h2>
              <div class="body">{_text_to_html(body)}</div>
            </section>
            """
        )

    warnings = ""
    if report.scrape_errors:
        warn_items = "".join(f"<li>{html.escape(e)}</li>" for e in report.scrape_errors[:8])
        warnings = f'<div class="warn"><strong>Scrape notes:</strong><ul>{warn_items}</ul></div>'

    return f"""<!DOCTYPE html>
<html lang="he" dir="auto">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Weekly Report — {html.escape(report.report_date)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.55;
      color: #1a1a2e;
      max-width: 720px;
      margin: 0 auto;
      padding: 24px 20px;
      background: #f8f9fc;
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,.06);
      padding: 28px 32px;
    }}
    h1 {{
      font-size: 1.5rem;
      margin: 0 0 4px;
      color: #0f3460;
    }}
    .meta {{
      color: #64748b;
      font-size: 0.875rem;
      margin-bottom: 24px;
    }}
    h2 {{
      font-size: 1.05rem;
      margin: 22px 0 8px;
      padding-bottom: 6px;
      border-bottom: 2px solid #e2e8f0;
      color: #16213e;
    }}
    .body p, .body ul {{ margin: 0 0 10px; }}
    .body ul {{ padding-left: 1.25rem; }}
    .takeaway {{
      background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
      border-radius: 8px;
      padding: 14px 16px;
      margin-top: 8px;
    }}
    .warn {{
      font-size: 0.8rem;
      color: #92400e;
      background: #fffbeb;
      border-radius: 8px;
      padding: 10px 14px;
      margin-top: 20px;
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
    <h1>🤖 AI Weekly Intelligence Report</h1>
    <p class="meta">{html.escape(report.report_date)} · {report.sources_used} sources scanned</p>
    {''.join(sections_html[:-1])}
    <section class="block takeaway" id="takeaway">
      <h2>Key Takeaway</h2>
      <div class="body">{_text_to_html(report.key_takeaway)}</div>
    </section>
    {warnings}
  </div>
  <footer>Automated report · GitHub Actions</footer>
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
