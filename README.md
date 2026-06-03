# AI Weekly Report

מערכת אוטומטית שסורקת מקורות AI חינמיים, מסכמת עם Claude, ושולחת דוח שבועי ביום חמישי במייל (HTML + גרסת טקסט).

## מבנה

| קובץ | תפקיד |
|------|--------|
| `scraper.py` | RSS, HN, ArXiv, Reddit, HF Papers, X (Nitter RSS) |
| `summarizer.py` | סיכום מובנה עם Claude |
| `sender.py` | Gmail SMTP |
| `main.py` | תזמור |
| `config.py` | הגדרות ו-URLs |

## הרצה מקומית

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="..."
export GMAIL_USER="you@gmail.com"
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
export EMAIL_RECIPIENT="you@gmail.com"  # אופציונלי

python main.py
```

## GitHub Actions

תזמון: `0 5 * * 4` — חמישי 08:00 שעון ישראל (UTC+3).

ניתן להריץ ידנית: **Actions → AI Weekly Report → Run workflow**.
