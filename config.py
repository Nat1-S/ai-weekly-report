"""Configuration loaded from environment variables (GitHub Secrets)."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# --- API keys & delivery ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL") or "claude-sonnet-4-20250514"

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT") or GMAIL_USER
EMAIL_SUBJECT_PREFIX = os.environ.get("EMAIL_SUBJECT_PREFIX") or "AI Weekly Report"

REPORT_LANGUAGE = os.environ.get("REPORT_LANGUAGE") or "hebrew"

# --- Scraping window ---
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))
LOCAL_TZ = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Jerusalem"))

# --- HTTP ---
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "25"))
ARXIV_TIMEOUT = int(os.environ.get("ARXIV_TIMEOUT", "60"))
USER_AGENT = os.environ.get("USER_AGENT", "AIWeeklyBriefBot/1.0")
HTTP_ACCEPT = "text/html,application/rss+xml,application/xml;q=0.9,*/*;q=0.8"
HTTP_ACCEPT_LANGUAGE = "en-US,en;q=0.9,he;q=0.8"
THROTTLE_DELAY_MIN = float(os.environ.get("THROTTLE_DELAY_MIN", "2"))
THROTTLE_DELAY_MAX = float(os.environ.get("THROTTLE_DELAY_MAX", "5"))
RATE_LIMIT_RETRY_SECONDS = int(os.environ.get("RATE_LIMIT_RETRY_SECONDS", "60"))
DOMAIN_THROTTLE_SECONDS = {
    "huggingface.co": float(os.environ.get("THROTTLE_HF_SECONDS", "10")),
    "venturebeat.com": float(os.environ.get("THROTTLE_VENTUREBEAT_SECONDS", "15")),
    "arxiv.org": float(os.environ.get("THROTTLE_ARXIV_SECONDS", "10")),
}
HTTP_CACHE_DIR = Path(os.environ.get("HTTP_CACHE_DIR", ".cache/http"))
HTTP_CACHE_ENABLED = os.environ.get("HTTP_CACHE_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", USER_AGENT)

# --- RSS feeds (free) ---
RSS_FEEDS = {
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "OpenAI News": "https://openai.com/blog/rss.xml",
    "Product Hunt": "https://www.producthunt.com/feed",
}

# HTML listing fallback when RSS fails (single page; no per-article fetches)
RSS_HTML_FALLBACKS = {
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/",
    "OpenAI News": "https://openai.com/news/",
}

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL")

REDDIT_SUBREDDITS = ("MachineLearning", "artificial", "OpenAI")
REDDIT_BASE = "https://www.reddit.com"

TWITTER_PROFILES = {
    "Sam Altman": "sama",
    "Andrej Karpathy": "karpathy",
    "Yann LeCun": "ylecun",
}

NITTER_RSS_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://xcancel.com",
]

MAX_ITEMS_PER_SOURCE = int(os.environ.get("MAX_ITEMS_PER_SOURCE", "12"))
MAX_TOTAL_ITEMS = int(os.environ.get("MAX_TOTAL_ITEMS", "120"))
MAX_ITEM_SUMMARY_CHARS = int(os.environ.get("MAX_ITEM_SUMMARY_CHARS", "400"))
