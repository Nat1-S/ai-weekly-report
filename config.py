"""Configuration loaded from environment variables (GitHub Secrets)."""

import os
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

# --- API keys & delivery ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT", GMAIL_USER)
EMAIL_SUBJECT_PREFIX = os.environ.get("EMAIL_SUBJECT_PREFIX", "AI Weekly Report")

REPORT_LANGUAGE = os.environ.get("REPORT_LANGUAGE", "hebrew")

# --- Scraping window ---
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))
LOCAL_TZ = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Jerusalem"))

# --- HTTP ---
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "25"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "AI-Weekly-Report/1.0 (GitHub Actions; +https://github.com)",
)

# --- RSS feeds (free) ---
RSS_FEEDS = {
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "Hugging Face Papers (community RSS)": "https://jamesg.blog/hf-papers.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "The Batch (DeepLearning.AI)": "https://www.deeplearning.ai/the-batch/feed/",
    "The Rundown AI": "https://www.therundown.ai/feed",
    "OpenAI News": "https://openai.com/blog/rss.xml",
    "Anthropic Newsroom": "https://www.anthropic.com/rss.xml",
    "Product Hunt": "https://www.producthunt.com/feed",
}

# Hugging Face trending papers API (free, no auth)
HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"

# Hacker News (Algolia API — free)
HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"

# ArXiv (free)
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL")

# Reddit (JSON endpoints — no OAuth for read-only)
REDDIT_SUBREDDITS = ("MachineLearning", "artificial", "OpenAI")
REDDIT_BASE = "https://www.reddit.com"

# Twitter/X via public Nitter-style RSS mirrors (no official API)
TWITTER_PROFILES = {
    "Sam Altman": "sama",
    "Andrej Karpathy": "karpathy",
    "Yann LeCun": "ylecun",
}

# Fallback instances; first success wins per profile
NITTER_RSS_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://xcancel.com",
]

# Max items per source before summarization
MAX_ITEMS_PER_SOURCE = int(os.environ.get("MAX_ITEMS_PER_SOURCE", "12"))
MAX_TOTAL_ITEMS = int(os.environ.get("MAX_TOTAL_ITEMS", "120"))
MAX_ITEM_SUMMARY_CHARS = int(os.environ.get("MAX_ITEM_SUMMARY_CHARS", "400"))
