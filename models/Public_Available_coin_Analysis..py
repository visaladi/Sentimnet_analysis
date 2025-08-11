# models/Public_Available_coin_Analysis.py
import os, io, base64, time, random
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

NITTER_MIRRORS = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.unixfox.eu",
    "https://nitter.lacontrevoie.fr",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)

def _extract_texts(soup: BeautifulSoup) -> List[str]:
    texts = []
    # common selectors across mirrors
    for sel in [".timeline .timeline-item .tweet-content",
                ".timeline-item .tweet-content",
                ".tweet-content"]:
        for node in soup.select(sel):
            s = node.get_text(" ", strip=True)
            if s:
                texts.append(s)
    return texts

def _next_cursor(soup: BeautifulSoup) -> str | None:
    more = soup.select_one("div.show-more a[href*='cursor=']")
    if not more or "href" not in more.attrs:
        return None
    import urllib.parse as up
    parsed = up.urlparse(more["href"])
    qd = up.parse_qs(parsed.query)
    vals = qd.get("cursor", [])
    return vals[0] if vals else None

def _scrape_nitter(query: str, limit: int) -> list[str]:
    texts, seen = [], set()
    session = requests.Session()
    session.headers.update(HEADERS)

    mirrors = NITTER_MIRRORS[:]  # copy and shuffle to distribute load
    random.shuffle(mirrors)

    for mirror in mirrors:
        cursor = None
        consecutive_empty = 0
        for _page in range(6):  # up to 6 pages per mirror
            params = {"f": "tweets", "q": query}
            if cursor:
                params["cursor"] = cursor
            try:
                r = session.get(f"{mirror}/search", params=params, timeout=15)
                if r.status_code in (429, 503):
                    time.sleep(2.0 + random.random())
                    continue
                r.raise_for_status()
            except Exception:
                # try next mirror
                break

            soup = BeautifulSoup(r.text, "html.parser")
            found = 0
            for s in _extract_texts(soup):
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                texts.append(s)
                found += 1
                if len(texts) >= limit:
                    return texts

            if found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            if consecutive_empty >= 2:
                # likely rate-limited/content blocked on this mirror/page
                break

            cursor = _next_cursor(soup)
            if not cursor:
                break

            time.sleep(0.8 + random.random() * 0.6)  # polite delay

        if len(texts) >= limit:
            break

    return texts

def _bar_image(pos_pct: float, neg_pct: float, pos: int, neg: int, total: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    pos_w = (pos / total) if total else 0; neg_w = (neg / total) if total else 0
    ax.barh([0], [pos_w], left=0, height=0.5)
    ax.barh([0], [neg_w], left=pos_w, height=0.5)
    ax.text(0, 0.85, f"{pos_pct:.2f}%", va="center", ha="left", fontsize=14)
    ax.text(1, 0.85, f"{neg_pct:.2f}%", va="center", ha="right", fontsize=14)
    ax.text(0, 1.35, "Community Mentions", va="center", ha="left", fontsize=16)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.8)
    buf = io.BytesIO(); plt.tight_layout(pad=1.0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def public_available_coin_search(query: str, max_results: int = 300) -> Dict[str, Any]:
    _ensure_vader()
    texts = _scrape_nitter(query, max_results)

    # sentiment
    sia = SentimentIntensityAnalyzer()
    pos = neg = 0
    for s in texts:
        c = sia.polarity_scores(s)["compound"]
        if c > 0.25: pos += 1
        elif c < -0.25: neg += 1
    total = len(texts)
    pos_pct = round(100 * pos / total, 2) if total else 0.0
    neg_pct = round(100 * neg / total, 2) if total else 0.0
    bar_b64 = _bar_image(pos_pct, neg_pct, pos, neg, total) if total else ""

    return {
        "query": query,
        "total_mentions": total,
        "positive": pos,
        "negative": neg,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "bar_image_base64": bar_b64,
        "sample_texts": texts[:10],
    }
