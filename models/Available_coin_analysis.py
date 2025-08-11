# models/Availble_coin_analysis.py
import os, io, base64, json, inspect
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from twikit import Client
from configurations import config

base_dir = os.path.dirname(__file__)
cookies_path = os.path.abspath(os.path.join(base_dir, "..", "configurations", "cookies.json"))

def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)

async def _maybe_await(x):
    # Await only if it's awaitable; otherwise just return the value.
    if inspect.isawaitable(x):
        return await x
    return x

async def _twikit_login_and_save(client: Client) -> None:
    email = os.getenv("TWIKIT_EMAIL", getattr(config, "TWITTER_EMAIL", None))
    username = os.getenv("TWIKIT_USERNAME", getattr(config, "TWITTER_USERNAME", None))
    password = os.getenv("TWIKIT_PASSWORD", getattr(config, "TWITTER_PASSWORD", None))
    if username and username.startswith("@"):
        username = username[1:]
    if not all([email, username, password]):
        raise RuntimeError("Twikit auth missing: no valid cookies and no credentials in env/config.")
    os.makedirs(os.path.dirname(cookies_path), exist_ok=True)
    await _maybe_await(client.login(auth_info_1=email, auth_info_2=username, password=password))
    await _maybe_await(client.save_cookies(cookies_path))

async def _load_or_login(client: Client) -> None:
    need_login = True
    if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        try:
            # validate JSON to avoid JSONDecodeError
            with open(cookies_path, "r", encoding="utf-8") as f:
                json.load(f)
            await _maybe_await(client.load_cookies(cookies_path))
            need_login = False
        except Exception:
            need_login = True
    if need_login:
        await _twikit_login_and_save(client)

async def available_coin_search(query: str, max_results: int = 300) -> Dict[str, Any]:
    _ensure_vader()

    client = Client("en-US")
    await _load_or_login(client)

    # --- fetch tweets (async or sync, depending on Twikit version) ---
    res = await _maybe_await(client.search_tweet(query=query, product="Latest"))

    texts, seen = [], set()

    async def handle_async_iter(it):
        nonlocal texts, seen
        async for t in it:
            s = (getattr(t, "text", "") or "").strip()
            if s and (s.lower() not in seen):
                seen.add(s.lower()); texts.append(s)
                if len(texts) >= max_results: break

    def handle_sync_iter(it):
        nonlocal texts, seen
        for t in (it or []):
            s = (getattr(t, "text", "") or "").strip()
            if s and (s.lower() not in seen):
                seen.add(s.lower()); texts.append(s)
                if len(texts) >= max_results: break

    if hasattr(res, "__aiter__"):
        await handle_async_iter(res)
    else:
        handle_sync_iter(res)

    # --- sentiment ---
    sia = SentimentIntensityAnalyzer()
    pos = neg = 0
    for s in texts:
        c = sia.polarity_scores(s)["compound"]
        if c > 0.25: pos += 1
        elif c < -0.25: neg += 1
    total = len(texts)
    pos_pct = round(100 * pos / total, 2) if total else 0.0
    neg_pct = round(100 * neg / total, 2) if total else 0.0

    # --- chart ---
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    pos_w = (pos / total) if total else 0; neg_w = (neg / total) if total else 0
    ax.barh([0], [pos_w], left=0, height=0.5)
    ax.barh([0], [neg_w], left=pos_w, height=0.5)
    ax.text(0, 0.85, f"{pos_pct:.2f}%", va="center", ha="left", fontsize=14)
    ax.text(1, 0.85, f"{neg_pct:.2f}%", va="center", ha="right", fontsize=14)
    ax.text(0, 1.35, "Community Mentions", va="center", ha="left", fontsize=16)
    ax.set_xlim(0,1); ax.set_ylim(-0.5,1.8)
    buf = io.BytesIO(); plt.tight_layout(pad=1.0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    bar_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

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