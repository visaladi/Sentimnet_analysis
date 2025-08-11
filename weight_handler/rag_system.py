# models/rag_system.py
import os, json, math, time
from typing import Dict, Any, List, Tuple
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test data"))

# ---- Expected files written by your pipelines (no changes to them) ----
PATHS = {
    # from models/News_handler.py
    "news_sentiment": os.path.join(DATA_DIR, "sentiment_output_for_news.json"),
    # from models/general_handler.py
    "general_sentiment": os.path.join(DATA_DIR, "sentiment_output_general.json"),
    # from models/coinflow_With_sentiment.py
    "focus_sentiment": os.path.join(DATA_DIR, "sentiment_output_for_coin_finder.json"),
    # from models/coinflow_Analysis.py
    "coin_flow": os.path.join(DATA_DIR, "Analysis_output_for_coin_flow.json"),
    # from models/coin_finder.py
    "coin_finder": os.path.join(DATA_DIR, "coin_keywords_extracted.json"),
    # optional: verified group result
    "verified_focus": os.path.join(DATA_DIR, "verified_sentiment_output_focus_group.json"),
    # optional cache from /coin-sentiment (Twikit) if you decide to save it
    "twitter_cache": os.path.join(DATA_DIR, "twitter_sentiment_cache.json"),
}

# In-memory index
_RAG_INDEX: Dict[str, Dict[str, Any]] = {}
_RAG_TS: float = 0.0

def _safe_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _norm(values: Dict[str, float]) -> Dict[str, float]:
    """Simple z-score like normalization (mean/std) with safety for 0 std."""
    if not values:
        return {}
    xs = list(values.values())
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return {k: (v - mean) / std for k, v in values.items()}

def _add(d: Dict[str, float], k: str, v: float):
    d[k] = d.get(k, 0.0) + v

def _push(lst_map: Dict[str, List[float]], k: str, v: float):
    lst_map.setdefault(k, []).append(v)

def _to_pct(pos: int, neg: int) -> Tuple[float, float]:
    total = max(1, pos + neg)  # ignore neutrals for pct
    return round(100*pos/total, 2), round(100*neg/total, 2)

def build_rag_index(weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Collect outputs from all pipelines, build per-coin profiles, normalize numeric features,
    compute composite scores with weights, and cache in memory.
    """
    global _RAG_INDEX, _RAG_TS
    weights = weights or {
        # Feature weights (tune freely)
        "news_sent": 0.25,           # avg FinBERT label: POS=+1, NEG=-1, NEU=0 over last N news items about coin (if present)
        "general_sent": 0.15,        # from general_handler (POS/NEG ratio)
        "focus_sent": 0.20,          # from focus pipeline avg sentiment per coin
        "flow": 0.25,                # z-scored aggregated net flow
        "mentions": 0.10,            # z-scored mentions from coin_finder keywords
        "twitter_sent": 0.05,        # cached /coin-sentiment if you decide to save it
    }

    profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "news_sent": None,
        "general_sent": None,
        "focus_sent": None,
        "flow": 0.0,
        "mentions": 0.0,
        "twitter_pos": 0,
        "twitter_neg": 0,
        "sources": [],
    })

    # ===== 1) Coin Flow (numeric) =====
    coin_flow = _safe_load(PATHS["coin_flow"])
    if coin_flow and isinstance(coin_flow, dict):
        # aggregated_flows: { "BTC": 12345, ... }
        agg = coin_flow.get("aggregated_flows", {})
        for coin, val in agg.items():
            profiles[coin]["flow"] = profiles[coin].get("flow", 0.0) + float(val)
            profiles[coin]["sources"].append("coin_flow")

    # ===== 2) Focus Sentiment per coin (numeric) =====
    focus_sent = _safe_load(PATHS["focus_sentiment"])
    if focus_sent and isinstance(focus_sent, dict):
        avg = focus_sent.get("average_sentiment", {})
        for coin, s in avg.items():
            profiles[coin]["focus_sent"] = float(s)
            profiles[coin]["sources"].append("focus_sentiment")

    # ===== 3) Coin Finder Mentions (numeric volume by unique keywords) =====
    cf = _safe_load(PATHS["coin_finder"])
    if cf and isinstance(cf, dict):
        ckf = cf.get("coin_keywords_filtered", {})
        # ckf is a dict keyed by some id-> {keyword:count,...}; we’ll count mentions of known coin-like words
        known_coins = {
            "Bitcoin","BTC","Ethereum","ETH","XRP","Solana","SOL","Ondo","Cronos","CRO","Binance","BNB",
            "Picoin","Cardano","ADA","Litecoin","LTC","XVG"
        }
        # accumulate keyword counts for coin-like tokens
        mention_scores: Dict[str, float] = defaultdict(float)
        for _, kw_counts in ckf.items():
            if not isinstance(kw_counts, dict):
                continue
            for word, cnt in kw_counts.items():
                w = (word or "").strip()
                if not w:
                    continue
                if w in known_coins or w.upper() in known_coins:
                    # Try map to canonical tickers/names
                    key = w.upper()
                    mention_scores[key] += float(cnt)

        # Push into profiles
        for coin, m in mention_scores.items():
            profiles[coin]["mentions"] = profiles[coin].get("mentions", 0.0) + m
            profiles[coin]["sources"].append("coin_finder")

    # ===== 4) General Sentiment (treat POS=+1, NEG=-1 average over last items) =====
    gen = _safe_load(PATHS["general_sentiment"])
    # general handler saves list of {text, sentiment}
    if isinstance(gen, list):
        # crude coin tag: look for $TICKER patterns and major names
        COIN_TAGS = ["$BTC","$ETH","$XRP","$SOL","$ADA","Bitcoin","Ethereum","XRP","Solana","Cardano"]
        label_map = {"POSITIVE":1, "NEGATIVE":-1, "NEUTRAL":0}
        per_coin_scores = defaultdict(list)
        for item in gen:
            text = (item.get("text") or "").lower()
            label = (item.get("sentiment") or "NEUTRAL").upper()
            score = label_map.get(label, 0)
            for tag in COIN_TAGS:
                if tag.lower() in text:
                    per_coin_scores[tag.upper()].append(score)
        for coin, ss in per_coin_scores.items():
            if ss:
                profiles[coin]["general_sent"] = sum(ss)/len(ss)
                profiles[coin]["sources"].append("general_sentiment")

    # ===== 5) News Sentiment (FinBERT) -> same mapping =====
    news = _safe_load(PATHS["news_sentiment"])
    if isinstance(news, list):
        label_map = {"POSITIVE":1, "NEGATIVE":-1, "NEUTRAL":0}
        COIN_TAGS = ["$BTC","$ETH","$XRP","$SOL","$ADA","Bitcoin","Ethereum","XRP","Solana","Cardano"]
        per_coin_scores = defaultdict(list)
        for item in news:
            text = (item.get("text") or "").lower()
            label = (item.get("dominant_sentiment") or "NEUTRAL").upper()
            score = label_map.get(label, 0)
            for tag in COIN_TAGS:
                if tag.lower() in text:
                    per_coin_scores[tag.upper()].append(score)
        for coin, ss in per_coin_scores.items():
            if ss:
                profiles[coin]["news_sent"] = sum(ss)/len(ss)
                profiles[coin]["sources"].append("news_sentiment")

    # ===== 6) Optional: cached twitter sentiment from /coin-sentiment =====
    tw = _safe_load(PATHS["twitter_cache"])
    # format assumed: [{query, positive, negative, ...}, ...]
    if isinstance(tw, list):
        for row in tw:
            q = (row.get("query") or "").upper()
            pos = int(row.get("positive", 0))
            neg = int(row.get("negative", 0))
            if q:
                profiles[q]["twitter_pos"] += pos
                profiles[q]["twitter_neg"] += neg
                profiles[q]["sources"].append("twitter_sentiment")

    # ===== Normalize numeric fields for ranking =====
    flow_map = {c: v["flow"] for c, v in profiles.items()}
    mentions_map = {c: v["mentions"] for c, v in profiles.items()}
    flow_z = _norm(flow_map)
    mentions_z = _norm(mentions_map)

    # compute twitter % -> sentiment in [-1,1]
    for coin, v in profiles.items():
        pos, neg = v["twitter_pos"], v["twitter_neg"]
        tw_total = pos + neg
        if tw_total > 0:
            tw_sent = (pos - neg) / tw_total
        else:
            tw_sent = None
        v["twitter_sent"] = tw_sent

    # ===== Final score =====
    for coin, v in profiles.items():
        score = 0.0
        detail = {}
        # Fill missing as 0 for sentiment-like
        ns = v["news_sent"] if v["news_sent"] is not None else 0.0
        gs = v["general_sent"] if v["general_sent"] is not None else 0.0
        fs = v["focus_sent"] if v["focus_sent"] is not None else 0.0
        tws = v["twitter_sent"] if v["twitter_sent"] is not None else 0.0
        fl = flow_z.get(coin, 0.0)
        mn = mentions_z.get(coin, 0.0)

        score += weights["news_sent"]   * ns;   detail["news_sent"] = ns
        score += weights["general_sent"]* gs;   detail["general_sent"] = gs
        score += weights["focus_sent"]  * fs;   detail["focus_sent"] = fs
        score += weights["flow"]        * fl;   detail["flow_z"] = fl
        score += weights["mentions"]    * mn;   detail["mentions_z"] = mn
        score += weights["twitter_sent"]* tws;  detail["twitter_sent"] = tws

        v["score"] = round(score, 4)
        v["score_breakdown"] = detail

    _RAG_INDEX = dict(sorted(profiles.items(), key=lambda kv: kv[1]["score"], reverse=True))
    _RAG_TS = time.time()
    return {"coins_indexed": len(_RAG_INDEX), "updated_at": _RAG_TS}

def rag_top(top_k: int = 10) -> List[Dict[str, Any]]:
    items = list(_RAG_INDEX.items())[:max(1, top_k)]
    out = []
    for coin, v in items:
        out.append({"coin": coin, "score": v["score"], **v})
    return out

def rag_explain(coin: str) -> Dict[str, Any]:
    c = coin.upper()
    v = _RAG_INDEX.get(c)
    if not v:
        # fuzzy fallback: try coin name like "BITCOIN" -> "$BTC" if present
        for k in _RAG_INDEX.keys():
            if c in k or k in c:
                v = _RAG_INDEX[k]
                c = k
                break
    if not v:
        return {"coin": coin, "found": False}
    # human-readable explanation
    ns = v["score_breakdown"].get("news_sent", 0.0)
    gs = v["score_breakdown"].get("general_sent", 0.0)
    fs = v["score_breakdown"].get("focus_sent", 0.0)
    fl = v["score_breakdown"].get("flow_z", 0.0)
    mn = v["score_breakdown"].get("mentions_z", 0.0)
    tw = v["score_breakdown"].get("twitter_sent", 0.0)
    why = []
    if ns: why.append(f"news_sent={ns:+.2f}")
    if gs: why.append(f"general_sent={gs:+.2f}")
    if fs: why.append(f"focus_sent={fs:+.2f}")
    if fl: why.append(f"flow_z={fl:+.2f}")
    if mn: why.append(f"mentions_z={mn:+.2f}")
    if tw: why.append(f"twitter_sent={tw:+.2f}")
    return {
        "coin": c,
        "found": True,
        "score": v["score"],
        "why": ", ".join(why) or "no strong signals",
        "sources": sorted(set(v.get("sources", []))),
        "raw": v
    }
