
#!/usr/bin/env python3
"""
sentiment_analysis.py

Fetch and analyze sentiment from Twitter, Reddit, and 4chan for a given symbol,
then plot pie charts of the positive/negative/neutral distribution.
"""

from dotenv import load_dotenv
load_dotenv()   # load variables from .env

import os
import logging
import requests
import html
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import tweepy
from tweepy.errors import TooManyRequests
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Initialize Twitter client ────────────────────────────────────────────────
_twitter_bearer = os.getenv("TWITTER_BEARER_TOKEN")
if _twitter_bearer:
    twitter_client = tweepy.Client(
        bearer_token=_twitter_bearer,
        wait_on_rate_limit=False  # disable auto-sleep
    )
    logger.info("Twitter client initialized.")
else:
    twitter_client = None
    logger.warning("TWITTER_BEARER_TOKEN not set; Twitter fetch disabled.")

# ─── Initialize Reddit client ─────────────────────────────────────────────────
_reddit_id     = os.getenv("REDDIT_CLIENT_ID")
_reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
_reddit_agent  = os.getenv("REDDIT_USER_AGENT")
if _reddit_id and _reddit_secret and _reddit_agent:
    reddit = praw.Reddit(
        client_id=_reddit_id,
        client_secret=_reddit_secret,
        user_agent=_reddit_agent
    )
    logger.info("Reddit client initialized.")
else:
    reddit = None
    logger.warning("Reddit credentials missing; Reddit fetch disabled.")

# ─── Sentiment analyzer ────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()


def fetch_twitter_texts(query: str, max_tweets: int = 50) -> List[str]:
    """
    Fetch recent English-language tweets matching `query` (excluding retweets).
    Returns a list of tweet texts, or [] if disabled or rate-limited.
    """
    if not twitter_client:
        return []
    try:
        resp = twitter_client.search_recent_tweets(
            query=f"{query} -is:retweet lang:en",
            max_results=max_tweets,
            tweet_fields=["text"]
        )
        return [t.text for t in (resp.data or [])]
    except TooManyRequests:
        logger.warning("Twitter rate limit reached—skipping tweets.")
        return []
    except Exception as e:
        logger.warning(f"Twitter fetch error: {e}")
        return []


def fetch_reddit_texts(
    subreddit_name: str,
    query: str,
    max_posts: int = 20,
    max_comments_per_post: int = 10
) -> List[str]:
    """
    Search r/<subreddit_name> for `query`. Return post titles, bodies, and top comments.
    """
    if not reddit:
        return []
    texts: List[str] = []
    try:
        sub = reddit.subreddit(subreddit_name)
        for submission in sub.search(query, limit=max_posts):
            texts.append(submission.title)
            if submission.selftext:
                texts.append(submission.selftext)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:max_comments_per_post]:
                texts.append(comment.body)
    except Exception as e:
        logger.warning(f"Reddit fetch error: {e}")
    return texts


def fetch_4chan_texts(
    board: str = "biz",
    max_threads: int = 3,
    posts_per_thread: int = 30
) -> List[str]:
    """
    Fetch the first `max_threads` threads from /{board}/catalog.json
    and return up to `posts_per_thread` HTML-decoded comment texts.
    """
    texts: List[str] = []
    try:
        catalog = requests.get(f"https://a.4cdn.org/{board}/catalog.json").json()
        thread_ids = []
        for page in catalog:
            for th in page["threads"]:
                thread_ids.append(th["no"])
                if len(thread_ids) >= max_threads:
                    break
            if len(thread_ids) >= max_threads:
                break

        for tid in thread_ids:
            thread = requests.get(f"https://a.4cdn.org/{board}/thread/{tid}.json").json()
            for post in thread.get("posts", [])[:posts_per_thread]:
                comment = post.get("com", "")
                if comment:
                    texts.append(html.unescape(comment))
    except Exception as e:
        logger.warning(f"4chan fetch error: {e}")
    return texts


def analyze_sentiments(texts: List[str]) -> Dict:
    """
    Run VADER sentiment on each text. Return:
      - avg_compound: average compound score
      - counts: dict with 'pos', 'neg', 'neu'
      - label: "Positive"/"Negative"/"Neutral"
    """
    scores = [sia.polarity_scores(t)["compound"] for t in texts]
    if not scores:
        return {"avg_compound": 0.0, "counts": {"pos":0,"neg":0,"neu":0}, "label":"Neutral"}

    pos = sum(1 for s in scores if s >=  0.05)
    neg = sum(1 for s in scores if s <= -0.05)
    neu = len(scores) - pos - neg
    avg = sum(scores) / len(scores)

    if avg >  0.05:
        label = "Positive"
    elif avg < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"avg_compound":avg, "counts":{"pos":pos,"neg":neg,"neu":neu}, "label":label}

def get_multi_social_sentiment(symbol: str) -> Dict[str, Dict]:
    # fetch each platform’s texts
    tw_texts = fetch_twitter_texts(symbol)
    rd_texts = fetch_reddit_texts("stocks", symbol)
    ch_texts = fetch_4chan_texts("biz")
    # build per-platform sentiment dicts
    tw_sent = analyze_sentiments(tw_texts)
    rd_sent = analyze_sentiments(rd_texts)
    ch_sent = analyze_sentiments(ch_texts)
    # overall across all texts
    all_sent = analyze_sentiments(tw_texts + rd_texts + ch_texts)

    return {
        "twitter": tw_sent,
        "reddit":  rd_sent,
        "4chan":   ch_sent,
        "overall": all_sent      # ← now this exists
    }





def plot_sentiment_pie(sent: Dict, platform: str) -> plt.Figure:
    """
    Given a sentiment dict with 'counts',
    produce a pie-chart Figure. If all counts are zero or NaN,
    shows a 'No data' placeholder.
    """
    counts = sent.get("counts", {})
    # Pull out and replace any NaN with 0
    pos = np.nan_to_num(counts.get("pos", 0))
    neg = np.nan_to_num(counts.get("neg", 0))
    neu = np.nan_to_num(counts.get("neu", 0))

    sizes = [pos, neg, neu]
    total = sum(sizes)

    fig, ax = plt.subplots()
    if total <= 0:
        # Nothing to show
        ax.text(
            0.5, 0.5, "No data",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            color="gray"
        )
        ax.set_title(f"{platform} Sentiment Distribution")
        ax.axis("off")
        return fig

    labels = ["Positive", "Negative", "Neutral"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"{platform} Sentiment Distribution")
    ax.axis("equal")
    return fig


def make_recommendation(avg_compound: float):
    """ Simple rule: positive => RISE/BUY, negative or zero => FALL/SELL """
    if avg_compound > 0:
        return "RISE", "BUY"
    else:
        return "FALL", "SELL"



