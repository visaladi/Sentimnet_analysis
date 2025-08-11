import json
import os
import pandas as pd

from extrctor.tweets_extractor import fetch_discord_messages
from model_loader.berta_models import load_deberta_sentiment_model
from services.tweet_converter import run_preprocessing_general

base_dir = os.path.dirname(__file__)

def analyze_general_tweet_sentiment():
    # === Step 1: Define paths and channel type ===
    channel_type = "general"
    raw_json_file = os.path.join(base_dir, "..", "test data", "raw_general_messages.json")
    output_json_file = os.path.join(base_dir, "..", "test data", "sentiment_output_general.json")

    # === Step 2: Fetch Discord messages ===
    fetch_discord_messages(channel_type, raw_json_file)

    # === Step 3: Preprocess messages ===
    preprocessed_path = run_preprocessing_general(input_path=raw_json_file)

    # === Step 4: Load preprocessed texts ===
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    # === Step 5: Extract plain texts ===
    tweet_texts = []
    for tweet in tweets:
        if isinstance(tweet, dict) and 'text' in tweet:
            tweet_texts.append(tweet['text'])
        elif isinstance(tweet, str):
            tweet_texts.append(tweet)

    # === Step 6: Load sentiment model ===
    sentiment_pipeline = load_deberta_sentiment_model()

    # === Step 7: Run sentiment analysis ===
    labels = []
    for text in tweet_texts:
        result = sentiment_pipeline(text[:512])[0]  # DeBERTa limit
        labels.append(result['label'].upper())

    # === Step 8: Display and return results ===
    df = pd.DataFrame({
        'text': tweet_texts,
        'sentiment': labels
    })

    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
        print(df)

    # === (Optional) Save to JSON ===
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient='records'), f, indent=2, ensure_ascii=False)

    print(f"Sentiment analysis results saved to '{output_json_file}'")
