import json
import pandas as pd
from extrctor.tweets_extractor import fetch_discord_messages
from model_loader.berta_models import load_finbert_sentiment_model
from services.tweet_converter import run_preprocessing_news
import os
base_dir = os.path.dirname(__file__)

def analyze_discord_news_sentiment():
    channel_type = "news"
    raw_json_file = os.path.join(base_dir, "..", "test data", "raw_news_messages.json")
    output_json_file= os.path.join(base_dir, "..", "test data","sentiment_output_for_news.json")
    # Step 1: Fetch Discord messages and save raw JSON
    fetch_discord_messages(channel_type, raw_json_file)

    # Step 2: Preprocess the fetched messages (input_path = raw_json_file)
    preprocessed_path = run_preprocessing_news(input_path=raw_json_file)

    # Step 3: Load cleaned/preprocessed news texts
    with open(preprocessed_path, "r", encoding='utf-8') as f:
        news_texts = json.load(f)

    # Step 4: Load FinBERT model
    sentiment_model = load_finbert_sentiment_model()

    # Step 5: Run sentiment analysis
    results = []
    for text in news_texts:
        trimmed = text[:512]
        result = sentiment_model(trimmed)[0]
        sentiment_label = result["label"].upper()
        results.append({
            "text": text,
            "dominant_sentiment": sentiment_label
        })

    # Step 6: Display and save to JSON
    df = pd.DataFrame(results)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
        print(df)

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Sentiment analysis results saved to '{output_json_file}'")
