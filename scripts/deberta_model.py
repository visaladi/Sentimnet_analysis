import json
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- Step 1: Load the Tweets ---

# Absolute path to your tweets JSON file.
tweets_file = r"C:\Users\visal Adikari\OneDrive\Desktop\Uni Sem 7\sentiment alaysis\data\preprocessed_data1.json"

with open(tweets_file, 'r', encoding='utf-8') as f:
    tweets = json.load(f)

print("Loaded", len(tweets), "tweets successfully.")

# --- Step 2: Extract Coin Flow Data ---
# Define a regex pattern that extracts:
#   1. The coin name (following a '$')
#   2. The sign of the price change ([+-])
#   3. The numeric value with an optional K (thousands) or M (millions) suffix.
flow_pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'

# Dictionaries to store detailed coin flows and aggregated texts for sentiment.
coin_price_data = {}
coin_texts = {}

# Iterate through each tweet.
# (Assuming each tweet is a JSON object. For coin flows, we search the tweet string.
#  For sentiment analysis, we search within tweet embeds.)
for tweet in tweets:
    # Extract coin flow information using the regex (from the tweet content if available).
    # If your tweets are strings directly, use tweet. If they are objects, adjust accordingly.
    # Here, we try to extract from the entire tweet (converted to string).
    tweet_str = json.dumps(tweet)
    flow_matches = re.findall(flow_pattern, tweet_str)
    for coin, sign, value_str in flow_matches:
        # Determine multiplier based on suffix.
        multiplier = 1
        if value_str.endswith('K'):
            multiplier = 1_000
            numeric_part = value_str[:-1]
        elif value_str.endswith('M'):
            multiplier = 1_000_000
            numeric_part = value_str[:-1]
        else:
            numeric_part = value_str
        try:
            numeric_value = float(numeric_part) * multiplier
        except ValueError:
            continue  # Skip if conversion fails.
        # Apply the sign.
        numeric_value = numeric_value if sign == '+' else -numeric_value
        # Append value to the coin's list.
        coin_price_data.setdefault(coin, []).append(numeric_value)

    # Extract textual information from tweet embeds (if any).
    if 'embeds' in tweet and tweet['embeds']:
        for embed in tweet['embeds']:
            text_parts = []
            if 'title' in embed and embed['title']:
                text_parts.append(embed['title'])
            if 'description' in embed and embed['description']:
                text_parts.append(embed['description'])
            combined_text = " ".join(text_parts)
            # Use regex to find coin symbols in the combined text.
            coins_found = re.findall(r'\$(\w+)', combined_text)
            for coin in coins_found:
                # Aggregate text per coin.
                if coin in coin_texts:
                    coin_texts[coin] += " " + combined_text
                else:
                    coin_texts[coin] = combined_text

# Aggregate the coin flows (e.g., sum of values for each coin).
aggregated_prices = {coin: sum(values) for coin, values in coin_price_data.items()}

# --- Step 3: Run Sentiment Analysis on Aggregated Text per Coin ---

# Load the DeBERTa V3 tokenizer and model.
# We disable the fast tokenizer to avoid conversion issues.
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Create a dictionary to store sentiment scores.
coin_sentiment_scores = {}
for coin, text in coin_texts.items():
    result = sentiment_analyzer(text)
    label = result[0]['label'].upper()
    score = result[0]['score']
    # Convert the sentiment to a numeric score: positive as +score, negative as -score.
    sentiment_score = score if label == "POSITIVE" else -score
    coin_sentiment_scores[coin] = sentiment_score

# --- Step 4: Combine and Save the Output Data ---

output_data = {
    "detailed_flows": coin_price_data,
    "aggregated_flows": aggregated_prices,
    "coin_sentiment_scores": coin_sentiment_scores
}

# Save the output data to a JSON file in your data folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, '..', 'data', 'coin_combined_data.json')
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(output_data, outfile, indent=4)

print("Combined coin data (flows and sentiment) saved to", output_file)
