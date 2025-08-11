import json
from inference import get_sentiment


def compute_weighted_sentiment(texts, weights):
    sentiments = [get_sentiment(text) for text in texts]
    weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
    total_weight = sum(weights)
    overall_sentiment = weighted_sum / total_weight if total_weight != 0 else 0
    return overall_sentiment, sentiments


def load_preprocessed_data(file_path):
    with open(file_path, "r") as f:
        texts = json.load(f)
    return texts


def main():
    # Load preprocessed texts
    texts = load_preprocessed_data("../data/preprocessed_data.json")
    # Assign a default weight of 1 for each text (adjust as needed)
    weights = [1 for _ in texts]

    overall_sentiment, sentiments = compute_weighted_sentiment(texts, weights)
    print("Overall weighted sentiment score:", overall_sentiment)

    # Optionally, save the sentiment scores to a file
    with open("../data/sentiment_scores.json", "w") as f:
        json.dump({"overall": overall_sentiment, "scores": sentiments}, f, indent=2)


if __name__ == "__main__":
    main()
