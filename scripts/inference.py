from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import torch

# Load tokenizer and ONNX model for sequence classification.
onnx_model_path = "../models/finetuned_sentiment_model_onnx"
tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
model = ORTModelForSequenceClassification.from_pretrained(
    onnx_model_path,
    provider="CUDAExecutionProvider"  # Use "CPUExecutionProvider" if no GPU is available.
)


def get_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run inference using the ONNX model
    outputs = model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()

    # For binary classification: assume label 1 is positive, label 0 is negative.
    pred = np.argmax(logits, axis=-1).item()
    sentiment = 1 if pred == 1 else -1  # Adjust mapping if using more classes.
    return sentiment


def run_inference():
    sample_text = "This is an amazing product!"
    sentiment = get_sentiment(sample_text)
    print("Sentiment:", sentiment)


if __name__ == "__main__":
    run_inference()
