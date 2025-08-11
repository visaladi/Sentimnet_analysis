from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os


def export_model(model_path, onnx_model_path):
    # Load the fine-tuned model and tokenizer from Transformers format,
    # then load the model into an ONNX Runtime wrapper (this converts it)
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path, from_transformers=True)
    # Save the ONNX model to the specified directory
    ort_model.save_pretrained(onnx_model_path)

    # Also save the tokenizer for inference
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(onnx_model_path)

    print(f"Model exported to ONNX format at {onnx_model_path}")


if __name__ == "__main__":
    # Make sure the fine-tuned model was saved using deberv3 as the base model.
    model_path = os.path.join("..", "models", "finetuned_sentiment_model")
    onnx_model_path = os.path.join("..", "models", "finetuned_sentiment_model_onnx")
    export_model(model_path, onnx_model_path)
