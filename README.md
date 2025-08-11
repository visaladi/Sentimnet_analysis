# Sentiment Analysis with Bi-LSTM + Attention

> A simple Bi-LSTM + Attention model for binary and multi-class sentiment analysis, with example data loading and training scripts.

## 📋 Description

This repository provides:

- **Data loaders** for common sentiment datasets (IMDB, SST-2, Twitter US Airline Sentiment).
- **Model code** implementing a Bi-LSTM encoder + attention layer.
- **Training scripts** with configurable hyperparameters.
- **Evaluation notebooks** for accuracy, F1, and confusion matrices.

## 🗂️ Data

We include example scripts to download and preprocess the following datasets:

### 1. IMDB Movie Reviews

- 50,000 labeled reviews (25k train / 25k test)
- Sentiment labels: positive (1) / negative (0)
- Source: [Stanford IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

### 2. Stanford Sentiment Treebank (SST-2)

- Binary sentiment on movie sentences
- 67,349 phrases / 11,855 test instances
- Source: [SST-2](https://nlp.stanford.edu/sentiment/index.html)

### 3. Twitter US Airline Sentiment

- 14,640 tweets labeled positive, negative, or neutral
- Source: [Kaggle: Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

Preprocessed datasets will be saved under `data/{imdb,sst2,twitter}` as PyTorch `.pt` or NumPy `.npz` files.

### ⚙️ Requirements

Python 3.8+

PyTorch 1.10+ or TensorFlow 2.5+

torchtext / transformers (optional)

numpy, pandas, scikit-learn, matplotlib

Install dependencies:

pip install -r requirements.txt

### ▶️ Usage

Training on IMDB

python train.py \
  --data_dir data/imdb \
  --model bilstm_attention \
  --embed_dim 300 \
  --hidden_dim 128 \
  --batch_size 64 \
  --epochs 5 \
  --lr 1e-3

### Evaluation

python evaluate.py \
  --model_path checkpoints/imdb_model.pt \
  --test_data data/imdb/test.pt

### 📖 Citation

If you use this code, please cite:

Visal S. Adikari, “Sentiment Analysis with Bi-LSTM and Attention,” GitHub repository, 2025

### 🏗️ Architecture

```text
Input Text → Tokenizer → Embedding Layer (GloVe, FastText, or trainable)
                    ↓
             Bi-LSTM Encoder
                    ↓
              Attention Layer
                    ↓
         Dropout → Dense → Softmax
                    ↓
             Predicted Label

Embedding: 100–300 dimensions (pre-trained or trainable)

Bi-LSTM: Hidden size configurable (e.g., 128 units each direction)

Attention: Weighted sum of hidden states

Output: Softmax over 2–5 classes.



