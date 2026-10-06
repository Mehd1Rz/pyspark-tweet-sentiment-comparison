# Sentiment Analysis on Sentiment140 (PySpark + Keras)

Binary sentiment classification (negative / positive) of tweets from the
[Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140),
comparing several feature/model combinations in one notebook.

## Dataset

Not included in this repo (1.6M tweets, ~230 MB). Download it from Kaggle:
<https://www.kaggle.com/datasets/kazanova/sentiment140>

The notebook works on a **2,000-tweet sample**, not the full dataset. To recreate it:

```bash
# after downloading + unzipping training.1600000.processed.noemoticon.csv
python prepare_data.py training.1600000.processed.noemoticon.csv
```

This writes `data/Sentiment_tweet4.csv`. The notebook was written for Colab and reads
`/content/Sentiment_tweet4.csv`, so either upload the file to `/content/` in Colab or
change the path in the data-loading cell to `data/Sentiment_tweet4.csv`.

## Pipeline

1. **Cleaning** (PySpark): lowercase, strip URLs and @mentions, remove punctuation
2. **Tokenizing** (spaCy): lemmatize, remove stop words (negations like "not"/"no" are kept)
3. **Features**: CountVectorizer or Word2Vec
4. **Models**: Spark MLP, Keras dense network, LSTM, CNN

## Results (2,000-tweet sample)

| Model | Features | Test acc. | Train acc. |
|---|---|---|---|
| Spark MLP | CountVectorizer | 64.7% | 99.8% |
| Spark MLP | Word2Vec | 56.1% | 54.8% |
| Keras dense | CountVectorizer | ~65% (val) | 98% |
| LSTM | Word2Vec | ~48-50% | ~50% |
| CNN | Word2Vec | ~48-50% | ~50% |

The CountVectorizer models overfit heavily on this small sample, and the LSTM/CNN
runs stay at chance level (see Known issues). A reference run on the full dataset
reaches ~79% accuracy: <https://www.kaggle.com/code/ashishjangra27/word-embedding-techniques-in-nlp>

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
PySpark also needs Java installed locally (Colab already has it).

## Known issues / TODO

- The "Model Evaluation on Data: 92.3%" for the dense model is measured on the training data, not held-out data. The validation accuracy (~65%) is the honest number.
- LSTM and CNN cells feed averaged Word2Vec vectors into an `Embedding` layer, which expects integer token ids, so they don't learn (about 50% accuracy). They need a `Tokenizer` + `pad_sequences` input instead.
- Some cells depend on imports and variables defined in later cells (`tf`, `Sequential`, `max_words`, `max_sequence_length`), so "Run all" on a fresh kernel will fail.
- Only 2,000 tweets are used, so results are noisy. Training on a larger sample would help.

## Data credit

Go, A., Bhayani, R., Huang, L. *Twitter Sentiment Classification using Distant Supervision* (2009).
