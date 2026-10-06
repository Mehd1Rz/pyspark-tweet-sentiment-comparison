"""Build Sentiment_tweet4.csv (the file the notebook reads) from the raw Sentiment140 CSV.

Usage:
    python prepare_data.py path/to/training.1600000.processed.noemoticon.csv

Output: data/Sentiment_tweet4.csv  (2000-row balanced sample, same column layout the notebook expects)
"""
import sys
from pathlib import Path

import pandas as pd

N_PER_CLASS = 1000
SEED = 42

raw_path = sys.argv[1] if len(sys.argv) > 1 else "training.1600000.processed.noemoticon.csv"

# The raw file has no header and is latin-1 encoded.
cols = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv(raw_path, encoding="latin-1", header=None, names=cols)

# target: 0 = negative, 4 = positive
sample = (
    df.groupby("target", group_keys=False)
      .apply(lambda g: g.sample(N_PER_CLASS, random_state=SEED))
      .sample(frac=1, random_state=SEED)   # shuffle
      .reset_index(drop=True)
)

Path("data").mkdir(exist_ok=True)
sample.to_csv("data/Sentiment_tweet4.csv", index=True)  # index becomes the "_c0" column
print(sample.shape, sample["target"].value_counts().to_dict())
