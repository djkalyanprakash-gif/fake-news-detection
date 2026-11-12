# Datasets

The training datasets are **not** included in this repository — together they are over 40 MB compressed, so they are kept out of the repo.

## What you need

1. **Fake and Real News Dataset** (Fake.csv + True.csv)
   Download from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. **The second Kaggle text-file collection** — 2,260 articles in plain text files (celebrity, fakeNewsDataset training, and testingSet folders). Download the zip from Kaggle and use it as-is.

## Setup

Put the two downloaded zip files in this folder, named:

```
data/archive_3.zip        # Fake.csv + True.csv
data/archive__1__.zip     # 2,260 .txt articles
```

Then run the training script from the repo root:

```bash
python train_model.py
```

The script reads the zips directly (no need to extract them), combines both datasets, de-duplicates, trains the model, and saves `model/fake_news_model.joblib` and `model/metrics.json`.

If no dataset is found in `data/`, training falls back to a small built-in sample set so it never crashes — but the resulting model is only a demo.
